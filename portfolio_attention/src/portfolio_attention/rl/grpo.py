"""GRPO-like policy-step helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Dirichlet

from ..common.net_return import apply_transaction_cost_to_returns
from ..common.win_rate import compute_win_rate_metrics
from ..config import EvaluationConfig, ModelConfig, TrainConfig
from ..model.allocation_distribution import logits_to_rl_post_train_dirichlet_alpha
from ..model.reward import (
    apply_turnover_reward_penalty,
    compute_dsr_day_reward,
    compute_dsr_warmup_stats,
    compute_return_reward,
    compute_rolling_sharpe_reward,
)
from . import algorithms as rl_algorithms


@dataclass(frozen=True)
class GRPOPolicyStepResult:
    policy_loss: torch.Tensor
    metrics: dict[str, torch.Tensor | float]
    summary: dict[str, torch.Tensor]
    batch_size: int


def run_grpo_like_policy_step_from_scored_tensors(
    *,
    scored_logits: torch.Tensor,
    scored_r_stock: torch.Tensor,
    scored_returns: torch.Tensor,
    scored_turnover: torch.Tensor,
    scored_previous_allocation: torch.Tensor,
    allocation_logits: torch.Tensor,
    raw_allocation: torch.Tensor,
    horizon_days: int,
    model_config: ModelConfig,
    train_config: TrainConfig,
    evaluation_config: EvaluationConfig,
) -> GRPOPolicyStepResult:
    alpha = logits_to_rl_post_train_dirichlet_alpha(
        scored_logits[:, -1, :],
        alpha_min=float(train_config.rl_training.alpha_min),
        alpha_max=float(train_config.rl_training.alpha_max),
        logit_scale=float(model_config.dirichlet_logit_scale),
        evidence_scale=float(train_config.rl_training.rl_post_train_evidence_scale),
    )
    dist = Dirichlet(alpha)
    group_size = int(train_config.rl_training.group_size)
    sampled_actions = dist.sample((group_size,))
    sampled_log_probs = dist.log_prob(sampled_actions)
    entropy = dist.entropy()

    with torch.no_grad():
        net_scored_returns = apply_transaction_cost_to_returns(
            scored_returns,
            scored_turnover,
            transaction_cost_rate=float(train_config.transaction_cost_rate),
        )
        sampled_stock_weights = sampled_actions[..., :-1]
        reward_stock_returns = scored_r_stock[:, -1, :].unsqueeze(0)
        gross_reward_return = (sampled_stock_weights * reward_stock_returns).sum(dim=-1)

        reward_previous_allocation = scored_previous_allocation[:, -1, :].unsqueeze(0)
        sampled_turnover = 0.5 * torch.abs(sampled_actions - reward_previous_allocation).sum(dim=-1)
        sampled_net_reward_return = apply_transaction_cost_to_returns(
            gross_reward_return,
            sampled_turnover,
            transaction_cost_rate=float(train_config.transaction_cost_rate),
        )

        warmup_net_returns = net_scored_returns[:, : horizon_days - 1]
        warmup_path = warmup_net_returns.unsqueeze(0).expand(group_size, -1, -1)
        sampled_prediction_returns = torch.cat(
            (warmup_path, sampled_net_reward_return.unsqueeze(-1)),
            dim=-1,
        )
        reward_type = str(train_config.rl_training.reward_type).strip().lower()
        if reward_type == "dsr_day_last":
            warmup_A0, warmup_B0 = compute_dsr_warmup_stats(
                net_scored_returns,
                rolling_horizon_days=horizon_days,
            )
            base_rewards = compute_dsr_day_reward(
                sampled_prediction_returns.reshape(-1, horizon_days),
                rolling_horizon_days=horizon_days,
                A0=warmup_A0.unsqueeze(0).expand(group_size, -1).reshape(-1),
                B0=warmup_B0.unsqueeze(0).expand(group_size, -1).reshape(-1),
                dsr_var_eps=float(train_config.rl_training.dsr_var_eps),
                reward_clip=float(train_config.rl_training.reward_clip),
            ).reshape(group_size, -1)
        elif reward_type == "rolling_sharpe":
            base_rewards = compute_rolling_sharpe_reward(
                sampled_prediction_returns.reshape(-1, horizon_days),
                reward_clip=float(train_config.rl_training.reward_clip),
            ).reshape(group_size, -1)
        elif reward_type == "return":
            base_rewards = compute_return_reward(
                sampled_prediction_returns.reshape(-1, horizon_days),
                reward_scale=float(train_config.rl_training.reward_scale),
                reward_clip=float(train_config.rl_training.reward_clip),
            ).reshape(group_size, -1)
        elif reward_type == "win_rate":
            win_rate_metrics = compute_win_rate_metrics(
                sampled_net_reward_return,
                reward_stock_returns,
                reward_previous_allocation,
                reward_baseline=str(evaluation_config.reward_baseline),
                transaction_cost_rate=float(train_config.transaction_cost_rate),
            )
            base_rewards = win_rate_metrics.binary_reward
        else:
            raise ValueError(f"Unsupported RL reward_type: {reward_type!r}.")
        rewards, turnover_reward_penalty = apply_turnover_reward_penalty(
            base_rewards,
            sampled_actions,
            reward_previous_allocation,
            turnover_penalty=float(train_config.turnover_penalty),
            turnover_penalty_norm=str(train_config.turnover_penalty_norm),
            reward_scale=float(train_config.rl_training.reward_scale),
        )

    action_dim = int(alpha.shape[-1])
    entropy_per_dim = entropy / float(action_dim)
    entropy_loss = -float(train_config.rl_training.entropy_coef) * entropy_per_dim.mean()
    policy_loss, advantages = rl_algorithms.compute_grpo_like_policy_loss(
        sampled_log_probs,  # no detach
        rewards,
        entropy=entropy,
        entropy_coef=float(train_config.rl_training.entropy_coef),
        entropy_normalizer=float(action_dim),
        group_dim=0,
    )

    scenario_final_returns = torch.prod(1.0 + net_scored_returns, dim=1) - 1.0
    summary = {
        "scenario_final_returns": scenario_final_returns,
        "mean_turnover": sampled_turnover.detach().mean(),
        "mean_turnover_reward_penalty": turnover_reward_penalty.detach().mean(),
        "return_mean_min": net_scored_returns.mean(dim=1).detach().min(),
        "return_mean_max": net_scored_returns.mean(dim=1).detach().max(),
        "return_std_min": net_scored_returns.std(dim=1, unbiased=True).detach().min(),
        "return_std_max": net_scored_returns.std(dim=1, unbiased=True).detach().max(),
        "allocation_logits_abs_max": allocation_logits.detach().abs().max(),
        "raw_allocation_min": raw_allocation.detach().min(),
        "raw_allocation_max": raw_allocation.detach().max(),
    }
    metrics = {
        "train_loss": policy_loss,
        "train_weight_loss": policy_loss.detach().new_zeros(()),
        "train_policy_loss": policy_loss,
        "train_entropy_loss": entropy_loss.detach(),
        "train_entropy_per_dim": entropy_per_dim.detach().mean(),
        "train_alpha_min": alpha.detach().min(),
        "train_alpha_max": alpha.detach().max(),
        "train_alpha_mean": alpha.detach().mean(),
        "train_reward_base": base_rewards.detach().mean(),
        "train_reward_final": rewards.detach().mean(),
        "train_reward_TO_penalty": turnover_reward_penalty.detach().mean(),
        "train_advantage_mean": advantages.detach().mean(),
        "train_advantage_std": advantages.detach().std(unbiased=False),
        "train_log_prob_mean": sampled_log_probs.detach().mean(),
        "train_log_prob_std": sampled_log_probs.detach().std(unbiased=False),
        "train_TO": sampled_turnover.detach().mean(),
        "train_return": sampled_net_reward_return.detach().mean(),
    }
    return GRPOPolicyStepResult(
        policy_loss=policy_loss,
        metrics=metrics,
        summary=summary,
        batch_size=int(sampled_actions.shape[1]),
    )
