"""RL training-step primitives shared by runtime frontends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.distributions import Dirichlet

from ..common.utils import apply_score_mask
from ..config import DataConfig, TrainConfig
from ..evaluation.metrics import apply_transaction_cost_to_returns
from ..model import PortfolioAttentionModel
from ..model.allocation_distribution import logits_to_rl_post_train_dirichlet_alpha
from ..model.reward import (
    compute_dsr_day_reward,
    compute_dsr_warmup_stats,
    compute_group_relative_advantage,
    compute_policy_gradient_objective,
    compute_rolling_sharpe_reward,
)


@dataclass(frozen=True)
class RLPolicyStepResult:
    policy_loss: torch.Tensor
    metrics: dict[str, torch.Tensor | float]
    summary: dict[str, torch.Tensor]
    batch_size: int


def run_rl_policy_step(
    model: PortfolioAttentionModel,
    batch: dict[str, Any],
    *,
    data_config: DataConfig,
    train_config: TrainConfig,
) -> RLPolicyStepResult:
    outputs = model(
        batch["x_stock"],
        batch["x_market"],
        batch["stock_indices"],
        target_returns=batch["r_stock"],
    )
    portfolio_return = _require_tensor(outputs, "portfolio_return")
    turnover = _require_tensor(outputs, "turnover")
    previous_allocation = _require_tensor(outputs, "previous_allocation")
    allocation_logits = _require_tensor(outputs, "allocation_logits")
    raw_allocation = _require_tensor(outputs, "raw_allocation")
    score_mask = _require_score_mask(batch)

    scored_returns = apply_score_mask(portfolio_return, score_mask)
    scored_turnover = apply_score_mask(turnover, score_mask)
    scored_previous_allocation = apply_score_mask(previous_allocation, score_mask)
    scored_logits = apply_score_mask(allocation_logits, score_mask)
    scored_r_stock = apply_score_mask(batch["r_stock"], score_mask)

    horizon_days = int(data_config.rolling_horizon_days)
    if int(scored_returns.shape[1]) != horizon_days:
        raise ValueError(
            "RL training expects score-masked returns with time dimension equal to rolling_horizon_days. "
            f"Received scored_returns={tuple(scored_returns.shape)} rolling_horizon_days={horizon_days}."
        )

    alpha = logits_to_rl_post_train_dirichlet_alpha(
        scored_logits[:, -1, :],
        alpha_min=float(train_config.rl_training.alpha_min),
        alpha_max=float(train_config.rl_training.alpha_max),
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
            rewards = compute_dsr_day_reward(
                sampled_prediction_returns.reshape(-1, horizon_days),
                rolling_horizon_days=horizon_days,
                A0=warmup_A0.unsqueeze(0).expand(group_size, -1).reshape(-1),
                B0=warmup_B0.unsqueeze(0).expand(group_size, -1).reshape(-1),
                dsr_var_eps=float(train_config.rl_training.dsr_var_eps),
                reward_clip=float(train_config.rl_training.reward_clip),
            ).reshape(group_size, -1)
        elif reward_type == "rolling_sharpe":
            rewards = compute_rolling_sharpe_reward(
                sampled_prediction_returns.reshape(-1, horizon_days),
                reward_clip=float(train_config.rl_training.reward_clip),
            ).reshape(group_size, -1)
        else:
            raise ValueError(f"Unsupported RL reward_type: {reward_type!r}.")
        advantages = compute_group_relative_advantage(rewards, group_dim=0)
    action_dim = int(alpha.shape[-1])
    entropy_per_dim = entropy / float(action_dim)
    entropy_loss = -float(train_config.rl_training.entropy_coef) * entropy_per_dim.mean()
    policy_loss = compute_policy_gradient_objective(
        sampled_log_probs, #no detach
        advantages,
        entropy=entropy,
        entropy_coef=float(train_config.rl_training.entropy_coef),
        entropy_normalizer=float(action_dim),
    )

    scenario_final_returns = torch.prod(1.0 + net_scored_returns, dim=1) - 1.0
    summary = {
        "scenario_final_returns": scenario_final_returns,
        "mean_turnover": sampled_turnover.detach().mean(),
        "return_mean_min": net_scored_returns.mean(dim=1).detach().min(),
        "return_mean_max": net_scored_returns.mean(dim=1).detach().max(),
        "return_std_min": net_scored_returns.std(dim=1, unbiased=True).detach().min(),
        "return_std_max": net_scored_returns.std(dim=1, unbiased=True).detach().max(),
        "allocation_logits_abs_max": allocation_logits.detach().abs().max(),
        "raw_allocation_min": raw_allocation.detach().min(),
        "raw_allocation_max": raw_allocation.detach().max(),
    }
    metrics = {
        "train_policy_loss": policy_loss,
        "train_entropy_loss": entropy_loss.detach(),
        "train_entropy_per_dim": entropy_per_dim.detach().mean(),
        "train_alpha_min": alpha.detach().min(),
        "train_alpha_max": alpha.detach().max(),
        "train_alpha_mean": alpha.detach().mean(),
        "train_reward_mean": rewards.detach().mean(),
        "train_reward_std": rewards.detach().std(unbiased=False),
        "train_advantage_mean": advantages.detach().mean(),
        "train_advantage_std": advantages.detach().std(unbiased=False),
        "train_log_prob_mean": sampled_log_probs.detach().mean(),
        "train_log_prob_std": sampled_log_probs.detach().std(unbiased=False),
        "train_OT": sampled_turnover.detach().mean(),
        "train_return": sampled_net_reward_return.detach().mean(),
    }
    return RLPolicyStepResult(
        policy_loss=policy_loss,
        metrics=metrics,
        summary=summary,
        batch_size=int(sampled_actions.shape[1]),
    )


def _require_tensor(outputs: dict[str, Any], key: str) -> torch.Tensor:
    value = outputs.get(key)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"Training batch requires model outputs to include {key} tensor.")
    return value


def _require_score_mask(batch: dict[str, Any]) -> torch.Tensor:
    score_mask = batch.get("score_mask")
    if not isinstance(score_mask, torch.Tensor):
        raise RuntimeError("RL training requires score_mask tensor in batch.")
    return score_mask.to(dtype=torch.bool)
