"""RL training-step primitives shared by runtime frontends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.distributions import Dirichlet

from ..common.net_return import apply_transaction_cost_to_returns
from ..common.utils import apply_score_mask
from ..common.win_rate import compute_win_rate_metrics
from ..config import DataConfig, EvaluationConfig, ModelConfig, TrainConfig
from ..model import PortfolioAttentionModel
from ..model.allocation_distribution import logits_to_rl_post_train_dirichlet_alpha
from ..model.reward import (
    apply_turnover_reward_penalty,
    compute_dsr_day_reward,
    compute_dsr_warmup_stats,
    compute_return_reward,
    compute_rolling_sharpe_reward,
)
from ..rl.rollout import sample_rollout_path
from ..rl.rollout_targets import (
    compute_discounted_reward_targets,
    compute_rollout_advantages_from_targets,
)
from . import rl_algorithms


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
    model_config: ModelConfig,
    train_config: TrainConfig,
    evaluation_config: EvaluationConfig | None = None,
) -> RLPolicyStepResult:
    resolved_evaluation_config = evaluation_config or EvaluationConfig()
    algorithm = str(train_config.rl_training.algorithm).strip().lower()
    forward_kwargs: dict[str, Any] = {"target_returns": batch["r_stock"]}
    if algorithm == "single_epoch_rollout_ppo":
        forward_kwargs["compute_value_prediction"] = True
    outputs = model(
        batch["x_stock"],
        batch["x_market"],
        batch["stock_indices"],
        **forward_kwargs,
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
    scored_value_prediction = None
    if algorithm == "single_epoch_rollout_ppo":
        value_prediction = _require_tensor(outputs, "value_prediction")
        scored_value_prediction = apply_score_mask(value_prediction, score_mask)

    horizon_days = int(data_config.rolling_horizon_days)
    if int(scored_returns.shape[1]) != horizon_days:
        raise ValueError(
            "RL training expects score-masked returns with time dimension equal to rolling_horizon_days. "
            f"Received scored_returns={tuple(scored_returns.shape)} rolling_horizon_days={horizon_days}."
        )

    if algorithm == "single_epoch_rollout_ppo":
        reward_type = str(train_config.rl_training.reward_type).strip().lower()
        if reward_type != "return":
            raise ValueError(
                "single_epoch_rollout_ppo currently requires reward_type='return'; "
                f"received {reward_type!r}."
            )
        if scored_value_prediction is None:
            raise RuntimeError("single_epoch_rollout_ppo requires value_prediction in model outputs.")
        return _run_single_epoch_rollout_ppo_policy_step(
            scored_logits=scored_logits,
            scored_r_stock=scored_r_stock,
            scored_returns=scored_returns,
            scored_turnover=scored_turnover,
            scored_previous_allocation=scored_previous_allocation,
            scored_value_prediction=scored_value_prediction,
            allocation_logits=allocation_logits,
            raw_allocation=raw_allocation,
            model_config=model_config,
            train_config=train_config,
        )

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
                reward_baseline=str(resolved_evaluation_config.reward_baseline),
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
    if algorithm == "grpo_like":
        policy_loss, advantages = rl_algorithms.compute_grpo_like_policy_loss(
            sampled_log_probs,  # no detach
            rewards,
            entropy=entropy,
            entropy_coef=float(train_config.rl_training.entropy_coef),
            entropy_normalizer=float(action_dim),
            group_dim=0,
        )
    else:
        raise ValueError(f"Unsupported RL training algorithm: {algorithm!r}.")

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
    return RLPolicyStepResult(
        policy_loss=policy_loss,
        metrics=metrics,
        summary=summary,
        batch_size=int(sampled_actions.shape[1]),
    )


def _run_single_epoch_rollout_ppo_policy_step(
    *,
    scored_logits: torch.Tensor,
    scored_r_stock: torch.Tensor,
    scored_returns: torch.Tensor,
    scored_turnover: torch.Tensor,
    scored_previous_allocation: torch.Tensor,
    scored_value_prediction: torch.Tensor,
    allocation_logits: torch.Tensor,
    raw_allocation: torch.Tensor,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> RLPolicyStepResult:
    rollout = sample_rollout_path(
        scored_logits=scored_logits,
        scored_r_stock=scored_r_stock,
        initial_previous_allocation=scored_previous_allocation[:, 0, :],
        value_prediction=scored_value_prediction,
        model_config=model_config,
        train_config=train_config,
    )
    targets = compute_discounted_reward_targets(
        rollout.sampled_rewards,
        gamma=float(train_config.rl_training.ppo_gamma),
    )
    advantages = compute_rollout_advantages_from_targets(
        targets,
        rollout.old_values,
        normalize=bool(train_config.rl_training.normalize_rollout_advantages),
    )

    new_log_probs: list[torch.Tensor] = []
    entropy: list[torch.Tensor] = []
    alpha_values: list[torch.Tensor] = []
    time_steps = int(scored_logits.shape[1])
    for timestep in range(time_steps):
        alpha_t = logits_to_rl_post_train_dirichlet_alpha(
            scored_logits[:, timestep, :],
            alpha_min=float(train_config.rl_training.alpha_min),
            alpha_max=float(train_config.rl_training.alpha_max),
            logit_scale=float(model_config.dirichlet_logit_scale),
            evidence_scale=float(train_config.rl_training.rl_post_train_evidence_scale),
        )
        dist_t = Dirichlet(alpha_t)
        alpha_values.append(alpha_t)
        new_log_probs.append(dist_t.log_prob(rollout.sampled_actions[:, timestep, :]))
        entropy.append(dist_t.entropy())

    alpha_tensor = torch.stack(alpha_values, dim=1)
    new_log_probs_tensor = torch.stack(new_log_probs, dim=1)
    entropy_tensor = torch.stack(entropy, dim=1)
    action_dim = int(scored_logits.shape[-1])
    entropy_per_dim = entropy_tensor / float(action_dim)
    entropy_loss = -float(train_config.rl_training.entropy_coef) * entropy_per_dim.mean()

    # single_epoch_rollout_ppo still collects and re-scores actions within one
    # update context, so PPO ratios are expected to stay near 1.0. PPO clipping
    # becomes fully meaningful only after adding rollout buffer reuse and
    # multi-epoch updates.
    ppo_policy_loss, ppo_ratio, ppo_clip_fraction = rl_algorithms.compute_ppo_clipped_policy_loss(
        new_log_probs_tensor,
        rollout.sampled_log_probs,
        advantages,
        clip_range=float(train_config.rl_training.ppo_clip_range),
    )
    value_loss = rl_algorithms.compute_value_loss(
        scored_value_prediction,
        targets.detach(),
    )
    policy_loss = (
        ppo_policy_loss
        + float(train_config.rl_training.value_loss_coef) * value_loss
        + entropy_loss
    )

    with torch.no_grad():
        net_scored_returns = apply_transaction_cost_to_returns(
            scored_returns,
            scored_turnover,
            transaction_cost_rate=float(train_config.transaction_cost_rate),
        )
    rollout_final_returns = torch.prod(1.0 + rollout.sampled_net_returns, dim=1) - 1.0
    scenario_final_returns = torch.prod(1.0 + net_scored_returns, dim=1) - 1.0
    summary = {
        "scenario_final_returns": scenario_final_returns,
        "rollout_final_returns": rollout_final_returns.detach(),
        "mean_turnover": rollout.sampled_turnover.detach().mean(),
        "mean_turnover_reward_penalty": rollout.sampled_reward_penalty.detach().mean(),
        "return_mean_min": net_scored_returns.mean(dim=1).detach().min(),
        "return_mean_max": net_scored_returns.mean(dim=1).detach().max(),
        "return_std_min": net_scored_returns.std(dim=1, unbiased=True).detach().min(),
        "return_std_max": net_scored_returns.std(dim=1, unbiased=True).detach().max(),
        "allocation_logits_abs_max": allocation_logits.detach().abs().max(),
        "raw_allocation_min": raw_allocation.detach().min(),
        "raw_allocation_max": raw_allocation.detach().max(),
    }
    metrics = {
        "train_policy_loss": ppo_policy_loss,
        "train_entropy_loss": entropy_loss.detach(),
        "train_entropy_per_dim": entropy_per_dim.detach().mean(),
        "train_alpha_min": alpha_tensor.detach().min(),
        "train_alpha_max": alpha_tensor.detach().max(),
        "train_alpha_mean": alpha_tensor.detach().mean(),
        "train_reward_base": rollout.sampled_base_rewards.detach().mean(),
        "train_reward_final": rollout.sampled_rewards.detach().mean(),
        "train_reward_TO_penalty": rollout.sampled_reward_penalty.detach().mean(),
        "train_advantage_mean": advantages.detach().mean(),
        "train_advantage_std": advantages.detach().std(unbiased=False),
        "train_log_prob_mean": new_log_probs_tensor.detach().mean(),
        "train_log_prob_std": new_log_probs_tensor.detach().std(unbiased=False),
        "train_TO": rollout.sampled_turnover.detach().mean(),
        "train_return": rollout.sampled_net_returns.detach().mean(),
        "train_rollout_value_loss": value_loss.detach(),
        "train_rollout_target_mean": targets.detach().mean(),
        "train_rollout_target_std": targets.detach().std(unbiased=False),
        "train_rollout_advantage_mean": advantages.detach().mean(),
        "train_rollout_advantage_std": advantages.detach().std(unbiased=False),
        "train_rollout_ppo_ratio_mean": ppo_ratio.detach().mean(),
        "train_rollout_ppo_clip_fraction": ppo_clip_fraction.detach(),
        "train_rollout_entropy_per_dim": entropy_per_dim.detach().mean(),
        "train_rollout_total_loss": policy_loss.detach(),
        "train_rollout_reward_base": rollout.sampled_base_rewards.detach().mean(),
        "train_rollout_reward_final": rollout.sampled_rewards.detach().mean(),
        "train_rollout_reward_TO_penalty": rollout.sampled_reward_penalty.detach().mean(),
        "train_rollout_return": rollout.sampled_net_returns.detach().mean(),
        "train_rollout_TO": rollout.sampled_turnover.detach().mean(),
        "train_rollout_final_returns": rollout_final_returns.detach().mean(),
    }
    return RLPolicyStepResult(
        policy_loss=policy_loss,
        metrics=metrics,
        summary=summary,
        batch_size=int(rollout.sampled_actions.shape[0]),
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
