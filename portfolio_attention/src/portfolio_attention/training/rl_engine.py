"""RL training-step primitives shared by runtime frontends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..common.net_return import apply_transaction_cost_to_returns
from ..common.utils import apply_score_mask
from ..config import DataConfig, EvaluationConfig, ModelConfig, TrainConfig
from ..model import PortfolioAttentionModel
from ..rl.grpo import run_grpo_like_policy_step_from_scored_tensors
from ..rl.ppo import (
    RolloutPPOBatch,
    RolloutPPOTrainingBatch,
    RolloutPPOUpdateResult,
    collect_rollout_ppo_batch,
    compute_rollout_ppo_update_loss,
)


ROLLOUT_PPO_ALGORITHMS = frozenset({"single_epoch_rollout_ppo", "multi_epoch_rollout_ppo"})


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
    if algorithm == "multi_epoch_rollout_ppo":
        raise RuntimeError(
            "multi_epoch_rollout_ppo requires an explicit optimizer loop; "
            "use collect_rollout_ppo_training_batch and run_rollout_ppo_update."
        )
    forward_kwargs: dict[str, Any] = {"target_returns": batch["r_stock"]}
    if algorithm in ROLLOUT_PPO_ALGORITHMS:
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
    if algorithm in ROLLOUT_PPO_ALGORITHMS:
        value_prediction = _require_tensor(outputs, "value_prediction")
        scored_value_prediction = apply_score_mask(value_prediction, score_mask)

    horizon_days = int(data_config.rolling_horizon_days)
    if int(scored_returns.shape[1]) != horizon_days:
        raise ValueError(
            "RL training expects score-masked returns with time dimension equal to rolling_horizon_days. "
            f"Received scored_returns={tuple(scored_returns.shape)} rolling_horizon_days={horizon_days}."
        )

    if algorithm in ROLLOUT_PPO_ALGORITHMS:
        reward_type = str(train_config.rl_training.reward_type).strip().lower()
        if reward_type != "return":
            raise ValueError(
                f"{algorithm} currently requires reward_type='return'; "
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

    if algorithm == "grpo_like":
        grpo_result = run_grpo_like_policy_step_from_scored_tensors(
            scored_logits=scored_logits,
            scored_r_stock=scored_r_stock,
            scored_returns=scored_returns,
            scored_turnover=scored_turnover,
            scored_previous_allocation=scored_previous_allocation,
            allocation_logits=allocation_logits,
            raw_allocation=raw_allocation,
            horizon_days=horizon_days,
            model_config=model_config,
            train_config=train_config,
            evaluation_config=resolved_evaluation_config,
        )
        return RLPolicyStepResult(
            policy_loss=grpo_result.policy_loss,
            metrics=grpo_result.metrics,
            summary=grpo_result.summary,
            batch_size=grpo_result.batch_size,
        )

    raise ValueError(f"Unsupported RL training algorithm: {algorithm!r}.")


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
    ppo_batch = collect_rollout_ppo_batch(
        scored_logits=scored_logits,
        scored_r_stock=scored_r_stock,
        initial_previous_allocation=scored_previous_allocation[:, 0, :],
        value_prediction=scored_value_prediction,
        model_config=model_config,
        train_config=train_config,
    )
    ppo_update = compute_rollout_ppo_update_loss(
        scored_logits=scored_logits,
        scored_value_prediction=scored_value_prediction,
        ppo_batch=ppo_batch,
        model_config=model_config,
        train_config=train_config,
    )
    policy_scoring = ppo_update.policy_scoring
    entropy_per_dim = policy_scoring.entropy / float(scored_logits.shape[-1])

    with torch.no_grad():
        net_scored_returns = apply_transaction_cost_to_returns(
            scored_returns,
            scored_turnover,
            transaction_cost_rate=float(train_config.transaction_cost_rate),
        )
    rollout_final_returns = torch.prod(1.0 + ppo_batch.sampled_net_returns, dim=1) - 1.0
    scenario_final_returns = torch.prod(1.0 + net_scored_returns, dim=1) - 1.0
    summary = {
        "scenario_final_returns": scenario_final_returns,
        "rollout_final_returns": rollout_final_returns.detach(),
        "mean_turnover": ppo_batch.sampled_turnover.detach().mean(),
        "mean_turnover_reward_penalty": ppo_batch.sampled_reward_penalty.detach().mean(),
        "return_mean_min": net_scored_returns.mean(dim=1).detach().min(),
        "return_mean_max": net_scored_returns.mean(dim=1).detach().max(),
        "return_std_min": net_scored_returns.std(dim=1, unbiased=True).detach().min(),
        "return_std_max": net_scored_returns.std(dim=1, unbiased=True).detach().max(),
        "allocation_logits_abs_max": allocation_logits.detach().abs().max(),
        "raw_allocation_min": raw_allocation.detach().min(),
        "raw_allocation_max": raw_allocation.detach().max(),
    }
    metrics = {
        "train_policy_loss": ppo_update.ppo_policy_loss,
        "train_entropy_loss": ppo_update.entropy_loss.detach(),
        "train_entropy_per_dim": entropy_per_dim.detach().mean(),
        "train_alpha_min": policy_scoring.alpha.detach().min(),
        "train_alpha_max": policy_scoring.alpha.detach().max(),
        "train_alpha_mean": policy_scoring.alpha.detach().mean(),
        "train_reward_base": ppo_batch.sampled_base_rewards.detach().mean(),
        "train_reward_final": ppo_batch.sampled_rewards.detach().mean(),
        "train_reward_TO_penalty": ppo_batch.sampled_reward_penalty.detach().mean(),
        "train_advantage_mean": ppo_batch.advantages.detach().mean(),
        "train_advantage_std": ppo_batch.advantages.detach().std(unbiased=False),
        "train_log_prob_mean": policy_scoring.new_log_probs.detach().mean(),
        "train_log_prob_std": policy_scoring.new_log_probs.detach().std(unbiased=False),
        "train_TO": ppo_batch.sampled_turnover.detach().mean(),
        "train_return": ppo_batch.sampled_net_returns.detach().mean(),
        "train_rollout_value_loss": ppo_update.value_loss.detach(),
        "train_rollout_target_mean": ppo_batch.targets.detach().mean(),
        "train_rollout_target_std": ppo_batch.targets.detach().std(unbiased=False),
        "train_rollout_advantage_mean": ppo_batch.advantages.detach().mean(),
        "train_rollout_advantage_std": ppo_batch.advantages.detach().std(unbiased=False),
        "train_rollout_ppo_ratio_mean": ppo_update.ppo_ratio.detach().mean(),
        "train_rollout_ppo_clip_fraction": ppo_update.ppo_clip_fraction.detach(),
        "train_rollout_entropy_per_dim": entropy_per_dim.detach().mean(),
        "train_rollout_total_loss": ppo_update.policy_loss.detach(),
        "train_rollout_reward_base": ppo_batch.sampled_base_rewards.detach().mean(),
        "train_rollout_reward_final": ppo_batch.sampled_rewards.detach().mean(),
        "train_rollout_reward_TO_penalty": ppo_batch.sampled_reward_penalty.detach().mean(),
        "train_rollout_return": ppo_batch.sampled_net_returns.detach().mean(),
        "train_rollout_TO": ppo_batch.sampled_turnover.detach().mean(),
        "train_rollout_final_returns": rollout_final_returns.detach().mean(),
    }
    return RLPolicyStepResult(
        policy_loss=ppo_update.policy_loss,
        metrics=metrics,
        summary=summary,
        batch_size=int(ppo_batch.sampled_actions.shape[0]),
    )


def run_rollout_ppo_update(
    model: PortfolioAttentionModel,
    batch: dict[str, Any],
    ppo_batch: RolloutPPOBatch,
    *,
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> RolloutPPOUpdateResult:
    outputs = model(
        batch["x_stock"],
        batch["x_market"],
        batch["stock_indices"],
        target_returns=batch["r_stock"],
        compute_value_prediction=True,
    )
    score_mask = _require_score_mask(batch)
    scored_logits = apply_score_mask(_require_tensor(outputs, "allocation_logits"), score_mask)
    scored_value_prediction = apply_score_mask(_require_tensor(outputs, "value_prediction"), score_mask)

    horizon_days = int(data_config.rolling_horizon_days)
    if int(scored_logits.shape[1]) != horizon_days:
        raise ValueError(
            "Rollout PPO update expects score-masked logits with time dimension equal to "
            "rolling_horizon_days. "
            f"Received scored_logits={tuple(scored_logits.shape)} rolling_horizon_days={horizon_days}."
        )

    return compute_rollout_ppo_update_loss(
        scored_logits=scored_logits,
        scored_value_prediction=scored_value_prediction,
        ppo_batch=ppo_batch,
        model_config=model_config,
        train_config=train_config,
    )


@torch.no_grad()
def collect_rollout_ppo_training_batch(
    model: PortfolioAttentionModel,
    batch: dict[str, Any],
    *,
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> RolloutPPOTrainingBatch:
    outputs = model(
        batch["x_stock"],
        batch["x_market"],
        batch["stock_indices"],
        target_returns=batch["r_stock"],
        compute_value_prediction=True,
    )
    score_mask = _require_score_mask(batch)
    scored_returns = apply_score_mask(_require_tensor(outputs, "portfolio_return"), score_mask)
    scored_turnover = apply_score_mask(_require_tensor(outputs, "turnover"), score_mask)
    scored_previous_allocation = apply_score_mask(_require_tensor(outputs, "previous_allocation"), score_mask)
    scored_logits = apply_score_mask(_require_tensor(outputs, "allocation_logits"), score_mask)
    scored_r_stock = apply_score_mask(batch["r_stock"], score_mask)
    scored_value_prediction = apply_score_mask(_require_tensor(outputs, "value_prediction"), score_mask)
    allocation_logits = _require_tensor(outputs, "allocation_logits")
    raw_allocation = _require_tensor(outputs, "raw_allocation")

    horizon_days = int(data_config.rolling_horizon_days)
    if int(scored_returns.shape[1]) != horizon_days:
        raise ValueError(
            "Rollout PPO collection expects score-masked returns with time dimension equal to "
            "rolling_horizon_days. "
            f"Received scored_returns={tuple(scored_returns.shape)} rolling_horizon_days={horizon_days}."
        )

    ppo_batch = collect_rollout_ppo_batch(
        scored_logits=scored_logits,
        scored_r_stock=scored_r_stock,
        initial_previous_allocation=scored_previous_allocation[:, 0, :],
        value_prediction=scored_value_prediction,
        model_config=model_config,
        train_config=train_config,
    )
    with torch.no_grad():
        net_scored_returns = apply_transaction_cost_to_returns(
            scored_returns,
            scored_turnover,
            transaction_cost_rate=float(train_config.transaction_cost_rate),
        )
    rollout_final_returns = torch.prod(1.0 + ppo_batch.sampled_net_returns, dim=1) - 1.0
    scenario_final_returns = torch.prod(1.0 + net_scored_returns, dim=1) - 1.0
    summary = {
        "scenario_final_returns": scenario_final_returns,
        "rollout_final_returns": rollout_final_returns.detach(),
        "mean_turnover": ppo_batch.sampled_turnover.detach().mean(),
        "mean_turnover_reward_penalty": ppo_batch.sampled_reward_penalty.detach().mean(),
        "return_mean_min": net_scored_returns.mean(dim=1).detach().min(),
        "return_mean_max": net_scored_returns.mean(dim=1).detach().max(),
        "return_std_min": net_scored_returns.std(dim=1, unbiased=True).detach().min(),
        "return_std_max": net_scored_returns.std(dim=1, unbiased=True).detach().max(),
        "allocation_logits_abs_max": allocation_logits.detach().abs().max(),
        "raw_allocation_min": raw_allocation.detach().min(),
        "raw_allocation_max": raw_allocation.detach().max(),
    }
    return RolloutPPOTrainingBatch(
        ppo_batch=ppo_batch,
        summary=summary,
        batch_size=int(ppo_batch.sampled_actions.shape[0]),
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
