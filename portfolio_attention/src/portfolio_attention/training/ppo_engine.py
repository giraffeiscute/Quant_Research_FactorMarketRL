"""Rollout PPO training-step helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..common.net_return import apply_transaction_cost_to_returns
from ..common.utils import apply_score_mask
from ..config import DataConfig, ModelConfig, TrainConfig
from ..model import PortfolioAttentionModel
from ..rl.ppo import (
    RolloutPPOBatch,
    RolloutPPOTrainingBatch,
    RolloutPPOUpdateResult,
    build_rollout_ppo_update_metrics,
    collect_rollout_ppo_batch,
    compute_rollout_ppo_update_loss,
)


@dataclass(frozen=True)
class RolloutPPOPolicyStepResult:
    policy_loss: torch.Tensor
    metrics: dict[str, torch.Tensor | float]
    summary: dict[str, torch.Tensor]
    batch_size: int


def run_rollout_ppo_policy_step(
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
) -> RolloutPPOPolicyStepResult:
    """Run a single-epoch rollout PPO policy step from scored model tensors."""
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
    metrics = build_rollout_ppo_update_metrics(
        ppo_batch,
        ppo_update,
        ppo_epoch=1,
    )
    return RolloutPPOPolicyStepResult(
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
    """Score a fixed rollout PPO batch against a fresh policy forward."""
    validate_rollout_ppo_config(model_config=model_config, train_config=train_config)
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
    """Collect a frozen rollout PPO batch for one or more PPO update epochs."""
    validate_rollout_ppo_config(model_config=model_config, train_config=train_config)
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
    scored_previous_allocation = apply_score_mask(
        _require_tensor(outputs, "previous_allocation"),
        score_mask,
    )
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


def validate_rollout_ppo_config(
    *,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> None:
    """Validate rollout PPO restrictions that are not global config invariants."""
    reward_type = str(train_config.rl_training.reward_type).strip().lower()
    if reward_type != "return":
        raise ValueError(
            "rollout_ppo currently requires reward_type='return'; "
            f"received {reward_type!r}."
        )
    if bool(getattr(model_config, "use_prev_weight_feature", False)):
        raise ValueError(
            "rollout_ppo currently requires ModelConfig.use_prev_weight_feature=False "
            "so the frozen PPO rollout state is not conditioned on previous weights."
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
