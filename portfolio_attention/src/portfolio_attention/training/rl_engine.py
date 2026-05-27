"""RL training-step dispatcher shared by runtime frontends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..common.utils import apply_score_mask
from ..config import DataConfig, EvaluationConfig, ModelConfig, TrainConfig
from ..model import PortfolioAttentionModel
from .grpo_engine import run_grpo_policy_step
from .ppo_engine import (
    collect_rollout_ppo_training_batch,
    run_rollout_ppo_policy_step,
    run_rollout_ppo_update,
    validate_rollout_ppo_config,
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
    model_config: ModelConfig,
    train_config: TrainConfig,
    evaluation_config: EvaluationConfig | None = None,
) -> RLPolicyStepResult:
    """Run the single-step RL algorithm path for frontends with automatic optimization."""
    resolved_evaluation_config = evaluation_config or EvaluationConfig()
    algorithm = str(train_config.rl_training.algorithm).strip().lower()
    if algorithm == "sac":
        raise NotImplementedError(
            "SAC RL training is recognized by config and uses the Lightning "
            "manual optimization path; call PortfolioLightningModule._training_step_sac "
            "instead of run_rl_policy_step."
        )
    if algorithm == "rollout_ppo" and int(train_config.rl_training.ppo.num_epochs) > 1:
        raise RuntimeError(
            "rollout_ppo with ppo_num_epochs > 1 requires an explicit/manual optimizer loop; "
            "use collect_rollout_ppo_training_batch and run_rollout_ppo_update."
        )
    if algorithm == "rollout_ppo":
        validate_rollout_ppo_config(model_config=model_config, train_config=train_config)

    forward_kwargs: dict[str, Any] = {"target_returns": batch["r_stock"]}
    if algorithm == "rollout_ppo":
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
    if algorithm == "rollout_ppo":
        value_prediction = _require_tensor(outputs, "value_prediction")
        scored_value_prediction = apply_score_mask(value_prediction, score_mask)

    horizon_days = int(data_config.rolling_horizon_days)
    if int(scored_returns.shape[1]) != horizon_days:
        raise ValueError(
            "RL training expects score-masked returns with time dimension equal to rolling_horizon_days. "
            f"Received scored_returns={tuple(scored_returns.shape)} rolling_horizon_days={horizon_days}."
        )

    if algorithm == "rollout_ppo":
        if scored_value_prediction is None:
            raise RuntimeError("rollout_ppo requires value_prediction in model outputs.")
        ppo_result = run_rollout_ppo_policy_step(
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
        return RLPolicyStepResult(
            policy_loss=ppo_result.policy_loss,
            metrics=ppo_result.metrics,
            summary=ppo_result.summary,
            batch_size=ppo_result.batch_size,
        )

    if algorithm == "grpo_like":
        grpo_result = run_grpo_policy_step(
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


_validate_rollout_ppo_config = validate_rollout_ppo_config
