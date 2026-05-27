"""GRPO-like training-step helpers."""

from __future__ import annotations

import torch

from ..config import EvaluationConfig, ModelConfig, TrainConfig
from ..rl.grpo import GRPOPolicyStepResult, run_grpo_like_policy_step_from_scored_tensors


def run_grpo_policy_step(
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
    """Run the GRPO-like policy step from score-masked model tensors."""
    return run_grpo_like_policy_step_from_scored_tensors(
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
        evaluation_config=evaluation_config,
    )
