"""Rollout utilities for RL post-training."""

from .distributions import RolloutPPOPolicyScoring, compute_dirichlet_log_probs_from_logits
from .grpo import GRPOPolicyStepResult, run_grpo_like_policy_step_from_scored_tensors
from .ppo import (
    RolloutPPOBatch,
    RolloutPPOTrainingBatch,
    RolloutPPOUpdateResult,
    build_rollout_ppo_update_metrics,
    collect_rollout_ppo_batch,
    compute_rollout_ppo_update_loss,
)
from .rollout import RolloutPathResult, sample_rollout_path

__all__ = [
    "RolloutPPOPolicyScoring",
    "compute_dirichlet_log_probs_from_logits",
    "GRPOPolicyStepResult",
    "run_grpo_like_policy_step_from_scored_tensors",
    "RolloutPPOBatch",
    "RolloutPPOTrainingBatch",
    "RolloutPPOUpdateResult",
    "build_rollout_ppo_update_metrics",
    "collect_rollout_ppo_batch",
    "compute_rollout_ppo_update_loss",
    "RolloutPathResult",
    "sample_rollout_path",
]
