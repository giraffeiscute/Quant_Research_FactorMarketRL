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
from .replay import (
    SACReplayBuffer,
    SACTransitionBatch,
    collect_sac_transitions_from_rollout,
)
from .sac_policy import (
    DirichletSACPolicy,
    build_dirichlet_sac_policy,
    sac_dirichlet_alpha_from_logits,
)
from .sac import (
    SACActorAlphaReplayLossResult,
    SACLossBatchResult,
    SACQLossResult,
    SACQReplayLossResult,
    build_sac_metrics,
    compute_sac_actor_loss,
    compute_sac_actor_alpha_loss_from_replay_batch,
    compute_sac_alpha_loss,
    compute_sac_losses_from_replay_batch,
    compute_sac_q_loss,
    compute_sac_q_loss_from_replay_batch,
    soft_update_targets,
)

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
    "SACReplayBuffer",
    "SACTransitionBatch",
    "collect_sac_transitions_from_rollout",
    "DirichletSACPolicy",
    "build_dirichlet_sac_policy",
    "sac_dirichlet_alpha_from_logits",
    "SACActorAlphaReplayLossResult",
    "SACLossBatchResult",
    "SACQLossResult",
    "SACQReplayLossResult",
    "build_sac_metrics",
    "compute_sac_actor_loss",
    "compute_sac_actor_alpha_loss_from_replay_batch",
    "compute_sac_alpha_loss",
    "compute_sac_losses_from_replay_batch",
    "compute_sac_q_loss",
    "compute_sac_q_loss_from_replay_batch",
    "soft_update_targets",
]
