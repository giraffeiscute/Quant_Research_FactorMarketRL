"""Policy distribution helpers for RL post-training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Dirichlet

from ..config import ModelConfig, TrainConfig
from ..model.allocation_distribution import logits_to_rl_post_train_dirichlet_alpha


@dataclass(frozen=True)
class RolloutPPOPolicyScoring:
    """Fresh policy distribution statistics for fixed rollout actions."""

    alpha: torch.Tensor
    new_log_probs: torch.Tensor
    entropy: torch.Tensor


def compute_dirichlet_log_probs_from_logits(
    scored_logits: torch.Tensor,
    actions: torch.Tensor,
    *,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> RolloutPPOPolicyScoring:
    if scored_logits.ndim != 3:
        raise ValueError(
            "scored_logits must have shape [B, T, A]. "
            f"Received {tuple(scored_logits.shape)}."
        )
    if tuple(scored_logits.shape) != tuple(actions.shape):
        raise ValueError(
            "scored_logits and actions must share shape [B, T, A]. "
            f"Received scored_logits={tuple(scored_logits.shape)} actions={tuple(actions.shape)}."
        )
    alpha_values: list[torch.Tensor] = []
    new_log_probs: list[torch.Tensor] = []
    entropy: list[torch.Tensor] = []
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
        new_log_probs.append(dist_t.log_prob(actions[:, timestep, :]))
        entropy.append(dist_t.entropy())
    return RolloutPPOPolicyScoring(
        alpha=torch.stack(alpha_values, dim=1),
        new_log_probs=torch.stack(new_log_probs, dim=1),
        entropy=torch.stack(entropy, dim=1),
    )
