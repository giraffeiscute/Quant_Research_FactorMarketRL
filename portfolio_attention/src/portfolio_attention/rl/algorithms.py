"""Policy-gradient algorithm helpers for RL portfolio training."""

from __future__ import annotations

import torch


def compute_group_relative_advantage(
    rewards: torch.Tensor,
    *,
    eps: float = 1e-6,
    group_dim: int = 0,
) -> torch.Tensor:
    if rewards.numel() == 0:
        raise ValueError("rewards must not be empty.")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be > 0, received {eps}.")
    mean = rewards.mean(dim=group_dim, keepdim=True)
    std = rewards.std(dim=group_dim, keepdim=True, unbiased=False)
    return (rewards - mean) / (std + float(eps))


def compute_policy_gradient_objective(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    entropy: torch.Tensor | None = None,
    entropy_coef: float = 0.0,
    entropy_normalizer: float = 1.0,
) -> torch.Tensor:
    if tuple(log_probs.shape) != tuple(advantages.shape):
        raise ValueError(
            "log_probs and advantages must share the same shape. "
            f"Received log_probs={tuple(log_probs.shape)} advantages={tuple(advantages.shape)}."
        )
    policy_loss = -(advantages.detach() * log_probs).mean()
    if entropy is None:
        return policy_loss
    normalizer = float(entropy_normalizer)
    if normalizer <= 0.0:
        raise ValueError(f"entropy_normalizer must be > 0, received {entropy_normalizer}.")
    return policy_loss - float(entropy_coef) * (entropy / normalizer).mean()


def compute_grpo_like_policy_loss(
    log_probs: torch.Tensor,
    rewards: torch.Tensor,
    *,
    entropy: torch.Tensor | None = None,
    entropy_coef: float = 0.0,
    entropy_normalizer: float = 1.0,
    advantage_eps: float = 1e-6,
    group_dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        advantages = compute_group_relative_advantage(
            rewards,
            eps=advantage_eps,
            group_dim=group_dim,
        )
    policy_loss = compute_policy_gradient_objective(
        log_probs,
        advantages,
        entropy=entropy,
        entropy_coef=entropy_coef,
        entropy_normalizer=entropy_normalizer,
    )
    return policy_loss, advantages


def compute_ppo_clipped_policy_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tuple(new_log_probs.shape) != tuple(old_log_probs.shape):
        raise ValueError(
            "new_log_probs and old_log_probs must share the same shape. "
            f"Received new_log_probs={tuple(new_log_probs.shape)} "
            f"old_log_probs={tuple(old_log_probs.shape)}."
        )
    if tuple(new_log_probs.shape) != tuple(advantages.shape):
        raise ValueError(
            "new_log_probs and advantages must share the same shape. "
            f"Received new_log_probs={tuple(new_log_probs.shape)} "
            f"advantages={tuple(advantages.shape)}."
        )
    clip_value = float(clip_range)
    if clip_value <= 0.0:
        raise ValueError(f"clip_range must be > 0, received {clip_range}.")

    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, min=1.0 - clip_value, max=1.0 + clip_value)
    detached_advantages = advantages.detach()
    unclipped_objective = ratio * detached_advantages
    clipped_objective = clipped_ratio * detached_advantages
    policy_loss = -torch.minimum(unclipped_objective, clipped_objective).mean()
    clip_fraction = (torch.abs(ratio - 1.0) > clip_value).to(dtype=ratio.dtype).mean()
    return policy_loss, ratio, clip_fraction


def compute_value_loss(
    values: torch.Tensor,
    value_targets: torch.Tensor,
) -> torch.Tensor:
    if tuple(values.shape) != tuple(value_targets.shape):
        raise ValueError(
            "values and value_targets must share the same shape. "
            f"Received values={tuple(values.shape)} value_targets={tuple(value_targets.shape)}."
        )
    return 0.5 * (values - value_targets.detach()).pow(2).mean()
