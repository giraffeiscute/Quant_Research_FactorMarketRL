"""Rollout target utilities for future rolling PPO."""

from __future__ import annotations

import torch


def compute_discounted_reward_targets(
    rewards: torch.Tensor,
    *,
    gamma: float,
    bootstrap_value: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute discounted reward targets from sampled rollout rewards."""

    _validate_rollout_matrix(rewards, name="rewards")
    gamma = float(gamma)
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError(f"gamma must be in [0, 1], received {gamma}.")

    batch_size, horizon = rewards.shape
    if bootstrap_value is None:
        running_target = rewards.new_zeros(batch_size)
    else:
        if tuple(bootstrap_value.shape) != (batch_size,):
            raise ValueError(
                "bootstrap_value must have shape [B] matching rewards, "
                f"received {tuple(bootstrap_value.shape)} for rewards shape {tuple(rewards.shape)}."
            )
        running_target = bootstrap_value

    targets = torch.empty_like(rewards)
    for timestep in range(horizon - 1, -1, -1):
        running_target = rewards[:, timestep] + gamma * running_target
        targets[:, timestep] = running_target

    return targets


def compute_rollout_advantages_from_targets(
    targets: torch.Tensor,
    old_values: torch.Tensor,
    *,
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute rollout advantages from reward targets and detached old values."""

    _validate_rollout_matrix(targets, name="targets")
    if tuple(targets.shape) != tuple(old_values.shape):
        raise ValueError(
            "targets and old_values must have identical shapes, "
            f"received {tuple(targets.shape)} and {tuple(old_values.shape)}."
        )
    eps = float(eps)
    if eps <= 0.0:
        raise ValueError(f"eps must be > 0, received {eps}.")

    advantages = targets - old_values.detach()
    if normalize:
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + eps
        )
    return advantages


def _validate_rollout_matrix(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.ndim != 2:
        raise ValueError(
            f"{name} must have shape [B, T], received shape {tuple(tensor.shape)}."
        )
    if tensor.numel() == 0:
        raise ValueError(f"{name} must not be empty.")
