"""Dirichlet policy helpers for SAC portfolio actions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Dirichlet

from ..config import ModelConfig, TrainConfig
from ..model.allocation_distribution import logits_to_rl_post_train_dirichlet_alpha


@dataclass(frozen=True)
class DirichletSACPolicy:
    """Batched Dirichlet policy over full portfolio allocation [stocks..., cash]."""

    alpha: torch.Tensor
    distribution: Dirichlet

    def sample_action(self) -> torch.Tensor:
        """Sample a detached Monte Carlo action with shape [B, T, N+1]."""
        return self.distribution.sample().detach()

    def rsample_action(self) -> torch.Tensor:
        """Sample a reparameterized action with shape [B, T, N+1]."""
        return self.distribution.rsample()

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Return log probability for full allocation actions with shape [B, T]."""
        _validate_actions(actions, self.alpha)
        return self.distribution.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        """Return Dirichlet entropy with shape [B, T]."""
        return self.distribution.entropy()


def build_dirichlet_sac_policy(
    allocation_logits: torch.Tensor,
    *,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> DirichletSACPolicy:
    """Build a SAC policy distribution from allocation logits [B, T, N+1]."""
    alpha = sac_dirichlet_alpha_from_logits(
        allocation_logits,
        model_config=model_config,
        train_config=train_config,
    )
    return DirichletSACPolicy(alpha=alpha, distribution=Dirichlet(alpha))


def sac_dirichlet_alpha_from_logits(
    allocation_logits: torch.Tensor,
    *,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> torch.Tensor:
    """Convert full allocation logits [B, T, N+1] to SAC Dirichlet alpha."""
    _validate_allocation_logits(allocation_logits)
    return logits_to_rl_post_train_dirichlet_alpha(
        allocation_logits,
        alpha_min=float(train_config.rl_training.alpha_min),
        alpha_max=float(train_config.rl_training.alpha_max),
        logit_scale=float(model_config.dirichlet_logit_scale),
        evidence_scale=float(train_config.rl_training.rl_post_train_evidence_scale),
    )


def _validate_allocation_logits(allocation_logits: torch.Tensor) -> None:
    if allocation_logits.ndim != 3:
        raise ValueError(
            "allocation_logits must have shape [B, T, N+1] including cash. "
            f"Received {tuple(allocation_logits.shape)}."
        )
    if int(allocation_logits.shape[-1]) < 2:
        raise ValueError(
            "allocation_logits must have shape [B, T, N+1] including cash; "
            "action dimension must be N+1 >= 2. "
            f"Received {tuple(allocation_logits.shape)}."
        )
    if not torch.is_floating_point(allocation_logits):
        raise TypeError("allocation_logits must be a floating point tensor.")
    if not torch.isfinite(allocation_logits).all():
        raise ValueError("allocation_logits must be finite.")


def _validate_actions(actions: torch.Tensor, alpha: torch.Tensor) -> None:
    if tuple(actions.shape) != tuple(alpha.shape):
        raise ValueError(
            "actions must have shape [B, T, N+1] and match policy alpha. "
            f"Received actions={tuple(actions.shape)} alpha={tuple(alpha.shape)}."
        )
    if not torch.is_floating_point(actions):
        raise TypeError("actions must be a floating point tensor.")
    if not torch.isfinite(actions).all():
        raise ValueError("actions must be finite.")
    if (actions < 0).any():
        raise ValueError("Dirichlet actions must be non-negative.")
    sums = actions.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-4, rtol=1e-4):
        raise ValueError("Dirichlet actions must sum to 1 over the last dimension.")
