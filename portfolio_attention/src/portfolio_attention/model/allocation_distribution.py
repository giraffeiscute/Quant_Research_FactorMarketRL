"""Portfolio allocation distribution modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Dirichlet


@dataclass
class AllocationDistributionResult:
    raw_allocation: torch.Tensor
    alpha: torch.Tensor | None = None
    debug_info: dict[str, Any] | None = None


class AllocationDistribution(nn.Module):
    """Convert allocation logits into raw portfolio allocation weights.

    Softmax is the deterministic baseline. Dirichlet mode uses reparameterized
    samples during training and the distribution mean during evaluation.
    """

    def __init__(
        self,
        *,
        allocation_distribution_type: str = "softmax",
        dirichlet_alpha_offset: float = 0.1,
    ) -> None:
        super().__init__()
        allocation_distribution_type = str(allocation_distribution_type).strip().lower()
        if allocation_distribution_type not in {"softmax", "dirichlet"}:
            raise ValueError(
                "allocation_distribution_type must be one of {'softmax', 'dirichlet'}, "
                f"received {allocation_distribution_type!r}."
            )

        dirichlet_alpha_offset = float(dirichlet_alpha_offset)
        if dirichlet_alpha_offset <= 0.0:
            raise ValueError(
                "dirichlet_alpha_offset must be > 0.0, "
                f"received {dirichlet_alpha_offset}."
            )

        self.allocation_distribution_type = allocation_distribution_type
        self.dirichlet_alpha_offset = dirichlet_alpha_offset

    def forward(self, logits: torch.Tensor) -> AllocationDistributionResult:
        if self.allocation_distribution_type == "softmax":
            raw_allocation = torch.softmax(logits, dim=-1)
            return AllocationDistributionResult(
                raw_allocation=raw_allocation,
                alpha=None,
                debug_info={
                    "allocation_distribution_type": "softmax",
                    "allocation_sampling_mode": "deterministic",
                    "dirichlet_alpha_offset": None,
                    "dirichlet_alpha_min": None,
                    "dirichlet_alpha_max": None,
                    "dirichlet_alpha_mean": None,
                    "dirichlet_alpha_sum_mean": None,
                },
            )

        alpha = F.softplus(logits) + self.dirichlet_alpha_offset
        if self.training:
            raw_allocation = Dirichlet(alpha).rsample()
            sampling_mode = "rsample"
        else:
            raw_allocation = alpha / alpha.sum(dim=-1, keepdim=True)
            sampling_mode = "mean"

        alpha_detached = alpha.detach()
        alpha_sum = alpha_detached.sum(dim=-1)
        return AllocationDistributionResult(
            raw_allocation=raw_allocation,
            alpha=alpha,
            debug_info={
                "allocation_distribution_type": "dirichlet",
                "allocation_sampling_mode": sampling_mode,
                "dirichlet_alpha_offset": self.dirichlet_alpha_offset,
                "dirichlet_alpha_min": alpha_detached.min().item(),
                "dirichlet_alpha_max": alpha_detached.max().item(),
                "dirichlet_alpha_mean": alpha_detached.mean().item(),
                "dirichlet_alpha_sum_mean": alpha_sum.mean().item(),
            },
        )
