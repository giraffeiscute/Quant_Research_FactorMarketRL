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


def _format_nonfinite_tensor_summary(tensor: torch.Tensor, *, name: str) -> str:
    detached = tensor.detach()
    finite_mask = torch.isfinite(detached)
    invalid_mask = ~finite_mask
    invalid_indices = torch.nonzero(invalid_mask, as_tuple=False)
    first_invalid_index = (
        tuple(int(value) for value in invalid_indices[0].tolist())
        if invalid_indices.numel() > 0
        else None
    )
    return (
        f"{name} contains non-finite values: "
        f"shape={tuple(detached.shape)} dtype={detached.dtype} device={detached.device} "
        f"first_invalid_index={first_invalid_index}"
    )


def _raise_if_not_finite(
    tensor: torch.Tensor,
    *,
    name: str,
    debug_context: str | None,
) -> None:
    if torch.isfinite(tensor).all():
        return
    context = f" context={debug_context}" if debug_context else ""
    raise FloatingPointError(
        f"{_format_nonfinite_tensor_summary(tensor, name=name)}.{context}"
    )


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

    def forward(
        self,
        logits: torch.Tensor,
        *,
        debug_context: str | None = None,
        logits_name: str = "allocation_logits",
    ) -> AllocationDistributionResult:
        _raise_if_not_finite(logits, name=logits_name, debug_context=debug_context)
        if self.allocation_distribution_type == "softmax":
            raw_allocation = torch.softmax(logits, dim=-1)
            _raise_if_not_finite(
                raw_allocation,
                name="raw_allocation",
                debug_context=debug_context,
            )
            return AllocationDistributionResult(
                raw_allocation=raw_allocation,
                alpha=None,
                debug_info={
                    "allocation_distribution_type": "softmax",
                    "allocation_sampling_mode": "deterministic",
                    "dirichlet_alpha_offset": None,
                },
            )

        alpha = F.softplus(logits) + self.dirichlet_alpha_offset
        _raise_if_not_finite(alpha, name="dirichlet_alpha", debug_context=debug_context)
        if self.training:
            raw_allocation = Dirichlet(alpha).rsample()
            sampling_mode = "rsample"
        else:
            raw_allocation = alpha / alpha.sum(dim=-1, keepdim=True)
            sampling_mode = "mean"
        _raise_if_not_finite(
            raw_allocation,
            name="raw_allocation",
            debug_context=debug_context,
        )

        return AllocationDistributionResult(
            raw_allocation=raw_allocation,
            alpha=alpha,
            debug_info={
                "allocation_distribution_type": "dirichlet",
                "allocation_sampling_mode": sampling_mode,
                "dirichlet_alpha_offset": self.dirichlet_alpha_offset,
            },
        )
