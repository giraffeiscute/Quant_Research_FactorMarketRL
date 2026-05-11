"""Portfolio allocation distribution modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class AllocationDistributionResult:
    raw_allocation: torch.Tensor
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
    """Convert allocation logits into softmax portfolio weights."""

    def forward(
        self,
        logits: torch.Tensor,
        *,
        debug_context: str | None = None,
        logits_name: str = "allocation_logits",
    ) -> AllocationDistributionResult:
        _raise_if_not_finite(logits, name=logits_name, debug_context=debug_context)
        raw_allocation = torch.softmax(logits, dim=-1)
        _raise_if_not_finite(
            raw_allocation,
            name="raw_allocation",
            debug_context=debug_context,
        )

        return AllocationDistributionResult(
            raw_allocation=raw_allocation,
            debug_info={
                "allocation_distribution_type": "softmax",
                "allocation_sampling_mode": "deterministic",
            },
        )
