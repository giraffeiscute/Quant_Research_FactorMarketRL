"""Portfolio allocation distribution modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


DEFAULT_DIRICHLET_ALPHA_MIN = 1e-4
DEFAULT_DIRICHLET_LOGIT_SCALE = 3.0
DEFAULT_RL_POST_TRAIN_EVIDENCE_SCALE = 0.5
DEFAULT_RL_POST_TRAIN_DIRICHLET_ALPHA_MIN = 0.05
DEFAULT_RL_POST_TRAIN_DIRICHLET_ALPHA_MAX = 50.0


@dataclass
class AllocationDistributionResult:
    raw_allocation: torch.Tensor
    alpha: torch.Tensor | None = None
    debug_info: dict[str, Any] | None = None


def logits_to_dirichlet_alpha(
    logits: torch.Tensor,
    *,
    alpha_min: float = DEFAULT_DIRICHLET_ALPHA_MIN,
    logit_scale: float = DEFAULT_DIRICHLET_LOGIT_SCALE,
) -> torch.Tensor:
    """Convert allocation logits to ordinary-training Dirichlet concentration parameters."""
    alpha_min = float(alpha_min)
    logit_scale = float(logit_scale)
    if alpha_min <= 0.0:
        raise ValueError(f"alpha_min must be positive, received {alpha_min}.")
    if logit_scale <= 0.0:
        raise ValueError(f"logit_scale must be positive, received {logit_scale}.")
    return F.softplus(logit_scale * logits) + alpha_min


def logits_to_rl_post_train_dirichlet_alpha(
    logits: torch.Tensor,
    *,
    alpha_min: float = DEFAULT_RL_POST_TRAIN_DIRICHLET_ALPHA_MIN,
    alpha_max: float = DEFAULT_RL_POST_TRAIN_DIRICHLET_ALPHA_MAX,
    logit_scale: float = DEFAULT_DIRICHLET_LOGIT_SCALE,
    evidence_scale: float = DEFAULT_RL_POST_TRAIN_EVIDENCE_SCALE,
) -> torch.Tensor:
    """Convert allocation logits to bounded RL post-training Dirichlet parameters."""
    alpha_min = float(alpha_min)
    alpha_max = float(alpha_max)
    logit_scale = float(logit_scale)
    evidence_scale = float(evidence_scale)

    if alpha_min <= 0.0:
        raise ValueError(f"alpha_min must be positive, received {alpha_min}.")
    if alpha_max <= 0.0:
        raise ValueError(f"alpha_max must be positive, received {alpha_max}.")
    if alpha_min > alpha_max:
        raise ValueError(
            "alpha_min must be <= alpha_max, "
            f"received alpha_min={alpha_min} alpha_max={alpha_max}."
        )
    if logit_scale <= 0.0:
        raise ValueError(f"logit_scale must be positive, received {logit_scale}.")
    if evidence_scale <= 0.0:
        raise ValueError(f"evidence_scale must be positive, received {evidence_scale}.")

    alpha = evidence_scale * F.softplus(logit_scale * logits)

    return torch.clamp(alpha, min=alpha_min, max=alpha_max)


def dirichlet_mean_from_logits(
    logits: torch.Tensor,
    *,
    alpha_min: float = DEFAULT_DIRICHLET_ALPHA_MIN,
    logit_scale: float = DEFAULT_DIRICHLET_LOGIT_SCALE,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    alpha = logits_to_dirichlet_alpha(
        logits,
        alpha_min=alpha_min,
        logit_scale=logit_scale,
    )
    mean = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    return mean, alpha


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
    """Convert allocation logits into deterministic portfolio weights."""

    _VALID_INFERENCE_ALLOCATION_MODES = frozenset({"softmax", "dirichlet_mean"})

    def __init__(
        self,
        *,
        inference_allocation_mode: str = "softmax",
        dirichlet_alpha_min: float = DEFAULT_DIRICHLET_ALPHA_MIN,
        dirichlet_logit_scale: float = DEFAULT_DIRICHLET_LOGIT_SCALE,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.inference_allocation_mode = str(inference_allocation_mode).strip().lower()
        if self.inference_allocation_mode not in self._VALID_INFERENCE_ALLOCATION_MODES:
            raise ValueError(
                "inference_allocation_mode must be one of "
                f"{sorted(self._VALID_INFERENCE_ALLOCATION_MODES)}, "
                f"received {self.inference_allocation_mode!r}."
            )
        self.eps = float(eps)
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, received {self.eps}.")
        self.dirichlet_alpha_min = float(dirichlet_alpha_min)
        self.dirichlet_logit_scale = float(dirichlet_logit_scale)
        if self.dirichlet_alpha_min <= 0.0:
            raise ValueError(
                "dirichlet_alpha_min must be positive, "
                f"received {self.dirichlet_alpha_min}."
            )
        if self.dirichlet_logit_scale <= 0.0:
            raise ValueError(
                "dirichlet_logit_scale must be positive, "
                f"received {self.dirichlet_logit_scale}."
            )

    def _deterministic_allocation(
        self,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.inference_allocation_mode == "softmax":
            return torch.softmax(logits, dim=-1), None
        if self.inference_allocation_mode == "dirichlet_mean":
            return dirichlet_mean_from_logits(
                logits,
                alpha_min=self.dirichlet_alpha_min,
                logit_scale=self.dirichlet_logit_scale,
                eps=self.eps,
            )
        raise ValueError(
            "inference_allocation_mode must be one of "
            f"{sorted(self._VALID_INFERENCE_ALLOCATION_MODES)}, "
            f"received {self.inference_allocation_mode!r}."
        )

    def forward(
        self,
        logits: torch.Tensor,
        *,
        debug_context: str | None = None,
        logits_name: str = "allocation_logits",
    ) -> AllocationDistributionResult:
        _raise_if_not_finite(logits, name=logits_name, debug_context=debug_context)
        raw_allocation, alpha = self._deterministic_allocation(logits)
        _raise_if_not_finite(
            raw_allocation,
            name="raw_allocation",
            debug_context=debug_context,
        )
        if alpha is not None:
            _raise_if_not_finite(
                alpha,
                name="dirichlet_alpha",
                debug_context=debug_context,
            )

        return AllocationDistributionResult(
            raw_allocation=raw_allocation,
            alpha=alpha,
            debug_info={
                "allocation_distribution_type": self.inference_allocation_mode,
                "allocation_sampling_mode": "deterministic",
            },
        )
