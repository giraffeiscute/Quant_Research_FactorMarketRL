"""Allocation smoothing and turnover helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class AllocationResult:
    raw_allocation: torch.Tensor
    allocation: torch.Tensor
    previous_allocation: torch.Tensor
    turnover: torch.Tensor
    initial_allocation: torch.Tensor


def compute_previous_allocation(
    allocation: torch.Tensor,
    *,
    initial_allocation: torch.Tensor,
    detach_prev_allocation: bool = False,
) -> torch.Tensor:
    if allocation.ndim != 3:
        raise ValueError("allocation must have shape [S, T, N+1].")
    if initial_allocation.shape != (allocation.shape[0], allocation.shape[2]):
        raise ValueError(
            "initial_allocation must have shape [S, N+1]. "
            f"Received {tuple(initial_allocation.shape)} expected {(allocation.shape[0], allocation.shape[2])}."
        )
    prev_allocation = torch.cat(
        [initial_allocation.unsqueeze(1), allocation[:, :-1, :]],
        dim=1,
    )
    if detach_prev_allocation:
        prev_allocation = prev_allocation.detach()
    return prev_allocation


def compute_turnover_from_allocation(
    allocation: torch.Tensor,
    *,
    initial_allocation: torch.Tensor,
    detach_prev_allocation: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    prev_allocation = compute_previous_allocation(
        allocation,
        initial_allocation=initial_allocation,
        detach_prev_allocation=detach_prev_allocation,
    )
    allocation_delta = allocation - prev_allocation
    turnover = 0.5 * torch.abs(allocation_delta).sum(dim=-1)
    return turnover, prev_allocation


class AllocationSmoother(nn.Module):
    """Convert raw portfolio weights into a smoothed allocation path."""

    def __init__(
        self,
        *,
        initial_allocation_mode: str,
        initial_random_concentration: float,
        allocation_smoothing_alpha: float,
        detach_prev_weight: bool,
    ) -> None:
        super().__init__()
        self.initial_allocation_mode = str(initial_allocation_mode).strip().lower()
        self.initial_random_concentration = float(initial_random_concentration)
        self.allocation_smoothing_alpha = float(allocation_smoothing_alpha)
        if not isinstance(detach_prev_weight, bool):
            raise ValueError(
                "detach_prev_weight must be a bool, "
                f"received {detach_prev_weight!r}."
            )
        self.detach_prev_weight = detach_prev_weight

        if self.initial_allocation_mode not in {"equal_weight", "random_dirichlet"}:
            raise ValueError(
                "initial_allocation_mode must be one of {'equal_weight', 'random_dirichlet'}, "
                f"received {self.initial_allocation_mode!r}."
            )
        if not 0.0 <= self.allocation_smoothing_alpha <= 1.0:
            raise ValueError(
                "allocation_smoothing_alpha must be in [0.0, 1.0], "
                f"received {self.allocation_smoothing_alpha}."
            )
        if self.initial_random_concentration <= 0.0:
            raise ValueError(
                "initial_random_concentration must be > 0.0, "
                f"received {self.initial_random_concentration}."
            )

    def initial_allocation(
        self,
        *,
        num_scenarios: int,
        total_assets: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.initial_allocation_mode == "equal_weight":
            return torch.full(
                (num_scenarios, total_assets),
                1.0 / total_assets,
                device=device,
                dtype=dtype,
            )

        if self.initial_allocation_mode == "random_dirichlet":
            concentration = torch.full(
                (total_assets,),
                self.initial_random_concentration,
                device=device,
                dtype=dtype,
            )
            return torch.distributions.Dirichlet(concentration).sample((num_scenarios,))

        raise ValueError(f"Unsupported initial_allocation_mode: {self.initial_allocation_mode!r}")

    def forward(
        self,
        raw_allocation: torch.Tensor,
        *,
        initial_allocation: torch.Tensor | None = None,
    ) -> AllocationResult:
        if raw_allocation.ndim != 3:
            raise ValueError("raw_allocation must have shape [S, T, N+1].")
        num_scenarios, time_steps, total_assets = raw_allocation.shape
        if initial_allocation is None:
            prev_weight = self.initial_allocation(
                num_scenarios=num_scenarios,
                total_assets=total_assets,
                device=raw_allocation.device,
                dtype=raw_allocation.dtype,
            )
            initial_weight = prev_weight
        else:
            prev_weight = initial_allocation
            if prev_weight.shape != (num_scenarios, total_assets):
                raise ValueError(
                    "initial_allocation must have shape [S, N+1]. "
                    f"Received {tuple(prev_weight.shape)} expected {(num_scenarios, total_assets)}."
                )
            initial_weight = prev_weight

        alpha = float(self.allocation_smoothing_alpha)
        raw_allocations: list[torch.Tensor] = []
        allocations: list[torch.Tensor] = []
        turnovers: list[torch.Tensor] = []
        previous_allocations: list[torch.Tensor] = []
        for time_index in range(time_steps):
            prev_weight_step = prev_weight.detach() if self.detach_prev_weight else prev_weight
            raw_t = raw_allocation[:, time_index, :]
            allocation_t = alpha * raw_t + (1.0 - alpha) * prev_weight_step
            allocation_delta_t = allocation_t - prev_weight_step
            turnover_t = 0.5 * torch.abs(allocation_delta_t).sum(dim=-1)
            raw_allocations.append(raw_t)
            allocations.append(allocation_t)
            turnovers.append(turnover_t)
            previous_allocations.append(prev_weight_step)
            prev_weight = allocation_t

        return AllocationResult(
            raw_allocation=torch.stack(raw_allocations, dim=1),
            allocation=torch.stack(allocations, dim=1),
            previous_allocation=torch.stack(previous_allocations, dim=1),
            turnover=torch.stack(turnovers, dim=1),
            initial_allocation=initial_weight,
        )
