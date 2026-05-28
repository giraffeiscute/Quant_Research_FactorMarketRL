"""Rebalance-interval helpers for RL rollouts.

These helpers assume one rollout time step corresponds to one trading day.
The public config keeps the user-facing name ``rebalance_interval_days``;
inside this module the value is used as a tensor-step interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RebalanceSchedule:
    """Half-open rebalance segments over a rollout time axis.

    ``starts`` contains decision time indices. Each action is held over the
    matching ``[start, end)`` segment.
    """

    starts: tuple[int, ...]
    ends: tuple[int, ...]

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(end - start for start, end in zip(self.starts, self.ends))

    @property
    def num_decisions(self) -> int:
        return len(self.starts)


def validate_rebalance_interval_days(value: int) -> int:
    interval = int(value)
    if interval <= 0:
        raise ValueError(
            "rebalance_interval_days must be positive, "
            f"received {interval}."
        )
    return interval


def validate_rebalance_schedule(schedule: RebalanceSchedule) -> None:
    """Validate schedule shape and contiguous half-open segments."""
    if schedule.num_decisions <= 0:
        raise ValueError("rebalance schedule must contain at least one decision.")
    if len(schedule.starts) != len(schedule.ends):
        raise ValueError("rebalance schedule starts and ends must have the same length.")
    expected_start = 0
    for start, end in zip(schedule.starts, schedule.ends):
        if int(start) != expected_start:
            raise ValueError(
                "rebalance schedule segments must be contiguous and start at 0. "
                f"Expected start={expected_start}, received start={start}."
            )
        if int(start) < 0 or int(end) <= int(start):
            raise ValueError(f"Invalid rebalance segment [{start}, {end}) in schedule.")
        expected_start = int(end)


def validate_schedule_with_time_dim(
    schedule: RebalanceSchedule,
    *,
    time_steps: int,
    name: str,
) -> None:
    """Validate that a schedule fits a tensor's time dimension."""
    validate_rebalance_schedule(schedule)
    resolved_time_steps = int(time_steps)
    if resolved_time_steps <= 0:
        raise ValueError(f"{name} time dimension must be positive, received {resolved_time_steps}.")
    schedule_end = int(schedule.ends[-1])
    if schedule_end > resolved_time_steps:
        raise ValueError(
            f"{name} time dimension is too short for rebalance schedule. "
            f"Received time_steps={resolved_time_steps}, schedule_end={schedule_end}."
        )


def build_rebalance_schedule(
    *,
    horizon_steps: int,
    rebalance_interval_days: int,
) -> RebalanceSchedule:
    """Build a rebalance schedule over rollout steps/trading days."""
    horizon = int(horizon_steps)
    if horizon <= 0:
        raise ValueError(f"horizon_steps must be positive, received {horizon}.")
    interval = validate_rebalance_interval_days(rebalance_interval_days)
    starts = tuple(range(0, horizon, interval))
    ends = tuple(min(start + interval, horizon) for start in starts)
    schedule = RebalanceSchedule(starts=starts, ends=ends)
    validate_schedule_with_time_dim(
        schedule,
        time_steps=horizon,
        name="horizon_steps",
    )
    return schedule


def decision_indices_tensor(
    schedule: RebalanceSchedule,
    *,
    device: torch.device,
) -> torch.Tensor:
    validate_rebalance_schedule(schedule)
    return torch.tensor(schedule.starts, dtype=torch.long, device=device)


def gather_decision_steps(
    values: torch.Tensor,
    *,
    schedule: RebalanceSchedule,
) -> torch.Tensor:
    if values.ndim < 2:
        raise ValueError(
            "values must include batch and time dimensions, "
            f"received {tuple(values.shape)}."
        )
    validate_schedule_with_time_dim(
        schedule,
        time_steps=int(values.shape[1]),
        name="values",
    )
    indices = decision_indices_tensor(schedule, device=values.device)
    return values.index_select(1, indices)


def expand_decision_steps(
    decision_values: torch.Tensor,
    *,
    schedule: RebalanceSchedule,
) -> torch.Tensor:
    if decision_values.ndim < 2:
        raise ValueError(
            "decision_values must include batch and decision dimensions, "
            f"received {tuple(decision_values.shape)}."
        )
    validate_rebalance_schedule(schedule)
    if int(decision_values.shape[1]) != schedule.num_decisions:
        raise ValueError(
            "decision_values decision dimension must match rebalance schedule. "
            f"Received decision_values={tuple(decision_values.shape)} "
            f"num_decisions={schedule.num_decisions}."
        )
    repeats = torch.tensor(schedule.lengths, dtype=torch.long, device=decision_values.device)
    return torch.repeat_interleave(decision_values, repeats, dim=1)


def compound_returns_by_schedule(
    daily_returns: torch.Tensor,
    *,
    schedule: RebalanceSchedule,
) -> torch.Tensor:
    if daily_returns.ndim != 2:
        raise ValueError(
            "daily_returns must have shape [B, T], "
            f"received {tuple(daily_returns.shape)}."
        )
    validate_schedule_with_time_dim(
        schedule,
        time_steps=int(daily_returns.shape[1]),
        name="daily_returns",
    )
    segment_returns = [
        daily_returns[:, start]
        if end - start == 1
        else torch.prod(1.0 + daily_returns[:, start:end], dim=1) - 1.0
        for start, end in zip(schedule.starts, schedule.ends)
    ]
    return torch.stack(segment_returns, dim=1)
