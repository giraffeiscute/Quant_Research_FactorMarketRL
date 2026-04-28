"""Time-window bounds and rolling-window layout helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeWindowLayout:
    train_segment_start_index: int
    train_segment_end_index: int
    validation_segment_start_index: int
    validation_segment_end_index: int
    test_segment_start_index: int
    test_segment_end_index: int
    train_segment_raw_length: int
    validation_segment_raw_length: int
    test_segment_raw_length: int
    train_segment_time_steps: int
    validation_segment_time_steps: int
    test_segment_time_steps: int
    train_context_time_steps: int
    train_score_time_steps: int
    train_warmup_time_steps: int
    train_windows_per_scenario: int
    max_time_steps: int


def context_bounds_for(
    split_name: str,
    *,
    train_segment_start_index: int,
    train_segment_end_index: int,
    parsed_t: int,
) -> tuple[int, int]:
    if split_name == "train":
        return train_segment_start_index, train_segment_end_index - 1
    if split_name == "validation":
        return 0, parsed_t - 1
    if split_name == "test":
        return 0, parsed_t - 1
    raise ValueError(f"Unsupported split_name: {split_name}")


def score_target_bounds_for(
    split_name: str,
    *,
    train_segment_start_index: int,
    train_segment_end_index: int,
    parsed_t: int,
    lookback_days: int,
) -> tuple[int, int]:
    if split_name == "train":
        return train_segment_start_index + 1, train_segment_end_index
    if split_name == "validation":
        return int(lookback_days) + 1, parsed_t
    if split_name == "test":
        return int(lookback_days) + 1, parsed_t
    raise ValueError(f"Unsupported split_name: {split_name}")


def resolve_time_window_layout(
    *,
    total_time_steps: int,
    lookback_days: int,
    rolling_horizon_days: int,
    rolling_stride_days: int,
) -> TimeWindowLayout:
    if total_time_steps < 2:
        raise ValueError(
            "Each scenario must contain at least 2 raw timestamps so that one-step "
            "target returns can be formed without future leakage."
        )

    train_segment_start_index = 0
    train_segment_end_index = int(total_time_steps)
    validation_segment_start_index = 0
    validation_segment_end_index = int(total_time_steps)
    test_segment_start_index = 0
    test_segment_end_index = int(total_time_steps)

    train_segment_raw_length = train_segment_end_index - train_segment_start_index
    validation_segment_raw_length = validation_segment_end_index - validation_segment_start_index
    test_segment_raw_length = test_segment_end_index - test_segment_start_index

    train_segment_time_steps = train_segment_raw_length - 1
    validation_context_start, validation_context_stop = context_bounds_for(
        "validation",
        train_segment_start_index=train_segment_start_index,
        train_segment_end_index=train_segment_end_index,
        parsed_t=int(total_time_steps),
    )
    test_context_start, test_context_stop = context_bounds_for(
        "test",
        train_segment_start_index=train_segment_start_index,
        train_segment_end_index=train_segment_end_index,
        parsed_t=int(total_time_steps),
    )
    validation_segment_time_steps = validation_context_stop - validation_context_start
    test_segment_time_steps = test_context_stop - test_context_start
    train_context_time_steps = int(lookback_days) + int(rolling_horizon_days)
    train_score_time_steps = int(rolling_horizon_days)
    train_warmup_time_steps = int(lookback_days)

    available_train_steps = train_segment_time_steps
    if available_train_steps < train_context_time_steps:
        raise ValueError(
            "Train scenario is too short for rolling_window mode. "
            f"Need at least lookback_days + rolling_horizon_days = {train_context_time_steps} "
            f"train time steps, but only found {available_train_steps}."
        )
    required_scored_steps = int(lookback_days) + 1
    if validation_segment_time_steps < required_scored_steps:
        raise ValueError(
            "Validation/test scenarios are too short for full-scenario evaluation. "
            f"Need parsed_t - 1 > lookback_days, but found parsed_t={total_time_steps} "
            f"and lookback_days={int(lookback_days)}."
        )
    train_windows_per_scenario = (
        ((available_train_steps - train_context_time_steps) // int(rolling_stride_days)) + 1
    )
    max_time_steps = max(
        train_context_time_steps,
        validation_segment_time_steps,
        test_segment_time_steps,
    )

    return TimeWindowLayout(
        train_segment_start_index=train_segment_start_index,
        train_segment_end_index=train_segment_end_index,
        validation_segment_start_index=validation_segment_start_index,
        validation_segment_end_index=validation_segment_end_index,
        test_segment_start_index=test_segment_start_index,
        test_segment_end_index=test_segment_end_index,
        train_segment_raw_length=train_segment_raw_length,
        validation_segment_raw_length=validation_segment_raw_length,
        test_segment_raw_length=test_segment_raw_length,
        train_segment_time_steps=train_segment_time_steps,
        validation_segment_time_steps=validation_segment_time_steps,
        test_segment_time_steps=test_segment_time_steps,
        train_context_time_steps=train_context_time_steps,
        train_score_time_steps=train_score_time_steps,
        train_warmup_time_steps=train_warmup_time_steps,
        train_windows_per_scenario=train_windows_per_scenario,
        max_time_steps=max_time_steps,
    )
