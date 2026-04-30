"""Stateless temporal utility functions for portfolio models."""

from __future__ import annotations

import math

import torch


def causal_running_mean(values: torch.Tensor) -> torch.Tensor:
    if values.ndim < 2:
        raise ValueError("Expected at least 2 dimensions for causal running mean.")
    steps = torch.arange(
        1,
        values.shape[1] + 1,
        device=values.device,
        dtype=values.dtype,
    )
    view_shape = [1, values.shape[1]] + [1] * (values.ndim - 2)
    return values.cumsum(dim=1) / steps.view(*view_shape)


def fixed_window_causal_mean(values: torch.Tensor, window_size: int) -> torch.Tensor:
    if values.ndim < 2:
        raise ValueError("Expected at least 2 dimensions for fixed-window causal mean.")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, received {window_size}.")

    time_steps = values.shape[1]
    if window_size >= time_steps:
        return causal_running_mean(values)

    cumsum = values.cumsum(dim=1)
    window_sums = cumsum.clone()
    window_sums[:, window_size:] = cumsum[:, window_size:] - cumsum[:, :-window_size]
    counts = torch.arange(
        1,
        time_steps + 1,
        device=values.device,
        dtype=values.dtype,
    ).clamp(max=window_size)
    view_shape = [1, time_steps] + [1] * (values.ndim - 2)
    return window_sums / counts.view(*view_shape)


def build_local_causal_window_mask(
    *,
    time_steps: int,
    window_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if time_steps <= 0:
        raise ValueError(f"time_steps must be positive, received {time_steps}.")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, received {window_size}.")

    query_positions = torch.arange(time_steps, device=device).unsqueeze(1)
    key_positions = torch.arange(time_steps, device=device).unsqueeze(0)
    earliest_allowed = query_positions - (window_size - 1)
    disallowed = (key_positions > query_positions) | (key_positions < earliest_allowed)
    mask = torch.zeros((time_steps, time_steps), device=device, dtype=dtype)
    return mask.masked_fill(disallowed, float("-inf"))


def build_sinusoidal_time_encoding(
    *,
    time_steps: int,
    embedding_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if time_steps <= 0:
        raise ValueError(f"time_steps must be positive, received {time_steps}.")
    if embedding_dim <= 0:
        raise ValueError(f"embedding_dim must be positive, received {embedding_dim}.")

    positions = torch.arange(time_steps, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embedding_dim, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / embedding_dim)
    )
    encoding = torch.zeros((time_steps, embedding_dim), device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(positions * div_term)
    encoding[:, 1::2] = torch.cos(positions * div_term[: encoding[:, 1::2].shape[1]])
    return encoding.unsqueeze(0)
