"""Deterministic stock sampling helpers for training windows."""

from __future__ import annotations

import math

import numpy as np


def validate_sample_num_stocks(sample_num_stocks: int, full_num_stocks: int) -> int:
    """Validate and return the per-window stock sample count."""
    resolved_sample_num_stocks = int(sample_num_stocks)
    resolved_full_num_stocks = int(full_num_stocks)
    if resolved_sample_num_stocks <= 0:
        raise ValueError(
            "sample_num_stocks must be positive, "
            f"received {sample_num_stocks}."
        )
    if resolved_full_num_stocks <= 0:
        raise ValueError(
            "full_num_stocks must be positive, "
            f"received {full_num_stocks}."
        )
    if resolved_sample_num_stocks > resolved_full_num_stocks:
        raise ValueError(
            "sample_num_stocks cannot exceed full_num_stocks. "
            f"sample_num_stocks={resolved_sample_num_stocks} "
            f"full_num_stocks={resolved_full_num_stocks}."
        )
    return resolved_sample_num_stocks


def coverage_cycle_stock_indices(
    window_ordinal: int,
    sample_num_stocks: int,
    full_num_stocks: int,
    base_seed: int = 0,
) -> np.ndarray:
    """Return global stock indices for one canonical training window."""
    resolved_full_num_stocks = int(full_num_stocks)
    resolved_sample_num_stocks = validate_sample_num_stocks(
        sample_num_stocks,
        resolved_full_num_stocks,
    )
    if int(window_ordinal) < 0:
        raise ValueError(f"window_ordinal must be non-negative, received {window_ordinal}.")
    if resolved_sample_num_stocks == resolved_full_num_stocks:
        return np.arange(resolved_full_num_stocks, dtype=np.int64)

    cycle_size = int(math.ceil(resolved_full_num_stocks / resolved_sample_num_stocks))
    cycle_id = int(window_ordinal) // cycle_size
    slot_id = int(window_ordinal) % cycle_size
    rng = np.random.default_rng(int(base_seed) + cycle_id)
    permutation = rng.permutation(resolved_full_num_stocks).astype(np.int64, copy=False)

    start = slot_id * resolved_sample_num_stocks
    end = start + resolved_sample_num_stocks
    if end <= resolved_full_num_stocks:
        return permutation[start:end].astype(np.int64, copy=False)

    tail = permutation[start:resolved_full_num_stocks]
    remaining = resolved_sample_num_stocks - int(tail.shape[0])
    return np.concatenate([tail, permutation[:remaining]]).astype(np.int64, copy=False)
