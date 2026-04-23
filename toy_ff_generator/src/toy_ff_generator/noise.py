from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def resolve_epsilon_sigma(
    epsilon_group: str,
    epsilon_levels: Mapping[str, float],
) -> float:
    """Return the configured epsilon sigma for the selected group."""

    try:
        return float(epsilon_levels[epsilon_group])
    except KeyError as exc:
        raise ValueError(
            f"epsilon_group must be one of {sorted(epsilon_levels)}. Received {epsilon_group!r}."
        ) from exc


def generate_noise(
    stock_ids: Sequence[str],
    time_columns: Sequence[str],
    epsilon_group: str,
    epsilon_levels: Mapping[str, float],
    rng: np.random.Generator,
    per_stock_epsilon_groups: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Generate epsilon draws with either a shared or fixed per-stock sigma group."""

    if per_stock_epsilon_groups is None:
        sigma_values = np.full(
            len(stock_ids),
            resolve_epsilon_sigma(
                epsilon_group=epsilon_group,
                epsilon_levels=epsilon_levels,
            ),
            dtype=float,
        )
    else:
        sigma_values = np.asarray(
            [
                resolve_epsilon_sigma(
                    epsilon_group=group_name,
                    epsilon_levels=epsilon_levels,
                )
                for group_name in per_stock_epsilon_groups
            ],
            dtype=float,
        )

    stock_count = len(stock_ids)
    time_count = len(time_columns)
    epsilon_matrix = rng.normal(
        loc=0.0,
        scale=sigma_values[:, np.newaxis],
        size=(stock_count, time_count),
    )
    return pd.DataFrame(
        {
            "stock_id": np.repeat(np.asarray(stock_ids, dtype=object), time_count),
            "t": np.tile(np.asarray(time_columns, dtype=object), stock_count),
            "epsilon": epsilon_matrix.reshape(stock_count * time_count),
        }
    )
