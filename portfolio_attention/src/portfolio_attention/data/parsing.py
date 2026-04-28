"""Parsing helpers for scenario panel files."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from .constants import SCENARIO_FILE_PATTERN


def parse_panel_dimensions(file_name: str) -> tuple[int, int]:
    """Parse `(N, T)` from a scenario file name like `{state}_{N}_{T}_PL_{idx}.parquet`."""

    match = SCENARIO_FILE_PATTERN.fullmatch(file_name)
    if not match:
        raise ValueError(f"Could not parse N/T from file name: {file_name}")
    return int(match.group("n")), int(match.group("t"))


def parse_scenario_file_info(file_name: str) -> dict[str, int | str]:
    match = SCENARIO_FILE_PATTERN.fullmatch(file_name)
    if not match:
        raise ValueError(f"Unsupported scenario file name: {file_name}")
    return {
        "state": match.group("state"),
        "parsed_n": int(match.group("n")),
        "parsed_t": int(match.group("t")),
        "scenario_index": int(match.group("scenario_index")),
    }


def _parse_time_label(raw_value: Any) -> int:
    if isinstance(raw_value, str):
        match = re.fullmatch(r"t_(\d+)", raw_value)
        if not match:
            raise ValueError(f"Unsupported time label: {raw_value}")
        return int(match.group(1))
    return int(raw_value)


def _parse_time_series(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        return pd.to_numeric(series, errors="raise").to_numpy(dtype=np.int64, copy=False)

    normalized = series.astype(str).str.strip()
    if not normalized.str.startswith("t_").all():
        invalid = normalized[~normalized.str.startswith("t_")].iloc[0]
        raise ValueError(f"Unsupported time label: {invalid}")
    return normalized.str.slice(2).astype(np.int64).to_numpy(copy=False)


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="raise")

    normalized = series.astype(str).str.strip()
    percent_mask = normalized.str.endswith("%")
    if percent_mask.any():
        normalized = normalized.str.replace("%", "", regex=False)
        numeric = pd.to_numeric(normalized, errors="raise")
        return numeric / 100.0
    return pd.to_numeric(normalized, errors="raise")
