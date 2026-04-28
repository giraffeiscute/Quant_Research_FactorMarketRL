"""Parquet scenario loading and dense panel materialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .constants import (
    BASE_REQUIRED_COLUMNS,
    LOADABLE_COLUMNS,
    MARKET_FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    OPTIONAL_RETURN_COLUMN,
    STOCK_FEATURE_COLUMNS,
)
from .parsing import _coerce_numeric_series, _parse_time_series
from .records import LoadedScenarioArrays, ScenarioFileRecord


@dataclass
class PreparedScenarioFrame:
    frame: pd.DataFrame
    source_path: Path
    has_return_column: bool
    stock_ids: list[str]
    time_index: list[int]


def read_scenario_frame(source_path: Path) -> pd.DataFrame:
    available_columns = set(pq.read_schema(source_path).names)
    columns = [column_name for column_name in LOADABLE_COLUMNS if column_name in available_columns]
    return pd.read_parquet(source_path, columns=columns)


def prepare_scenario_frame(
    scenario_record: ScenarioFileRecord,
    *,
    raise_if_interrupted: Callable[[], None] | None = None,
) -> PreparedScenarioFrame:
    if raise_if_interrupted is not None:
        raise_if_interrupted()
    source_path = Path(scenario_record.source_path)
    frame = read_scenario_frame(source_path)
    if raise_if_interrupted is not None:
        raise_if_interrupted()

    missing_columns = [column for column in BASE_REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {source_path}: {missing_columns}")

    has_return_column = OPTIONAL_RETURN_COLUMN in frame.columns
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = _coerce_numeric_series(frame[column])
    frame["time_index"] = _parse_time_series(frame["t"])

    if frame.duplicated(["stock_id", "time_index"]).any():
        raise ValueError(f"Scenario contains duplicated (stock_id, t) rows: {source_path}")

    stock_ids = sorted(frame["stock_id"].unique().tolist())
    time_index = sorted(frame["time_index"].unique().tolist())

    if len(stock_ids) != scenario_record.parsed_n:
        raise ValueError(
            f"Parsed N={scenario_record.parsed_n} from file name but CSV contains {len(stock_ids)} stocks."
        )
    if len(time_index) != scenario_record.parsed_t:
        raise ValueError(
            f"Parsed T={scenario_record.parsed_t} from file name but CSV contains {len(time_index)} times."
        )

    return PreparedScenarioFrame(
        frame=frame,
        source_path=source_path,
        has_return_column=has_return_column,
        stock_ids=stock_ids,
        time_index=time_index,
    )


def materialize_scenario_arrays(
    prepared: PreparedScenarioFrame,
    *,
    scenario_record: ScenarioFileRecord,
    reference_stock_ids: list[str],
    reference_time_index_array: np.ndarray,
    selected_stock_indices: np.ndarray,
    parsed_t: int,
    raise_if_interrupted: Callable[[], None] | None = None,
) -> LoadedScenarioArrays:
    frame = prepared.frame
    expected_row_count = scenario_record.parsed_n * scenario_record.parsed_t
    if len(frame) != expected_row_count:
        raise ValueError(f"Scenario is incomplete: {prepared.source_path}")

    stock_position = pd.Categorical(
        frame["stock_id"],
        categories=reference_stock_ids,
        ordered=True,
    ).codes
    if (stock_position < 0).any():
        raise ValueError("Scenario contains unknown stock IDs compared with the reference universe.")

    time_values = frame["time_index"].to_numpy(dtype=np.int64, copy=False)
    time_position = np.searchsorted(reference_time_index_array, time_values)
    if (
        (time_position >= len(reference_time_index_array)).any()
        or not np.array_equal(reference_time_index_array[time_position], time_values)
    ):
        raise ValueError("Scenario contains unexpected time indices compared with the reference grid.")

    linear_index = stock_position.astype(np.int64) * int(parsed_t) + time_position.astype(np.int64)
    coverage = np.bincount(linear_index, minlength=expected_row_count)
    if coverage.shape[0] != expected_row_count or not np.all(coverage == 1):
        raise ValueError(f"Scenario is incomplete after position mapping: {prepared.source_path}")

    if raise_if_interrupted is not None:
        raise_if_interrupted()
    stock_feature_values = frame[STOCK_FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    stock_feature_grid = np.empty((expected_row_count, len(STOCK_FEATURE_COLUMNS)), dtype=np.float32)
    stock_feature_grid[linear_index] = stock_feature_values
    stock_features_raw = stock_feature_grid.reshape(
        scenario_record.parsed_n,
        scenario_record.parsed_t,
        len(STOCK_FEATURE_COLUMNS),
    ).transpose(1, 0, 2)

    if raise_if_interrupted is not None:
        raise_if_interrupted()
    market_values = frame[MARKET_FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    market_grid = np.empty((expected_row_count, len(MARKET_FEATURE_COLUMNS)), dtype=np.float32)
    market_grid[linear_index] = market_values
    market_cube = market_grid.reshape(
        scenario_record.parsed_n,
        scenario_record.parsed_t,
        len(MARKET_FEATURE_COLUMNS),
    ).transpose(1, 0, 2)
    if not np.allclose(market_cube, market_cube[:, :1, :], atol=0.0, rtol=0.0):
        raise ValueError("FF3 factors are not identical across stocks within the same day.")
    market_features_raw = market_cube[:, 0, :]

    if prepared.has_return_column:
        return_values = frame[OPTIONAL_RETURN_COLUMN].to_numpy(dtype=np.float32, copy=True)
        return_grid = np.empty((expected_row_count,), dtype=np.float32)
        return_grid[linear_index] = return_values
        stock_returns_raw = return_grid.reshape(
            scenario_record.parsed_n,
            scenario_record.parsed_t,
        ).transpose(1, 0)
    else:
        price_array = stock_features_raw[..., -1]
        stock_returns_raw = np.zeros_like(price_array)
        stock_returns_raw[1:] = (price_array[1:] / price_array[:-1]) - 1.0

    return LoadedScenarioArrays(
        record=scenario_record,
        stock_ids=prepared.stock_ids,
        time_index=prepared.time_index,
        stock_features_raw=stock_features_raw[:, selected_stock_indices, :],
        market_features_raw=market_features_raw,
        stock_returns_raw=stock_returns_raw[:, selected_stock_indices],
    )
