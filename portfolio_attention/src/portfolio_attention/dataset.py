"""Scenario-aware dataset loading and validation for portfolio_attention."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from .config import DataConfig

BASE_REQUIRED_COLUMNS = [
    "stock_id",
    "t",
    "characteristic_1",
    "characteristic_2",
    "characteristic_3",
    "MKT",
    "SMB",
    "HML",
    "price",
]
OPTIONAL_RETURN_COLUMN = "return"
STOCK_FEATURE_COLUMNS = [
    "characteristic_1",
    "characteristic_2",
    "characteristic_3",
    "price",
]
PRICE_FEATURE_INDEX = STOCK_FEATURE_COLUMNS.index("price")
VALID_PRICE_NORMALIZATION_MODES = ("none", "relative_to_anchor")
RELATIVE_TO_ANCHOR_PRICE_NORMALIZATION_MODE = "relative_to_anchor"
ANCHOR_PRICE_EPSILON = 1e-12
MARKET_FEATURE_COLUMNS = ["MKT", "SMB", "HML"]
NUMERIC_COLUMNS = STOCK_FEATURE_COLUMNS + MARKET_FEATURE_COLUMNS + [OPTIONAL_RETURN_COLUMN]
LOADABLE_COLUMNS = BASE_REQUIRED_COLUMNS + [OPTIONAL_RETURN_COLUMN]
SCENARIO_FILE_PATTERN = re.compile(
    r"^(?P<state>.+?)_(?P<n>\d+)_(?P<t>\d+)_PL_(?P<scenario_index>\d+)\.parquet$"
)


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


class Standardizer:
    """Simple ndarray standardizer fit only on train-scenario train-segment rows."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "Standardizer":
        if values.size == 0:
            raise ValueError("Cannot fit a scaler on empty values.")
        self.mean = values.mean(axis=0)
        std = values.std(axis=0)
        self.std = np.where(std < 1e-6, 1.0, std)
        return self

    def set_statistics(self, mean: np.ndarray, std: np.ndarray) -> "Standardizer":
        self.mean = mean.astype(np.float32)
        self.std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer must be fit before transform.")
        return (values - self.mean) / self.std


class RunningMoments:
    """Streaming moments helper used to avoid fitting scalers on validation/test rows."""

    def __init__(self, feature_dim: int) -> None:
        self.feature_dim = feature_dim
        self.count = 0
        self.sum = np.zeros((feature_dim,), dtype=np.float64)
        self.sum_sq = np.zeros((feature_dim,), dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected values with shape [*, {self.feature_dim}], received {values.shape}."
            )
        self.count += int(values.shape[0])
        self.sum += values.sum(axis=0, dtype=np.float64)
        self.sum_sq += np.square(values, dtype=np.float64).sum(axis=0, dtype=np.float64)

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count <= 0:
            raise ValueError("Cannot finalize moments without any observations.")
        mean = self.sum / float(self.count)
        variance = (self.sum_sq / float(self.count)) - np.square(mean)
        variance = np.maximum(variance, 1e-12)
        return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _slice_stock_features_for_context(
    stock_features_raw: np.ndarray,
    *,
    context_feature_start: int,
    context_feature_stop: int,
) -> np.ndarray:
    context_stock_features = np.asarray(
        stock_features_raw[context_feature_start:context_feature_stop],
        dtype=np.float32,
    )
    if context_stock_features.ndim != 3 or context_stock_features.shape[-1] != len(STOCK_FEATURE_COLUMNS):
        raise ValueError(
            "Expected stock feature slice with shape [time, stock, feature]. "
            f"Received {context_stock_features.shape}."
        )
    if context_stock_features.shape[0] <= 0:
        raise ValueError("Context stock feature slice must contain at least one time step.")
    return context_stock_features


def _compute_relative_price_feature(context_stock_features: np.ndarray) -> np.ndarray:
    anchor_prices = context_stock_features[0, :, PRICE_FEATURE_INDEX].astype(np.float64, copy=False)
    if np.any(np.abs(anchor_prices) < ANCHOR_PRICE_EPSILON):
        raise ValueError(
            "Anchor price must be non-zero before relative_to_anchor normalization."
        )
    relative_prices = (
        context_stock_features[..., PRICE_FEATURE_INDEX].astype(np.float64)
        / anchor_prices[np.newaxis, :]
    ) - 1.0
    return relative_prices.astype(np.float32)


def transform_stock_feature_context_array(
    context_stock_features: np.ndarray,
    *,
    price_normalization_mode: str,
) -> np.ndarray:
    if price_normalization_mode not in VALID_PRICE_NORMALIZATION_MODES:
        raise ValueError(
            "Unsupported price_normalization_mode for stock feature transformation: "
            f"{price_normalization_mode!r}."
        )
    context_stock_features = np.asarray(context_stock_features, dtype=np.float32)
    if context_stock_features.ndim != 3 or context_stock_features.shape[-1] != len(STOCK_FEATURE_COLUMNS):
        raise ValueError(
            "Expected context_stock_features with shape [time, stock, feature]. "
            f"Received {context_stock_features.shape}."
        )
    if context_stock_features.shape[0] <= 0:
        raise ValueError("context_stock_features must contain at least one time step.")

    transformed = context_stock_features.copy()
    if price_normalization_mode == RELATIVE_TO_ANCHOR_PRICE_NORMALIZATION_MODE:
        transformed[..., PRICE_FEATURE_INDEX] = _compute_relative_price_feature(context_stock_features)
    return transformed


def transform_stock_features_for_context(
    stock_features_raw: np.ndarray,
    *,
    context_feature_start: int,
    context_feature_stop: int,
    price_normalization_mode: str,
) -> np.ndarray:
    context_stock_features = _slice_stock_features_for_context(
        stock_features_raw,
        context_feature_start=context_feature_start,
        context_feature_stop=context_feature_stop,
    )
    return transform_stock_feature_context_array(
        context_stock_features,
        price_normalization_mode=price_normalization_mode,
    )


def scale_stock_feature_context_array(
    context_stock_features: np.ndarray,
    *,
    price_normalization_mode: str,
    stock_mean: np.ndarray,
    stock_std: np.ndarray,
) -> np.ndarray:
    transformed = transform_stock_feature_context_array(
        context_stock_features,
        price_normalization_mode=price_normalization_mode,
    )
    return ((transformed - stock_mean.reshape(1, 1, -1)) / stock_std.reshape(1, 1, -1)).astype(np.float32)


def scale_stock_features_for_context(
    stock_features_raw: np.ndarray,
    *,
    context_feature_start: int,
    context_feature_stop: int,
    price_normalization_mode: str,
    stock_mean: np.ndarray,
    stock_std: np.ndarray,
) -> np.ndarray:
    transformed = transform_stock_features_for_context(
        stock_features_raw,
        context_feature_start=context_feature_start,
        context_feature_stop=context_feature_stop,
        price_normalization_mode=price_normalization_mode,
    )
    return scale_stock_feature_context_array(
        transformed,
        price_normalization_mode="none",
        stock_mean=stock_mean,
        stock_std=stock_std,
    )


@dataclass
class ScenarioFileRecord:
    scenario_id: str
    source_path: str
    state: str
    scenario_index: int
    parsed_n: int
    parsed_t: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScenarioSegmentRecord:
    """One full scenario segment.

    Tensor layout for a single record:
    - `x_stock`: [T_split, N, F_stock]
    - `x_market`: [T_split, F_market]
    - `r_stock`: [T_split, N]
    - `stock_indices`: [N]

    A DataLoader stacks these into:
    - `x_stock`: [S, T_split, N, F_stock]
    - `x_market`: [S, T_split, F_market]
    - `r_stock`: [S, T_split, N]
    - `stock_indices`: [S, N]

    `S` is the scenario batch dimension and must never be flattened together with
    the time dimension `T_split`.
    """

    scenario_id: str
    source_path: str
    split_name: str
    feature_time_indices: np.ndarray
    target_time_indices: np.ndarray
    score_mask: np.ndarray
    x_stock: np.ndarray
    x_market: np.ndarray
    r_stock: np.ndarray
    stock_indices: np.ndarray
    x_stock_raw: np.ndarray | None = None


@dataclass
class ScenarioDatasetMetadata:
    scenario_dir: str
    scenario_glob: str
    state: str
    lookback_mode: str
    lookback_days: int
    rolling_horizon_days: int
    rolling_stride_days: int
    price_normalization_mode: str
    shuffle_scenario_splits: bool
    scenario_split_seed: int
    total_scenarios_found: int
    num_train_scenarios: int
    num_validation_scenarios: int
    num_test_scenarios: int
    train_scenarios: list[str]
    validation_scenarios: list[str]
    test_scenarios: list[str]
    total_num_days: int
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
    validation_context_time_steps: int
    test_context_time_steps: int
    max_context_time_steps: int
    train_context_feature_start_index: int | None
    train_context_feature_end_index: int | None
    validation_context_feature_start_index: int
    validation_context_feature_end_index: int
    test_context_feature_start_index: int
    test_context_feature_end_index: int
    train_context_target_start_index: int | None
    train_context_target_end_index: int | None
    validation_context_target_start_index: int
    validation_context_target_end_index: int
    test_context_target_start_index: int
    test_context_target_end_index: int
    train_score_target_start_index: int | None
    train_score_target_end_index: int | None
    validation_score_target_start_index: int
    validation_score_target_end_index: int
    test_score_target_start_index: int
    test_score_target_end_index: int
    train_score_feature_start_index: int | None
    train_score_feature_end_index: int | None
    validation_score_feature_start_index: int
    validation_score_feature_end_index: int
    test_score_feature_start_index: int
    test_score_feature_end_index: int
    train_score_time_steps: int
    validation_score_time_steps: int
    test_score_time_steps: int
    train_warmup_time_steps: int
    validation_warmup_time_steps: int
    test_warmup_time_steps: int
    train_windows_per_scenario: int
    train_window_count: int
    train_dataset_is_lazy_rolling: bool
    rolling_train_dataset_mode: str
    train_batch_size: int
    shuffle_train_scenarios: bool
    selected_num_stocks: int
    parsed_n: int
    parsed_t: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoadedScenarioArrays:
    record: ScenarioFileRecord
    stock_ids: list[str]
    time_index: list[int]
    stock_features_raw: np.ndarray
    market_features_raw: np.ndarray
    stock_returns_raw: np.ndarray


@dataclass
class PrecomputedTrainScenarioArrays:
    scenario_id: str
    source_path: str
    time_index: np.ndarray
    stock_features_raw: np.ndarray | None
    scaled_stock_features: np.ndarray | None
    scaled_market_features: np.ndarray
    stock_returns: np.ndarray
    stock_indices: np.ndarray


class ScenarioSegmentDataset(Dataset):
    """Dataset returning one sample per item.

    Depending on the train configuration, a sample is either a full scenario
    segment or a rolling train window.
    """

    def __init__(self, scenario_segments: list[ScenarioSegmentRecord]) -> None:
        self.scenario_segments = scenario_segments

    def __len__(self) -> int:
        return len(self.scenario_segments)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        item = self.scenario_segments[index]
        payload: dict[str, torch.Tensor | str] = {
            "scenario_id": item.scenario_id,
            "source_path": item.source_path,
            "split_name": item.split_name,
            "feature_time_indices": torch.from_numpy(item.feature_time_indices),
            "target_time_indices": torch.from_numpy(item.target_time_indices),
            "score_mask": torch.from_numpy(item.score_mask),
            "x_stock": torch.from_numpy(item.x_stock),
            "x_market": torch.from_numpy(item.x_market),
            "r_stock": torch.from_numpy(item.r_stock),
            "stock_indices": torch.from_numpy(item.stock_indices),
        }
        if item.x_stock_raw is not None:
            payload["x_stock_raw"] = torch.from_numpy(item.x_stock_raw)
        return payload


class RollingTrainWindowDataset(Dataset):
    """Lazy rolling train dataset backed by scenario-level caches."""

    def __init__(
        self,
        scenario_arrays_by_id: dict[str, PrecomputedTrainScenarioArrays],
        window_index: list[tuple[str, int]],
        *,
        lookback_days: int,
        rolling_horizon_days: int,
        price_normalization_mode: str,
        stock_scaler_mean: np.ndarray,
        stock_scaler_std: np.ndarray,
    ) -> None:
        self.scenario_arrays_by_id = scenario_arrays_by_id
        self.window_index = window_index
        self.lookback_days = int(lookback_days)
        self.rolling_horizon_days = int(rolling_horizon_days)
        self.price_normalization_mode = str(price_normalization_mode)
        self.stock_scaler_mean = np.asarray(stock_scaler_mean, dtype=np.float32)
        self.stock_scaler_std = np.asarray(stock_scaler_std, dtype=np.float32)
        self.context_time_steps = self.lookback_days + self.rolling_horizon_days
        self.score_mask_template = np.zeros((self.context_time_steps,), dtype=np.bool_)
        self.score_mask_template[self.lookback_days :] = True

    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        scenario_id, feature_start = self.window_index[index]
        arrays = self.scenario_arrays_by_id[scenario_id]
        feature_stop = feature_start + self.context_time_steps
        target_start = feature_start + 1
        target_stop = feature_stop + 1

        feature_time_indices = arrays.time_index[feature_start:feature_stop]
        target_time_indices = arrays.time_index[target_start:target_stop]
        if arrays.scaled_stock_features is not None:
            x_stock = arrays.scaled_stock_features[feature_start:feature_stop]
        elif arrays.stock_features_raw is not None:
            x_stock = scale_stock_features_for_context(
                arrays.stock_features_raw,
                context_feature_start=feature_start,
                context_feature_stop=feature_stop,
                price_normalization_mode=self.price_normalization_mode,
                stock_mean=self.stock_scaler_mean,
                stock_std=self.stock_scaler_std,
            )
        else:
            raise ValueError("Rolling train scenario cache must provide either raw or scaled stock features.")

        return {
            "scenario_id": arrays.scenario_id,
            "source_path": arrays.source_path,
            "split_name": "train",
            "feature_time_indices": torch.from_numpy(feature_time_indices),
            "target_time_indices": torch.from_numpy(target_time_indices),
            "score_mask": torch.from_numpy(self.score_mask_template),
            "x_stock": torch.from_numpy(x_stock),
            "x_market": torch.from_numpy(arrays.scaled_market_features[feature_start:feature_stop]),
            "r_stock": torch.from_numpy(arrays.stock_returns[target_start:target_stop]),
            "stock_indices": torch.from_numpy(arrays.stock_indices),
        }


class PortfolioPanelDataset:
    """Scenario-only dataset manager.

    Each scenario file represents one path. Train/validation/test scenario groups
    are derived from a stable file ordering and may be shuffled by a dedicated
    split seed. Within each scenario, train windows are built across the full
    scenario, while validation/test remain scenario-level records that score the
    full scenario after a lookback warmup prefix.
    """

    def __init__(
        self,
        config: DataConfig,
        progress_callback: Callable[[str], None] | None = None,
        interrupt_checker: Callable[[], None] | None = None,
    ) -> None:
        self.config = config
        self._progress_callback = progress_callback
        self._interrupt_checker = interrupt_checker
        self.scenario_dir = Path(config.scenario_dir)
        if not self.scenario_dir.exists():
            raise FileNotFoundError(f"Scenario directory not found: {self.scenario_dir}")

        self.loaded_stock_feature_columns = list(STOCK_FEATURE_COLUMNS)
        self.loaded_market_feature_columns = list(MARKET_FEATURE_COLUMNS)
        self.ignored_extra_columns: list[str] = []
        self.state = self.config.state
        self.stock_scaler = Standardizer()
        self.market_scaler = Standardizer()

        self.train_segment_records: list[ScenarioSegmentRecord] = []
        self.validation_segment_records: list[ScenarioSegmentRecord] = []
        self.test_segment_records: list[ScenarioSegmentRecord] = []
        self._scenario_arrays_cache: dict[str, LoadedScenarioArrays] = {}
        self._rolling_train_scenario_cache: dict[str, PrecomputedTrainScenarioArrays] = {}
        self._train_window_index: list[tuple[str, int]] = []
        self.standardizer_fit_count = 0

        self._emit_progress("Discovering scenario files.")
        self._discover_scenarios()
        self._emit_progress("Fitting train-only feature standardizers.")
        self._fit_standardizers_on_train_scenarios()
        self._emit_progress("Preparing training dataset windows.")
        self._prepare_train_dataset()
        self._emit_progress("Materializing validation/test scenario segments.")
        self._materialize_scenario_segments()
        self._emit_progress("Building dataset metadata summary.")
        self._build_metadata()
        self._emit_progress("Dataset build complete.")

    def _emit_progress(self, message: str) -> None:
        if self._progress_callback is not None:
            self._progress_callback(str(message))

    def _raise_if_interrupted(self) -> None:
        if self._interrupt_checker is None:
            return
        self._interrupt_checker()

    @staticmethod
    def _should_emit_progress_step(index: int, total: int, interval: int = 20) -> bool:
        if total <= 0:
            return False
        if index == total:
            return True
        return index % max(1, int(interval)) == 0

    def _use_lazy_rolling_train_dataset(self) -> bool:
        return str(self.config.rolling_train_dataset_mode) == "lazy"

    def _uses_relative_price_normalization(self) -> bool:
        return (
            str(self.config.price_normalization_mode)
            == RELATIVE_TO_ANCHOR_PRICE_NORMALIZATION_MODE
        )

    def _context_bounds_for(self, split_name: str) -> tuple[int, int]:
        if split_name == "train":
            return self.train_segment_start_index, self.train_segment_end_index - 1
        if split_name == "validation":
            return 0, self.parsed_t - 1
        if split_name == "test":
            return 0, self.parsed_t - 1
        raise ValueError(f"Unsupported split_name: {split_name}")

    def _score_target_bounds_for(self, split_name: str) -> tuple[int, int]:
        if split_name == "train":
            return self.train_segment_start_index + 1, self.train_segment_end_index
        if split_name == "validation":
            return int(self.config.lookback_days) + 1, self.parsed_t
        if split_name == "test":
            return int(self.config.lookback_days) + 1, self.parsed_t
        raise ValueError(f"Unsupported split_name: {split_name}")

    def _discover_scenarios(self) -> None:
        scenario_glob = self.config.resolved_scenario_glob
        matched_paths = sorted(
            self.scenario_dir.glob(scenario_glob),
            key=self._scenario_sort_key,
        )
        records: list[ScenarioFileRecord] = []
        for path in matched_paths:
            info = parse_scenario_file_info(path.name)
            if str(info["state"]) != self.state:
                continue
            records.append(
                ScenarioFileRecord(
                    scenario_id=path.stem,
                    source_path=str(path),
                    state=str(info["state"]),
                    scenario_index=int(info["scenario_index"]),
                    parsed_n=int(info["parsed_n"]),
                    parsed_t=int(info["parsed_t"]),
                )
            )

        expected_total = self.config.expected_total_scenarios
        if len(records) != expected_total:
            raise ValueError(
                f"Expected exactly {expected_total} scenario files in {self.scenario_dir}, "
                f"but found {len(records)} matching '{scenario_glob}'."
            )

        self._emit_progress(
            (
                f"Discovered {len(records)} scenario files in {self.scenario_dir} "
                f"with glob '{scenario_glob}'."
            )
        )

        split_records = list(records)
        if self.config.shuffle_scenario_splits:
            generator = np.random.default_rng(int(self.config.scenario_split_seed))
            permutation = generator.permutation(len(split_records))
            split_records = [split_records[int(index)] for index in permutation.tolist()]

        self.scenario_records = split_records
        self.train_scenario_records = split_records[: self.config.num_train_scenarios]
        val_start = self.config.num_train_scenarios
        val_end = val_start + self.config.num_validation_scenarios
        self.validation_scenario_records = split_records[val_start:val_end]
        self.test_scenario_records = split_records[val_end:]
        self._emit_progress(
            (
                "Scenario splits ready: "
                f"train={len(self.train_scenario_records)} "
                f"validation={len(self.validation_scenario_records)} "
                f"test={len(self.test_scenario_records)}."
            )
        )

    @staticmethod
    def _scenario_sort_key(path: Path) -> tuple[str, int]:
        info = parse_scenario_file_info(path.name)
        return str(info["state"]), int(info["scenario_index"])

    def _validate_reference_schema(
        self,
        *,
        record: ScenarioFileRecord,
        stock_ids: list[str],
        time_index: list[int],
    ) -> None:
        if not hasattr(self, "parsed_n"):
            self.parsed_n = record.parsed_n
            self.parsed_t = record.parsed_t
            self.reference_stock_ids = list(stock_ids)
            self.reference_stock_id_to_position = {
                stock_id: index for index, stock_id in enumerate(self.reference_stock_ids)
            }
            self.reference_time_index = list(time_index)
            self.reference_time_index_array = np.asarray(self.reference_time_index, dtype=np.int64)
            self.selected_stock_indices = self._resolve_selected_stock_indices(len(stock_ids))
            self.selected_stock_ids = [
                stock_ids[index] for index in self.selected_stock_indices.tolist()
            ]
            self._resolve_time_segment_lengths(self.parsed_t)
            return

        if record.parsed_n != self.parsed_n or record.parsed_t != self.parsed_t:
            raise ValueError(
                "All scenarios must share the same parsed N/T. "
                f"Expected ({self.parsed_n}, {self.parsed_t}), "
                f"received ({record.parsed_n}, {record.parsed_t}) "
                f"for {record.source_path}."
            )
        if stock_ids != self.reference_stock_ids:
            raise ValueError(
                "All scenarios must share the same stock universe ordering after sorting."
            )
        if time_index != self.reference_time_index:
            raise ValueError("All scenarios must share the same time index ordering after sorting.")

    def _resolve_selected_stock_indices(self, actual_num_stocks: int) -> np.ndarray:
        requested_num_stocks = self.config.num_stocks
        if requested_num_stocks is None:
            return np.arange(actual_num_stocks, dtype=np.int64)
        if requested_num_stocks <= 0:
            raise ValueError("DataConfig.num_stocks must be positive when provided.")
        if requested_num_stocks > actual_num_stocks:
            raise ValueError(
                f"Requested fixed num_stocks={requested_num_stocks}, "
                f"but scenario data only provides {actual_num_stocks} stocks."
            )
        return np.arange(requested_num_stocks, dtype=np.int64)

    def _resolve_time_segment_lengths(self, total_time_steps: int) -> None:
        if total_time_steps < 2:
            raise ValueError(
                "Each scenario must contain at least 2 raw timestamps so that one-step "
                "target returns can be formed without future leakage."
            )

        self.train_segment_start_index = 0
        self.train_segment_end_index = total_time_steps
        self.validation_segment_start_index = 0
        self.validation_segment_end_index = total_time_steps
        self.test_segment_start_index = 0
        self.test_segment_end_index = total_time_steps

        self.train_segment_raw_length = (
            self.train_segment_end_index - self.train_segment_start_index
        )
        self.validation_segment_raw_length = (
            self.validation_segment_end_index - self.validation_segment_start_index
        )
        self.test_segment_raw_length = self.test_segment_end_index - self.test_segment_start_index

        self.train_segment_time_steps = self.train_segment_raw_length - 1
        validation_context_start, validation_context_stop = self._context_bounds_for("validation")
        test_context_start, test_context_stop = self._context_bounds_for("test")
        self.validation_segment_time_steps = validation_context_stop - validation_context_start
        self.test_segment_time_steps = test_context_stop - test_context_start
        self.train_context_time_steps = (
            int(self.config.lookback_days) + int(self.config.rolling_horizon_days)
        )
        self.train_score_time_steps = int(self.config.rolling_horizon_days)
        self.train_warmup_time_steps = int(self.config.lookback_days)
        available_train_steps = self.train_segment_time_steps
        if available_train_steps < self.train_context_time_steps:
            raise ValueError(
                "Train scenario is too short for rolling_window mode. "
                f"Need at least lookback_days + rolling_horizon_days = {self.train_context_time_steps} "
                f"train time steps, but only found {available_train_steps}."
            )
        required_scored_steps = int(self.config.lookback_days) + 1
        if self.validation_segment_time_steps < required_scored_steps:
            raise ValueError(
                "Validation/test scenarios are too short for full-scenario evaluation. "
                f"Need parsed_t - 1 > lookback_days, but found parsed_t={total_time_steps} "
                f"and lookback_days={int(self.config.lookback_days)}."
            )
        self.train_windows_per_scenario = (
            ((available_train_steps - self.train_context_time_steps) // int(self.config.rolling_stride_days))
            + 1
        )

        self.max_time_steps = max(
            self.train_context_time_steps,
            self.validation_segment_time_steps,
            self.test_segment_time_steps,
        )

    def _read_scenario_frame(self, source_path: Path) -> pd.DataFrame:
        available_columns = set(pq.read_schema(source_path).names)
        columns = [column_name for column_name in LOADABLE_COLUMNS if column_name in available_columns]
        return pd.read_parquet(source_path, columns=columns)

    def _load_scenario_arrays_uncached(self, scenario_record: ScenarioFileRecord) -> LoadedScenarioArrays:
        self._raise_if_interrupted()
        source_path = Path(scenario_record.source_path)
        frame = self._read_scenario_frame(source_path)
        self._raise_if_interrupted()
        header = frame.columns.tolist()
        missing_columns = [column for column in BASE_REQUIRED_COLUMNS if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in {source_path}: {missing_columns}")
        if not self.ignored_extra_columns:
            self.ignored_extra_columns = []

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

        self._validate_reference_schema(
            record=scenario_record,
            stock_ids=stock_ids,
            time_index=time_index,
        )

        expected_row_count = scenario_record.parsed_n * scenario_record.parsed_t
        if len(frame) != expected_row_count:
            raise ValueError(f"Scenario is incomplete: {source_path}")
        stock_position = pd.Categorical(
            frame["stock_id"],
            categories=self.reference_stock_ids,
            ordered=True,
        ).codes
        if (stock_position < 0).any():
            raise ValueError("Scenario contains unknown stock IDs compared with the reference universe.")

        time_values = frame["time_index"].to_numpy(dtype=np.int64, copy=False)
        time_position = np.searchsorted(self.reference_time_index_array, time_values)
        if (
            (time_position >= len(self.reference_time_index_array)).any()
            or not np.array_equal(self.reference_time_index_array[time_position], time_values)
        ):
            raise ValueError("Scenario contains unexpected time indices compared with the reference grid.")

        linear_index = stock_position.astype(np.int64) * self.parsed_t + time_position.astype(np.int64)
        coverage = np.bincount(linear_index, minlength=expected_row_count)
        if coverage.shape[0] != expected_row_count or not np.all(coverage == 1):
            raise ValueError(f"Scenario is incomplete after position mapping: {source_path}")

        self._raise_if_interrupted()
        stock_feature_values = frame[STOCK_FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
        stock_feature_grid = np.empty((expected_row_count, len(STOCK_FEATURE_COLUMNS)), dtype=np.float32)
        stock_feature_grid[linear_index] = stock_feature_values
        stock_features_raw = stock_feature_grid.reshape(
            self.parsed_n,
            self.parsed_t,
            len(STOCK_FEATURE_COLUMNS),
        ).transpose(1, 0, 2)

        self._raise_if_interrupted()
        market_values = frame[MARKET_FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
        market_grid = np.empty((expected_row_count, len(MARKET_FEATURE_COLUMNS)), dtype=np.float32)
        market_grid[linear_index] = market_values
        market_cube = market_grid.reshape(
            self.parsed_n,
            self.parsed_t,
            len(MARKET_FEATURE_COLUMNS),
        ).transpose(1, 0, 2)
        if not np.allclose(market_cube, market_cube[:, :1, :], atol=0.0, rtol=0.0):
            raise ValueError("FF3 factors are not identical across stocks within the same day.")
        market_features_raw = market_cube[:, 0, :]

        if has_return_column:
            return_values = frame[OPTIONAL_RETURN_COLUMN].to_numpy(dtype=np.float32, copy=True)
            return_grid = np.empty((expected_row_count,), dtype=np.float32)
            return_grid[linear_index] = return_values
            stock_returns_raw = return_grid.reshape(self.parsed_n, self.parsed_t).transpose(1, 0)
        else:
            price_array = stock_features_raw[..., -1]
            stock_returns_raw = np.zeros_like(price_array)
            stock_returns_raw[1:] = (price_array[1:] / price_array[:-1]) - 1.0

        return LoadedScenarioArrays(
            record=scenario_record,
            stock_ids=stock_ids,
            time_index=time_index,
            stock_features_raw=stock_features_raw[:, self.selected_stock_indices, :],
            market_features_raw=market_features_raw,
            stock_returns_raw=stock_returns_raw[:, self.selected_stock_indices],
        )

    def _load_scenario_arrays(self, scenario_record: ScenarioFileRecord) -> LoadedScenarioArrays:
        cached = self._scenario_arrays_cache.get(scenario_record.scenario_id)
        if cached is not None:
            return cached
        arrays = self._load_scenario_arrays_uncached(scenario_record)
        self._scenario_arrays_cache[scenario_record.scenario_id] = arrays
        return arrays

    def _fit_standardizers_on_train_scenarios(self) -> None:
        self.standardizer_fit_count += 1
        stock_characteristic_moments = RunningMoments(PRICE_FEATURE_INDEX)
        stock_price_moments = RunningMoments(1)
        market_moments = RunningMoments(len(MARKET_FEATURE_COLUMNS))
        stride = int(self.config.rolling_stride_days)
        window_length = int(self.config.lookback_days) + int(self.config.rolling_horizon_days)

        total = len(self.train_scenario_records)
        for index, scenario_record in enumerate(self.train_scenario_records, start=1):
            self._raise_if_interrupted()
            if self._should_emit_progress_step(index, total):
                self._emit_progress(
                    (
                        "Fitting standardizers on train scenarios: "
                        f"{index}/{total} ({scenario_record.scenario_id})."
                    )
                )
            arrays = self._load_scenario_arrays(scenario_record)
            train_stock_values = arrays.stock_features_raw[
                self.train_segment_start_index : self.train_segment_end_index
            ].reshape(-1, len(STOCK_FEATURE_COLUMNS))
            train_market_values = arrays.market_features_raw[
                self.train_segment_start_index : self.train_segment_end_index
            ]
            stock_characteristic_moments.update(train_stock_values[:, :PRICE_FEATURE_INDEX])
            if self._uses_relative_price_normalization():
                for window_index in range(self.train_windows_per_scenario):
                    feature_start = self.train_segment_start_index + (window_index * stride)
                    feature_stop = feature_start + window_length
                    relative_prices = _compute_relative_price_feature(
                        _slice_stock_features_for_context(
                            arrays.stock_features_raw,
                            context_feature_start=feature_start,
                            context_feature_stop=feature_stop,
                        )
                    )
                    stock_price_moments.update(relative_prices.reshape(-1, 1))
            else:
                stock_price_moments.update(train_stock_values[:, PRICE_FEATURE_INDEX : PRICE_FEATURE_INDEX + 1])
            market_moments.update(train_market_values)

        stock_characteristic_mean, stock_characteristic_std = stock_characteristic_moments.finalize()
        stock_price_mean, stock_price_std = stock_price_moments.finalize()
        stock_mean = np.concatenate([stock_characteristic_mean, stock_price_mean], axis=0)
        stock_std = np.concatenate([stock_characteristic_std, stock_price_std], axis=0)
        market_mean, market_std = market_moments.finalize()
        self.stock_scaler.set_statistics(stock_mean, stock_std)
        self.market_scaler.set_statistics(np.zeros_like(market_mean), market_std)

    def build_debug_summary(self) -> dict[str, Any]:
        if not hasattr(self, "metadata"):
            raise RuntimeError("Dataset metadata is unavailable before dataset build completes.")
        if self.stock_scaler.mean is None or self.stock_scaler.std is None:
            raise RuntimeError("Stock scaler statistics are unavailable before dataset build completes.")
        if self.market_scaler.mean is None or self.market_scaler.std is None:
            raise RuntimeError("Market scaler statistics are unavailable before dataset build completes.")

        metadata = self.metadata
        scenario_split_summary = {
            "train_scenarios": list(metadata.train_scenarios),
            "validation_scenarios": list(metadata.validation_scenarios),
            "test_scenarios": list(metadata.test_scenarios),
            "train_window_count": int(metadata.train_window_count),
            "validation_scenario_count": int(metadata.num_validation_scenarios),
            "test_scenario_count": int(metadata.num_test_scenarios),
        }
        scaler_summary = {
            "stock_mean": np.asarray(self.stock_scaler.mean, dtype=np.float32).round(8).tolist(),
            "stock_std": np.asarray(self.stock_scaler.std, dtype=np.float32).round(8).tolist(),
            "market_mean": np.asarray(self.market_scaler.mean, dtype=np.float32).round(8).tolist(),
            "market_std": np.asarray(self.market_scaler.std, dtype=np.float32).round(8).tolist(),
        }
        return {
            "scenario_split_hash": self._stable_hash_payload(scenario_split_summary),
            "stock_scaler_hash": self._stable_hash_payload(
                {
                    "stock_mean": scaler_summary["stock_mean"],
                    "stock_std": scaler_summary["stock_std"],
                }
            ),
            "market_scaler_hash": self._stable_hash_payload(
                {
                    "market_mean": scaler_summary["market_mean"],
                    "market_std": scaler_summary["market_std"],
                }
            ),
            "scenario_split_summary": scenario_split_summary,
            "scaler_summary": scaler_summary,
            "standardizer_fit_count": int(self.standardizer_fit_count),
        }

    @staticmethod
    def _stable_hash_payload(payload: Any) -> str:
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _prepare_train_dataset(self) -> None:
        stock_indices = np.arange(len(self.selected_stock_ids), dtype=np.int64)
        stride = int(self.config.rolling_stride_days)
        window_length = int(self.config.lookback_days) + int(self.config.rolling_horizon_days)
        total = len(self.train_scenario_records)
        for index, scenario_record in enumerate(self.train_scenario_records, start=1):
            self._raise_if_interrupted()
            if self._should_emit_progress_step(index, total):
                self._emit_progress(
                    (
                        "Preparing rolling train windows: "
                        f"{index}/{total} ({scenario_record.scenario_id})."
                    )
                )
            arrays = self._load_scenario_arrays(scenario_record)
            if self._use_lazy_rolling_train_dataset():
                scaled_market = self.market_scaler.transform(arrays.market_features_raw)
                if self._uses_relative_price_normalization():
                    stock_features_raw = arrays.stock_features_raw.astype(np.float32)
                    scaled_stock: np.ndarray | None = None
                else:
                    stock_features_raw = None
                    scaled_stock = self.stock_scaler.transform(
                        arrays.stock_features_raw.reshape(-1, len(STOCK_FEATURE_COLUMNS))
                    ).reshape(arrays.stock_features_raw.shape)
                self._rolling_train_scenario_cache[scenario_record.scenario_id] = PrecomputedTrainScenarioArrays(
                    scenario_id=scenario_record.scenario_id,
                    source_path=scenario_record.source_path,
                    time_index=np.asarray(arrays.time_index, dtype=np.int64),
                    stock_features_raw=stock_features_raw,
                    scaled_stock_features=(
                        None if scaled_stock is None else scaled_stock.astype(np.float32)
                    ),
                    scaled_market_features=scaled_market.astype(np.float32),
                    stock_returns=arrays.stock_returns_raw.astype(np.float32),
                    stock_indices=stock_indices,
                )
            for window_index in range(self.train_windows_per_scenario):
                self._raise_if_interrupted()
                feature_start = self.train_segment_start_index + (window_index * stride)
                feature_stop = feature_start + window_length
                if feature_stop + 1 > self.train_segment_end_index:
                    raise ValueError("Rolling train window exceeds the train segment boundary.")
                self._train_window_index.append((scenario_record.scenario_id, feature_start))
                if not self._use_lazy_rolling_train_dataset():
                    score_target_start = feature_start + int(self.config.lookback_days) + 1
                    score_target_stop = feature_stop + 1
                    self.train_segment_records.append(
                        self._build_segment_record_from_bounds(
                            arrays,
                            split_name="train",
                            context_feature_start=feature_start,
                            context_feature_stop=feature_stop,
                            score_target_start=score_target_start,
                            score_target_stop=score_target_stop,
                        )
                    )
            self._scenario_arrays_cache.pop(scenario_record.scenario_id, None)

    def _build_segment_record_from_bounds(
        self,
        arrays: LoadedScenarioArrays,
        *,
        split_name: str,
        context_feature_start: int,
        context_feature_stop: int,
        score_target_start: int,
        score_target_stop: int,
    ) -> ScenarioSegmentRecord:
        context_target_start = context_feature_start + 1
        context_target_stop = context_feature_stop + 1
        if self.stock_scaler.mean is None or self.stock_scaler.std is None:
            raise RuntimeError("Stock scaler statistics must be fitted before segment construction.")
        if self.market_scaler.mean is None or self.market_scaler.std is None:
            raise RuntimeError("Market scaler statistics must be fitted before segment construction.")
        scaled_stock = scale_stock_features_for_context(
            arrays.stock_features_raw,
            context_feature_start=context_feature_start,
            context_feature_stop=context_feature_stop,
            price_normalization_mode=str(self.config.price_normalization_mode),
            stock_mean=self.stock_scaler.mean,
            stock_std=self.stock_scaler.std,
        )
        scaled_market = self.market_scaler.transform(arrays.market_features_raw)

        x_stock = scaled_stock
        x_market = scaled_market[context_feature_start:context_feature_stop]
        r_stock = arrays.stock_returns_raw[context_target_start:context_target_stop]
        x_stock_raw = None
        if split_name in {"validation", "test"}:
            x_stock_raw = _slice_stock_features_for_context(
                arrays.stock_features_raw,
                context_feature_start=context_feature_start,
                context_feature_stop=context_feature_stop,
            )
        feature_time_indices = np.asarray(
            arrays.time_index[context_feature_start:context_feature_stop],
            dtype=np.int64,
        )
        target_time_indices = np.asarray(
            arrays.time_index[context_target_start:context_target_stop],
            dtype=np.int64,
        )
        score_mask = (
            (target_time_indices >= int(arrays.time_index[score_target_start]))
            & (target_time_indices <= int(arrays.time_index[score_target_stop - 1]))
        )
        stock_indices = np.arange(len(self.selected_stock_ids), dtype=np.int64)

        expected_time_steps = context_feature_stop - context_feature_start
        assert x_stock.shape == (
            expected_time_steps,
            len(self.selected_stock_ids),
            len(STOCK_FEATURE_COLUMNS),
        )
        assert x_market.shape == (expected_time_steps, len(MARKET_FEATURE_COLUMNS))
        assert r_stock.shape == (expected_time_steps, len(self.selected_stock_ids))
        assert feature_time_indices.shape == target_time_indices.shape == (expected_time_steps,)
        assert score_mask.shape == (expected_time_steps,)
        if x_stock_raw is not None:
            assert x_stock_raw.shape == x_stock.shape
        if int(score_mask.sum()) <= 0:
            raise ValueError(f"{split_name} score_mask must contain at least one time step.")

        return ScenarioSegmentRecord(
            scenario_id=arrays.record.scenario_id,
            source_path=arrays.record.source_path,
            split_name=split_name,
            feature_time_indices=feature_time_indices,
            target_time_indices=target_time_indices,
            score_mask=score_mask.astype(bool),
            x_stock=x_stock.astype(np.float32),
            x_market=x_market.astype(np.float32),
            r_stock=r_stock.astype(np.float32),
            stock_indices=stock_indices,
            x_stock_raw=None if x_stock_raw is None else x_stock_raw.astype(np.float32),
        )

    def _build_segment_record(
        self,
        arrays: LoadedScenarioArrays,
        split_name: str,
    ) -> ScenarioSegmentRecord:
        context_feature_start, context_feature_stop = self._context_bounds_for(split_name)
        score_target_start, score_target_stop = self._score_target_bounds_for(split_name)
        return self._build_segment_record_from_bounds(
            arrays,
            split_name=split_name,
            context_feature_start=context_feature_start,
            context_feature_stop=context_feature_stop,
            score_target_start=score_target_start,
            score_target_stop=score_target_stop,
        )

    def _materialize_scenario_segments(self) -> None:
        split_map = {
            "validation": self.validation_scenario_records,
            "test": self.test_scenario_records,
        }
        target_lists = {
            "validation": self.validation_segment_records,
            "test": self.test_segment_records,
        }

        for split_name, records in split_map.items():
            total = len(records)
            if total == 0:
                self._emit_progress(f"Skipping {split_name} segment materialization; no scenario-level records.")
                continue
            for index, scenario_record in enumerate(records, start=1):
                self._raise_if_interrupted()
                if self._should_emit_progress_step(index, total):
                    self._emit_progress(
                        (
                            f"Materializing {split_name} segments: "
                            f"{index}/{total} ({scenario_record.scenario_id})."
                        )
                    )
                arrays = self._load_scenario_arrays(scenario_record)
                target_lists[split_name].append(self._build_segment_record(arrays, split_name))
        self._scenario_arrays_cache.clear()

    def _build_metadata(self) -> None:
        validation_context_feature_start, validation_context_feature_stop = self._context_bounds_for(
            "validation"
        )
        test_context_feature_start, test_context_feature_stop = self._context_bounds_for("test")
        validation_score_target_start, validation_score_target_stop = self._score_target_bounds_for(
            "validation"
        )
        test_score_target_start, test_score_target_stop = self._score_target_bounds_for("test")
        validation_score_time_steps = validation_score_target_stop - validation_score_target_start
        test_score_time_steps = test_score_target_stop - test_score_target_start
        validation_context_time_steps = validation_context_feature_stop - validation_context_feature_start
        test_context_time_steps = test_context_feature_stop - test_context_feature_start
        train_context_feature_start: int | None = None
        train_context_feature_end: int | None = None
        train_context_target_start: int | None = None
        train_context_target_end: int | None = None
        train_score_target_start: int | None = None
        train_score_target_end: int | None = None
        train_score_feature_start: int | None = None
        train_score_feature_end: int | None = None
        self.metadata = ScenarioDatasetMetadata(
            scenario_dir=str(self.scenario_dir),
            scenario_glob=self.config.resolved_scenario_glob,
            state=self.state,
            lookback_mode="rolling_window",
            lookback_days=int(self.config.lookback_days),
            rolling_horizon_days=int(self.config.rolling_horizon_days),
            rolling_stride_days=int(self.config.rolling_stride_days),
            price_normalization_mode=str(self.config.price_normalization_mode),
            shuffle_scenario_splits=bool(self.config.shuffle_scenario_splits),
            scenario_split_seed=int(self.config.scenario_split_seed),
            total_scenarios_found=len(self.scenario_records),
            num_train_scenarios=len(self.train_scenario_records),
            num_validation_scenarios=len(self.validation_scenario_records),
            num_test_scenarios=len(self.test_scenario_records),
            train_scenarios=[record.scenario_id for record in self.train_scenario_records],
            validation_scenarios=[record.scenario_id for record in self.validation_scenario_records],
            test_scenarios=[record.scenario_id for record in self.test_scenario_records],
            total_num_days=self.parsed_t,
            train_segment_start_index=self.train_segment_start_index,
            train_segment_end_index=self.train_segment_end_index - 1,
            validation_segment_start_index=self.validation_segment_start_index,
            validation_segment_end_index=self.validation_segment_end_index - 1,
            test_segment_start_index=self.test_segment_start_index,
            test_segment_end_index=self.test_segment_end_index - 1,
            train_segment_raw_length=self.train_segment_raw_length,
            validation_segment_raw_length=self.validation_segment_raw_length,
            test_segment_raw_length=self.test_segment_raw_length,
            train_segment_time_steps=self.train_segment_time_steps,
            validation_segment_time_steps=self.validation_segment_time_steps,
            test_segment_time_steps=self.test_segment_time_steps,
            train_context_time_steps=self.train_context_time_steps,
            validation_context_time_steps=validation_context_time_steps,
            test_context_time_steps=test_context_time_steps,
            max_context_time_steps=self.max_time_steps,
            train_context_feature_start_index=train_context_feature_start,
            train_context_feature_end_index=train_context_feature_end,
            validation_context_feature_start_index=validation_context_feature_start,
            validation_context_feature_end_index=validation_context_feature_stop - 1,
            test_context_feature_start_index=test_context_feature_start,
            test_context_feature_end_index=test_context_feature_stop - 1,
            train_context_target_start_index=train_context_target_start,
            train_context_target_end_index=train_context_target_end,
            validation_context_target_start_index=validation_context_feature_start + 1,
            validation_context_target_end_index=validation_context_feature_stop,
            test_context_target_start_index=test_context_feature_start + 1,
            test_context_target_end_index=test_context_feature_stop,
            train_score_target_start_index=train_score_target_start,
            train_score_target_end_index=train_score_target_end,
            validation_score_target_start_index=validation_score_target_start,
            validation_score_target_end_index=validation_score_target_stop - 1,
            test_score_target_start_index=test_score_target_start,
            test_score_target_end_index=test_score_target_stop - 1,
            train_score_feature_start_index=train_score_feature_start,
            train_score_feature_end_index=train_score_feature_end,
            validation_score_feature_start_index=validation_score_target_start - 1,
            validation_score_feature_end_index=validation_score_target_stop - 2,
            test_score_feature_start_index=test_score_target_start - 1,
            test_score_feature_end_index=test_score_target_stop - 2,
            train_score_time_steps=self.train_score_time_steps,
            validation_score_time_steps=validation_score_time_steps,
            test_score_time_steps=test_score_time_steps,
            train_warmup_time_steps=self.train_warmup_time_steps,
            validation_warmup_time_steps=validation_context_time_steps - validation_score_time_steps,
            test_warmup_time_steps=test_context_time_steps - test_score_time_steps,
            train_windows_per_scenario=int(self.train_windows_per_scenario),
            train_window_count=len(self._train_window_index),
            train_dataset_is_lazy_rolling=bool(self._use_lazy_rolling_train_dataset()),
            rolling_train_dataset_mode=str(self.config.rolling_train_dataset_mode),
            train_batch_size=int(self.config.train_batch_size),
            shuffle_train_scenarios=bool(self.config.shuffle_train_scenarios),
            selected_num_stocks=len(self.selected_stock_ids),
            parsed_n=self.parsed_n,
            parsed_t=self.parsed_t,
        )

    @property
    def num_stocks(self) -> int:
        return len(self.selected_stock_ids)

    @property
    def num_times(self) -> int:
        return self.parsed_t

    def build_train_validation_test_datasets(
        self,
    ) -> tuple[Dataset, ScenarioSegmentDataset, ScenarioSegmentDataset]:
        if self._use_lazy_rolling_train_dataset():
            if self.stock_scaler.mean is None or self.stock_scaler.std is None:
                raise RuntimeError("Stock scaler statistics must be available before building datasets.")
            train_dataset: Dataset = RollingTrainWindowDataset(
                self._rolling_train_scenario_cache,
                list(self._train_window_index),
                lookback_days=int(self.config.lookback_days),
                rolling_horizon_days=int(self.config.rolling_horizon_days),
                price_normalization_mode=str(self.config.price_normalization_mode),
                stock_scaler_mean=self.stock_scaler.mean,
                stock_scaler_std=self.stock_scaler.std,
            )
        else:
            train_dataset = ScenarioSegmentDataset(list(self.train_segment_records))
        return (
            train_dataset,
            ScenarioSegmentDataset(list(self.validation_segment_records)),
            ScenarioSegmentDataset(list(self.test_segment_records)),
        )

    def build_train_validation_backtest_datasets(
        self,
    ) -> tuple[Dataset, ScenarioSegmentDataset, ScenarioSegmentDataset]:
        return self.build_train_validation_test_datasets()

    def build_shared_train_validation_test_datasets(
        self,
    ) -> tuple[Dataset, ScenarioSegmentDataset, ScenarioSegmentDataset]:
        """Return dataset views backed by this dataset instance's prepared caches."""
        return self.build_train_validation_test_datasets()

    def get_split_dataset(self, split_name: str) -> Dataset:
        if split_name == "train":
            if self._use_lazy_rolling_train_dataset():
                if self.stock_scaler.mean is None or self.stock_scaler.std is None:
                    raise RuntimeError("Stock scaler statistics must be available before building datasets.")
                return RollingTrainWindowDataset(
                    self._rolling_train_scenario_cache,
                    list(self._train_window_index),
                    lookback_days=int(self.config.lookback_days),
                    rolling_horizon_days=int(self.config.rolling_horizon_days),
                    price_normalization_mode=str(self.config.price_normalization_mode),
                    stock_scaler_mean=self.stock_scaler.mean,
                    stock_scaler_std=self.stock_scaler.std,
                )
            return ScenarioSegmentDataset(list(self.train_segment_records))
        if split_name == "validation":
            return ScenarioSegmentDataset(list(self.validation_segment_records))
        if split_name == "test":
            return ScenarioSegmentDataset(list(self.test_segment_records))
        raise ValueError(f"Unsupported split_name: {split_name}")
