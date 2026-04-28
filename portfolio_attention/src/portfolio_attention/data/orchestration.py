"""Scenario-aware dataset orchestration for portfolio_attention."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import hashlib
import json

import numpy as np
from torch.utils.data import Dataset

from ..config import DataConfig
from .constants import (
    MARKET_FEATURE_COLUMNS,
    PRICE_FEATURE_INDEX,
    RELATIVE_TO_ANCHOR_PRICE_NORMALIZATION_MODE,
    STOCK_FEATURE_COLUMNS,
)
from .loader import materialize_scenario_arrays, prepare_scenario_frame
from .records import (
    LoadedScenarioArrays,
    PrecomputedTrainScenarioArrays,
    ScenarioDatasetMetadata,
    ScenarioFileRecord,
    ScenarioSegmentRecord,
)
from .standardization import (
    RunningMoments,
    Standardizer,
    _compute_relative_price_feature,
    _slice_stock_features_for_context,
    scale_stock_features_for_context,
)
from .scenario_split import discover_scenario_records, scenario_sort_key, split_scenario_records
from .torch_datasets import RollingTrainWindowDataset, ScenarioSegmentDataset
from .windows import context_bounds_for, resolve_time_window_layout, score_target_bounds_for


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
        return context_bounds_for(
            split_name,
            train_segment_start_index=self.train_segment_start_index,
            train_segment_end_index=self.train_segment_end_index,
            parsed_t=self.parsed_t,
        )

    def _score_target_bounds_for(self, split_name: str) -> tuple[int, int]:
        return score_target_bounds_for(
            split_name,
            train_segment_start_index=self.train_segment_start_index,
            train_segment_end_index=self.train_segment_end_index,
            parsed_t=self.parsed_t,
            lookback_days=int(self.config.lookback_days),
        )

    def _discover_scenarios(self) -> None:
        scenario_glob = self.config.resolved_scenario_glob
        records = discover_scenario_records(
            scenario_dir=self.scenario_dir,
            scenario_glob=scenario_glob,
            state=self.state,
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

        (
            self.scenario_records,
            self.train_scenario_records,
            self.validation_scenario_records,
            self.test_scenario_records,
        ) = split_scenario_records(records, self.config)
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
        return scenario_sort_key(path)

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
        layout = resolve_time_window_layout(
            total_time_steps=int(total_time_steps),
            lookback_days=int(self.config.lookback_days),
            rolling_horizon_days=int(self.config.rolling_horizon_days),
            rolling_stride_days=int(self.config.rolling_stride_days),
        )
        for field_name, value in layout.__dict__.items():
            setattr(self, field_name, value)

    def _load_scenario_arrays_uncached(self, scenario_record: ScenarioFileRecord) -> LoadedScenarioArrays:
        prepared = prepare_scenario_frame(
            scenario_record,
            raise_if_interrupted=self._raise_if_interrupted,
        )
        if not self.ignored_extra_columns:
            self.ignored_extra_columns = []
        self._validate_reference_schema(
            record=scenario_record,
            stock_ids=prepared.stock_ids,
            time_index=prepared.time_index,
        )
        return materialize_scenario_arrays(
            prepared,
            scenario_record=scenario_record,
            reference_stock_ids=self.reference_stock_ids,
            reference_time_index_array=self.reference_time_index_array,
            selected_stock_indices=self.selected_stock_indices,
            parsed_t=self.parsed_t,
            raise_if_interrupted=self._raise_if_interrupted,
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
