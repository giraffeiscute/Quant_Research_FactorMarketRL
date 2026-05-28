"""Record types shared by scenario data loading and PyTorch datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


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
    rebalance_interval_days: int
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
    sample_num_stocks: int
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
