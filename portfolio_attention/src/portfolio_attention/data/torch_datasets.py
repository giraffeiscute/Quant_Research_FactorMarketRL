"""PyTorch Dataset adapters for prepared scenario records and rolling windows."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .records import PrecomputedTrainScenarioArrays, ScenarioSegmentRecord
from .standardization import scale_stock_features_for_context
from .stock_sampling import coverage_cycle_stock_indices, validate_sample_num_stocks


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
        sample_num_stocks: int,
        stock_sampling_base_seed: int = 0,
    ) -> None:
        self.scenario_arrays_by_id = scenario_arrays_by_id
        self.window_index = window_index
        self.lookback_days = int(lookback_days)
        self.rolling_horizon_days = int(rolling_horizon_days)
        self.price_normalization_mode = str(price_normalization_mode)
        self.stock_scaler_mean = np.asarray(stock_scaler_mean, dtype=np.float32)
        self.stock_scaler_std = np.asarray(stock_scaler_std, dtype=np.float32)
        self.sample_num_stocks = int(sample_num_stocks)
        self.stock_sampling_base_seed = int(stock_sampling_base_seed)
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
        full_num_stocks = int(arrays.stock_indices.shape[0])
        sample_num_stocks = validate_sample_num_stocks(self.sample_num_stocks, full_num_stocks)
        sampled_stock_indices = coverage_cycle_stock_indices(
            window_ordinal=int(index),
            sample_num_stocks=sample_num_stocks,
            full_num_stocks=full_num_stocks,
            base_seed=self.stock_sampling_base_seed,
        )

        feature_time_indices = arrays.time_index[feature_start:feature_stop]
        target_time_indices = arrays.time_index[target_start:target_stop]
        if arrays.scaled_stock_features is not None:
            x_stock = arrays.scaled_stock_features[feature_start:feature_stop, sampled_stock_indices, :]
        elif arrays.stock_features_raw is not None:
            x_stock = scale_stock_features_for_context(
                arrays.stock_features_raw[:, sampled_stock_indices, :],
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
            "r_stock": torch.from_numpy(arrays.stock_returns[target_start:target_stop, sampled_stock_indices]),
            "stock_indices": torch.from_numpy(arrays.stock_indices[sampled_stock_indices]),
        }
