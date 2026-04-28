"""Backward-compatible data API for portfolio_attention.

The implementation is split across focused modules under ``portfolio_attention.data``;
this module re-exports the historical public surface so existing imports continue
to work.
"""

from __future__ import annotations

from .constants import *
from .orchestration import PortfolioPanelDataset
from .parsing import (
    _coerce_numeric_series,
    _parse_time_label,
    _parse_time_series,
    parse_panel_dimensions,
    parse_scenario_file_info,
)
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
    scale_stock_feature_context_array,
    scale_stock_features_for_context,
    transform_stock_feature_context_array,
    transform_stock_features_for_context,
)
from .torch_datasets import RollingTrainWindowDataset, ScenarioSegmentDataset
