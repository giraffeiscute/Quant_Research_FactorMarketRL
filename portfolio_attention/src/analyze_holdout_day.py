#!/usr/bin/env python3
"""Analyze one holdout day for a single scenario from final artifacts or replay."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from portfolio_attention.config import (
    DataConfig,
    ModelConfig,
    PathsConfig,
    normalize_model_config_dict,
)
from portfolio_attention.config_validation import LEGACY_LOOKBACK_MODES, normalize_lookback_mode
from portfolio_attention.dataset import (
    LOADABLE_COLUMNS,
    MARKET_FEATURE_COLUMNS,
    OPTIONAL_RETURN_COLUMN,
    STOCK_FEATURE_COLUMNS,
    PortfolioPanelDataset,
    ScenarioSegmentDataset,
    _coerce_numeric_series,
    _parse_time_series,
    scale_stock_features_for_context,
)
from portfolio_attention.evaluate import (
    ROLLING_ONE_STEP_EVALUATION_MODE,
    _compute_backtest_portfolio_sr,
    _extract_exported_train_config,
    _is_weight_above_threshold,
    _validate_checkpoint_metadata,
    format_allocation_group_label,
)
from portfolio_attention.evaluation_presentation import get_aux_lookup, load_aux_frame
from portfolio_attention.model import PortfolioAttentionModel
from portfolio_attention.utils import apply_score_mask, ensure_output_dirs, resolve_device, save_json


# Analysis settings: edit these values before running the script.
STATE = "bull"
LOSS_NAME = "sharpe"
SCENARIO_ID = "bull_4860_200_PL_48"
TARGET_DAY = 197
CHECKPOINT_PATH: str | Path | None = None

# Set MONITORING_EPOCH to inspect the canonical monitoring checkpoint for that epoch.
# In monitoring mode, the checkpoint path is derived from the epoch automatically.
# CHECKPOINT_PATH is optional here and, when provided, is only used for a consistency check.
MONITORING_EPOCH: int | None = 2400
DEVICE_NAME = "auto"
WEIGHT_THRESHOLD = 0.001
CHART_TOP_K: int | None = 100
HIDE_X_AXIS = True
OUTPUT_DIR: str | Path | None = None
SHOW_PROGRESS = True


VALID_LOSSES = ("return", "sharpe", "dsr", "sortino", "mdd", "cvar")
COLOR_MAP_NAMES = ("tab20", "tab20b", "tab20c")
STOCK_ROW_COLUMNS = [
    "global_rank",
    "category_rank",
    "category_order",
    "plot_position",
    "stock_id",
    "weight",
    "is_selected",
    "mu",
    "epsilon_variance",
    "alpha",
    "category_label",
]
GROUPED_ROW_COLUMNS = [
    "category_rank",
    "category_label",
    "mu",
    "epsilon_variance",
    "alpha",
    "total_weight",
    "selected_stock_count",
]


@dataclass(frozen=True)
class HoldoutDayAnalysisConfig:
    state: str
    loss_name: str
    scenario_id: str
    target_day: int
    checkpoint_path: Path | None = None
    monitoring_epoch: int | None = None
    device_name: str = "auto"
    weight_threshold: float = 0.001
    chart_top_k: int | None = 100
    hide_x_axis: bool = True
    output_dir: Path | None = None
    show_progress: bool = True


@dataclass(frozen=True)
class ResolvedAnalysisInputs:
    analysis_source: str
    source_path: Path
    stock_ids: list[str]
    scored_outputs: dict[str, torch.Tensor]
    train_config: dict[str, object]
    checkpoint_path: str | None


def _build_runtime_config() -> HoldoutDayAnalysisConfig:
    resolved_loss_name = str(LOSS_NAME).strip().lower()
    if resolved_loss_name not in VALID_LOSSES:
        raise ValueError(f"Unsupported LOSS_NAME={LOSS_NAME!r}. Valid options: {sorted(VALID_LOSSES)}.")
    resolved_state = str(STATE).strip().lower()
    if not resolved_state:
        raise ValueError("STATE must be non-empty.")
    resolved_scenario_id = str(SCENARIO_ID).strip()
    if not resolved_scenario_id:
        raise ValueError("SCENARIO_ID must be non-empty.")
    resolved_target_day = int(TARGET_DAY)
    resolved_weight_threshold = float(WEIGHT_THRESHOLD)
    if resolved_weight_threshold < 0.0:
        raise ValueError(f"WEIGHT_THRESHOLD must be non-negative, received {WEIGHT_THRESHOLD}.")
    resolved_chart_top_k = None if CHART_TOP_K is None else int(CHART_TOP_K)
    if resolved_chart_top_k is not None and resolved_chart_top_k <= 0:
        raise ValueError(f"CHART_TOP_K must be positive when provided, received {CHART_TOP_K}.")
    resolved_monitoring_epoch = (
        None if MONITORING_EPOCH in {None, ""} else int(MONITORING_EPOCH)
    )
    if resolved_monitoring_epoch is not None and resolved_monitoring_epoch <= 0:
        raise ValueError(
            f"MONITORING_EPOCH must be positive when provided, received {MONITORING_EPOCH}."
        )
    resolved_checkpoint_path = None if CHECKPOINT_PATH in {None, ""} else Path(str(CHECKPOINT_PATH))
    resolved_output_dir = None if OUTPUT_DIR in {None, ""} else Path(str(OUTPUT_DIR))
    return HoldoutDayAnalysisConfig(
        state=resolved_state,
        loss_name=resolved_loss_name,
        scenario_id=resolved_scenario_id,
        target_day=resolved_target_day,
        checkpoint_path=resolved_checkpoint_path,
        monitoring_epoch=resolved_monitoring_epoch,
        device_name=str(DEVICE_NAME),
        weight_threshold=resolved_weight_threshold,
        chart_top_k=resolved_chart_top_k,
        hide_x_axis=bool(HIDE_X_AXIS),
        output_dir=resolved_output_dir,
        show_progress=bool(SHOW_PROGRESS),
    )


def _load_checkpoint(
    checkpoint_path: Path,
    *,
    map_location: str | torch.device,
) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a dict payload: {checkpoint_path}")
    return checkpoint


def _fail_if_legacy_checkpoint_replay_uses_removed_lookback_mode(checkpoint: dict[str, Any]) -> None:
    checkpoint_data_config = checkpoint.get("data_config", {})
    checkpoint_metadata = checkpoint.get("metadata", {})
    candidate_modes: list[object] = []
    if isinstance(checkpoint_data_config, dict):
        candidate_modes.append(checkpoint_data_config.get("lookback_mode"))
    if isinstance(checkpoint_metadata, dict):
        candidate_modes.append(checkpoint_metadata.get("lookback_mode"))

    for raw_lookback_mode in candidate_modes:
        if raw_lookback_mode is None:
            continue
        lookback_mode = normalize_lookback_mode(raw_lookback_mode)
        if lookback_mode not in LEGACY_LOOKBACK_MODES:
            continue
        raise ValueError(
            "Legacy checkpoint replay supports reading old metadata but cannot rebuild "
            f"datasets for lookback_mode={lookback_mode!r}. Legacy non-rolling modes are read-only."
        )


def _emit_progress(
    config: HoldoutDayAnalysisConfig,
    *,
    phase: str,
    step_index: int,
    step_count: int,
    message: str,
    scenario_id: str | None = None,
    epoch: int | None = None,
    artifact_source: str | None = None,
) -> None:
    if not config.show_progress:
        return
    details: list[str] = []
    if scenario_id is not None:
        details.append(f"scenario_id={scenario_id}")
    if epoch is not None:
        details.append(f"epoch={epoch}")
    if artifact_source is not None:
        details.append(f"source={artifact_source}")
    suffix = f" | {' '.join(details)}" if details else ""
    print(f"[{step_index}/{step_count}] {phase}: {message}{suffix}", flush=True)


def _canonicalize_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _resolve_monitoring_checkpoint_path(paths: PathsConfig, config: HoldoutDayAnalysisConfig) -> Path:
    if config.monitoring_epoch is None:
        raise ValueError("monitoring_epoch must be set to resolve a monitoring checkpoint.")
    return (
        paths.checkpoints_dir
        / f"{config.state}_train_monitoring_{config.loss_name}_epoch_{int(config.monitoring_epoch)}.pt"
    )


def _resolve_monitoring_manifest_path(paths: PathsConfig, config: HoldoutDayAnalysisConfig) -> Path:
    if config.monitoring_epoch is None:
        raise ValueError("monitoring_epoch must be set to resolve a monitoring manifest.")
    return (
        paths.get_state_predictions_dir(config.state)
        / f"{int(config.monitoring_epoch)}_holdout_backtest"
        / f"{config.loss_name}_monitoring_holdout_backtest.json"
    )


def _resolve_prediction_json_path(paths: PathsConfig, config: HoldoutDayAnalysisConfig) -> Path:
    return (
        paths.get_state_predictions_dir(config.state)
        / f"{config.loss_name}_{config.scenario_id}_prediction.json"
    )


def _load_json_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _coerce_tensor(raw_value: object, *, name: str, expected_ndim: int) -> torch.Tensor:
    tensor = raw_value.detach().cpu() if isinstance(raw_value, torch.Tensor) else torch.as_tensor(raw_value)
    if tensor.ndim != expected_ndim:
        raise ValueError(
            f"{name} must have ndim={expected_ndim}, received shape={tuple(tensor.shape)}."
        )
    return tensor


def _load_day_weight_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Day-weight artifact not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Day-weight artifact must contain a dict payload: {path}")
    required_keys = [
        "artifact_type",
        "scenario_id",
        "state",
        "loss_name",
        "source_path",
        "stock_ids",
        "target_time_indices",
        "stock_weights",
        "cash_weights",
        "portfolio_returns",
    ]
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(f"Day-weight artifact is missing keys {missing}: {path}")
    if payload.get("artifact_type") != "holdout_scenario_day_weights":
        raise ValueError(f"Unexpected day-weight artifact type in {path}: {payload.get('artifact_type')!r}")
    payload["target_time_indices"] = _coerce_tensor(
        payload["target_time_indices"],
        name="target_time_indices",
        expected_ndim=1,
    ).to(dtype=torch.long)
    payload["stock_weights"] = _coerce_tensor(
        payload["stock_weights"],
        name="stock_weights",
        expected_ndim=2,
    ).to(dtype=torch.float32)
    payload["cash_weights"] = _coerce_tensor(
        payload["cash_weights"],
        name="cash_weights",
        expected_ndim=1,
    ).to(dtype=torch.float32)
    payload["portfolio_returns"] = _coerce_tensor(
        payload["portfolio_returns"],
        name="portfolio_returns",
        expected_ndim=1,
    ).to(dtype=torch.float32)
    stock_ids = payload.get("stock_ids")
    if not isinstance(stock_ids, list) or not all(isinstance(item, str) for item in stock_ids):
        raise ValueError(f"Day-weight artifact must provide stock_ids as a list[str]: {path}")
    if payload["stock_weights"].shape[1] != len(stock_ids):
        raise ValueError(
            "Day-weight artifact stock_weights width must match stock_ids length. "
            f"Received shape={tuple(payload['stock_weights'].shape)} len(stock_ids)={len(stock_ids)}."
        )
    if payload["stock_weights"].shape[0] != payload["cash_weights"].shape[0]:
        raise ValueError("Day-weight artifact stock_weights and cash_weights must share the same time dimension.")
    if payload["portfolio_returns"].shape[0] != payload["cash_weights"].shape[0]:
        raise ValueError(
            "Day-weight artifact portfolio_returns and cash_weights must share the same time dimension."
        )
    if payload["target_time_indices"].shape[0] != payload["cash_weights"].shape[0]:
        raise ValueError(
            "Day-weight artifact target_time_indices and cash_weights must share the same time dimension."
        )
    return payload


def _build_model_from_checkpoint_metadata(
    checkpoint: dict[str, Any],
    *,
    num_stocks: int,
    max_time_steps: int,
    device: torch.device,
) -> PortfolioAttentionModel:
    checkpoint_model_config = checkpoint.get("model_config", {})
    if not isinstance(checkpoint_model_config, dict):
        raise ValueError("Checkpoint is missing a valid model_config payload.")
    if "stock_temporal_encoder_type" not in checkpoint_model_config:
        raise ValueError(
            "Checkpoint model_config is missing 'stock_temporal_encoder_type'. "
            "This checkpoint was saved with an older architecture and is not compatible with the current model."
        )

    normalized_model_config = normalize_model_config_dict(checkpoint_model_config)
    filtered_model_config = {
        key: value
        for key, value in normalized_model_config.items()
        if key in ModelConfig.__dataclass_fields__
    }
    model_config = ModelConfig(**filtered_model_config)
    max_lookback = checkpoint.get("max_lookback")
    if max_lookback is None:
        max_lookback = checkpoint.get("metadata", {}).get("max_context_time_steps")
    if max_lookback is None:
        max_lookback = max_time_steps

    model = PortfolioAttentionModel(
        model_config,
        num_stocks=num_stocks,
        max_lookback=int(max_lookback),
        stock_temporal_attention_window=int(
            checkpoint.get("data_config", {}).get("lookback_days", max_time_steps)
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    *,
    dataset: PortfolioPanelDataset,
    device: torch.device,
) -> PortfolioAttentionModel:
    return _build_model_from_checkpoint_metadata(
        checkpoint,
        num_stocks=dataset.num_stocks,
        max_time_steps=dataset.max_time_steps,
        device=device,
    )


def _validate_checkpoint_matches_request(
    checkpoint: dict[str, Any],
    *,
    state: str,
    loss_name: str,
) -> None:
    checkpoint_data_config = checkpoint.get("data_config", {})
    checkpoint_train_config = checkpoint.get("train_config", {})
    checkpoint_state = str(checkpoint_data_config.get("state", "")).strip().lower()
    checkpoint_loss_name = str(checkpoint_train_config.get("loss_name", "")).strip().lower()
    if checkpoint_state != state:
        raise ValueError(
            f"Checkpoint state mismatch: expected {state!r}, received {checkpoint_state!r}."
        )
    if checkpoint_loss_name != loss_name:
        raise ValueError(
            f"Checkpoint loss mismatch: expected {loss_name!r}, received {checkpoint_loss_name!r}."
        )


def _resolve_output_dir(paths: PathsConfig, config: HoldoutDayAnalysisConfig) -> Path:
    if config.monitoring_epoch is not None:
        return (
            paths.get_state_predictions_dir(config.state)
            / f"{int(config.monitoring_epoch)}_holdout_backtest"
            / "day_analysis"
        )
    if config.output_dir is not None:
        return config.output_dir
    return paths.get_state_predictions_dir(config.state) / "day_analysis"


def _checkpoint_supports_single_scenario_replay(checkpoint: dict[str, Any]) -> bool:
    scaler_state = checkpoint.get("scaler_state")
    selected_stock_ids = checkpoint.get("selected_stock_ids")
    if not isinstance(scaler_state, dict):
        return False
    required_scalers = ["stock_mean", "stock_std", "market_mean", "market_std"]
    if any(key not in scaler_state or scaler_state[key] is None for key in required_scalers):
        return False
    return isinstance(selected_stock_ids, list) and all(isinstance(item, str) for item in selected_stock_ids)


def _resolve_checkpoint_selected_stock_ids(checkpoint: dict[str, Any]) -> list[str]:
    selected_stock_ids = checkpoint.get("selected_stock_ids")
    if not isinstance(selected_stock_ids, list) or not all(isinstance(item, str) for item in selected_stock_ids):
        raise ValueError("Checkpoint must provide selected_stock_ids as a list[str] for single-scenario replay.")
    return [str(item) for item in selected_stock_ids]


def _resolve_checkpoint_scaler_state(
    checkpoint: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scaler_state = checkpoint.get("scaler_state")
    if not isinstance(scaler_state, dict):
        raise ValueError("Checkpoint is missing scaler_state required for single-scenario replay.")
    try:
        stock_mean = np.asarray(scaler_state["stock_mean"], dtype=np.float32)
        stock_std = np.asarray(scaler_state["stock_std"], dtype=np.float32)
        market_mean = np.asarray(scaler_state["market_mean"], dtype=np.float32)
        market_std = np.asarray(scaler_state["market_std"], dtype=np.float32)
    except KeyError as exc:
        raise ValueError(
            f"Checkpoint scaler_state is missing required key {exc.args[0]!r}."
        ) from exc
    return stock_mean, stock_std, market_mean, market_std


def _resolve_checkpoint_price_normalization_mode(checkpoint: dict[str, Any]) -> str:
    checkpoint_data_config = checkpoint.get("data_config", {})
    if isinstance(checkpoint_data_config, dict):
        raw_mode = checkpoint_data_config.get("price_normalization_mode")
        if raw_mode not in {None, ""}:
            return str(raw_mode).strip().lower()

    metadata = checkpoint.get("metadata", {})
    if isinstance(metadata, dict):
        raw_mode = metadata.get("price_normalization_mode")
        if raw_mode not in {None, ""}:
            return str(raw_mode).strip().lower()

    return "none"


def _resolve_checkpoint_source_path(
    checkpoint: dict[str, Any],
    *,
    scenario_id: str,
) -> Path:
    checkpoint_data_config = checkpoint.get("data_config", {})
    if not isinstance(checkpoint_data_config, dict):
        raise ValueError("Checkpoint is missing a valid data_config payload.")
    scenario_dir = checkpoint_data_config.get("scenario_dir")
    if scenario_dir in {None, ""}:
        raise ValueError("Checkpoint data_config is missing scenario_dir.")
    source_path = Path(str(scenario_dir)) / f"{scenario_id}.parquet"
    if not source_path.exists():
        raise FileNotFoundError(f"Scenario parquet not found for scenario_id={scenario_id!r}: {source_path}")
    return source_path


def _load_single_scenario_arrays(
    source_path: Path,
    *,
    selected_stock_ids: list[str],
) -> dict[str, Any]:
    available_columns = set(pq.read_schema(source_path).names)
    columns = [column for column in LOADABLE_COLUMNS if column in available_columns]
    frame = pd.read_parquet(source_path, columns=columns)
    required_columns = ["stock_id", "t", *STOCK_FEATURE_COLUMNS, *MARKET_FEATURE_COLUMNS]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Scenario parquet is missing required columns in {source_path}: {missing_columns}")

    numeric_columns = [*STOCK_FEATURE_COLUMNS, *MARKET_FEATURE_COLUMNS]
    if OPTIONAL_RETURN_COLUMN in frame.columns:
        numeric_columns.append(OPTIONAL_RETURN_COLUMN)
    for column in numeric_columns:
        frame[column] = _coerce_numeric_series(frame[column])
    frame["time_index"] = _parse_time_series(frame["t"])

    if frame.duplicated(["stock_id", "time_index"]).any():
        raise ValueError(f"Scenario contains duplicated (stock_id, t) rows: {source_path}")

    selected_stock_id_set = set(selected_stock_ids)
    filtered = frame[frame["stock_id"].isin(selected_stock_id_set)].copy()
    missing_stock_ids = sorted(selected_stock_id_set - set(filtered["stock_id"].unique().tolist()))
    if missing_stock_ids:
        raise ValueError(
            "Scenario parquet is missing selected stocks required by the checkpoint. "
            f"missing={missing_stock_ids[:10]}"
        )

    time_index = sorted(filtered["time_index"].unique().tolist())
    selected_count = len(selected_stock_ids)
    expected_row_count = len(time_index) * selected_count
    if len(filtered) != expected_row_count:
        raise ValueError(
            f"Scenario parquet is incomplete for selected stocks: {source_path}. "
            f"Expected {expected_row_count} rows, received {len(filtered)}."
        )

    stock_position_by_id = {stock_id: index for index, stock_id in enumerate(selected_stock_ids)}
    filtered["stock_position"] = filtered["stock_id"].map(stock_position_by_id)
    if filtered["stock_position"].isna().any():
        raise ValueError("Scenario parquet contains stock IDs that could not be mapped to selected_stock_ids.")

    filtered.sort_values(["time_index", "stock_position"], kind="stable", inplace=True)
    counts_by_time = filtered.groupby("time_index")["stock_position"].nunique()
    if not (counts_by_time == selected_count).all():
        raise ValueError(f"Scenario parquet does not contain a complete selected universe for every day: {source_path}")

    stock_features_raw = filtered[STOCK_FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True).reshape(
        len(time_index),
        selected_count,
        len(STOCK_FEATURE_COLUMNS),
    )
    market_features_raw = (
        filtered.drop_duplicates("time_index", keep="first")
        .sort_values("time_index")[MARKET_FEATURE_COLUMNS]
        .to_numpy(dtype=np.float32, copy=True)
    )
    if market_features_raw.shape != (len(time_index), len(MARKET_FEATURE_COLUMNS)):
        raise ValueError(
            "Scenario parquet produced unexpected market feature shape. "
            f"Received {market_features_raw.shape}."
        )
    if OPTIONAL_RETURN_COLUMN in filtered.columns:
        stock_returns_raw = filtered[OPTIONAL_RETURN_COLUMN].to_numpy(dtype=np.float32, copy=True).reshape(
            len(time_index),
            selected_count,
        )
    else:
        price_array = stock_features_raw[..., -1]
        stock_returns_raw = np.zeros_like(price_array)
        stock_returns_raw[1:] = (price_array[1:] / price_array[:-1]) - 1.0

    return {
        "time_index": np.asarray(time_index, dtype=np.int64),
        "stock_features_raw": stock_features_raw,
        "market_features_raw": market_features_raw,
        "stock_returns_raw": stock_returns_raw.astype(np.float32),
    }


def _build_single_scenario_item_from_checkpoint(
    checkpoint: dict[str, Any],
    *,
    scenario_id: str,
) -> tuple[dict[str, Any], list[str], Path]:
    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint is missing a valid metadata payload.")
    test_scenarios = metadata.get("test_scenarios")
    if not isinstance(test_scenarios, list) or scenario_id not in test_scenarios:
        raise ValueError(
            "Requested scenario_id is not present in the holdout/test split. "
            f"scenario_id={scenario_id!r} available={test_scenarios}"
        )

    selected_stock_ids = _resolve_checkpoint_selected_stock_ids(checkpoint)
    source_path = _resolve_checkpoint_source_path(checkpoint, scenario_id=scenario_id)
    arrays = _load_single_scenario_arrays(source_path, selected_stock_ids=selected_stock_ids)
    stock_mean, stock_std, market_mean, market_std = _resolve_checkpoint_scaler_state(checkpoint)

    if stock_mean.shape != (len(STOCK_FEATURE_COLUMNS),) or stock_std.shape != (len(STOCK_FEATURE_COLUMNS),):
        raise ValueError("Checkpoint stock scaler statistics do not match expected stock feature dimensions.")
    if market_mean.shape != (len(MARKET_FEATURE_COLUMNS),) or market_std.shape != (len(MARKET_FEATURE_COLUMNS),):
        raise ValueError("Checkpoint market scaler statistics do not match expected market feature dimensions.")

    context_feature_start = int(metadata["test_context_feature_start_index"])
    context_feature_stop = int(metadata["test_context_feature_end_index"]) + 1
    score_target_start = int(metadata["test_score_target_start_index"])
    score_target_stop = int(metadata["test_score_target_end_index"]) + 1
    context_target_start = context_feature_start + 1
    context_target_stop = context_feature_stop + 1
    price_normalization_mode = _resolve_checkpoint_price_normalization_mode(checkpoint)

    scaled_stock = scale_stock_features_for_context(
        arrays["stock_features_raw"],
        context_feature_start=context_feature_start,
        context_feature_stop=context_feature_stop,
        price_normalization_mode=price_normalization_mode,
        stock_mean=stock_mean,
        stock_std=stock_std,
    )
    scaled_market = ((arrays["market_features_raw"] - market_mean) / market_std).astype(np.float32)
    x_stock = scaled_stock
    x_market = scaled_market[context_feature_start:context_feature_stop]
    r_stock = arrays["stock_returns_raw"][context_target_start:context_target_stop]
    target_time_indices = arrays["time_index"][context_target_start:context_target_stop]
    score_mask = (
        (target_time_indices >= int(arrays["time_index"][score_target_start]))
        & (target_time_indices <= int(arrays["time_index"][score_target_stop - 1]))
    )
    if int(score_mask.sum()) <= 0:
        raise ValueError("Single-scenario replay produced an empty score mask.")

    lookback_days = int(
        checkpoint.get("data_config", {}).get(
            "lookback_days",
            metadata.get("lookback_days", 0),
        )
    )
    scenario_item: dict[str, Any] = {
        "scenario_id": scenario_id,
        "source_path": str(source_path),
        "split_name": "test",
        "feature_time_indices": torch.from_numpy(arrays["time_index"][context_feature_start:context_feature_stop]),
        "target_time_indices": torch.from_numpy(target_time_indices),
        "score_mask": torch.from_numpy(score_mask.astype(bool)),
        "x_stock": torch.from_numpy(x_stock),
        "x_market": torch.from_numpy(x_market),
        "r_stock": torch.from_numpy(r_stock),
        "stock_indices": torch.from_numpy(np.arange(len(selected_stock_ids), dtype=np.int64)),
        "evaluation_mode": ROLLING_ONE_STEP_EVALUATION_MODE,
        "rolling_window_lookback_days": lookback_days,
    }
    return scenario_item, selected_stock_ids, source_path


def _validate_monitoring_manifest_scenario(manifest_payload: dict[str, Any], *, scenario_id: str) -> None:
    scenario_artifacts = manifest_payload.get("scenario_artifacts")
    if not isinstance(scenario_artifacts, list):
        raise ValueError("Monitoring manifest must provide a scenario_artifacts list.")
    available_scenarios = [
        str(item.get("scenario_id"))
        for item in scenario_artifacts
        if isinstance(item, dict) and item.get("scenario_id") is not None
    ]
    if scenario_id not in available_scenarios:
        raise ValueError(
            "Requested scenario_id is not present in the monitoring holdout backtest manifest. "
            f"scenario_id={scenario_id!r} available={available_scenarios}"
        )


def _resolve_fast_path_inputs(
    *,
    paths: PathsConfig,
    config: HoldoutDayAnalysisConfig,
) -> ResolvedAnalysisInputs | None:
    prediction_path = _resolve_prediction_json_path(paths, config)
    if not prediction_path.exists():
        return None
    prediction_payload = _load_json_payload(prediction_path)
    if prediction_payload.get("artifact_type") != "holdout_scenario_prediction":
        raise ValueError(f"Unexpected prediction artifact type in {prediction_path}: {prediction_payload.get('artifact_type')!r}")
    raw_day_weight_artifact = prediction_payload.get("day_weight_artifact")
    if raw_day_weight_artifact in {None, ""}:
        return None
    day_weight_artifact_path = Path(str(raw_day_weight_artifact))
    if not day_weight_artifact_path.exists():
        return None
    day_weight_payload = _load_day_weight_artifact(day_weight_artifact_path)
    if str(day_weight_payload["scenario_id"]) != config.scenario_id:
        raise ValueError(
            f"Day-weight artifact scenario_id mismatch: expected {config.scenario_id!r}, "
            f"received {day_weight_payload['scenario_id']!r}."
        )
    if str(day_weight_payload["state"]).strip().lower() != config.state:
        raise ValueError(
            f"Day-weight artifact state mismatch: expected {config.state!r}, "
            f"received {day_weight_payload['state']!r}."
        )
    if str(day_weight_payload["loss_name"]).strip().lower() != config.loss_name:
        raise ValueError(
            f"Day-weight artifact loss mismatch: expected {config.loss_name!r}, "
            f"received {day_weight_payload['loss_name']!r}."
        )
    return ResolvedAnalysisInputs(
        analysis_source="final_evaluation_artifact",
        source_path=Path(str(day_weight_payload["source_path"])),
        stock_ids=list(day_weight_payload["stock_ids"]),
        scored_outputs={
            "stock_weights": day_weight_payload["stock_weights"],
            "cash_weights": day_weight_payload["cash_weights"],
            "portfolio_returns": day_weight_payload["portfolio_returns"],
            "target_time_indices": day_weight_payload["target_time_indices"],
        },
        train_config=dict(prediction_payload.get("train_config", {})),
        checkpoint_path=str(
            day_weight_payload.get("checkpoint_path", prediction_payload.get("checkpoint_path", ""))
        )
        or None,
    )
def _find_holdout_scenario_item(
    holdout_dataset: ScenarioSegmentDataset,
    scenario_id: str,
) -> dict[str, Any]:
    available_scenarios = [segment.scenario_id for segment in holdout_dataset.scenario_segments]
    for index, segment in enumerate(holdout_dataset.scenario_segments):
        if segment.scenario_id == scenario_id:
            return holdout_dataset[index]
    raise ValueError(
        "Requested scenario_id is not present in the holdout/test split. "
        f"scenario_id={scenario_id!r} available={available_scenarios}"
    )


def _collect_scored_scenario_outputs(
    *,
    model: PortfolioAttentionModel,
    scenario_item: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    score_mask = scenario_item["score_mask"]
    if not isinstance(score_mask, torch.Tensor):
        raise RuntimeError("Scenario item must provide a tensor score_mask.")
    target_time_indices = scenario_item["target_time_indices"]
    if not isinstance(target_time_indices, torch.Tensor):
        raise RuntimeError("Scenario item must provide tensor target_time_indices.")

    x_stock = scenario_item["x_stock"]
    x_market = scenario_item["x_market"]
    stock_indices = scenario_item["stock_indices"]
    target_returns = scenario_item["r_stock"]
    if not all(isinstance(value, torch.Tensor) for value in (x_stock, x_market, stock_indices, target_returns)):
        raise RuntimeError("Scenario item tensors are malformed.")

    evaluation_mode = str(
        scenario_item.get("evaluation_mode", ROLLING_ONE_STEP_EVALUATION_MODE)
    ).strip().lower()
    if evaluation_mode == ROLLING_ONE_STEP_EVALUATION_MODE:
        lookback_days = int(scenario_item.get("rolling_window_lookback_days", 0))
        scored_positions = torch.nonzero(score_mask.to(dtype=torch.bool), as_tuple=False).flatten()
        if scored_positions.numel() <= 0:
            raise RuntimeError("Checkpoint replay produced an empty score mask.")

        scored_stock_weights_rows: list[torch.Tensor] = []
        scored_cash_weights_rows: list[torch.Tensor] = []
        scored_portfolio_return_rows: list[torch.Tensor] = []
        scored_target_time_indices_rows: list[torch.Tensor] = []

        model.eval()
        for scored_position in scored_positions.tolist():
            window_start = int(scored_position) - lookback_days
            window_stop = int(scored_position) + 1
            if window_start < 0:
                raise RuntimeError(
                    "Checkpoint replay rolling window starts before the lookback warmup. "
                    f"position={scored_position} lookback_days={lookback_days}."
                )

            with torch.no_grad():
                outputs = model(
                    x_stock[window_start:window_stop].unsqueeze(0).to(device),
                    x_market[window_start:window_stop].unsqueeze(0).to(device),
                    stock_indices.unsqueeze(0).to(device),
                    target_returns=target_returns[window_start:window_stop].unsqueeze(0).to(device),
                )
            if outputs["portfolio_return"] is None:
                raise RuntimeError("Checkpoint replay did not produce portfolio_return.")

            scored_stock_weights_rows.append(outputs["stock_weights"][0, -1].detach().cpu())
            scored_cash_weights_rows.append(outputs["cash_weight"][0, -1].detach().cpu())
            scored_portfolio_return_rows.append(outputs["portfolio_return"][0, -1].detach().cpu())
            scored_target_time_indices_rows.append(target_time_indices[int(scored_position)].detach().cpu())

        scored_stock_weights = torch.stack(scored_stock_weights_rows, dim=0)
        scored_cash_weights = torch.stack(scored_cash_weights_rows, dim=0)
        scored_portfolio_returns = torch.stack(scored_portfolio_return_rows, dim=0)
        scored_target_time_indices = torch.stack(scored_target_time_indices_rows, dim=0).to(
            dtype=torch.long
        )
    else:
        model.eval()
        with torch.no_grad():
            outputs = model(
                x_stock.unsqueeze(0).to(device),
                x_market.unsqueeze(0).to(device),
                stock_indices.unsqueeze(0).to(device),
                target_returns=target_returns.unsqueeze(0).to(device),
            )

        if outputs["portfolio_return"] is None:
            raise RuntimeError("Checkpoint replay did not produce portfolio_return.")

        score_mask_device = score_mask.unsqueeze(0).to(device=device, dtype=torch.bool)
        scored_stock_weights = apply_score_mask(outputs["stock_weights"], score_mask_device).detach().cpu()[0]
        scored_cash_weights = apply_score_mask(outputs["cash_weight"], score_mask_device).detach().cpu()[0]
        scored_portfolio_returns = apply_score_mask(outputs["portfolio_return"], score_mask_device).detach().cpu()[0]
        scored_target_time_indices = apply_score_mask(
            target_time_indices.unsqueeze(0).to(device=device),
            score_mask_device,
        ).detach().cpu()[0]

    return {
        "stock_weights": scored_stock_weights,
        "cash_weights": scored_cash_weights,
        "portfolio_returns": scored_portfolio_returns,
        "target_time_indices": scored_target_time_indices,
    }


def _resolve_target_day_position(
    target_time_indices: torch.Tensor,
    target_day: int,
) -> int:
    resolved_target_day = int(target_day)
    available_days = [int(value) for value in target_time_indices.tolist()]
    matches = [index for index, value in enumerate(available_days) if value == resolved_target_day]
    if len(matches) != 1:
        raise ValueError(
            "TARGET_DAY is not present in the scored holdout time indices. "
            f"target_day={resolved_target_day} available={available_days}"
        )
    return int(matches[0])


def _build_category_label(
    *,
    mu: object,
    epsilon_variance: object,
    alpha: object,
) -> str:
    return format_allocation_group_label(
        {
            "mu": mu,
            "epsilon_variance": epsilon_variance,
            "alpha": alpha,
        }
    )


def build_stock_rows(
    *,
    stock_ids: list[str],
    day_stock_weights: torch.Tensor | np.ndarray | list[float],
    aux_frame: pd.DataFrame,
    target_day: int,
    weight_threshold: float,
) -> list[dict[str, object]]:
    if isinstance(day_stock_weights, torch.Tensor):
        weights = [float(value) for value in day_stock_weights.detach().cpu().tolist()]
    else:
        weights = [float(value) for value in np.asarray(day_stock_weights).tolist()]

    if len(weights) != len(stock_ids):
        raise ValueError(
            "day_stock_weights length must match stock_ids. "
            f"Received len(weights)={len(weights)} len(stock_ids)={len(stock_ids)}."
        )

    aux_lookup = get_aux_lookup(aux_frame)
    rows: list[dict[str, object]] = []
    for stock_id, weight in zip(stock_ids, weights):
        match = aux_lookup.get((str(stock_id), int(target_day)))
        if match is None:
            raise ValueError(
                f"Could not find aux metadata for stock_id={stock_id!r} at target_day={int(target_day)}."
            )
        rows.append(
            {
                "stock_id": str(stock_id),
                "weight": float(weight),
                "is_selected": bool(
                    _is_weight_above_threshold(float(weight), threshold=float(weight_threshold))
                ),
                "mu": match["mu"],
                "epsilon_variance": match["epsilon_variance"],
                "alpha": match["alpha"],
                "category_label": _build_category_label(
                    mu=match["mu"],
                    epsilon_variance=match["epsilon_variance"],
                    alpha=match["alpha"],
                ),
                "global_rank": None,
                "category_rank": None,
                "category_order": None,
                "plot_position": None,
            }
        )

    for rank, row in enumerate(
        sorted(rows, key=lambda item: (-float(item["weight"]), str(item["stock_id"]))),
        start=1,
    ):
        row["global_rank"] = rank

    rows_by_category: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        rows_by_category.setdefault(str(row["category_label"]), []).append(row)
    for category_rows in rows_by_category.values():
        for rank, row in enumerate(
            sorted(category_rows, key=lambda item: (-float(item["weight"]), str(item["stock_id"]))),
            start=1,
        ):
            row["category_rank"] = rank

    return sorted(rows, key=lambda item: int(item["global_rank"]))


def build_grouped_category_rows(
    stock_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for row in stock_rows:
        if not bool(row["is_selected"]):
            continue
        category_label = str(row["category_label"])
        if category_label not in grouped:
            grouped[category_label] = {
                "category_rank": None,
                "category_label": category_label,
                "mu": row["mu"],
                "epsilon_variance": row["epsilon_variance"],
                "alpha": row["alpha"],
                "total_weight": 0.0,
                "selected_stock_count": 0,
            }
        grouped_row = grouped[category_label]
        grouped_row["total_weight"] = float(grouped_row["total_weight"]) + float(row["weight"])
        grouped_row["selected_stock_count"] = int(grouped_row["selected_stock_count"]) + 1

    grouped_rows = sorted(
        grouped.values(),
        key=lambda item: (-float(item["total_weight"]), str(item["category_label"])),
    )
    for rank, row in enumerate(grouped_rows, start=1):
        row["category_rank"] = rank
    return grouped_rows


def prepare_plot_rows(
    stock_rows: list[dict[str, object]],
    grouped_rows: list[dict[str, object]],
    *,
    chart_top_k: int | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    category_order = {
        str(row["category_label"]): int(row["category_rank"])
        for row in grouped_rows
    }
    selected_rows = [
        row
        for row in stock_rows
        if bool(row["is_selected"]) and str(row["category_label"]) in category_order
    ]
    plot_rows = sorted(
        selected_rows,
        key=lambda row: (
            category_order[str(row["category_label"])],
            -float(row["weight"]),
            str(row["stock_id"]),
        ),
    )
    if chart_top_k is not None:
        plot_rows = plot_rows[: int(chart_top_k)]

    plot_rows = [dict(row) for row in plot_rows]
    for index, row in enumerate(plot_rows, start=1):
        row["plot_position"] = index
        row["category_order"] = category_order[str(row["category_label"])]

    plotted_counts: dict[str, int] = {}
    for row in plot_rows:
        category_label = str(row["category_label"])
        plotted_counts[category_label] = plotted_counts.get(category_label, 0) + 1

    plotted_group_rows: list[dict[str, object]] = []
    cursor = 0
    for grouped_row in grouped_rows:
        category_label = str(grouped_row["category_label"])
        plotted_stock_count = plotted_counts.get(category_label, 0)
        if plotted_stock_count <= 0:
            continue
        start_index = cursor
        end_index = cursor + plotted_stock_count - 1
        plotted_group_rows.append(
            {
                **grouped_row,
                "plotted_stock_count": plotted_stock_count,
                "start_index": start_index,
                "end_index": end_index,
            }
        )
        cursor += plotted_stock_count
    return plot_rows, plotted_group_rows


def _merge_plot_metadata_into_stock_rows(
    stock_rows: list[dict[str, object]],
    grouped_rows: list[dict[str, object]],
    plot_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped_row_by_label = {
        str(row["category_label"]): row
        for row in grouped_rows
    }
    plot_position_by_stock = {
        str(row["stock_id"]): int(row["plot_position"])
        for row in plot_rows
    }

    exported_rows: list[dict[str, object]] = []
    for row in stock_rows:
        category_label = str(row["category_label"])
        grouped_row = grouped_row_by_label.get(category_label)
        exported_row = dict(row)
        exported_row["category_order"] = (
            int(grouped_row["category_rank"]) if grouped_row is not None else None
        )
        exported_row["plot_position"] = plot_position_by_stock.get(str(row["stock_id"]))
        exported_rows.append(exported_row)
    return exported_rows


def _build_category_color_map(category_labels: list[str]) -> dict[str, tuple[float, float, float, float]]:
    unique_labels = sorted(set(category_labels))
    palette: list[tuple[float, float, float, float]] = []
    for cmap_name in COLOR_MAP_NAMES:
        cmap = plt.get_cmap(cmap_name)
        if hasattr(cmap, "colors"):
            palette.extend([tuple(color) for color in cmap.colors])  # type: ignore[arg-type]
        else:
            palette.extend([tuple(cmap(index / max(1, cmap.N - 1))) for index in range(cmap.N)])
    if not palette:
        raise RuntimeError("Could not build a category palette.")
    return {
        label: palette[index % len(palette)]
        for index, label in enumerate(unique_labels)
    }


def render_anonymous_stock_bar_chart(
    *,
    state: str,
    loss_name: str,
    scenario_id: str,
    target_day: int,
    plot_rows: list[dict[str, object]],
    plotted_group_rows: list[dict[str, object]],
    selected_stock_count: int,
    total_selected_weight: float,
    cash_weight: float,
    hide_x_axis: bool,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not plot_rows:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            "No stocks above threshold",
            ha="center",
            va="center",
            fontsize=20,
            transform=ax.transAxes,
        )
        ax.set_title(
            f"Holdout Day Allocation: state={state} loss={loss_name} "
            f"scenario={scenario_id} day={int(target_day)}"
        )
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    plotted_category_labels = [str(row["category_label"]) for row in plot_rows]
    category_color_map = _build_category_color_map(plotted_category_labels)
    x_values = np.arange(len(plot_rows), dtype=np.float32)
    y_values = np.asarray([float(row["weight"]) for row in plot_rows], dtype=np.float32)
    colors = [category_color_map[str(row["category_label"])] for row in plot_rows]
    max_categories = max(1, len(plotted_group_rows))
    fig_width = max(14.0, min(32.0, 8.0 + len(plot_rows) * 0.18))
    fig_height = max(8.0, min(18.0, 6.5 + max_categories * 0.35))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.bar(x_values, y_values, color=colors, width=0.92)
    ax.set_xlim(-0.6, len(plot_rows) - 0.4)
    ax.set_xticks([])
    ax.tick_params(axis="x", length=0, bottom=False, labelbottom=False)
    if hide_x_axis:
        ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("Weight")
    if not hide_x_axis:
        ax.set_xlabel("Selected stocks ordered by category total weight, then stock weight")
    ax.set_title(
        f"Holdout Day Allocation: state={state} loss={loss_name} "
        f"scenario={scenario_id} day={int(target_day)}"
    )

    top_category_share = 0.0
    if plotted_group_rows and total_selected_weight > 0.0:
        top_category_share = float(plotted_group_rows[0]["total_weight"]) / total_selected_weight
    metrics_text = (
        f"Cash Weight: {float(cash_weight):.6f}\n"
        f"Selected Stocks: {int(selected_stock_count)}\n"
        f"Plotted Stocks: {len(plot_rows)}\n"
        f"Top Category Share: {top_category_share:.6f}"
    )
    ax.text(
        0.99,
        0.99,
        metrics_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.92,
        },
    )

    for index, grouped_row in enumerate(plotted_group_rows):
        if index > 0:
            ax.axvline(
                float(grouped_row["start_index"]) - 0.5,
                color="0.35",
                linestyle="--",
                linewidth=1.0,
                alpha=0.85,
            )

    if not hide_x_axis:
        label_transform = blended_transform_factory(ax.transData, ax.transAxes)
        for grouped_row in plotted_group_rows:
            center = (float(grouped_row["start_index"]) + float(grouped_row["end_index"])) / 2.0
            label = (
                f"{grouped_row['category_label']}\n"
                f"n={int(grouped_row['plotted_stock_count'])}"
            )
            ax.text(
                center,
                -0.10,
                label,
                transform=label_transform,
                ha="right",
                va="top",
                rotation=35,
                fontsize=8,
            )

    legend_handles = [
        Patch(
            facecolor=category_color_map[str(grouped_row["category_label"])],
            label=str(grouped_row["category_label"]),
        )
        for grouped_row in plotted_group_rows
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=10,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(bottom=0.12 if hide_x_axis else 0.33, right=0.75)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _format_terminal_summary(summary_payload: dict[str, object]) -> str:
    lines = [
        f"analysis_source: {summary_payload['analysis_source']}",
        f"checkpoint_path: {summary_payload['checkpoint_path']}",
        f"state: {summary_payload['state']}",
        f"loss_name: {summary_payload['loss_name']}",
        f"scenario_id: {summary_payload['scenario_id']}",
        f"target_day: {summary_payload['target_day']}",
        f"cash_weight: {float(summary_payload['cash_weight']):.8f}",
        f"selected_stock_count: {summary_payload['selected_stock_count']}",
        f"plotted_stock_count: {summary_payload['plotted_stock_count']}",
        f"portfolio_return_to_day: {float(summary_payload['portfolio_return_to_day']):.8f}",
        f"backtest_portfolio_sr: {float(summary_payload['backtest_portfolio_sr']):.8f}",
        "top_selected_stocks:",
    ]
    for item in summary_payload["top_selected_stocks"]:
        lines.append(
            "  "
            f"{item['stock_id']} | weight={float(item['weight']):.8f} | category={item['category_label']}"
        )
    lines.append("top_categories:")
    for item in summary_payload["top_categories"]:
        lines.append(
            "  "
            f"{item['category_label']} | total_weight={float(item['total_weight']):.8f} | "
            f"selected_stock_count={int(item['selected_stock_count'])}"
        )
    lines.append("output_paths:")
    for key, value in summary_payload["output_paths"].items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def _build_summary_payload(
    *,
    config: HoldoutDayAnalysisConfig,
    resolved_inputs: ResolvedAnalysisInputs,
    output_dir: Path,
    aux_frame: pd.DataFrame,
) -> dict[str, object]:
    scored_outputs = resolved_inputs.scored_outputs
    target_time_indices = scored_outputs["target_time_indices"]
    day_position = _resolve_target_day_position(target_time_indices, config.target_day)
    day_stock_weights = scored_outputs["stock_weights"][day_position]
    cash_weight = float(scored_outputs["cash_weights"][day_position].item())
    stock_rows = build_stock_rows(
        stock_ids=resolved_inputs.stock_ids,
        day_stock_weights=day_stock_weights,
        aux_frame=aux_frame,
        target_day=config.target_day,
        weight_threshold=config.weight_threshold,
    )
    grouped_rows = build_grouped_category_rows(stock_rows)
    plot_rows, plotted_group_rows = prepare_plot_rows(
        stock_rows,
        grouped_rows,
        chart_top_k=config.chart_top_k,
    )
    total_selected_weight = float(sum(float(row["weight"]) for row in stock_rows if bool(row["is_selected"])))
    selected_stock_count = int(sum(int(row["is_selected"]) for row in stock_rows))
    exported_stock_rows = _merge_plot_metadata_into_stock_rows(
        stock_rows,
        grouped_rows,
        plot_rows,
    )

    artifact_stem = f"{config.loss_name}_{config.scenario_id}_day_{int(config.target_day)}"
    stock_csv_path = output_dir / f"{artifact_stem}_stock_weights.csv"
    grouped_csv_path = output_dir / f"{artifact_stem}_grouped_allocations.csv"
    chart_path = output_dir / f"{artifact_stem}_stock_bar.png"
    summary_path = output_dir / f"{artifact_stem}_summary.json"

    pd.DataFrame(exported_stock_rows).reindex(columns=STOCK_ROW_COLUMNS).to_csv(stock_csv_path, index=False)
    pd.DataFrame(grouped_rows).reindex(columns=GROUPED_ROW_COLUMNS).to_csv(grouped_csv_path, index=False)
    render_anonymous_stock_bar_chart(
        state=config.state,
        loss_name=config.loss_name,
        scenario_id=config.scenario_id,
        target_day=config.target_day,
        plot_rows=plot_rows,
        plotted_group_rows=plotted_group_rows,
        selected_stock_count=selected_stock_count,
        total_selected_weight=total_selected_weight,
        cash_weight=cash_weight,
        hide_x_axis=config.hide_x_axis,
        output_path=chart_path,
    )

    portfolio_returns = scored_outputs["portfolio_returns"]
    portfolio_return_to_day = float(torch.prod(1.0 + portfolio_returns[: day_position + 1]).item() - 1.0)
    backtest_portfolio_sr = _compute_backtest_portfolio_sr(portfolio_returns)
    top_selected_stocks = [
        {
            "stock_id": row["stock_id"],
            "weight": float(row["weight"]),
            "category_label": row["category_label"],
        }
        for row in exported_stock_rows
        if bool(row["is_selected"])
    ][:10]
    top_categories = [
        {
            "category_label": row["category_label"],
            "total_weight": float(row["total_weight"]),
            "selected_stock_count": int(row["selected_stock_count"]),
        }
        for row in grouped_rows[:10]
    ]
    summary_payload: dict[str, object] = {
        "analysis_source": resolved_inputs.analysis_source,
        "checkpoint_path": resolved_inputs.checkpoint_path,
        "state": config.state,
        "loss_name": config.loss_name,
        "scenario_id": config.scenario_id,
        "target_day": int(config.target_day),
        "target_day_position": int(day_position),
        "available_target_time_indices": [int(value) for value in target_time_indices.tolist()],
        "weight_threshold": float(config.weight_threshold),
        "chart_top_k": config.chart_top_k,
        "hide_x_axis": config.hide_x_axis,
        "cash_weight": cash_weight,
        "selected_stock_count": selected_stock_count,
        "plotted_stock_count": len(plot_rows),
        "total_selected_weight": total_selected_weight,
        "portfolio_return_to_day": portfolio_return_to_day,
        "backtest_portfolio_sr": backtest_portfolio_sr,
        "train_config": dict(resolved_inputs.train_config),
        "source_path": str(resolved_inputs.source_path),
        "output_paths": {
            "stock_weights_csv": str(stock_csv_path),
            "grouped_allocations_csv": str(grouped_csv_path),
            "stock_bar_png": str(chart_path),
            "summary_json": str(summary_path),
        },
        "top_selected_stocks": top_selected_stocks,
        "top_categories": top_categories,
    }
    save_json(summary_payload, summary_path)
    return summary_payload


def analyze_holdout_day(config: HoldoutDayAnalysisConfig) -> dict[str, object]:
    paths = PathsConfig()
    ensure_output_dirs(paths)
    output_dir = _resolve_output_dir(paths, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device_name)

    if config.monitoring_epoch is not None:
        total_steps = 9
        _emit_progress(
            config,
            phase="resolve_inputs",
            step_index=1,
            step_count=total_steps,
            message="Preparing monitoring day analysis request.",
            scenario_id=config.scenario_id,
            epoch=config.monitoring_epoch,
        )
        _emit_progress(
            config,
            phase="resolve_source",
            step_index=2,
            step_count=total_steps,
            message="Resolving canonical monitoring artifact locations.",
            scenario_id=config.scenario_id,
            epoch=config.monitoring_epoch,
            artifact_source="monitoring",
        )
        _emit_progress(
            config,
            phase="resolve_monitoring_manifest",
            step_index=3,
            step_count=total_steps,
            message="Validating monitoring manifest and scenario membership.",
            scenario_id=config.scenario_id,
            epoch=config.monitoring_epoch,
        )
        _emit_progress(
            config,
            phase="load_checkpoint",
            step_index=4,
            step_count=total_steps,
            message="Loading canonical monitoring checkpoint.",
            scenario_id=config.scenario_id,
            epoch=config.monitoring_epoch,
        )
        monitoring_manifest_path = _resolve_monitoring_manifest_path(paths, config)
        monitoring_checkpoint_path = _resolve_monitoring_checkpoint_path(paths, config)
        if config.checkpoint_path is not None and _canonicalize_path(config.checkpoint_path) != _canonicalize_path(
            monitoring_checkpoint_path
        ):
            raise ValueError(
                "CHECKPOINT_PATH does not match the canonical monitoring checkpoint for the requested MONITORING_EPOCH. "
                f"expected={monitoring_checkpoint_path} received={config.checkpoint_path}"
            )
        manifest_payload = _load_json_payload(monitoring_manifest_path)
        if manifest_payload.get("artifact_type") != "monitoring_holdout_backtest_manifest":
            raise ValueError(
                "Unexpected monitoring manifest artifact_type. "
                f"path={monitoring_manifest_path} artifact_type={manifest_payload.get('artifact_type')!r}"
            )
        _validate_monitoring_manifest_scenario(manifest_payload, scenario_id=config.scenario_id)
        checkpoint = _load_checkpoint(monitoring_checkpoint_path, map_location=device)
        _validate_checkpoint_matches_request(checkpoint, state=config.state, loss_name=config.loss_name)
        if _checkpoint_supports_single_scenario_replay(checkpoint):
            _emit_progress(
                config,
                phase="load_single_scenario",
                step_index=5,
                step_count=total_steps,
                message="Loading target scenario only for monitoring replay.",
                scenario_id=config.scenario_id,
                epoch=config.monitoring_epoch,
            )
            scenario_item, stock_ids, source_path = _build_single_scenario_item_from_checkpoint(
                checkpoint,
                scenario_id=config.scenario_id,
            )
            _emit_progress(
                config,
                phase="replay_model",
                step_index=6,
                step_count=total_steps,
                message="Replaying model on the single holdout scenario.",
                scenario_id=config.scenario_id,
                epoch=config.monitoring_epoch,
            )
            model = _build_model_from_checkpoint_metadata(
                checkpoint,
                num_stocks=len(stock_ids),
                max_time_steps=int(scenario_item["x_stock"].shape[0]),  # type: ignore[index]
                device=device,
            )
            scored_outputs = _collect_scored_scenario_outputs(
                model=model,
                scenario_item=scenario_item,
                device=device,
            )
            resolved_inputs = ResolvedAnalysisInputs(
                analysis_source="monitoring_single_scenario_replay",
                source_path=source_path,
                stock_ids=stock_ids,
                scored_outputs=scored_outputs,
                train_config=_extract_exported_train_config(checkpoint),
                checkpoint_path=str(monitoring_checkpoint_path),
            )
        else:
            raise ValueError(
                "Checkpoint lacks single-scenario replay metadata, and rebuilding legacy "
                "internal-split datasets is no longer supported."
            )

        _emit_progress(
            config,
            phase="load_aux_frame",
            step_index=7,
            step_count=total_steps,
            message="Loading auxiliary grouping columns for the target day.",
            scenario_id=config.scenario_id,
            epoch=config.monitoring_epoch,
        )
        aux_frame = load_aux_frame(resolved_inputs.source_path)
        _emit_progress(
            config,
            phase="export_outputs",
            step_index=8,
            step_count=total_steps,
            message="Writing CSV, PNG, and summary outputs.",
            scenario_id=config.scenario_id,
            epoch=config.monitoring_epoch,
        )
        summary_payload = _build_summary_payload(
            config=config,
            resolved_inputs=resolved_inputs,
            output_dir=output_dir,
            aux_frame=aux_frame,
        )
        _emit_progress(
            config,
            phase="completed",
            step_index=9,
            step_count=total_steps,
            message="Monitoring day analysis completed.",
            scenario_id=config.scenario_id,
            epoch=config.monitoring_epoch,
        )
        return summary_payload

    _emit_progress(
        config,
        phase="resolve_inputs",
        step_index=1,
        step_count=7,
        message="Preparing holdout day analysis request.",
        scenario_id=config.scenario_id,
    )
    _emit_progress(
        config,
        phase="resolve_source",
        step_index=2,
        step_count=7,
        message="Checking for reusable final-evaluation day-weight artifacts.",
        scenario_id=config.scenario_id,
        artifact_source="final_evaluation",
    )
    fast_path_inputs = _resolve_fast_path_inputs(paths=paths, config=config)
    if fast_path_inputs is not None:
        _emit_progress(
            config,
            phase="load_prediction_artifact",
            step_index=3,
            step_count=7,
            message="Loading final evaluation prediction artifact.",
            scenario_id=config.scenario_id,
            artifact_source="final_evaluation",
        )
        _emit_progress(
            config,
            phase="load_day_weight_artifact",
            step_index=4,
            step_count=7,
            message="Loading compact day-weight artifact.",
            scenario_id=config.scenario_id,
            artifact_source="final_evaluation",
        )
        _emit_progress(
            config,
            phase="load_aux_frame",
            step_index=5,
            step_count=7,
            message="Loading auxiliary grouping columns for the target day.",
            scenario_id=config.scenario_id,
        )
        aux_frame = load_aux_frame(fast_path_inputs.source_path)
        _emit_progress(
            config,
            phase="export_outputs",
            step_index=6,
            step_count=7,
            message="Writing CSV, PNG, and summary outputs.",
            scenario_id=config.scenario_id,
        )
        summary_payload = _build_summary_payload(
            config=config,
            resolved_inputs=fast_path_inputs,
            output_dir=output_dir,
            aux_frame=aux_frame,
        )
        _emit_progress(
            config,
            phase="completed",
            step_index=7,
            step_count=7,
            message="Holdout day analysis completed.",
            scenario_id=config.scenario_id,
            artifact_source="final_evaluation",
        )
        return summary_payload

    if config.checkpoint_path is None:
        raise ValueError(
            "No reusable final-evaluation day-weight artifact was found and CHECKPOINT_PATH is not configured."
        )

    total_steps = 8
    _emit_progress(
        config,
        phase="load_checkpoint",
        step_index=3,
        step_count=total_steps,
        message="Loading checkpoint for replay fallback.",
        scenario_id=config.scenario_id,
    )
    checkpoint = _load_checkpoint(config.checkpoint_path, map_location=device)
    _validate_checkpoint_matches_request(checkpoint, state=config.state, loss_name=config.loss_name)

    if _checkpoint_supports_single_scenario_replay(checkpoint):
        _emit_progress(
            config,
            phase="load_single_scenario",
            step_index=4,
            step_count=total_steps,
            message="Loading target scenario only for replay.",
            scenario_id=config.scenario_id,
        )
        scenario_item, stock_ids, source_path = _build_single_scenario_item_from_checkpoint(
            checkpoint,
            scenario_id=config.scenario_id,
        )
        _emit_progress(
            config,
            phase="replay_model",
            step_index=5,
            step_count=total_steps,
            message="Replaying model on the single holdout scenario.",
            scenario_id=config.scenario_id,
        )
        model = _build_model_from_checkpoint_metadata(
            checkpoint,
            num_stocks=len(stock_ids),
            max_time_steps=int(scenario_item["x_stock"].shape[0]),  # type: ignore[index]
            device=device,
        )
        scored_outputs = _collect_scored_scenario_outputs(model=model, scenario_item=scenario_item, device=device)
        resolved_inputs = ResolvedAnalysisInputs(
            analysis_source="checkpoint_single_scenario_replay",
            source_path=source_path,
            stock_ids=stock_ids,
            scored_outputs=scored_outputs,
            train_config=_extract_exported_train_config(checkpoint),
            checkpoint_path=str(config.checkpoint_path),
        )
    else:
        _fail_if_legacy_checkpoint_replay_uses_removed_lookback_mode(checkpoint)
        raise ValueError(
            "Checkpoint lacks single-scenario replay metadata, and rebuilding legacy "
            "internal-split datasets is no longer supported."
        )

    _emit_progress(
        config,
        phase="load_aux_frame",
        step_index=6,
        step_count=total_steps,
        message="Loading auxiliary grouping columns for the target day.",
        scenario_id=config.scenario_id,
    )
    aux_frame = load_aux_frame(resolved_inputs.source_path)
    _emit_progress(
        config,
        phase="export_outputs",
        step_index=7,
        step_count=total_steps,
        message="Writing CSV, PNG, and summary outputs.",
        scenario_id=config.scenario_id,
    )
    summary_payload = _build_summary_payload(
        config=config,
        resolved_inputs=resolved_inputs,
        output_dir=output_dir,
        aux_frame=aux_frame,
    )
    _emit_progress(
        config,
        phase="completed",
        step_index=8,
        step_count=total_steps,
        message="Holdout day analysis completed.",
        scenario_id=config.scenario_id,
        artifact_source=resolved_inputs.analysis_source,
    )
    return summary_payload


def main() -> None:
    config = _build_runtime_config()
    summary_payload = analyze_holdout_day(config)
    print(_format_terminal_summary(summary_payload))


if __name__ == "__main__":
    main()
