"""Shared run-metadata keys, validation, and mutation helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

KEY_EPOCH = "epoch"
KEY_LOSS_NAME = "loss_name"
KEY_STATE = "state"
KEY_HISTORY = "history"
KEY_FINAL_BACKTEST = "final_backtest"

KEY_TRAIN_LOSS = "train_loss"
KEY_TRAIN_MEAN_FINAL_RETURN = "train_mean_final_return"
KEY_VAL_LOSS = "val_loss"
KEY_VAL_MEAN_FINAL_RETURN = "val_mean_final_return"

KEY_CURRENT_WINDOW_BEST_EPOCH = "current_window_best_epoch"
KEY_CURRENT_WINDOW_BEST_VAL_LOSS = "current_window_best_val_loss"
KEY_GLOBAL_BEST_VAL_LOSS = "global_best_val_loss"
KEY_GLOBAL_BEST_CHECKPOINT_UPDATED = "global_best_checkpoint_updated"
KEY_EPOCHS_WITHOUT_IMPROVEMENT = "epochs_without_improvement"

KEY_HOLDOUT_BACKTEST_RAN = "holdout_backtest_ran"
KEY_HOLDOUT_BACKTEST_EPOCH = "holdout_backtest_epoch"
KEY_HOLDOUT_BACKTEST_LOSS = "holdout_backtest_loss"
KEY_HOLDOUT_BACKTEST_MEAN_FINAL_RETURN = "holdout_backtest_mean_final_return"
KEY_HOLDOUT_BACKTEST_STD_FINAL_RETURN = "holdout_backtest_std_final_return"
KEY_HOLDOUT_BACKTEST_MEDIAN_FINAL_RETURN = "holdout_backtest_median_final_return"
KEY_HOLDOUT_BACKTEST_WORST_SCENARIO_FINAL_RETURN = "holdout_backtest_worst_scenario_final_return"
KEY_HOLDOUT_BACKTEST_BEST_SCENARIO_FINAL_RETURN = "holdout_backtest_best_scenario_final_return"
KEY_HOLDOUT_BACKTEST_BEST_SCENARIO_ID = "holdout_backtest_best_scenario_id"
KEY_HOLDOUT_BACKTEST_OUTPUT_DIR = "holdout_backtest_output_dir"
KEY_HOLDOUT_BACKTEST_OVERVIEW_PATHS = "holdout_backtest_overview_paths"
KEY_MONITORING_CHECKPOINT_PATH = "monitoring_checkpoint_path"

KEY_OVERVIEW_LOSS_ORDER = "overview_loss_order"
KEY_SCENARIO_ARTIFACTS = "scenario_artifacts"
KEY_SCENARIO_ID = "scenario_id"
KEY_WEIGHT_TRAJECTORY_OVERVIEW_CHART = "weight_trajectory_overview_chart"

RESUME_FLOAT_MATCH_KEYS = (
    KEY_TRAIN_LOSS,
    KEY_TRAIN_MEAN_FINAL_RETURN,
    KEY_VAL_LOSS,
    KEY_VAL_MEAN_FINAL_RETURN,
)
RESUME_EXACT_MATCH_KEYS = (
    KEY_HOLDOUT_BACKTEST_RAN,
    KEY_HOLDOUT_BACKTEST_EPOCH,
    KEY_HOLDOUT_BACKTEST_OUTPUT_DIR,
    KEY_MONITORING_CHECKPOINT_PATH,
)


def validate_history_list(history: Any) -> list[dict[str, Any]]:
    if not isinstance(history, list):
        raise ValueError("Payload must provide history as a list.")
    validated: list[dict[str, Any]] = []
    for item in history:
        if not isinstance(item, dict):
            raise ValueError("History entries must be objects.")
        validated.append(item)
    return validated


def validate_manifest_scenario_artifacts(manifest_payload: dict[str, Any]) -> list[dict[str, Any]]:
    scenario_artifacts = manifest_payload.get(KEY_SCENARIO_ARTIFACTS)
    if not isinstance(scenario_artifacts, list):
        raise ValueError("Monitoring manifest must provide a scenario_artifacts list.")
    validated: list[dict[str, Any]] = []
    for item in scenario_artifacts:
        if not isinstance(item, dict):
            raise ValueError("Monitoring manifest scenario_artifacts entries must be objects.")
        validated.append(item)
    return validated


def create_epoch_metrics(
    *,
    epoch: int,
    train_loss: float,
    train_mean_final_return: float,
    val_loss: float,
    val_mean_final_return: float,
    validation_epoch_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        KEY_EPOCH: epoch,
        KEY_TRAIN_LOSS: train_loss,
        KEY_TRAIN_MEAN_FINAL_RETURN: train_mean_final_return,
        KEY_VAL_LOSS: val_loss,
        KEY_VAL_MEAN_FINAL_RETURN: val_mean_final_return,
        **validation_epoch_metadata,
        KEY_HOLDOUT_BACKTEST_RAN: False,
        KEY_HOLDOUT_BACKTEST_EPOCH: None,
        KEY_HOLDOUT_BACKTEST_LOSS: None,
        KEY_HOLDOUT_BACKTEST_OUTPUT_DIR: None,
        KEY_HOLDOUT_BACKTEST_OVERVIEW_PATHS: [],
        KEY_MONITORING_CHECKPOINT_PATH: None,
    }


def apply_monitoring_backtest_to_epoch_metrics(
    epoch_metrics: dict[str, Any],
    *,
    epoch: int,
    monitoring_backtest: dict[str, Any],
) -> None:
    epoch_metrics.update(
        {
            KEY_HOLDOUT_BACKTEST_RAN: True,
            KEY_HOLDOUT_BACKTEST_EPOCH: epoch,
            KEY_HOLDOUT_BACKTEST_LOSS: monitoring_backtest[KEY_HOLDOUT_BACKTEST_LOSS],
            KEY_HOLDOUT_BACKTEST_MEAN_FINAL_RETURN: monitoring_backtest["mean_final_return"],
            KEY_HOLDOUT_BACKTEST_STD_FINAL_RETURN: monitoring_backtest["std_final_return"],
            KEY_HOLDOUT_BACKTEST_MEDIAN_FINAL_RETURN: monitoring_backtest["median_final_return"],
            KEY_HOLDOUT_BACKTEST_WORST_SCENARIO_FINAL_RETURN: monitoring_backtest[
                "worst_scenario_final_return"
            ],
            KEY_HOLDOUT_BACKTEST_BEST_SCENARIO_FINAL_RETURN: monitoring_backtest[
                "best_scenario_final_return"
            ],
            KEY_HOLDOUT_BACKTEST_BEST_SCENARIO_ID: monitoring_backtest["best_scenario_id"],
            KEY_HOLDOUT_BACKTEST_OUTPUT_DIR: monitoring_backtest[KEY_HOLDOUT_BACKTEST_OUTPUT_DIR],
            KEY_HOLDOUT_BACKTEST_OVERVIEW_PATHS: list(
                monitoring_backtest[KEY_HOLDOUT_BACKTEST_OVERVIEW_PATHS]
            ),
        }
    )


def set_monitoring_checkpoint_path(epoch_metrics: dict[str, Any], checkpoint_path: Path | str) -> None:
    epoch_metrics[KEY_MONITORING_CHECKPOINT_PATH] = str(checkpoint_path)


def inject_best_state_fields(
    metrics_payload: dict[str, Any],
    *,
    current_window_best_epoch: int,
    current_window_best_val_loss: float,
    global_best_val_loss: float,
    global_best_checkpoint_updated: bool,
    epochs_without_improvement: int,
) -> None:
    metrics_payload[KEY_CURRENT_WINDOW_BEST_EPOCH] = current_window_best_epoch
    metrics_payload[KEY_CURRENT_WINDOW_BEST_VAL_LOSS] = current_window_best_val_loss
    metrics_payload[KEY_GLOBAL_BEST_VAL_LOSS] = global_best_val_loss
    metrics_payload[KEY_GLOBAL_BEST_CHECKPOINT_UPDATED] = global_best_checkpoint_updated
    metrics_payload[KEY_EPOCHS_WITHOUT_IMPROVEMENT] = epochs_without_improvement


def update_monitoring_manifest_overview_paths(
    manifest_payload: dict[str, Any],
    *,
    overview_path_by_scenario: dict[str, str],
    loss_order: tuple[str, ...],
) -> bool:
    scenario_artifacts = validate_manifest_scenario_artifacts(manifest_payload)
    changed = False

    resolved_loss_order = list(loss_order)
    if manifest_payload.get(KEY_OVERVIEW_LOSS_ORDER) != resolved_loss_order:
        manifest_payload[KEY_OVERVIEW_LOSS_ORDER] = resolved_loss_order
        changed = True

    ordered_paths: list[str] = []
    for item in scenario_artifacts:
        scenario_id = str(item.get(KEY_SCENARIO_ID, ""))
        overview_path = overview_path_by_scenario.get(scenario_id)
        ordered_paths.append(overview_path if overview_path is not None else "")
        if item.get(KEY_WEIGHT_TRAJECTORY_OVERVIEW_CHART) != overview_path:
            item[KEY_WEIGHT_TRAJECTORY_OVERVIEW_CHART] = overview_path
            changed = True

    resolved_paths = [path for path in ordered_paths if path]
    if manifest_payload.get(KEY_HOLDOUT_BACKTEST_OVERVIEW_PATHS) != resolved_paths:
        manifest_payload[KEY_HOLDOUT_BACKTEST_OVERVIEW_PATHS] = resolved_paths
        changed = True
    return changed


def resolve_payload_state(payload: dict[str, Any]) -> str:
    final_backtest = payload.get(KEY_FINAL_BACKTEST)
    nested_state = final_backtest.get(KEY_STATE, "") if isinstance(final_backtest, dict) else ""
    return str(payload.get(KEY_STATE) or nested_state or "").lower()


def update_train_metrics_history_overview_paths(
    payload: dict[str, Any],
    *,
    state: str,
    epoch: int,
    holdout_backtest_output_dir: str,
    overview_paths: list[str],
) -> bool:
    payload_state = resolve_payload_state(payload)
    if payload_state and payload_state != state.lower():
        return False

    history = payload.get(KEY_HISTORY)
    if not isinstance(history, list):
        return False

    changed = False
    for item in history:
        if not isinstance(item, dict):
            continue
        if int(item.get(KEY_EPOCH, -1)) != int(epoch):
            continue
        if str(item.get(KEY_HOLDOUT_BACKTEST_OUTPUT_DIR) or "") != holdout_backtest_output_dir:
            continue
        if item.get(KEY_HOLDOUT_BACKTEST_OVERVIEW_PATHS) == overview_paths:
            continue
        item[KEY_HOLDOUT_BACKTEST_OVERVIEW_PATHS] = list(overview_paths)
        changed = True
    return changed


def set_scenario_artifacts_overview_path(
    payload: dict[str, Any],
    *,
    scenario_id: str | None,
    overview_path: str | None,
) -> bool:
    scenario_artifacts = payload.get(KEY_SCENARIO_ARTIFACTS)
    if not isinstance(scenario_artifacts, list):
        return False

    changed = False
    for item in scenario_artifacts:
        if not isinstance(item, dict):
            continue
        if scenario_id is not None and str(item.get(KEY_SCENARIO_ID)) != scenario_id:
            continue
        if item.get(KEY_WEIGHT_TRAJECTORY_OVERVIEW_CHART) == overview_path:
            continue
        item[KEY_WEIGHT_TRAJECTORY_OVERVIEW_CHART] = overview_path
        changed = True
    return changed


def update_payload_overview_paths(
    payload: dict[str, Any],
    *,
    state: str,
    scenario_id: str | None,
    overview_path: str | None,
) -> bool:
    payload_state = resolve_payload_state(payload)
    if payload_state and payload_state != state.lower():
        return False

    changed = set_scenario_artifacts_overview_path(
        payload,
        scenario_id=scenario_id,
        overview_path=overview_path,
    )
    final_backtest = payload.get(KEY_FINAL_BACKTEST)
    if isinstance(final_backtest, dict):
        changed = (
            set_scenario_artifacts_overview_path(
                final_backtest,
                scenario_id=scenario_id,
                overview_path=overview_path,
            )
            or changed
        )
    return changed


def resume_metric_matches(history_value: Any, checkpoint_value: Any) -> bool:
    try:
        return math.isclose(float(history_value), float(checkpoint_value), rel_tol=1e-9, abs_tol=1e-9)
    except (TypeError, ValueError):
        return history_value == checkpoint_value


def resume_history_item_matches_checkpoint(
    history_item: dict[str, Any],
    checkpoint_metrics: dict[str, Any],
    *,
    history_epoch: int | None,
) -> bool:
    compared_metric = False
    for key in RESUME_FLOAT_MATCH_KEYS:
        if key not in checkpoint_metrics or key not in history_item:
            continue
        compared_metric = True
        if not resume_metric_matches(history_item.get(key), checkpoint_metrics.get(key)):
            return False

    for key in RESUME_EXACT_MATCH_KEYS:
        if key not in checkpoint_metrics or key not in history_item:
            continue
        compared_metric = True
        if history_item.get(key) != checkpoint_metrics.get(key):
            return False

    if not compared_metric and KEY_EPOCH in checkpoint_metrics:
        if not resume_metric_matches(history_epoch, checkpoint_metrics.get(KEY_EPOCH)):
            return False
    return True
