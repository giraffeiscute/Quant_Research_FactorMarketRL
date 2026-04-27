"""Legacy compatibility exports for evaluate module split helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch

from portfolio_attention.cli import evaluate_rebuild
from portfolio_attention.config import PathsConfig
from portfolio_attention.evaluation import (
    artifacts as evaluation_artifacts,
    presentation as evaluation_presentation,
    shared as evaluation_shared,
)

MOVED_SYMBOL_FACADE_EXPORTS = (
    "_normalize_overview_loss_order",
    "_compute_backtest_portfolio_sr",
    "_extract_exported_train_config",
    "_get_aux_lookup",
    "_is_weight_above_threshold",
    "_load_aux_frame",
    "enrich_top_k_positions",
    "format_allocation_group_label",
    "refresh_existing_scenario_artifacts",
    "rebuild_monitoring_holdout_backtest_overviews",
    "cleanup_monitoring_holdout_backtest_artifacts",
    "rebuild_multi_loss_weight_trajectory_overviews",
    "cleanup_multi_loss_weight_trajectory_overviews",
    "backfill_monitoring_holdout_backtest_overviews",
)


def _normalize_overview_loss_order(loss_order: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return evaluation_shared.normalize_overview_loss_order(loss_order)


def _compute_backtest_portfolio_sr(portfolio_returns: torch.Tensor) -> float:
    return evaluation_artifacts._compute_backtest_portfolio_sr(portfolio_returns)


def _extract_exported_train_config(checkpoint: dict[str, Any]) -> dict[str, object]:
    return evaluation_artifacts.extract_exported_train_config(checkpoint)


def _get_aux_lookup(aux_frame: pd.DataFrame) -> dict[tuple[str, int], dict[str, object]]:
    return evaluation_presentation.get_aux_lookup(aux_frame)


def _is_weight_above_threshold(weight: float, *, threshold: float) -> bool:
    return evaluation_shared.is_weight_above_threshold(weight, threshold=threshold)


def _load_aux_frame(source_path):
    return evaluation_presentation.load_aux_frame(source_path)


def enrich_top_k_positions(*args, **kwargs):
    return evaluation_presentation.enrich_top_k_positions(*args, **kwargs)


def format_allocation_group_label(grouped_allocation: dict[str, object]) -> str:
    return evaluation_presentation.format_allocation_group_label(grouped_allocation)


def refresh_existing_scenario_artifacts(*args, **kwargs):
    return evaluate_rebuild.refresh_existing_scenario_artifacts(*args, **kwargs)


def rebuild_monitoring_holdout_backtest_overviews(*args, **kwargs):
    return evaluate_rebuild.rebuild_monitoring_holdout_backtest_overviews(*args, **kwargs)


def cleanup_monitoring_holdout_backtest_artifacts(*, paths: PathsConfig, state: str) -> None:
    evaluate_rebuild.cleanup_monitoring_holdout_backtest_artifacts(paths=paths, state=state)


def rebuild_multi_loss_weight_trajectory_overviews(*args, **kwargs):
    return evaluate_rebuild.rebuild_multi_loss_weight_trajectory_overviews(*args, **kwargs)


def cleanup_multi_loss_weight_trajectory_overviews(*, paths: PathsConfig, state: str) -> None:
    evaluate_rebuild.cleanup_multi_loss_weight_trajectory_overviews(paths=paths, state=state)


def backfill_monitoring_holdout_backtest_overviews(*args, **kwargs):
    return evaluate_rebuild.backfill_monitoring_holdout_backtest_overviews(*args, **kwargs)
