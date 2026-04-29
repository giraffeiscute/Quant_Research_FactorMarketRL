"""Evaluation artifact payload/persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from . import shared as evaluation_shared
from ..config import EvaluationConfig
from ..data.dataset import PortfolioPanelDataset
from .metrics import compute_average_turnover_from_weights
from .types import (
    BenchmarkMetrics,
    RuntimePayloadAdapter,
    ScenarioCashStats,
    ScenarioEvalResult,
    ScenarioIdentity,
    ScenarioReturnStats,
    ScenarioRuntimeTensors,
    ScenarioSelectionStats,
    ScenarioWindowMeta,
)
from ..model.losses import build_loss, sharpe_loss
from ..common.utils import save_json
from . import presentation as evaluation_presentation

EXPORTED_TRAIN_CONFIG_KEYS = [
    "num_epochs",
    "weight_decay",
    "grad_clip_norm",
    "early_stopping_patience",
]
SCENARIO_FILENAME_PATTERN = evaluation_shared.SCENARIO_FILENAME_PATTERN
BENCHMARK_COMPARISON_FIELD_KEYS = (
    "benchmark_market_index_csv",
    "benchmark_excess_return",
    "benchmark_information_ratio",
)


def _cleanup_stale_prediction_artifacts(output_dir: Path, loss_name: str) -> None:
    patterns = [
        f"*_{loss_name}_holdout_predictions.json",
        f"{loss_name}_*_prediction.json",
        f"{loss_name}_*_all_stock_weights.csv",
        f"{loss_name}_*_day_weights.pt",
        f"{loss_name}_*_weight_trajectory.png",
        f"{loss_name}_best_backtest_scenario_*.json",
        f"{loss_name}_best_backtest_scenario_*.png",
        f"{loss_name}_best_backtest_scenario_*.csv",
    ]
    evaluation_shared.unlink_artifacts_by_patterns(output_dir, patterns)


def _strip_transient_scenario_tensor_fields(per_scenario_payloads: list[dict[str, Any]]) -> None:
    evaluation_shared.strip_transient_scenario_tensor_fields(per_scenario_payloads)


def _extract_exported_train_config(checkpoint: dict[str, Any]) -> dict[str, object]:
    checkpoint_train_config = checkpoint.get("train_config", {})
    checkpoint_data_config = checkpoint.get("data_config", {})
    exported = {
        key: checkpoint_train_config[key]
        for key in EXPORTED_TRAIN_CONFIG_KEYS
        if key in checkpoint_train_config
    }
    if "train_batch_size" in checkpoint_data_config:
        exported["train_batch_size"] = checkpoint_data_config["train_batch_size"]
    elif "scenario_batch_size" in checkpoint_data_config:
        exported["train_batch_size"] = checkpoint_data_config["scenario_batch_size"]
    return exported


def _compute_backtest_portfolio_sr(portfolio_returns: torch.Tensor) -> float:
    return float((-sharpe_loss(portfolio_returns.detach().cpu()).item()))


def _compute_average_turnover(stock_weights: torch.Tensor, cash_weights: torch.Tensor) -> float:
    return compute_average_turnover_from_weights(stock_weights, cash_weights)


def _resolve_benchmark_market_index_path(source_path: Path) -> Path:
    match = SCENARIO_FILENAME_PATTERN.fullmatch(source_path.name)
    if match is None:
        raise ValueError(
            "Could not resolve market index path from source_path name. "
            f"Expected pattern '<state>_<num_stocks>_<num_time_steps>_PL_<id>.parquet', "
            f"received {source_path.name!r}."
        )
    state = match.group("state")
    num_stocks = match.group("num_stocks")
    num_time_steps = match.group("num_time_steps")
    scenario = match.group("scenario")
    return source_path.parent / f"{state}_{num_stocks}_{num_time_steps}_market_index_{scenario}.csv"


def _load_aligned_benchmark_returns(
    *,
    source_path: Path,
    target_time_indices: torch.Tensor | list[object] | tuple[object, ...] | np.ndarray,
) -> tuple[str | None, np.ndarray | None]:
    try:
        market_index_path = _resolve_benchmark_market_index_path(source_path)
    except ValueError:
        return None, None
    if not market_index_path.exists():
        return None, None

    try:
        market_frame = pd.read_csv(market_index_path, usecols=["t", "market_index"])
    except ValueError:
        return str(market_index_path), None

    if "t" not in market_frame.columns or "market_index" not in market_frame.columns:
        return str(market_index_path), None

    resolved_target_time_indices = evaluation_presentation.coerce_plot_series(
        target_time_indices,
        field_name="target_time_indices",
    ).astype(np.int64, copy=False)
    if resolved_target_time_indices.size == 0:
        return str(market_index_path), np.asarray([], dtype=np.float64)

    try:
        market_frame["t"] = pd.to_numeric(market_frame["t"], errors="raise").astype("int64")
        market_frame["market_index"] = pd.to_numeric(
            market_frame["market_index"],
            errors="raise",
        ).astype("float64")
    except (TypeError, ValueError):
        return str(market_index_path), None

    market_index_by_time = market_frame.set_index("t")["market_index"]
    returns: list[float] = []
    for time_index in resolved_target_time_indices.tolist():
        current_value = market_index_by_time.get(int(time_index))
        previous_value = market_index_by_time.get(int(time_index) - 1)
        if current_value is None or previous_value is None:
            return str(market_index_path), None
        previous_value = float(previous_value)
        current_value = float(current_value)
        if not np.isfinite(previous_value) or not np.isfinite(current_value) or np.isclose(previous_value, 0.0):
            return str(market_index_path), None
        returns.append(current_value / previous_value - 1.0)
    return str(market_index_path), np.asarray(returns, dtype=np.float64)


def _compute_information_ratio(
    active_returns: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float | None:
    resolved_active_returns = np.asarray(active_returns, dtype=np.float64)
    if resolved_active_returns.ndim != 1 or resolved_active_returns.size < 2:
        return None
    if not np.isfinite(resolved_active_returns).all():
        return None
    active_std = float(resolved_active_returns.std(ddof=1))
    if not np.isfinite(active_std) or active_std <= eps:
        return None
    active_mean = float(resolved_active_returns.mean())
    if not np.isfinite(active_mean):
        return None
    return active_mean / active_std


def _compute_benchmark_comparison_metrics(
    *,
    source_path: Path,
    portfolio_returns: torch.Tensor | list[object] | tuple[object, ...] | np.ndarray,
    target_time_indices: torch.Tensor | list[object] | tuple[object, ...] | np.ndarray,
) -> dict[str, str | float | None]:
    portfolio_return_array = evaluation_presentation.coerce_plot_series(
        portfolio_returns,
        field_name="portfolio_returns",
    ).astype(np.float64, copy=False)
    benchmark_market_index_csv, benchmark_returns = _load_aligned_benchmark_returns(
        source_path=source_path,
        target_time_indices=target_time_indices,
    )
    if benchmark_returns is None or benchmark_returns.shape != portfolio_return_array.shape:
        return {
            "benchmark_market_index_csv": benchmark_market_index_csv,
            "benchmark_excess_return": None,
            "benchmark_information_ratio": None,
        }

    benchmark_final_return = float(np.prod(1.0 + benchmark_returns, dtype=np.float64) - 1.0)
    portfolio_final_return = float(np.prod(1.0 + portfolio_return_array, dtype=np.float64) - 1.0)
    active_returns = portfolio_return_array - benchmark_returns
    return {
        "benchmark_market_index_csv": benchmark_market_index_csv,
        "benchmark_excess_return": portfolio_final_return - benchmark_final_return,
        "benchmark_information_ratio": _compute_information_ratio(active_returns),
    }


def _update_benchmark_comparison_metrics(
    payload: dict[str, Any],
    *,
    benchmark_metrics: dict[str, str | float | None],
) -> bool:
    changed = False
    for field_name in BENCHMARK_COMPARISON_FIELD_KEYS:
        resolved_value = benchmark_metrics.get(field_name)
        if payload.get(field_name) == resolved_value:
            continue
        payload[field_name] = resolved_value
        changed = True
    return changed


def _build_day_weight_artifact_payload(
    *,
    scenario_payload: dict[str, Any],
    dataset: PortfolioPanelDataset,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
) -> dict[str, object]:
    return {
        "artifact_type": "holdout_scenario_day_weights",
        "scenario_id": str(scenario_payload["scenario_id"]),
        "state": str(dataset.state),
        "loss_name": str(scenario_payload["loss_name"]),
        "evaluation_mode": scenario_payload.get("evaluation_mode"),
        "source_path": str(scenario_payload["source_path"]),
        "checkpoint_path": str(checkpoint_path),
        "train_config": _extract_exported_train_config(checkpoint),
        "rolling_window_lookback_days": scenario_payload.get("rolling_window_lookback_days"),
        "rolling_window_horizon_days": scenario_payload.get("rolling_window_horizon_days"),
        "rolling_window_stride_days": scenario_payload.get("rolling_window_stride_days"),
        "num_rolling_windows": scenario_payload.get("num_rolling_windows"),
        "stock_ids": list(dataset.selected_stock_ids),
        "target_time_indices": scenario_payload["_target_time_indices_tensor"].detach().cpu().clone(),
        "stock_weights": scenario_payload["_stock_weights_tensor"].detach().cpu().clone(),
        "cash_weights": scenario_payload["_cash_weights_tensor"].detach().cpu().clone(),
        "portfolio_returns": scenario_payload["_portfolio_returns_tensor"].detach().cpu().clone(),
    }


def _load_day_weight_artifact_payload(path: Path) -> dict[str, Any]:
    return evaluation_shared.PersistedArtifactLoader.load_day_weight_artifact(path)


def _build_weight_trajectory_export_data(
    *,
    grouped_weight_trajectories: list[dict[str, object]],
    target_time_indices: torch.Tensor,
) -> dict[str, object]:
    return evaluation_shared.build_weight_trajectory_export_data(
        grouped_weight_trajectories=grouped_weight_trajectories,
        target_time_indices=target_time_indices,
    )


def _load_weight_trajectory_export_data(
    payload: dict[str, object],
) -> tuple[list[dict[str, object]], torch.Tensor]:
    return evaluation_shared.load_weight_trajectory_export_data(payload)


def _build_monitoring_scenario_artifact(
    *,
    scenario_index: int,
    scenario_payload: dict[str, Any],
    grouped_weight_trajectories: list[dict[str, object]],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    scenario_result = RuntimePayloadAdapter.from_legacy_payload(
        scenario_payload,
        require_runtime_tensors=True,
    )
    runtime_tensors = scenario_result.runtime_tensors
    if runtime_tensors is None or runtime_tensors.target_time_indices is None:
        raise RuntimeError("Monitoring scenario payload is missing runtime target_time_indices tensor.")
    scenario_id = scenario_result.identity.scenario_id
    loss_name = scenario_result.identity.loss_name
    metrics_text = evaluation_presentation.build_chart_metrics_text(
        loss_name=loss_name,
        portfolio_return=float(scenario_result.return_stats.final_return),
        portfolio_sr=float(scenario_result.return_stats.backtest_portfolio_sr),
        benchmark_excess_return=scenario_result.benchmark_metrics.benchmark_excess_return,
        benchmark_information_ratio=scenario_result.benchmark_metrics.benchmark_information_ratio,
        average_turnover=float(scenario_result.return_stats.average_turnover),
        selected_stock_count=int(scenario_result.selection_stats.total_selected_stock_count),
        stock_count_weight_threshold=float(scenario_result.selection_stats.stock_count_weight_threshold),
    )
    weight_trajectory_chart = None
    if output_dir is not None:
        weight_trajectory_path = evaluation_presentation.monitoring_weight_trajectory_chart_path(
            output_dir,
            loss_name=loss_name,
            scenario_id=scenario_id,
        )
        evaluation_presentation.render_weight_trajectory_chart(
            scenario_id=scenario_id,
            grouped_weight_trajectories=grouped_weight_trajectories,
            target_time_indices=runtime_tensors.target_time_indices,
            output_path=weight_trajectory_path,
            metrics_text=metrics_text,
        )
        weight_trajectory_chart = str(weight_trajectory_path)
    return {
        "scenario_index": int(scenario_index),
        "scenario_id": scenario_id,
        "evaluation_mode": scenario_result.window_meta.evaluation_mode,
        "final_return": float(scenario_result.return_stats.final_return),
        "backtest_portfolio_sr": float(scenario_result.return_stats.backtest_portfolio_sr),
        "average_turnover": float(scenario_result.return_stats.average_turnover),
        "benchmark_market_index_csv": scenario_result.benchmark_metrics.benchmark_market_index_csv,
        "benchmark_excess_return": scenario_result.benchmark_metrics.benchmark_excess_return,
        "benchmark_information_ratio": scenario_result.benchmark_metrics.benchmark_information_ratio,
        "rolling_window_lookback_days": scenario_result.window_meta.rolling_window_lookback_days,
        "rolling_window_horizon_days": scenario_result.window_meta.rolling_window_horizon_days,
        "rolling_window_stride_days": scenario_result.window_meta.rolling_window_stride_days,
        "num_rolling_windows": scenario_result.window_meta.num_rolling_windows,
        "total_selected_stock_count": int(scenario_result.selection_stats.total_selected_stock_count),
        "stock_count_weight_threshold": float(scenario_result.selection_stats.stock_count_weight_threshold),
        "metrics_text": metrics_text,
        "weight_trajectory_data": _build_weight_trajectory_export_data(
            grouped_weight_trajectories=grouped_weight_trajectories,
            target_time_indices=runtime_tensors.target_time_indices,
        ),
        "weight_trajectory_chart": weight_trajectory_chart,
        "weight_trajectory_overview_chart": None,
    }


def _compute_monitoring_holdout_backtest_loss(
    per_scenario_payloads: list[dict[str, Any]],
    *,
    loss_name: str,
) -> float:
    if not per_scenario_payloads:
        raise RuntimeError("Monitoring holdout loss requires at least one per-scenario payload.")

    stacked_returns: list[torch.Tensor] = []
    expected_shape: tuple[int, ...] | None = None
    for payload in per_scenario_payloads:
        scenario_result = RuntimePayloadAdapter.from_legacy_payload(
            payload,
            require_runtime_tensors=True,
        )
        runtime_tensors = scenario_result.runtime_tensors
        portfolio_returns = (
            runtime_tensors.portfolio_returns if runtime_tensors is not None else None
        )
        if not isinstance(portfolio_returns, torch.Tensor):
            raise RuntimeError(
                "Monitoring holdout payload is missing _portfolio_returns_tensor required for holdout loss."
            )
        returns_cpu = portfolio_returns.detach().cpu()
        current_shape = tuple(int(dim) for dim in returns_cpu.shape)
        if expected_shape is None:
            expected_shape = current_shape
        elif current_shape != expected_shape:
            raise RuntimeError(
                "Monitoring holdout scenarios must share the same scored return shape to compute "
                f"aggregate holdout loss. Expected {expected_shape}, received {current_shape}."
            )
        stacked_returns.append(returns_cpu)

    holdout_loss = build_loss(loss_name, torch.stack(stacked_returns, dim=0))
    return float(holdout_loss.detach().cpu().item())


def _strip_monitoring_transient_tensor_fields(per_scenario_payloads: list[dict[str, Any]]) -> None:
    _strip_transient_scenario_tensor_fields(per_scenario_payloads)


def _build_holdout_summary_payload(
    per_scenario_payloads: list[dict[str, Any]],
    *,
    dataset: PortfolioPanelDataset,
    loss_name: str,
    evaluation_split: str,
) -> dict[str, Any]:
    if not per_scenario_payloads:
        raise RuntimeError("Holdout evaluation produced no per-scenario payloads.")

    scenario_results = [
        RuntimePayloadAdapter.from_legacy_payload(payload) for payload in per_scenario_payloads
    ]
    final_returns = np.asarray(
        [float(item.return_stats.final_return) for item in scenario_results],
        dtype=np.float64,
    )
    average_turnovers = np.asarray(
        [float(item.return_stats.average_turnover) for item in scenario_results],
        dtype=np.float64,
    )
    best_index = int(final_returns.argmax())
    worst_index = int(final_returns.argmin())
    best_payload = scenario_results[best_index]
    worst_payload = scenario_results[worst_index]
    rolling_metadata_fields = (
        "evaluation_mode",
        "rolling_window_lookback_days",
        "rolling_window_horizon_days",
        "rolling_window_stride_days",
        "num_rolling_windows",
        "evaluation_price_anchor_mode",
    )
    rolling_metadata: dict[str, Any] = {}
    for field_name in rolling_metadata_fields:
        values = {
            getattr(payload.window_meta, field_name)
            for payload in scenario_results
            if getattr(payload.window_meta, field_name) is not None
        }
        if len(values) == 1:
            rolling_metadata[field_name] = next(iter(values))
    return {
        "state": dataset.state,
        "loss_name": loss_name,
        "evaluation_split": evaluation_split,
        "num_holdout_scenarios": len(scenario_results),
        "mean_final_return": float(final_returns.mean()),
        "mean_average_turnover": float(average_turnovers.mean()),
        "std_final_return": float(final_returns.std(ddof=0)),
        "median_final_return": float(np.median(final_returns)),
        "worst_scenario_final_return": float(final_returns.min()),
        "best_scenario_final_return": float(final_returns.max()),
        "best_scenario_id": best_payload.identity.scenario_id,
        "worst_scenario_id": worst_payload.identity.scenario_id,
        "best_scenario_source_path": best_payload.identity.source_path,
        "worst_scenario_source_path": worst_payload.identity.source_path,
        **rolling_metadata,
    }


def _populate_prediction_benchmark_metrics_from_day_weight_artifact(payload: dict[str, Any]) -> bool:
    raw_day_weight_artifact = payload.get("day_weight_artifact")
    if raw_day_weight_artifact in {None, ""}:
        return False
    day_weight_artifact_path = Path(str(raw_day_weight_artifact))
    if not day_weight_artifact_path.exists():
        return False

    try:
        day_weight_payload = _load_day_weight_artifact_payload(day_weight_artifact_path)
    except (FileNotFoundError, ValueError, RuntimeError):
        return False

    raw_source_path = payload.get("source_path") or day_weight_payload.get("source_path")
    if raw_source_path in {None, ""}:
        return False
    portfolio_returns = day_weight_payload.get("portfolio_returns")
    target_time_indices = day_weight_payload.get("target_time_indices")
    if portfolio_returns is None or target_time_indices is None:
        return False

    benchmark_metrics = _compute_benchmark_comparison_metrics(
        source_path=Path(str(raw_source_path)),
        portfolio_returns=portfolio_returns,
        target_time_indices=target_time_indices,
    )
    return _update_benchmark_comparison_metrics(
        payload,
        benchmark_metrics=benchmark_metrics,
    )


def _build_per_scenario_payload(
    *,
    scenario_id: str,
    source_path: Path,
    loss_name: str,
    checkpoint: dict[str, Any],
    context_target_time_indices: torch.Tensor,
    target_time_indices: torch.Tensor,
    portfolio_returns: torch.Tensor,
    stock_weights: torch.Tensor,
    cash_weights: torch.Tensor,
    dataset: PortfolioPanelDataset,
    evaluation_config: EvaluationConfig,
    warmup_time_steps: int | None = None,
    evaluation_mode: str | None = None,
    rolling_window_lookback_days: int | None = None,
    rolling_window_horizon_days: int | None = None,
    rolling_window_stride_days: int | None = None,
    num_rolling_windows: int | None = None,
    evaluation_price_anchor_mode: str | None = None,
) -> dict[str, Any]:
    path_returns_cpu = portfolio_returns.detach().cpu()
    context_target_time_indices_cpu = context_target_time_indices.detach().cpu()
    stock_weights_cpu = stock_weights.detach().cpu()
    cash_weights_cpu = cash_weights.detach().cpu()
    target_time_indices_cpu = target_time_indices.detach().cpu()

    final_return = float(torch.prod(1.0 + path_returns_cpu).item() - 1.0)
    backtest_portfolio_sr = _compute_backtest_portfolio_sr(path_returns_cpu)
    final_cash_weight = float(cash_weights_cpu[-1].item())
    mean_cash_weight = float(cash_weights_cpu.mean().item())
    mean_step_return = float(path_returns_cpu.mean().item())
    std_step_return = float(path_returns_cpu.std(unbiased=False).item())
    average_turnover = _compute_average_turnover(stock_weights_cpu, cash_weights_cpu)

    top_k = min(5, dataset.num_stocks)
    final_stock_weights = stock_weights_cpu[-1]
    stock_count_weight_threshold = float(evaluation_config.stock_count_weight_threshold)
    stock_count_min_active_days = int(evaluation_config.stock_count_min_active_days)
    selected_stock_mask, stock_count_lookback_days, effective_stock_count_min_active_days = (
        evaluation_presentation.build_selected_stock_mask(
            stock_weights_cpu,
            threshold=stock_count_weight_threshold,
            min_active_days=stock_count_min_active_days,
        )
    )
    total_selected_stock_count = int(selected_stock_mask.sum())
    top_values, top_indices = torch.topk(final_stock_weights, k=top_k)
    top_positions = [
        {
            "stock_id": dataset.selected_stock_ids[int(index)],
            "weight": float(weight.item()),
        }
        for weight, index in zip(top_values, top_indices)
    ]

    analysis_time_index = int(target_time_indices_cpu[-1].item())
    aux_frame = evaluation_presentation.load_aux_frame(source_path)
    enriched_top_positions = evaluation_presentation.enrich_positions(
        aux_frame=aux_frame,
        analysis_time_index=analysis_time_index,
        positions=top_positions,
    )
    benchmark_metrics = _compute_benchmark_comparison_metrics(
        source_path=source_path,
        portfolio_returns=path_returns_cpu,
        target_time_indices=target_time_indices_cpu,
    )
    resolved_warmup_time_steps = (
        int(warmup_time_steps)
        if warmup_time_steps is not None
        else int(context_target_time_indices_cpu.shape[0] - path_returns_cpu.shape[0])
    )
    if resolved_warmup_time_steps < 0:
        raise ValueError(
            "warmup_time_steps must be non-negative after payload construction. "
            f"Received {resolved_warmup_time_steps}."
        )

    scenario_eval_result = ScenarioEvalResult(
        identity=ScenarioIdentity(
            scenario_id=scenario_id,
            source_path=str(source_path),
            loss_name=loss_name,
            state=dataset.state,
            evaluation_split="holdout_test",
        ),
        window_meta=ScenarioWindowMeta(
            num_time_steps=int(path_returns_cpu.shape[0]),
            scored_num_time_steps=int(path_returns_cpu.shape[0]),
            context_num_time_steps=int(context_target_time_indices_cpu.shape[0]),
            warmup_time_steps=resolved_warmup_time_steps,
            analysis_time_index=analysis_time_index,
            feature_time_start_index=int(target_time_indices_cpu[0].item()) - 1,
            feature_time_end_index=int(target_time_indices_cpu[-1].item()) - 1,
            target_time_start_index=int(target_time_indices_cpu[0].item()),
            target_time_end_index=int(target_time_indices_cpu[-1].item()),
            scored_feature_time_start_index=int(target_time_indices_cpu[0].item()) - 1,
            scored_feature_time_end_index=int(target_time_indices_cpu[-1].item()) - 1,
            scored_target_time_start_index=int(target_time_indices_cpu[0].item()),
            scored_target_time_end_index=int(target_time_indices_cpu[-1].item()),
            context_feature_time_start_index=int(context_target_time_indices_cpu[0].item()) - 1,
            context_feature_time_end_index=int(context_target_time_indices_cpu[-1].item()) - 1,
            context_target_time_start_index=int(context_target_time_indices_cpu[0].item()),
            context_target_time_end_index=int(context_target_time_indices_cpu[-1].item()),
            evaluation_mode=str(evaluation_mode) if evaluation_mode is not None else None,
            rolling_window_lookback_days=int(rolling_window_lookback_days)
            if rolling_window_lookback_days is not None
            else None,
            rolling_window_horizon_days=int(rolling_window_horizon_days)
            if rolling_window_horizon_days is not None
            else None,
            rolling_window_stride_days=int(rolling_window_stride_days)
            if rolling_window_stride_days is not None
            else None,
            num_rolling_windows=int(num_rolling_windows) if num_rolling_windows is not None else None,
            evaluation_price_anchor_mode=str(evaluation_price_anchor_mode)
            if evaluation_price_anchor_mode is not None
            else None,
        ),
        return_stats=ScenarioReturnStats(
            final_return=final_return,
            backtest_portfolio_sr=backtest_portfolio_sr,
            mean_step_return=mean_step_return,
            std_step_return=std_step_return,
            average_turnover=average_turnover,
            sharpe_like=backtest_portfolio_sr if loss_name == "sharpe" else None,
        ),
        cash_stats=ScenarioCashStats(
            final_cash_weight=final_cash_weight,
            mean_cash_weight=mean_cash_weight,
        ),
        selection_stats=ScenarioSelectionStats(
            allocation_group_top_n=int(evaluation_config.allocation_group_top_n),
            stock_count_weight_threshold=stock_count_weight_threshold,
            stock_count_min_active_days=stock_count_min_active_days,
            effective_stock_count_min_active_days=effective_stock_count_min_active_days,
            stock_count_lookback_days=stock_count_lookback_days,
            total_selected_stock_count=total_selected_stock_count,
        ),
        benchmark_metrics=BenchmarkMetrics(
            benchmark_market_index_csv=str(benchmark_metrics.get("benchmark_market_index_csv"))
            if benchmark_metrics.get("benchmark_market_index_csv") not in {None, ""}
            else None,
            benchmark_excess_return=evaluation_shared.coerce_optional_float(benchmark_metrics.get("benchmark_excess_return")),
            benchmark_information_ratio=evaluation_shared.coerce_optional_float(
                benchmark_metrics.get("benchmark_information_ratio")
            ),
        ),
        train_config=_extract_exported_train_config(checkpoint),
        top_k_stock_weights=enriched_top_positions,
        runtime_tensors=ScenarioRuntimeTensors(
            final_stock_weights=final_stock_weights,
            stock_weights=stock_weights_cpu,
            cash_weights=cash_weights_cpu,
            portfolio_returns=path_returns_cpu,
            target_time_indices=target_time_indices_cpu,
        ),
    )
    return RuntimePayloadAdapter.to_legacy_payload(scenario_eval_result)


def _export_scenario_payload(
    *,
    scenario_payload: dict[str, Any],
    checkpoint: dict[str, Any],
    dataset: PortfolioPanelDataset,
    output_dir: Path,
    evaluation_config: EvaluationConfig,
    loss_name: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
    scenario_result = RuntimePayloadAdapter.from_legacy_payload(
        scenario_payload,
        require_runtime_tensors=True,
    )
    runtime_tensors = scenario_result.runtime_tensors
    if runtime_tensors is None:
        raise RuntimeError("Expected runtime tensors to be present when exporting scenario payload.")
    if (
        runtime_tensors.final_stock_weights is None
        or runtime_tensors.stock_weights is None
        or runtime_tensors.cash_weights is None
        or runtime_tensors.target_time_indices is None
    ):
        raise RuntimeError("Scenario payload is missing runtime tensors required for export.")

    scenario_id = scenario_result.identity.scenario_id
    source_path = Path(scenario_result.identity.source_path)
    artifact_stem = f"{loss_name}_{scenario_id}"
    aux_frame = evaluation_presentation.load_aux_frame(source_path)
    stock_count_weight_threshold = float(
        scenario_payload.get(
            "stock_count_weight_threshold",
            scenario_result.selection_stats.stock_count_weight_threshold
            or evaluation_config.stock_count_weight_threshold,
        )
    )
    stock_count_min_active_days = int(
        scenario_payload.get(
            "stock_count_min_active_days",
            scenario_result.selection_stats.stock_count_min_active_days
            or evaluation_config.stock_count_min_active_days,
        )
    )
    stock_count_lookback_days = int(
        scenario_payload.get(
            "stock_count_lookback_days",
            scenario_result.selection_stats.stock_count_lookback_days
            or runtime_tensors.stock_weights.shape[0],
        )
    )
    effective_stock_count_min_active_days = int(
        scenario_payload.get(
            "effective_stock_count_min_active_days",
            min(stock_count_min_active_days, stock_count_lookback_days),
        )
    )
    total_selected_stock_count = int(
        scenario_payload.get(
            "total_selected_stock_count",
            evaluation_presentation.build_selected_stock_mask(
                runtime_tensors.stock_weights,
                threshold=stock_count_weight_threshold,
                min_active_days=stock_count_min_active_days,
            )[0].sum(),
        )
    )
    weight_trajectory_metrics_text = evaluation_presentation.build_chart_metrics_text(
        loss_name=loss_name,
        portfolio_return=float(scenario_result.return_stats.final_return),
        portfolio_sr=float(scenario_result.return_stats.backtest_portfolio_sr),
        average_turnover=float(scenario_result.return_stats.average_turnover),
        selected_stock_count=total_selected_stock_count,
        stock_count_weight_threshold=stock_count_weight_threshold,
    )
    allocation_payload = evaluation_presentation.export_allocation_artifacts(
        aux_frame=aux_frame,
        analysis_time_index=int(scenario_result.window_meta.analysis_time_index),
        stock_ids=dataset.selected_stock_ids,
        stock_weights=runtime_tensors.final_stock_weights,
        cash_weight=float(scenario_result.cash_stats.final_cash_weight),
        allocation_group_top_n=evaluation_config.allocation_group_top_n,
    )
    grouped_weight_trajectories = evaluation_presentation.build_grouped_weight_trajectories(
        aux_frame=aux_frame,
        analysis_time_index=int(scenario_result.window_meta.analysis_time_index),
        stock_ids=dataset.selected_stock_ids,
        stock_weights=runtime_tensors.stock_weights,
        cash_weights=runtime_tensors.cash_weights,
        grouped_allocations_top_n=list(allocation_payload["grouped_allocations_top_n"]),
        stock_count_weight_threshold=stock_count_weight_threshold,
        stock_count_min_active_days=stock_count_min_active_days,
    )
    weight_trajectory_path = output_dir / f"{artifact_stem}_weight_trajectory.png"
    day_weight_artifact_path = output_dir / f"{artifact_stem}_day_weights.pt"
    evaluation_presentation.render_weight_trajectory_chart(
        scenario_id=scenario_id,
        grouped_weight_trajectories=grouped_weight_trajectories,
        target_time_indices=runtime_tensors.target_time_indices,
        output_path=weight_trajectory_path,
        metrics_text=weight_trajectory_metrics_text,
    )
    runtime_legacy_payload = RuntimePayloadAdapter.to_legacy_payload(scenario_result)
    torch.save(
        _build_day_weight_artifact_payload(
            scenario_payload=runtime_legacy_payload,
            dataset=dataset,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
        ),
        day_weight_artifact_path,
    )

    prediction_json_path = output_dir / f"{artifact_stem}_prediction.json"
    exported_payload = RuntimePayloadAdapter.to_legacy_payload(
        scenario_result,
        include_runtime_tensors=False,
    )
    exported_payload.update(
        {
            "artifact_type": "holdout_scenario_prediction",
            "train_config": _extract_exported_train_config(checkpoint),
            "checkpoint_path": str(checkpoint_path),
            "all_stock_weights": allocation_payload["all_stock_weights"],
            "all_stock_weights_csv": allocation_payload["all_stock_weights_csv"],
            "allocation_groups": allocation_payload["grouped_allocations"],
            "grouped_allocations": allocation_payload["grouped_allocations"],
            "grouped_allocations_top_n": allocation_payload["grouped_allocations_top_n"],
            "allocation_groups_top_n_plus_others": allocation_payload[
                "allocation_groups_top_n_plus_others"
            ],
            "allocation_group_top_n": allocation_payload["allocation_group_top_n"],
            "day_weight_artifact": str(day_weight_artifact_path),
            "weight_trajectory_chart": str(weight_trajectory_path),
            "weight_trajectory_data": _build_weight_trajectory_export_data(
                grouped_weight_trajectories=grouped_weight_trajectories,
                target_time_indices=runtime_tensors.target_time_indices,
            ),
            "weight_trajectory_overview_chart": None,
        }
    )
    save_json(exported_payload, prediction_json_path)

    return {
        "scenario_id": scenario_id,
        "evaluation_mode": scenario_result.window_meta.evaluation_mode,
        "final_return": float(scenario_result.return_stats.final_return),
        "backtest_portfolio_sr": float(scenario_result.return_stats.backtest_portfolio_sr),
        "average_turnover": float(scenario_result.return_stats.average_turnover),
        "benchmark_market_index_csv": scenario_result.benchmark_metrics.benchmark_market_index_csv,
        "benchmark_excess_return": scenario_result.benchmark_metrics.benchmark_excess_return,
        "benchmark_information_ratio": scenario_result.benchmark_metrics.benchmark_information_ratio,
        "rolling_window_lookback_days": scenario_result.window_meta.rolling_window_lookback_days,
        "rolling_window_horizon_days": scenario_result.window_meta.rolling_window_horizon_days,
        "rolling_window_stride_days": scenario_result.window_meta.rolling_window_stride_days,
        "num_rolling_windows": scenario_result.window_meta.num_rolling_windows,
        "prediction_json_path": str(prediction_json_path),
        "all_stock_weights_csv": allocation_payload["all_stock_weights_csv"],
        "day_weight_artifact": str(day_weight_artifact_path),
        "stock_count_weight_threshold": stock_count_weight_threshold,
        "stock_count_min_active_days": stock_count_min_active_days,
        "effective_stock_count_min_active_days": effective_stock_count_min_active_days,
        "stock_count_lookback_days": stock_count_lookback_days,
        "total_selected_stock_count": total_selected_stock_count,
        "weight_trajectory_chart": str(weight_trajectory_path),
        "weight_trajectory_overview_chart": None,
    }


# Public helper APIs used by pipeline/monitoring/rebuild modules.
def cleanup_stale_prediction_artifacts(output_dir: Path, loss_name: str) -> None:
    _cleanup_stale_prediction_artifacts(output_dir, loss_name)


def strip_transient_scenario_tensor_fields(per_scenario_payloads: list[dict[str, Any]]) -> None:
    _strip_transient_scenario_tensor_fields(per_scenario_payloads)


def extract_exported_train_config(checkpoint: dict[str, Any]) -> dict[str, object]:
    return _extract_exported_train_config(checkpoint)


def build_per_scenario_payload(
    *,
    scenario_id: str,
    source_path: Path,
    loss_name: str,
    checkpoint: dict[str, Any],
    context_target_time_indices: torch.Tensor,
    target_time_indices: torch.Tensor,
    portfolio_returns: torch.Tensor,
    stock_weights: torch.Tensor,
    cash_weights: torch.Tensor,
    dataset: PortfolioPanelDataset,
    evaluation_config: EvaluationConfig,
    warmup_time_steps: int | None = None,
    evaluation_mode: str | None = None,
    rolling_window_lookback_days: int | None = None,
    rolling_window_horizon_days: int | None = None,
    rolling_window_stride_days: int | None = None,
    num_rolling_windows: int | None = None,
    evaluation_price_anchor_mode: str | None = None,
) -> dict[str, Any]:
    return _build_per_scenario_payload(
        scenario_id=scenario_id,
        source_path=source_path,
        loss_name=loss_name,
        checkpoint=checkpoint,
        context_target_time_indices=context_target_time_indices,
        target_time_indices=target_time_indices,
        portfolio_returns=portfolio_returns,
        stock_weights=stock_weights,
        cash_weights=cash_weights,
        dataset=dataset,
        evaluation_config=evaluation_config,
        warmup_time_steps=warmup_time_steps,
        evaluation_mode=evaluation_mode,
        rolling_window_lookback_days=rolling_window_lookback_days,
        rolling_window_horizon_days=rolling_window_horizon_days,
        rolling_window_stride_days=rolling_window_stride_days,
        num_rolling_windows=num_rolling_windows,
        evaluation_price_anchor_mode=evaluation_price_anchor_mode,
    )


def export_scenario_payload(
    *,
    scenario_payload: dict[str, Any],
    checkpoint: dict[str, Any],
    dataset: PortfolioPanelDataset,
    output_dir: Path,
    evaluation_config: EvaluationConfig,
    loss_name: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
    return _export_scenario_payload(
        scenario_payload=scenario_payload,
        checkpoint=checkpoint,
        dataset=dataset,
        output_dir=output_dir,
        evaluation_config=evaluation_config,
        loss_name=loss_name,
        checkpoint_path=checkpoint_path,
    )


def build_holdout_summary_payload(
    per_scenario_payloads: list[dict[str, Any]],
    *,
    dataset: PortfolioPanelDataset,
    loss_name: str,
    evaluation_split: str,
) -> dict[str, Any]:
    return _build_holdout_summary_payload(
        per_scenario_payloads,
        dataset=dataset,
        loss_name=loss_name,
        evaluation_split=evaluation_split,
    )


def compute_monitoring_holdout_backtest_loss(
    per_scenario_payloads: list[dict[str, Any]],
    *,
    loss_name: str,
) -> float:
    return _compute_monitoring_holdout_backtest_loss(per_scenario_payloads, loss_name=loss_name)


def compute_average_turnover(stock_weights: torch.Tensor, cash_weights: torch.Tensor) -> float:
    return _compute_average_turnover(stock_weights, cash_weights)


def strip_monitoring_transient_tensor_fields(per_scenario_payloads: list[dict[str, Any]]) -> None:
    _strip_monitoring_transient_tensor_fields(per_scenario_payloads)


def build_monitoring_scenario_artifact(
    *,
    scenario_index: int,
    scenario_payload: dict[str, Any],
    grouped_weight_trajectories: list[dict[str, object]],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    return _build_monitoring_scenario_artifact(
        scenario_index=scenario_index,
        scenario_payload=scenario_payload,
        grouped_weight_trajectories=grouped_weight_trajectories,
        output_dir=output_dir,
    )


def load_weight_trajectory_export_data(
    payload: dict[str, object],
) -> tuple[list[dict[str, object]], torch.Tensor]:
    return _load_weight_trajectory_export_data(payload)


def populate_prediction_benchmark_metrics_from_day_weight_artifact(payload: dict[str, Any]) -> bool:
    return _populate_prediction_benchmark_metrics_from_day_weight_artifact(payload)
