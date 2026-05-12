"""Pure evaluation result construction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import presentation as evaluation_presentation
from . import shared as evaluation_shared
from ..config import EvaluationConfig
from ..data.dataset import PortfolioPanelDataset
from .metrics import apply_transaction_cost_to_returns, compute_average_turnover_from_weights
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
from ..model.losses import sharpe_loss

EXPORTED_TRAIN_CONFIG_KEYS = [
    "num_epochs",
    "weight_decay",
    "grad_clip_norm",
    "early_stopping_patience",
]
SCENARIO_FILENAME_PATTERN = evaluation_shared.SCENARIO_FILENAME_PATTERN


def extract_exported_train_config(checkpoint: dict[str, Any]) -> dict[str, object]:
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


def compute_backtest_portfolio_sr(portfolio_returns: torch.Tensor) -> float:
    return float((-sharpe_loss(portfolio_returns.detach().cpu()).item()))


def compute_average_turnover(stock_weights: torch.Tensor, cash_weights: torch.Tensor) -> float:
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
        import pandas as pd

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


def compute_benchmark_comparison_metrics(
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


def build_scenario_eval_result(
    *,
    scenario_id: str,
    source_path: Path,
    loss_name: str,
    checkpoint: dict[str, Any],
    context_target_time_indices: torch.Tensor,
    target_time_indices: torch.Tensor,
    portfolio_returns: torch.Tensor,
    turnover: torch.Tensor,
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
) -> ScenarioEvalResult:
    transaction_cost_rate = float(evaluation_config.evaluation_transaction_cost_rate)
    path_returns_cpu = apply_transaction_cost_to_returns(
        portfolio_returns.detach().cpu(),
        turnover.detach().cpu(),
        transaction_cost_rate=transaction_cost_rate,
    )
    context_target_time_indices_cpu = context_target_time_indices.detach().cpu()
    stock_weights_cpu = stock_weights.detach().cpu()
    cash_weights_cpu = cash_weights.detach().cpu()
    target_time_indices_cpu = target_time_indices.detach().cpu()

    final_return = float(torch.prod(1.0 + path_returns_cpu).item() - 1.0)
    backtest_portfolio_sr = compute_backtest_portfolio_sr(path_returns_cpu)
    final_cash_weight = float(cash_weights_cpu[-1].item())
    mean_cash_weight = float(cash_weights_cpu.mean().item())
    mean_step_return = float(path_returns_cpu.mean().item())
    std_step_return = float(path_returns_cpu.std(unbiased=False).item())
    average_turnover = compute_average_turnover(stock_weights_cpu, cash_weights_cpu)

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
    benchmark_metrics = compute_benchmark_comparison_metrics(
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

    return ScenarioEvalResult(
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
            benchmark_excess_return=evaluation_shared.coerce_optional_float(
                benchmark_metrics.get("benchmark_excess_return")
            ),
            benchmark_information_ratio=evaluation_shared.coerce_optional_float(
                benchmark_metrics.get("benchmark_information_ratio")
            ),
        ),
        train_config=extract_exported_train_config(checkpoint),
        top_k_stock_weights=enriched_top_positions,
        extra_payload={"evaluation_transaction_cost_rate": transaction_cost_rate},
        runtime_tensors=ScenarioRuntimeTensors(
            final_stock_weights=final_stock_weights,
            stock_weights=stock_weights_cpu,
            cash_weights=cash_weights_cpu,
            portfolio_returns=path_returns_cpu,
            target_time_indices=target_time_indices_cpu,
        ),
    )


def build_legacy_per_scenario_payload(**kwargs: Any) -> dict[str, Any]:
    return RuntimePayloadAdapter.to_legacy_payload(build_scenario_eval_result(**kwargs))

