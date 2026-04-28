"""Evaluation presentation and rendering helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from . import shared as evaluation_shared
from ..config import EvaluationConfig
from ..data.dataset import PortfolioPanelDataset
from .types import RuntimePayloadAdapter

REQUIRED_AUX_COLUMNS = ["stock_id", "t", "mu", "alpha", "epsilon_variance"]
WEIGHT_TRAJECTORY_REFERENCE_DAY = evaluation_shared.WEIGHT_TRAJECTORY_REFERENCE_DAY
WEIGHT_TRAJECTORY_OVERVIEW_LOSS_ORDER = evaluation_shared.WEIGHT_TRAJECTORY_OVERVIEW_LOSS_ORDER


def load_aux_frame(source_path: Path) -> pd.DataFrame:
    header = pq.read_schema(source_path).names
    missing_columns = [column for column in REQUIRED_AUX_COLUMNS if column not in header]
    if missing_columns:
        raise ValueError(
            "Evaluation export requires source panel columns: "
            f"{REQUIRED_AUX_COLUMNS}. Missing: {missing_columns}"
        )

    aux_frame = pd.read_parquet(source_path, columns=REQUIRED_AUX_COLUMNS)
    aux_frame["analysis_time_index"] = aux_frame["t"].map(evaluation_shared.parse_source_time_to_index)
    return aux_frame


def _add_metrics_text_box(
    ax: plt.Axes,
    metrics_text: str,
    *,
    x: float,
    y: float,
    ha: str = "left",
    va: str = "top",
) -> None:
    ax.text(
        x,
        y,
        metrics_text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.92,
        },
    )


def build_selected_stock_mask(
    stock_weights: torch.Tensor,
    *,
    threshold: float,
    min_active_days: int,
) -> tuple[np.ndarray, int, int]:
    if stock_weights.ndim != 2:
        raise ValueError("stock_weights must have shape [T, N].")
    resolved_min_active_days = int(min_active_days)
    if resolved_min_active_days <= 0:
        raise ValueError(f"min_active_days must be positive, received {min_active_days}.")

    weight_array = stock_weights.detach().cpu().numpy()
    backtest_days = int(weight_array.shape[0])
    if backtest_days <= 0:
        raise ValueError("stock_weights must include at least one backtest day.")

    above_threshold = (weight_array > float(threshold)) & (
        ~np.isclose(weight_array, float(threshold), rtol=0.0, atol=1e-9)
    )
    effective_min_active_days = min(resolved_min_active_days, backtest_days)
    selected_mask = above_threshold.sum(axis=0) >= effective_min_active_days
    return selected_mask.astype(bool), backtest_days, effective_min_active_days


def coerce_plot_series(
    values: torch.Tensor | list[object] | tuple[object, ...] | np.ndarray,
    *,
    field_name: str,
) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{field_name} must have shape [T].")
    return array


def _plot_weight_trajectory_axes(
    ax: plt.Axes,
    *,
    grouped_weight_trajectories: list[dict[str, object]],
    target_time_indices: torch.Tensor | list[object] | tuple[object, ...] | np.ndarray,
    metrics_text: str,
    title: str | None,
    legend_fontsize: int = 9,
) -> None:
    x_axis = coerce_plot_series(target_time_indices, field_name="target_time_indices")
    for item in grouped_weight_trajectories:
        weights = item["weights"]
        if not isinstance(weights, (torch.Tensor, list, tuple, np.ndarray)):
            raise ValueError("Each trajectory entry must provide a 1D series in 'weights'.")
        y_axis = coerce_plot_series(weights, field_name="weights")
        if y_axis.shape != x_axis.shape:
            raise ValueError(
                "Each weight trajectory must match target_time_indices. "
                f"Received weights.shape={y_axis.shape} target_time_indices.shape={x_axis.shape}."
            )
        label = str(item["label"])
        linestyle = "--" if label == "Cash" else "-"
        ax.plot(
            x_axis,
            y_axis,
            label=label,
            linestyle=linestyle,
            linewidth=2 if label == "Cash" else 1.5,
        )
    ax.axvline(
        WEIGHT_TRAJECTORY_REFERENCE_DAY,
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
    )
    ax.set_xlabel("Target Time Index")
    ax.set_ylabel("Weight")
    if title is not None:
        ax.set_title(title)
    ax.legend(loc="upper right", fontsize=legend_fontsize)
    _add_metrics_text_box(ax, metrics_text, x=0.02, y=0.98, ha="left", va="top")


def get_aux_lookup(aux_frame: pd.DataFrame) -> dict[tuple[str, int], dict[str, object]]:
    cached_lookup = aux_frame.attrs.get("_position_lookup")
    if cached_lookup is not None:
        return cached_lookup

    duplicated = aux_frame.duplicated(["stock_id", "analysis_time_index"], keep=False)
    if duplicated.any():
        duplicate_rows = (
            aux_frame.loc[duplicated, ["stock_id", "analysis_time_index"]]
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            "Evaluation export found multiple source rows for the same "
            "(stock_id, analysis_time_index) keys. "
        f"Examples: {duplicate_rows}"
        )

    lookup: dict[tuple[str, int], dict[str, object]] = {}
    for row in aux_frame.itertuples(index=False):
        lookup[(str(row.stock_id), int(row.analysis_time_index))] = {
            "mu": row.mu,
            "alpha": row.alpha,
            "epsilon_variance": row.epsilon_variance,
        }
    aux_frame.attrs["_position_lookup"] = lookup
    return lookup


def enrich_positions(
    *,
    aux_frame: pd.DataFrame,
    analysis_time_index: int,
    positions: list[dict[str, object]],
) -> list[dict[str, object]]:
    aux_lookup = get_aux_lookup(aux_frame)
    enriched: list[dict[str, object]] = []
    for rank, position in enumerate(positions, start=1):
        stock_id = str(position["stock_id"])
        match = aux_lookup.get((stock_id, analysis_time_index))
        if match is None:
            raise ValueError(
                f"Evaluation export could not find exactly one source row for stock_id={stock_id} "
                f"at analysis_time_index={analysis_time_index}."
            )
        enriched.append(
            {
                "rank": rank,
                "stock_id": stock_id,
                "weight": float(position["weight"]),
                "mu": match["mu"],
                "alpha": match["alpha"],
                "epsilon_variance": match["epsilon_variance"],
            }
        )
    return enriched


def enrich_top_k_positions(
    *,
    source_path: Path,
    metadata: dict[str, Any],
    top_positions: list[dict[str, object]],
) -> list[dict[str, object]]:
    analysis_time_index = int(metadata["analysis_time_index"])
    return enrich_positions(
        aux_frame=load_aux_frame(source_path),
        analysis_time_index=analysis_time_index,
        positions=top_positions,
    )


def build_all_stock_positions(
    *,
    stock_ids: list[str],
    stock_weights: torch.Tensor,
) -> list[dict[str, object]]:
    positions = [
        {
            "stock_id": stock_id,
            "weight": float(weight),
        }
        for stock_id, weight in zip(stock_ids, stock_weights.tolist())
        if float(weight) > 0.0
    ]
    return sorted(positions, key=lambda item: item["weight"], reverse=True)


def group_allocations_by_state(all_stock_positions: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], dict[str, object]] = {}
    for position in all_stock_positions:
        key = (
            str(position["mu"]),
            str(position["epsilon_variance"]),
            str(position["alpha"]),
        )
        if key not in grouped:
            grouped[key] = {
                "mu": key[0],
                "epsilon_variance": key[1],
                "alpha": key[2],
                "total_weight": 0.0,
                "stock_count": 0,
            }
        grouped[key]["total_weight"] = float(grouped[key]["total_weight"]) + float(position["weight"])
        grouped[key]["stock_count"] = int(grouped[key]["stock_count"]) + 1
    return sorted(grouped.values(), key=lambda item: float(item["total_weight"]), reverse=True)


def append_cash_allocation(
    grouped_allocations: list[dict[str, object]],
    cash_weight: float,
) -> list[dict[str, object]]:
    if cash_weight < 0.0:
        raise ValueError(f"cash_weight must be non-negative, received {cash_weight}.")
    if cash_weight == 0.0:
        return list(grouped_allocations)
    return [
        *grouped_allocations,
        {
            "mu": "Cash",
            "epsilon_variance": "Cash",
            "alpha": "Cash",
            "total_weight": float(cash_weight),
            "stock_count": 0,
        },
    ]


def summarize_grouped_allocations(
    grouped_allocations: list[dict[str, object]],
    top_n: int = 10,
) -> list[dict[str, object]]:
    if top_n <= 0:
        raise ValueError("top_n must be positive.")
    cash_allocations = [item for item in grouped_allocations if str(item["mu"]) == "Cash"]
    non_cash_allocations = [item for item in grouped_allocations if str(item["mu"]) != "Cash"]
    if len(non_cash_allocations) <= top_n:
        return non_cash_allocations + cash_allocations

    head = non_cash_allocations[:top_n]
    tail = non_cash_allocations[top_n:]
    others = {
        "mu": "Others",
        "epsilon_variance": "Others",
        "alpha": "Others",
        "total_weight": float(sum(float(item["total_weight"]) for item in tail)),
        "stock_count": int(sum(int(item["stock_count"]) for item in tail)),
    }
    return head + [others] + cash_allocations


def _allocation_group_key(mu: object, epsilon_variance: object, alpha: object) -> tuple[str, str, str]:
    return (str(mu), str(epsilon_variance), str(alpha))


def format_allocation_group_label(grouped_allocation: dict[str, object]) -> str:
    mu = str(grouped_allocation["mu"])
    epsilon_variance = str(grouped_allocation["epsilon_variance"])
    alpha = str(grouped_allocation["alpha"])
    if mu in {"Cash", "Others"} and mu == epsilon_variance == alpha:
        return mu
    return f"mu={mu} | eps={epsilon_variance} | alpha={alpha}"


def _format_grouped_weight_trajectory_label(
    grouped_allocation: dict[str, object],
    *,
    selected_stock_count: int,
) -> str:
    base_label = format_allocation_group_label(grouped_allocation)
    if str(grouped_allocation["mu"]) == "Cash":
        return base_label
    return f"{base_label} | stocks={int(selected_stock_count)}"


def build_grouped_weight_trajectories(
    *,
    aux_frame: pd.DataFrame,
    analysis_time_index: int,
    stock_ids: list[str],
    stock_weights: torch.Tensor,
    cash_weights: torch.Tensor,
    grouped_allocations_top_n: list[dict[str, object]],
    stock_count_weight_threshold: float,
    stock_count_min_active_days: int,
) -> list[dict[str, object]]:
    if stock_weights.ndim != 2:
        raise ValueError("stock_weights must have shape [T, N].")
    if cash_weights.ndim != 1:
        raise ValueError("cash_weights must have shape [T].")
    if stock_weights.shape[0] != cash_weights.shape[0]:
        raise ValueError(
            "stock_weights and cash_weights must share the same time dimension. "
            f"Received stock_weights.shape={tuple(stock_weights.shape)} and "
            f"cash_weights.shape={tuple(cash_weights.shape)}."
        )
    if stock_weights.shape[1] != len(stock_ids):
        raise ValueError(
            "stock_weights.shape[1] must match len(stock_ids). "
            f"Received {stock_weights.shape[1]} stocks and {len(stock_ids)} ids."
        )

    aux_lookup = get_aux_lookup(aux_frame)
    zero_series = torch.zeros(stock_weights.shape[0], dtype=stock_weights.dtype, device=stock_weights.device)
    selected_stock_mask, _, _ = build_selected_stock_mask(
        stock_weights,
        threshold=stock_count_weight_threshold,
        min_active_days=stock_count_min_active_days,
    )
    grouped_series: dict[tuple[str, str, str], torch.Tensor] = {}
    grouped_selected_stock_counts: dict[tuple[str, str, str], int] = {}
    for index, stock_id in enumerate(stock_ids):
        match = aux_lookup.get((str(stock_id), analysis_time_index))
        if match is None:
            raise ValueError(
                f"Evaluation export could not find exactly one source row for stock_id={stock_id} "
                f"at analysis_time_index={analysis_time_index}."
            )
        key = _allocation_group_key(
            match["mu"],
            match["epsilon_variance"],
            match["alpha"],
        )
        if key not in grouped_series:
            grouped_series[key] = zero_series.clone()
        grouped_series[key] = grouped_series[key] + stock_weights[:, index]
        if bool(selected_stock_mask[index]):
            grouped_selected_stock_counts[key] = grouped_selected_stock_counts.get(key, 0) + 1

    trajectories: list[dict[str, object]] = []
    for item in grouped_allocations_top_n:
        mu = str(item["mu"])
        if mu == "Cash":
            weights = cash_weights
            label = format_allocation_group_label(item)
        elif mu == "Others":
            continue
        else:
            key = _allocation_group_key(item["mu"], item["epsilon_variance"], item["alpha"])
            weights = grouped_series.get(key)
            if weights is None:
                raise ValueError(
                    "Grouped allocation summary referenced a group that is missing from the "
                    f"trajectory aggregation: {format_allocation_group_label(item)}."
                )
            label = _format_grouped_weight_trajectory_label(
                item,
                selected_stock_count=grouped_selected_stock_counts.get(key, 0),
            )
        trajectories.append(
            {
                "label": label,
                "weights": weights,
            }
        )
    return trajectories


def save_all_stock_weights_csv(
    all_stock_positions: list[dict[str, object]],
    output_path: Path,
) -> None:
    frame = pd.DataFrame(all_stock_positions)
    frame = frame.reindex(columns=["rank", "stock_id", "weight", "mu", "alpha", "epsilon_variance"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def export_allocation_artifacts(
    *,
    aux_frame: pd.DataFrame,
    analysis_time_index: int,
    stock_ids: list[str],
    stock_weights: torch.Tensor,
    cash_weight: float,
    allocation_group_top_n: int,
) -> dict[str, object]:
    all_stock_positions = enrich_positions(
        aux_frame=aux_frame,
        analysis_time_index=analysis_time_index,
        positions=build_all_stock_positions(stock_ids=stock_ids, stock_weights=stock_weights),
    )
    grouped_allocations = append_cash_allocation(
        group_allocations_by_state(all_stock_positions),
        cash_weight,
    )
    grouped_allocations_top_n = summarize_grouped_allocations(
        grouped_allocations,
        top_n=allocation_group_top_n,
    )
    return {
        "all_stock_weights": None,
        "all_stock_weights_csv": None,
        "grouped_allocations": grouped_allocations,
        "grouped_allocations_top_n": grouped_allocations_top_n,
        "allocation_groups_top_n_plus_others": grouped_allocations_top_n,
        "allocation_group_top_n": allocation_group_top_n,
    }


def render_weight_trajectory_chart(
    *,
    scenario_id: str,
    grouped_weight_trajectories: list[dict[str, object]],
    target_time_indices: torch.Tensor,
    output_path: Path,
    metrics_text: str,
) -> None:
    if target_time_indices.ndim != 1:
        raise ValueError("target_time_indices must have shape [T].")
    if not grouped_weight_trajectories:
        raise ValueError("grouped_weight_trajectories must be non-empty.")

    fig, ax = plt.subplots(figsize=(14, 7))
    _plot_weight_trajectory_axes(
        ax,
        grouped_weight_trajectories=grouped_weight_trajectories,
        target_time_indices=target_time_indices,
        metrics_text=metrics_text,
        title=f"Holdout Scenario Group Weight Trajectory: {scenario_id}",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_weight_trajectory_overview_chart(
    *,
    scenario_id: str,
    per_loss_chart_data: dict[str, dict[str, object]],
    output_path: Path,
    loss_order: list[str] | tuple[str, ...] | None = None,
) -> None:
    resolved_loss_order = evaluation_shared.normalize_overview_loss_order(
        list(loss_order) if loss_order is not None else list(WEIGHT_TRAJECTORY_OVERVIEW_LOSS_ORDER)
    )
    missing_losses = [
        loss_name for loss_name in resolved_loss_order if loss_name not in per_loss_chart_data
    ]
    if missing_losses:
        raise ValueError(f"Missing overview chart data for losses: {missing_losses}")

    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    fig.suptitle(f"Holdout Scenario Group Weight Trajectory: {scenario_id}")
    for ax, loss_name in zip(axes.flatten(), resolved_loss_order):
        chart_data = per_loss_chart_data[loss_name]
        grouped_weight_trajectories = chart_data.get("grouped_weight_trajectories")
        target_time_indices = chart_data.get("target_time_indices")
        metrics_text = chart_data.get("metrics_text")
        if not isinstance(grouped_weight_trajectories, list):
            raise ValueError(f"Overview chart data for loss={loss_name} must provide grouped trajectories.")
        if target_time_indices is None:
            raise ValueError(f"Overview chart data for loss={loss_name} must provide target_time_indices.")
        if not isinstance(metrics_text, str):
            raise ValueError(f"Overview chart data for loss={loss_name} must provide metrics_text.")
        _plot_weight_trajectory_axes(
            ax,
            grouped_weight_trajectories=grouped_weight_trajectories,
            target_time_indices=target_time_indices,
            metrics_text=metrics_text,
            title=None,
            legend_fontsize=8,
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_monitoring_weight_trajectory_overview_chart(
    *,
    epoch: int,
    scenario_id: str,
    grouped_weight_trajectories: list[dict[str, object]],
    target_time_indices: torch.Tensor,
    output_path: Path,
    metrics_text: str,
) -> None:
    if target_time_indices.ndim != 1:
        raise ValueError("target_time_indices must have shape [T].")
    if not grouped_weight_trajectories:
        raise ValueError("grouped_weight_trajectories must be non-empty.")

    fig, ax = plt.subplots(figsize=(16, 8))
    _plot_weight_trajectory_axes(
        ax,
        grouped_weight_trajectories=grouped_weight_trajectories,
        target_time_indices=target_time_indices,
        metrics_text=metrics_text,
        title=f"Monitoring Holdout Weight Trajectory: epoch={epoch} scenario={scenario_id}",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_monitoring_multi_loss_weight_trajectory_overview_chart(
    *,
    epoch: int,
    scenario_id: str,
    per_loss_chart_data: dict[str, dict[str, object]],
    output_path: Path,
    loss_order: list[str] | tuple[str, ...] | None = None,
) -> None:
    resolved_loss_order = evaluation_shared.normalize_overview_loss_order(
        list(loss_order) if loss_order is not None else list(WEIGHT_TRAJECTORY_OVERVIEW_LOSS_ORDER)
    )
    missing_losses = [
        loss_name for loss_name in resolved_loss_order if loss_name not in per_loss_chart_data
    ]
    if missing_losses:
        raise ValueError(f"Missing monitoring overview chart data for losses: {missing_losses}")

    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    fig.suptitle(f"Monitoring Holdout Weight Trajectory: epoch={int(epoch)} scenario={scenario_id}")
    for ax, loss_name in zip(axes.flatten(), resolved_loss_order):
        chart_data = per_loss_chart_data[loss_name]
        grouped_weight_trajectories = chart_data.get("grouped_weight_trajectories")
        target_time_indices = chart_data.get("target_time_indices")
        metrics_text = chart_data.get("metrics_text")
        if not isinstance(grouped_weight_trajectories, list):
            raise ValueError(f"Monitoring overview chart data for loss={loss_name} must provide grouped trajectories.")
        if target_time_indices is None:
            raise ValueError(f"Monitoring overview chart data for loss={loss_name} must provide target_time_indices.")
        if not isinstance(metrics_text, str):
            raise ValueError(f"Monitoring overview chart data for loss={loss_name} must provide metrics_text.")
        _plot_weight_trajectory_axes(
            ax,
            grouped_weight_trajectories=grouped_weight_trajectories,
            target_time_indices=target_time_indices,
            metrics_text=metrics_text,
            title=None,
            legend_fontsize=8,
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def monitoring_weight_trajectory_chart_path(
    output_dir: Path,
    *,
    loss_name: str,
    scenario_id: str,
) -> Path:
    return output_dir / f"{loss_name}_{scenario_id}_weight_trajectory.png"


def _build_monitoring_grouped_weight_trajectories(
    *,
    scenario_payload: dict[str, Any],
    dataset: PortfolioPanelDataset,
    evaluation_config: EvaluationConfig,
) -> list[dict[str, object]]:
    scenario_result = RuntimePayloadAdapter.from_legacy_payload(
        scenario_payload,
        require_runtime_tensors=True,
    )
    runtime_tensors = scenario_result.runtime_tensors
    if (
        runtime_tensors is None
        or runtime_tensors.final_stock_weights is None
        or runtime_tensors.stock_weights is None
        or runtime_tensors.cash_weights is None
    ):
        raise RuntimeError("Monitoring grouped trajectories require full runtime tensors.")
    source_path = Path(str(scenario_result.identity.source_path))
    aux_frame = load_aux_frame(source_path)
    all_stock_positions = enrich_positions(
        aux_frame=aux_frame,
        analysis_time_index=int(scenario_result.window_meta.analysis_time_index),
        positions=build_all_stock_positions(
            stock_ids=dataset.selected_stock_ids,
            stock_weights=runtime_tensors.final_stock_weights,
        ),
    )
    grouped_allocations = append_cash_allocation(
        group_allocations_by_state(all_stock_positions),
        float(scenario_result.cash_stats.final_cash_weight),
    )
    grouped_allocations_top_n = summarize_grouped_allocations(
        grouped_allocations,
        top_n=evaluation_config.allocation_group_top_n,
    )
    return build_grouped_weight_trajectories(
        aux_frame=aux_frame,
        analysis_time_index=int(scenario_result.window_meta.analysis_time_index),
        stock_ids=dataset.selected_stock_ids,
        stock_weights=runtime_tensors.stock_weights,
        cash_weights=runtime_tensors.cash_weights,
        grouped_allocations_top_n=grouped_allocations_top_n,
        stock_count_weight_threshold=float(scenario_result.selection_stats.stock_count_weight_threshold),
        stock_count_min_active_days=int(scenario_result.selection_stats.stock_count_min_active_days),
    )


# Public helper APIs.
def build_chart_metrics_text(
    *,
    loss_name: str,
    portfolio_return: float,
    portfolio_sr: float,
    benchmark_excess_return: float | None = None,
    benchmark_information_ratio: float | None = None,
    average_turnover: float | None = None,
    selected_stock_count: int | None = None,
    stock_count_weight_threshold: float | None = None,
) -> str:
    return evaluation_shared.build_chart_metrics_text(
        loss_name=loss_name,
        portfolio_return=portfolio_return,
        portfolio_sr=portfolio_sr,
        benchmark_excess_return=benchmark_excess_return,
        benchmark_information_ratio=benchmark_information_ratio,
        average_turnover=average_turnover,
        selected_stock_count=selected_stock_count,
        stock_count_weight_threshold=stock_count_weight_threshold,
    )


def build_monitoring_grouped_weight_trajectories(
    *,
    scenario_payload: dict[str, Any],
    dataset: PortfolioPanelDataset,
    evaluation_config: EvaluationConfig,
) -> list[dict[str, object]]:
    return _build_monitoring_grouped_weight_trajectories(
        scenario_payload=scenario_payload,
        dataset=dataset,
        evaluation_config=evaluation_config,
    )
