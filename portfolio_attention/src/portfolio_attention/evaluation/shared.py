"""Shared helpers for evaluation outputs and rebuild pipelines."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import torch

from .types import RuntimePayloadAdapter, WeightTrajectoryExportData

WEIGHT_TRAJECTORY_REFERENCE_DAY = 187
WEIGHT_TRAJECTORY_OVERVIEW_LOSS_ORDER = ("mdd", "return", "sharpe", "sortino")
MULTI_LOSS_WEIGHT_TRAJECTORY_OVERVIEW_COUNT = 4
WEIGHT_TRAJECTORY_OVERVIEW_FILENAME_SUFFIX = "_weight_trajectory_overview.png"
SCENARIO_FILENAME_PATTERN = re.compile(
    r"(?P<state>[^_]+)_(?P<num_stocks>\d+)_(?P<num_time_steps>\d+)_PL_(?P<scenario>\d+)\.parquet"
)


def canonicalize_overview_loss_order(loss_order: tuple[str, ...]) -> tuple[str, ...]:
    if (
        len(loss_order) == len(WEIGHT_TRAJECTORY_OVERVIEW_LOSS_ORDER)
        and set(loss_order) == set(WEIGHT_TRAJECTORY_OVERVIEW_LOSS_ORDER)
    ):
        return WEIGHT_TRAJECTORY_OVERVIEW_LOSS_ORDER
    return loss_order


def normalize_overview_loss_order(
    loss_order: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    resolved = tuple(str(loss).strip().lower() for loss in loss_order)
    if len(resolved) != MULTI_LOSS_WEIGHT_TRAJECTORY_OVERVIEW_COUNT:
        raise ValueError(
            "weight trajectory overview requires exactly "
            f"{MULTI_LOSS_WEIGHT_TRAJECTORY_OVERVIEW_COUNT} losses, "
            f"received {len(resolved)}."
        )
    if any(not loss for loss in resolved):
        raise ValueError("weight trajectory overview losses must be non-empty strings.")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"weight trajectory overview losses must be unique, received {resolved}.")
    return canonicalize_overview_loss_order(resolved)


def resolve_multi_loss_overview_loss_order(
    available_losses: set[str],
    *,
    preferred_loss_order: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, ...] | None:
    normalized_available_losses = {
        str(loss_name).strip().lower() for loss_name in available_losses if str(loss_name).strip()
    }
    if preferred_loss_order is not None:
        resolved_preferred_order = normalize_overview_loss_order(preferred_loss_order)
        if not set(resolved_preferred_order).issubset(normalized_available_losses):
            return None
        return resolved_preferred_order
    if len(normalized_available_losses) != MULTI_LOSS_WEIGHT_TRAJECTORY_OVERVIEW_COUNT:
        return None
    return canonicalize_overview_loss_order(tuple(sorted(normalized_available_losses)))


def resolve_existing_prediction_overview_path(
    selected_scenario_payloads: dict[str, tuple[Path, dict[str, Any]]],
) -> Path | None:
    existing_paths: list[Path] = []
    for _, payload in selected_scenario_payloads.values():
        raw_overview_path = payload.get("weight_trajectory_overview_chart")
        if not isinstance(raw_overview_path, str) or not raw_overview_path:
            return None
        overview_path = Path(raw_overview_path)
        if not overview_path.exists():
            return None
        existing_paths.append(overview_path)

    unique_paths = {str(path) for path in existing_paths}
    if len(unique_paths) != 1:
        return None
    return existing_paths[0]


def monitoring_output_dir_has_existing_overview_png(output_dir: Path) -> bool:
    return any(output_dir.glob(f"*{WEIGHT_TRAJECTORY_OVERVIEW_FILENAME_SUFFIX}"))


def monitoring_manifest_loss_name(
    manifest_path: Path,
    *,
    manifest_suffix: str,
) -> str:
    if not manifest_path.name.endswith(manifest_suffix):
        raise ValueError(f"Unexpected monitoring manifest file name: {manifest_path.name}")
    return manifest_path.name[: -len(manifest_suffix)].lower()


def resolve_monitoring_manifest_loss_order(
    manifest_payloads: dict[str, tuple[Path, dict[str, Any]]],
    *,
    preferred_loss_order: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, ...] | None:
    available_losses = set(manifest_payloads)
    resolved_loss_order = resolve_multi_loss_overview_loss_order(
        available_losses,
        preferred_loss_order=preferred_loss_order,
    )
    if resolved_loss_order is not None:
        return resolved_loss_order

    stored_loss_orders: set[tuple[str, ...]] = set()
    for _, payload in manifest_payloads.values():
        raw_loss_order = payload.get("overview_loss_order")
        if not isinstance(raw_loss_order, list):
            continue
        try:
            normalized_loss_order = normalize_overview_loss_order(
                [str(loss_name) for loss_name in raw_loss_order]
            )
        except ValueError:
            continue
        if set(normalized_loss_order).issubset(available_losses):
            stored_loss_orders.add(normalized_loss_order)

    if len(stored_loss_orders) > 1:
        raise ValueError(f"Conflicting overview_loss_order values found in {manifest_payloads}.")
    if stored_loss_orders:
        return next(iter(stored_loss_orders))
    return resolve_multi_loss_overview_loss_order(available_losses)


def parse_source_time_to_index(raw_value: object) -> int:
    if isinstance(raw_value, str):
        match = re.fullmatch(r"t_(\d+)", raw_value.strip())
        if not match:
            raise ValueError(f"Unsupported source time label: {raw_value}")
        return int(match.group(1))
    return int(raw_value)


def unlink_artifacts_by_patterns(output_dir: Path, patterns: list[str]) -> None:
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            path.unlink(missing_ok=True)


def strip_transient_scenario_tensor_fields(per_scenario_payloads: list[dict[str, Any]]) -> None:
    RuntimePayloadAdapter.strip_runtime_fields(per_scenario_payloads)


def is_weight_above_threshold(weight: float, *, threshold: float) -> bool:
    resolved_weight = float(weight)
    resolved_threshold = float(threshold)
    return resolved_weight > resolved_threshold and not np.isclose(
        resolved_weight,
        resolved_threshold,
        rtol=0.0,
        atol=1e-9,
    )


def format_optional_metric(value: float | None) -> str:
    if value is None or not np.isfinite(float(value)):
        return "N/A"
    return f"{float(value):.4f}"


def coerce_optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value:
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(resolved):
        return None
    return resolved


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
    lines = [
        f"Loss: {loss_name}",
        f"Portfolio Return: {portfolio_return:.4f}",
        f"Portfolio SR: {portfolio_sr:.4f}",
        f"Benchmark Excess Return: {format_optional_metric(benchmark_excess_return)}",
        f"Benchmark IR: {format_optional_metric(benchmark_information_ratio)}",
    ]
    if average_turnover is not None:
        lines.append(f"Avg Turnover: {float(average_turnover):.4f}")
    if selected_stock_count is not None:
        if stock_count_weight_threshold is None:
            raise ValueError("stock_count_weight_threshold is required when selected_stock_count is provided.")
        lines.append(f"Stocks Bought: {selected_stock_count}")
    return "\n".join(lines)


def parse_num_stocks_from_source_path(source_path: Path) -> int:
    match = SCENARIO_FILENAME_PATTERN.fullmatch(source_path.name)
    if match is None:
        raise ValueError(
            "Could not infer num_stocks from source_path. Expected file name pattern "
            "<state>_<num_stocks>_<num_time_steps>_PL_<scenario>.parquet, "
            f"received {source_path.name!r}."
        )
    return int(match.group("num_stocks"))


def build_weight_trajectory_export_data(
    *,
    grouped_weight_trajectories: list[dict[str, object]],
    target_time_indices: torch.Tensor,
) -> WeightTrajectoryExportData:
    series: list[dict[str, object]] = []
    for item in grouped_weight_trajectories:
        weights = item["weights"]
        if not isinstance(weights, torch.Tensor):
            raise ValueError("Each trajectory entry must provide a tensor in 'weights'.")
        series.append(
            {
                "label": str(item["label"]),
                "weights": [float(value) for value in weights.detach().cpu().tolist()],
            }
        )
    return {
        "reference_day": WEIGHT_TRAJECTORY_REFERENCE_DAY,
        "target_time_indices": [int(value) for value in target_time_indices.detach().cpu().tolist()],
        "series": series,
    }


def load_weight_trajectory_export_data(
    payload: Mapping[str, object],
) -> tuple[list[dict[str, object]], torch.Tensor]:
    raw_target_time_indices = payload.get("target_time_indices")
    raw_series = payload.get("series")
    if not isinstance(raw_target_time_indices, list) or not isinstance(raw_series, list):
        raise ValueError("weight_trajectory_data must provide 'target_time_indices' and 'series' lists.")
    grouped_weight_trajectories: list[dict[str, object]] = []
    for item in raw_series:
        if not isinstance(item, Mapping):
            raise ValueError("weight_trajectory_data.series entries must be objects.")
        label = item.get("label")
        weights = item.get("weights")
        if not isinstance(weights, list):
            raise ValueError("weight_trajectory_data.series weights must be a list.")
        grouped_weight_trajectories.append(
            {
                "label": str(label),
                "weights": torch.tensor(weights, dtype=torch.float32),
            }
        )
    target_time_indices = torch.tensor(raw_target_time_indices, dtype=torch.int64)
    return grouped_weight_trajectories, target_time_indices


class PersistedArtifactLoader:
    """Persistence boundary: tolerant file loading/parsing for artifacts."""

    @staticmethod
    def load_json_object(path: Path, *, expected_artifact_type: str | None = None) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Artifact must contain a JSON object: {path}")
        if expected_artifact_type is None:
            return payload
        artifact_type = payload.get("artifact_type")
        if artifact_type is not None and artifact_type != expected_artifact_type:
            raise ValueError(
                f"Unexpected artifact_type in {path}: expected {expected_artifact_type!r}, "
                f"received {artifact_type!r}."
            )
        return payload

    @staticmethod
    def load_day_weight_artifact(path: Path) -> dict[str, Any]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"day_weight_artifact must contain a dict payload: {path}")
        return payload
