"""Monitoring holdout workflow orchestration."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from . import artifact_paths, evaluation_shared, run_metadata
from .config import DataConfig, EvaluationConfig, ModelConfig, PathsConfig, TrainConfig
from .dataset import PortfolioPanelDataset
from .evaluation_artifacts import (
    build_holdout_summary_payload,
    build_monitoring_scenario_artifact,
    compute_monitoring_holdout_backtest_loss,
    strip_monitoring_transient_tensor_fields,
)
from .evaluation_presentation import build_monitoring_grouped_weight_trajectories
from .evaluate_rebuild import rebuild_monitoring_holdout_backtest_overviews
from .evaluation_runtime import _collect_holdout_per_scenario_payloads
from .model import PortfolioAttentionModel
from .utils import ensure_output_dirs, save_json

WEIGHT_TRAJECTORY_OVERVIEW_FILENAME_SUFFIX = evaluation_shared.WEIGHT_TRAJECTORY_OVERVIEW_FILENAME_SUFFIX


def _monitoring_holdout_backtest_manifest_path(output_dir: Path, loss_name: str) -> Path:
    return artifact_paths.monitoring_manifest_path(output_dir, loss_name)


def _runtime_config_snapshot_path(paths: PathsConfig, state: str, loss_name: str) -> Path:
    return paths.get_state_predictions_dir(state) / f"{loss_name}_runtime_config.json"


def _to_json_compatible_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_json_compatible_value(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_to_json_compatible_value(inner) for inner in value]
    if isinstance(value, tuple):
        return [_to_json_compatible_value(inner) for inner in value]
    return value


def _serialize_runtime_config(config: object) -> dict[str, Any]:
    serialized = asdict(config)  # type: ignore[arg-type]
    return _to_json_compatible_value(serialized)


def _save_runtime_config_snapshot(
    *,
    paths: PathsConfig,
    state: str,
    loss_name: str,
    epoch: int,
    holdout_backtest_output_dir: str,
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> None:
    payload = {
        "artifact_type": "runtime_config_snapshot",
        "generated_from": "monitoring_holdout_backtest",
        "state": state,
        "loss_name": loss_name,
        "epoch": int(epoch),
        "holdout_backtest_output_dir": holdout_backtest_output_dir,
        "data_config": _serialize_runtime_config(data_config),
        "model_config": _serialize_runtime_config(model_config),
        "train_config": _serialize_runtime_config(train_config),
    }
    save_json(payload, _runtime_config_snapshot_path(paths, state, loss_name))


def _set_scenario_artifacts_overview_path(
    payload: dict[str, Any],
    *,
    scenario_id: str | None,
    overview_path: str | None,
) -> bool:
    return run_metadata.set_scenario_artifacts_overview_path(
        payload,
        scenario_id=scenario_id,
        overview_path=overview_path,
    )


def _update_metrics_payload_overview_path(
    payload_path: Path,
    *,
    state: str,
    scenario_id: str | None,
    overview_path: str | None,
) -> None:
    if not payload_path.exists():
        return
    payload = evaluation_shared.PersistedArtifactLoader.load_json_object(payload_path)
    if run_metadata.update_payload_overview_paths(
        payload,
        state=state,
        scenario_id=scenario_id,
        overview_path=overview_path,
    ):
        save_json(payload, payload_path)


def _update_monitoring_manifest_overview_paths(
    manifest_payload: dict[str, Any],
    *,
    overview_path_by_scenario: dict[str, str],
    loss_order: tuple[str, ...],
) -> bool:
    return run_metadata.update_monitoring_manifest_overview_paths(
        manifest_payload,
        overview_path_by_scenario=overview_path_by_scenario,
        loss_order=loss_order,
    )


def _resolve_payload_state(payload: dict[str, Any]) -> str:
    return run_metadata.resolve_payload_state(payload)


def _update_train_metrics_monitoring_overview_paths(
    payload_path: Path,
    *,
    state: str,
    epoch: int,
    holdout_backtest_output_dir: str,
    overview_paths: list[str],
) -> None:
    if not payload_path.exists():
        return
    payload = evaluation_shared.PersistedArtifactLoader.load_json_object(payload_path)
    if run_metadata.update_train_metrics_history_overview_paths(
        payload,
        state=state,
        epoch=epoch,
        holdout_backtest_output_dir=holdout_backtest_output_dir,
        overview_paths=overview_paths,
    ):
        save_json(payload, payload_path)


def compute_monitoring_holdout_backtest_payload(
    *,
    model: PortfolioAttentionModel,
    dataset: PortfolioPanelDataset,
    holdout_dataset: Dataset,
    loss_name: str,
    epoch: int,
    device: torch.device,
    evaluation_config: EvaluationConfig | None = None,
) -> dict[str, Any]:
    resolved_evaluation_config = evaluation_config or EvaluationConfig()
    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=1,
        shuffle=False,
    )
    checkpoint = {
        "train_config": {"loss_name": loss_name},
        "data_config": {"train_batch_size": int(dataset.metadata.train_batch_size)},
    }
    per_scenario_payloads = _collect_holdout_per_scenario_payloads(
        model=model,
        holdout_loader=holdout_loader,
        device=device,
        dataset=dataset,
        loss_name=loss_name,
        evaluation_config=resolved_evaluation_config,
        checkpoint=checkpoint,
    )
    if len(per_scenario_payloads) != len(holdout_dataset):
        raise RuntimeError(
            "Holdout monitoring backtest did not produce a per-scenario payload for every holdout scenario."
        )

    monitoring_summary = build_holdout_summary_payload(
        per_scenario_payloads,
        dataset=dataset,
        loss_name=loss_name,
        evaluation_split="holdout_test_monitoring",
    )
    holdout_backtest_loss = compute_monitoring_holdout_backtest_loss(
        per_scenario_payloads,
        loss_name=loss_name,
    )
    return {
        "epoch": int(epoch),
        "loss_name": loss_name,
        "per_scenario_payloads": per_scenario_payloads,
        "monitoring_summary": monitoring_summary,
        "holdout_backtest_loss": holdout_backtest_loss,
    }


def _export_monitoring_holdout_payloads(
    *,
    monitoring_backtest_payload: dict[str, Any],
    dataset: PortfolioPanelDataset,
    paths: PathsConfig,
    evaluation_config: EvaluationConfig,
) -> dict[str, Any]:
    ensure_output_dirs(paths)
    per_scenario_payloads = list(monitoring_backtest_payload["per_scenario_payloads"])
    monitoring_summary = dict(monitoring_backtest_payload["monitoring_summary"])
    holdout_backtest_loss = float(monitoring_backtest_payload["holdout_backtest_loss"])
    epoch = int(monitoring_backtest_payload["epoch"])
    loss_name = str(monitoring_backtest_payload["loss_name"])
    output_dir = paths.get_state_predictions_dir(dataset.state) / f"{int(epoch)}_holdout_backtest"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_scenario_artifacts: list[dict[str, Any]] = []
    for scenario_index, scenario_payload in enumerate(per_scenario_payloads, start=1):
        grouped_weight_trajectories = build_monitoring_grouped_weight_trajectories(
            scenario_payload=scenario_payload,
            dataset=dataset,
            evaluation_config=evaluation_config,
        )
        manifest_scenario_artifacts.append(
            build_monitoring_scenario_artifact(
                scenario_index=scenario_index,
                scenario_payload=scenario_payload,
                grouped_weight_trajectories=grouped_weight_trajectories,
                output_dir=output_dir,
            )
        )

    manifest_path = _monitoring_holdout_backtest_manifest_path(output_dir, loss_name)
    manifest_payload = {
        "artifact_type": "monitoring_holdout_backtest_manifest",
        "evaluation_split": "holdout_test_monitoring",
        "state": dataset.state,
        "epoch": int(epoch),
        "loss_name": loss_name,
        "holdout_backtest_loss": holdout_backtest_loss,
        "holdout_backtest_output_dir": str(output_dir),
        "overview_loss_order": [loss_name],
        "scenario_ids": [str(item["scenario_id"]) for item in manifest_scenario_artifacts],
        "holdout_backtest_overview_paths": [],
        "scenario_artifacts": manifest_scenario_artifacts,
    }
    for field_name in (
        "evaluation_mode",
        "rolling_window_lookback_days",
        "rolling_window_horizon_days",
        "rolling_window_stride_days",
        "num_rolling_windows",
        "mean_average_turnover",
    ):
        if field_name in monitoring_summary:
            manifest_payload[field_name] = monitoring_summary[field_name]
    save_json(manifest_payload, manifest_path)

    generated_paths = rebuild_monitoring_holdout_backtest_overviews(
        paths,
        state=dataset.state,
        epoch=int(epoch),
    )
    overview_path_by_scenario = {
        path_obj.name[: -len(WEIGHT_TRAJECTORY_OVERVIEW_FILENAME_SUFFIX)]: str(path_obj)
        for path_obj in (Path(path) for path in generated_paths)
        if path_obj.name.endswith(WEIGHT_TRAJECTORY_OVERVIEW_FILENAME_SUFFIX)
    }
    overview_paths = [
        overview_path_by_scenario[scenario_id]
        for scenario_id in manifest_payload["scenario_ids"]
        if scenario_id in overview_path_by_scenario
    ]
    scenario_artifacts = [
        {
            "scenario_index": int(item["scenario_index"]),
            "scenario_id": str(item["scenario_id"]),
            "evaluation_mode": item.get("evaluation_mode"),
            "final_return": float(item["final_return"]),
            "backtest_portfolio_sr": float(item["backtest_portfolio_sr"]),
            "average_turnover": float(item["average_turnover"]),
            "benchmark_market_index_csv": item.get("benchmark_market_index_csv"),
            "benchmark_excess_return": evaluation_shared.coerce_optional_float(
                item.get("benchmark_excess_return")
            ),
            "benchmark_information_ratio": evaluation_shared.coerce_optional_float(
                item.get("benchmark_information_ratio")
            ),
            "rolling_window_lookback_days": item.get("rolling_window_lookback_days"),
            "rolling_window_horizon_days": item.get("rolling_window_horizon_days"),
            "rolling_window_stride_days": item.get("rolling_window_stride_days"),
            "num_rolling_windows": item.get("num_rolling_windows"),
            "weight_trajectory_chart": item.get("weight_trajectory_chart"),
            "weight_trajectory_overview_chart": overview_path_by_scenario.get(str(item["scenario_id"])),
        }
        for item in manifest_scenario_artifacts
    ]

    strip_monitoring_transient_tensor_fields(per_scenario_payloads)
    return {
        **monitoring_summary,
        "epoch": int(epoch),
        "holdout_backtest_loss": holdout_backtest_loss,
        "holdout_backtest_output_dir": str(output_dir),
        "holdout_backtest_overview_paths": overview_paths,
        "scenario_artifacts": scenario_artifacts,
    }


def run_monitoring_holdout_backtest(
    *,
    model: PortfolioAttentionModel,
    dataset: PortfolioPanelDataset,
    holdout_dataset: Dataset,
    loss_name: str,
    epoch: int,
    paths: PathsConfig,
    device: torch.device,
    evaluation_config: EvaluationConfig | None = None,
    data_config: DataConfig | None = None,
    model_config: ModelConfig | None = None,
    train_config: TrainConfig | None = None,
) -> dict[str, Any]:
    ensure_output_dirs(paths)
    resolved_evaluation_config = evaluation_config or EvaluationConfig()
    monitoring_backtest_payload = compute_monitoring_holdout_backtest_payload(
        model=model,
        dataset=dataset,
        holdout_dataset=holdout_dataset,
        loss_name=loss_name,
        epoch=int(epoch),
        device=device,
        evaluation_config=resolved_evaluation_config,
    )
    monitoring_result = _export_monitoring_holdout_payloads(
        monitoring_backtest_payload=monitoring_backtest_payload,
        dataset=dataset,
        paths=paths,
        evaluation_config=resolved_evaluation_config,
    )
    if data_config is not None and model_config is not None and train_config is not None:
        _save_runtime_config_snapshot(
            paths=paths,
            state=dataset.state,
            loss_name=str(monitoring_result["loss_name"]),
            epoch=int(monitoring_result["epoch"]),
            holdout_backtest_output_dir=str(monitoring_result["holdout_backtest_output_dir"]),
            data_config=data_config,
            model_config=model_config,
            train_config=train_config,
        )
    return monitoring_result
