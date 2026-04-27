"""Evaluation refresh/rebuild/backfill/cleanup helpers."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Callable

from . import artifact_paths, evaluation_shared, run_metadata
from .config import DataConfig, EvaluationConfig, PathsConfig
from .config_validation import LEGACY_LOOKBACK_MODES, LOOKBACK_MODE_ROLLING_WINDOW, normalize_lookback_mode
from .evaluation_artifacts import (
    populate_prediction_benchmark_metrics_from_day_weight_artifact,
)
from .evaluation_presentation import (
    render_monitoring_multi_loss_weight_trajectory_overview_chart,
    render_weight_trajectory_overview_chart,
)
from .utils import ensure_output_dirs, save_json

MONITORING_HOLDOUT_BACKTEST_MANIFEST_SUFFIX = artifact_paths.MONITORING_HOLDOUT_BACKTEST_MANIFEST_SUFFIX


def _metrics_dir_for_state(paths: PathsConfig, state: str | None) -> Path:
    return artifact_paths.metrics_dir_for_state(paths, state)


def _train_metrics_path(paths: PathsConfig, loss_name: str, *, state: str | None = None) -> Path:
    return artifact_paths.train_metrics_path(paths, loss_name, state=state)


def _evaluation_metrics_path(
    paths: PathsConfig,
    loss_name: str,
    *,
    state: str | None = None,
) -> Path:
    return artifact_paths.evaluation_metrics_path(paths, loss_name, state=state)


def _candidate_train_metrics_paths(
    paths: PathsConfig,
    loss_name: str,
    *,
    state: str | None = None,
) -> list[Path]:
    return artifact_paths.candidate_train_metrics_paths(paths, loss_name, state=state)

def _fail_if_refresh_uses_legacy_lookback_mode(metadata: dict[str, object]) -> None:
    raw_lookback_mode = metadata.get("lookback_mode")
    if raw_lookback_mode is None:
        return
    lookback_mode = normalize_lookback_mode(raw_lookback_mode)
    if lookback_mode not in LEGACY_LOOKBACK_MODES:
        return
    raise ValueError(
        "Refresh supports reading legacy metadata but cannot rebuild artifacts for "
        f"lookback_mode={lookback_mode!r}. Legacy non-rolling modes are read-only."
    )


def _build_refresh_data_config(
    *,
    source_path: Path,
    train_batch_size: object | None,
    train_metrics_metadata: dict[str, object] | None = None,
) -> DataConfig:
    default_config = DataConfig()
    metadata = train_metrics_metadata or {}
    _fail_if_refresh_uses_legacy_lookback_mode(metadata)
    legacy_ratio_keys = (
        "scenario_train_split_ratio",
        "scenario_validation_split_ratio",
        "scenario_test_split_ratio",
    )
    legacy_ratio_payload = {
        key: metadata[key] for key in legacy_ratio_keys if key in metadata
    }
    if legacy_ratio_payload:
        raise ValueError(
            "Refresh does not support rebuilding artifacts from legacy scenario-internal "
            f"time split metadata: {sorted(legacy_ratio_payload)}."
        )
    resolved_batch_size = (
        int(train_batch_size)
        if train_batch_size is not None
        else int(metadata.get("train_batch_size", metadata.get("scenario_batch_size", default_config.train_batch_size)))
    )
    resolved_lookback_mode = normalize_lookback_mode(
        metadata.get("lookback_mode", LOOKBACK_MODE_ROLLING_WINDOW)
    )
    if resolved_lookback_mode != LOOKBACK_MODE_ROLLING_WINDOW:
        raise ValueError(
            "Refresh only supports rolling-window metadata, "
            f"received lookback_mode={resolved_lookback_mode!r}."
        )
    return DataConfig(
        state=source_path.parent.name,
        scenario_dir=source_path.parent,
        scenario_glob=default_config.scenario_glob,
        num_train_scenarios=int(metadata.get("num_train_scenarios", default_config.num_train_scenarios)),
        num_validation_scenarios=int(
            metadata.get("num_validation_scenarios", default_config.num_validation_scenarios)
        ),
        num_test_scenarios=int(metadata.get("num_test_scenarios", default_config.num_test_scenarios)),
        train_batch_size=resolved_batch_size,
        shuffle_scenario_splits=bool(
            metadata.get("shuffle_scenario_splits", default_config.shuffle_scenario_splits)
        ),
        scenario_split_seed=int(metadata.get("scenario_split_seed", default_config.scenario_split_seed)),
        shuffle_train_scenarios=default_config.shuffle_train_scenarios,
        shuffle_train_scenarios_seed=default_config.shuffle_train_scenarios_seed,
        lookback_days=int(
            metadata.get("lookback_days", metadata.get("max_lookback_days", default_config.lookback_days))
        ),
        rolling_horizon_days=int(
            metadata.get("rolling_horizon_days", default_config.rolling_horizon_days)
        ),
        rolling_stride_days=int(metadata.get("rolling_stride_days", default_config.rolling_stride_days)),
        rolling_train_dataset_mode=str(
            metadata.get("rolling_train_dataset_mode", default_config.rolling_train_dataset_mode)
        ),
        price_normalization_mode=str(
            metadata.get("price_normalization_mode", default_config.price_normalization_mode)
        ),
        num_stocks=int(
            metadata.get(
                "selected_num_stocks",
                evaluation_shared.parse_num_stocks_from_source_path(source_path),
            )
        ),
    )


def _load_refresh_train_metrics_metadata(
    *,
    paths: PathsConfig,
    loss_name: str,
    source_path: Path,
    state: str | None = None,
) -> dict[str, object] | None:
    seen_paths: set[Path] = set()
    for train_metrics_path in _candidate_train_metrics_paths(paths, loss_name, state=state):
        resolved_path = train_metrics_path.resolve()
        if resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)
        if not train_metrics_path.exists():
            continue

        train_metrics_payload = evaluation_shared.PersistedArtifactLoader.load_json_object(
            train_metrics_path
        )
        metadata = train_metrics_payload.get("metadata")
        if not isinstance(metadata, dict):
            continue
        if str(metadata.get("scenario_dir", "")) != str(source_path.parent):
            continue
        return metadata
    return None


def _monitoring_holdout_backtest_manifest_path(output_dir: Path, loss_name: str) -> Path:
    return artifact_paths.monitoring_manifest_path(output_dir, loss_name)


def _monitoring_holdout_backtest_overview_path(output_dir: Path, scenario_id: str) -> Path:
    return artifact_paths.monitoring_overview_path(output_dir, scenario_id)


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


def _rebuild_monitoring_holdout_backtest_directory(
    paths: PathsConfig,
    *,
    state: str,
    output_dir: Path,
    loss_order: list[str] | tuple[str, ...] | None = None,
    interrupt_checker: Callable[[], None] | None = None,
) -> list[str]:
    if interrupt_checker is not None:
        interrupt_checker()
    manifest_payloads: dict[str, tuple[Path, dict[str, Any]]] = {}
    for manifest_path in sorted(output_dir.glob(f"*{MONITORING_HOLDOUT_BACKTEST_MANIFEST_SUFFIX}")):
        if interrupt_checker is not None:
            interrupt_checker()
        payload = evaluation_shared.PersistedArtifactLoader.load_json_object(
            manifest_path,
            expected_artifact_type="monitoring_holdout_backtest_manifest",
        )
        if payload.get("artifact_type") != "monitoring_holdout_backtest_manifest":
            raise ValueError(f"Unexpected monitoring manifest payload in {manifest_path}.")
        manifest_payloads[
            evaluation_shared.monitoring_manifest_loss_name(
                manifest_path,
                manifest_suffix=MONITORING_HOLDOUT_BACKTEST_MANIFEST_SUFFIX,
            )
        ] = (manifest_path, payload)

    resolved_loss_order = evaluation_shared.resolve_monitoring_manifest_loss_order(
        manifest_payloads,
        preferred_loss_order=loss_order,
    )
    if resolved_loss_order is None:
        return []
    selected_manifest_payloads = {
        loss_name: manifest_payloads[loss_name] for loss_name in resolved_loss_order
    }

    base_manifest = selected_manifest_payloads[resolved_loss_order[0]][1]
    scenario_ids = base_manifest.get("scenario_ids")
    if not isinstance(scenario_ids, list) or not all(isinstance(item, str) for item in scenario_ids):
        raise ValueError(f"Monitoring manifest {output_dir} must provide scenario_ids as a list of strings.")
    epoch = int(base_manifest.get("epoch", -1))
    if epoch < 0:
        raise ValueError(f"Monitoring manifest {output_dir} must provide a non-negative epoch.")

    per_loss_artifacts: dict[str, dict[str, dict[str, Any]]] = {}
    for loss_name, (_, payload) in selected_manifest_payloads.items():
        if interrupt_checker is not None:
            interrupt_checker()
        payload_state = str(payload.get("state", "")).lower()
        if payload_state and payload_state != state.lower():
            return []
        if int(payload.get("epoch", epoch)) != epoch:
            raise ValueError(f"Monitoring manifest epoch mismatch in {output_dir}.")
        payload_scenario_ids = payload.get("scenario_ids")
        if not isinstance(payload_scenario_ids, list) or set(payload_scenario_ids) != set(scenario_ids):
            raise ValueError(f"Monitoring manifest scenario set mismatch in {output_dir}.")

        scenario_artifacts = payload.get("scenario_artifacts")
        if not isinstance(scenario_artifacts, list):
            raise ValueError(f"Monitoring manifest {output_dir} must provide scenario_artifacts.")
        artifacts_by_scenario: dict[str, dict[str, Any]] = {}
        for item in scenario_artifacts:
            if not isinstance(item, dict):
                raise ValueError("Monitoring manifest scenario_artifacts entries must be objects.")
            scenario_id = str(item.get("scenario_id", ""))
            if not scenario_id:
                raise ValueError("Monitoring manifest scenario_artifacts must include scenario_id.")
            artifacts_by_scenario[scenario_id] = item
        if set(artifacts_by_scenario) != set(scenario_ids):
            raise ValueError(f"Monitoring manifest scenario_artifacts mismatch in {output_dir}.")
        per_loss_artifacts[loss_name] = artifacts_by_scenario

    # Multi-loss: remove individual per-loss weight_trajectory.png files; only the overview will be kept.
    evaluation_shared.unlink_artifacts_by_patterns(
        output_dir,
        [
            f"*{evaluation_shared.WEIGHT_TRAJECTORY_OVERVIEW_FILENAME_SUFFIX}",
            "*_weight_trajectory.png",
        ],
    )

    generated_paths: list[str] = []
    overview_path_by_scenario: dict[str, str] = {}
    for scenario_id in scenario_ids:
        if interrupt_checker is not None:
            interrupt_checker()
        per_loss_chart_data: dict[str, dict[str, object]] = {}
        for loss_name in resolved_loss_order:
            if interrupt_checker is not None:
                interrupt_checker()
            artifact = per_loss_artifacts[loss_name][scenario_id]
            weight_trajectory_data = artifact.get("weight_trajectory_data")
            if not isinstance(weight_trajectory_data, dict):
                raise ValueError(
                    f"Monitoring manifest for scenario={scenario_id} loss={loss_name} is missing weight_trajectory_data."
                )
            grouped_weight_trajectories, target_time_indices = evaluation_shared.load_weight_trajectory_export_data(
                dict(weight_trajectory_data)
            )
            total_selected_stock_count = artifact.get("total_selected_stock_count")
            stock_count_weight_threshold = artifact.get("stock_count_weight_threshold")
            if total_selected_stock_count is not None and not isinstance(total_selected_stock_count, int):
                total_selected_stock_count = int(total_selected_stock_count)
            if stock_count_weight_threshold is not None:
                stock_count_weight_threshold = float(stock_count_weight_threshold)
            if total_selected_stock_count is not None and stock_count_weight_threshold is None:
                total_selected_stock_count = None
            average_turnover = evaluation_shared.coerce_optional_float(
                artifact.get("average_turnover")
            )
            metrics_text = evaluation_shared.build_chart_metrics_text(
                loss_name=loss_name,
                portfolio_return=float(artifact["final_return"]),
                portfolio_sr=float(artifact["backtest_portfolio_sr"]),
                benchmark_excess_return=evaluation_shared.coerce_optional_float(
                    artifact.get("benchmark_excess_return")
                ),
                benchmark_information_ratio=evaluation_shared.coerce_optional_float(
                    artifact.get("benchmark_information_ratio")
                ),
                average_turnover=average_turnover,
                selected_stock_count=total_selected_stock_count,
                stock_count_weight_threshold=stock_count_weight_threshold,
            )
            per_loss_chart_data[loss_name] = {
                "grouped_weight_trajectories": grouped_weight_trajectories,
                "target_time_indices": target_time_indices,
                "metrics_text": metrics_text,
            }

        overview_path = _monitoring_holdout_backtest_overview_path(output_dir, scenario_id)
        if interrupt_checker is not None:
            interrupt_checker()
        render_monitoring_multi_loss_weight_trajectory_overview_chart(
            epoch=epoch,
            scenario_id=scenario_id,
            per_loss_chart_data=per_loss_chart_data,
            output_path=overview_path,
            loss_order=resolved_loss_order,
        )
        if interrupt_checker is not None:
            interrupt_checker()
        overview_path_str = str(overview_path)
        generated_paths.append(overview_path_str)
        overview_path_by_scenario[scenario_id] = overview_path_str

    for loss_name, (manifest_path, payload) in selected_manifest_payloads.items():
        if interrupt_checker is not None:
            interrupt_checker()
        if _update_monitoring_manifest_overview_paths(
            payload,
            overview_path_by_scenario=overview_path_by_scenario,
            loss_order=resolved_loss_order,
        ):
            save_json(payload, manifest_path)
        _update_train_metrics_monitoring_overview_paths(
            _train_metrics_path(paths, loss_name, state=state),
            state=state,
            epoch=epoch,
            holdout_backtest_output_dir=str(output_dir),
            overview_paths=generated_paths,
        )
    return generated_paths


def rebuild_monitoring_holdout_backtest_overviews(
    paths: PathsConfig,
    *,
    state: str | None = None,
    epoch: int | None = None,
    output_dirs: list[Path] | None = None,
    loss_order: list[str] | tuple[str, ...] | None = None,
    existing_only: bool = False,
    interrupt_checker: Callable[[], None] | None = None,
) -> list[str]:
    if interrupt_checker is not None:
        interrupt_checker()
    if output_dirs is None:
        if state is None:
            raise ValueError("state is required when output_dirs is not provided.")
        state_predictions_dir = paths.get_state_predictions_dir(state)
        if not state_predictions_dir.exists():
            return []
        if epoch is not None:
            resolved_output_dirs = [state_predictions_dir / f"{int(epoch)}_holdout_backtest"]
        else:
            resolved_output_dirs = sorted(
                path for path in state_predictions_dir.glob("*_holdout_backtest") if path.is_dir()
            )
    else:
        resolved_output_dirs = sorted({Path(path) for path in output_dirs})

    generated_paths: list[str] = []
    for output_dir in resolved_output_dirs:
        if interrupt_checker is not None:
            interrupt_checker()
        if not output_dir.exists():
            continue
        if existing_only and not evaluation_shared.monitoring_output_dir_has_existing_overview_png(
            output_dir
        ):
            continue
        resolved_state = state or output_dir.parent.name
        if not resolved_state:
            continue
        generated_paths.extend(
            _rebuild_monitoring_holdout_backtest_directory(
                paths,
                state=resolved_state,
                output_dir=output_dir,
                loss_order=loss_order,
                interrupt_checker=interrupt_checker,
            )
        )
    return generated_paths


def cleanup_monitoring_holdout_backtest_artifacts(
    paths: PathsConfig,
    *,
    state: str,
) -> None:
    state_predictions_dir = paths.get_state_predictions_dir(state)
    if not state_predictions_dir.exists():
        return

    for output_dir in state_predictions_dir.glob("*_holdout_backtest"):
        if not output_dir.is_dir():
            continue
        evaluation_shared.unlink_artifacts_by_patterns(
            output_dir,
            [
                f"*{MONITORING_HOLDOUT_BACKTEST_MANIFEST_SUFFIX}",
                f"*{evaluation_shared.WEIGHT_TRAJECTORY_OVERVIEW_FILENAME_SUFFIX}",
                "*_weight_trajectory.png",
            ],
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


def cleanup_multi_loss_weight_trajectory_overviews(
    paths: PathsConfig,
    *,
    state: str,
) -> None:
    state_predictions_dir = paths.get_state_predictions_dir(state)
    if state_predictions_dir.exists():
        for path in state_predictions_dir.glob("*_weight_trajectory_overview.png"):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        for prediction_path in state_predictions_dir.glob("*_prediction.json"):
            payload = evaluation_shared.PersistedArtifactLoader.load_json_object(prediction_path)
            if payload.get("weight_trajectory_overview_chart") is None:
                continue
            payload["weight_trajectory_overview_chart"] = None
            save_json(payload, prediction_path)

    metrics_dir = _metrics_dir_for_state(paths, state)
    metrics_paths = sorted(metrics_dir.glob("evaluation_metrics_*.json")) + sorted(
        metrics_dir.glob("train_metrics_*.json")
    )
    for metrics_path in metrics_paths:
        _update_metrics_payload_overview_path(
            metrics_path,
            state=state,
            scenario_id=None,
            overview_path=None,
        )


def rebuild_multi_loss_weight_trajectory_overviews(
    paths: PathsConfig,
    *,
    state: str,
    evaluation_config: EvaluationConfig | None = None,
    loss_order: list[str] | tuple[str, ...] | None = None,
    existing_only: bool = False,
) -> list[str]:
    state_predictions_dir = paths.get_state_predictions_dir(state)
    if not state_predictions_dir.exists():
        return []
    resolved_evaluation_config = evaluation_config or EvaluationConfig()

    per_scenario_payloads: dict[str, dict[str, tuple[Path, dict[str, Any]]]] = {}
    for prediction_path in sorted(state_predictions_dir.glob("*_prediction.json")):
        payload = evaluation_shared.PersistedArtifactLoader.load_json_object(prediction_path)
        if payload.get("artifact_type") != "holdout_scenario_prediction":
            continue
        weight_trajectory_data = payload.get("weight_trajectory_data")
        if not isinstance(weight_trajectory_data, dict):
            continue
        scenario_id = str(payload.get("scenario_id", ""))
        payload_loss_name = str(payload.get("loss_name", "")).strip().lower()
        if not scenario_id or not payload_loss_name:
            continue
        per_scenario_payloads.setdefault(scenario_id, {})[payload_loss_name] = (prediction_path, payload)

    generated_paths: list[str] = []
    for scenario_id in sorted(per_scenario_payloads):
        scenario_payloads = per_scenario_payloads[scenario_id]
        resolved_loss_order = evaluation_shared.resolve_multi_loss_overview_loss_order(
            set(scenario_payloads),
            preferred_loss_order=loss_order,
        )
        if resolved_loss_order is None:
            continue

        per_loss_chart_data: dict[str, dict[str, object]] = {}
        selected_scenario_payloads = {
            loss_name: scenario_payloads[loss_name] for loss_name in resolved_loss_order
        }
        overview_path = state_predictions_dir / f"{scenario_id}_weight_trajectory_overview.png"
        if existing_only:
            existing_overview_path = evaluation_shared.resolve_existing_prediction_overview_path(
                selected_scenario_payloads
            )
            if existing_overview_path is None:
                continue
            overview_path = existing_overview_path
        for loss_name in resolved_loss_order:
            _, payload = scenario_payloads[loss_name]
            grouped_weight_trajectories, target_time_indices = evaluation_shared.load_weight_trajectory_export_data(
                dict(payload["weight_trajectory_data"])
            )
            payload_changed = populate_prediction_benchmark_metrics_from_day_weight_artifact(payload)
            stock_count_weight_threshold = float(
                payload.get(
                    "stock_count_weight_threshold",
                    resolved_evaluation_config.stock_count_weight_threshold,
                )
            )
            total_selected_stock_count = int(
                payload.get(
                    "total_selected_stock_count",
                    sum(
                        1
                        for item in (payload.get("all_stock_weights") or [])
                        if isinstance(item, dict)
                        and evaluation_shared.is_weight_above_threshold(
                            float(item.get("weight", 0.0)),
                            threshold=stock_count_weight_threshold,
                        )
                    ),
                )
            )
            per_loss_chart_data[loss_name] = {
                "grouped_weight_trajectories": grouped_weight_trajectories,
                "target_time_indices": target_time_indices,
                "metrics_text": evaluation_shared.build_chart_metrics_text(
                    loss_name=loss_name,
                    portfolio_return=float(payload["final_return"]),
                    portfolio_sr=float(payload["backtest_portfolio_sr"]),
                    benchmark_excess_return=evaluation_shared.coerce_optional_float(
                        payload.get("benchmark_excess_return")
                    ),
                    benchmark_information_ratio=evaluation_shared.coerce_optional_float(
                        payload.get("benchmark_information_ratio")
                    ),
                    average_turnover=evaluation_shared.coerce_optional_float(
                        payload.get("average_turnover")
                    ),
                    selected_stock_count=total_selected_stock_count,
                    stock_count_weight_threshold=stock_count_weight_threshold,
                ),
            }
            if payload_changed:
                scenario_payloads[loss_name] = (scenario_payloads[loss_name][0], payload)

        render_weight_trajectory_overview_chart(
            scenario_id=scenario_id,
            per_loss_chart_data=per_loss_chart_data,
            output_path=overview_path,
            loss_order=resolved_loss_order,
        )
        overview_path_str = str(overview_path)
        generated_paths.append(overview_path_str)

        for _, payload in selected_scenario_payloads.values():
            payload["weight_trajectory_overview_chart"] = overview_path_str
        for prediction_path, payload in selected_scenario_payloads.values():
            save_json(payload, prediction_path)
        for loss_name in resolved_loss_order:
            _update_metrics_payload_overview_path(
                _evaluation_metrics_path(paths, loss_name, state=state),
                state=state,
                scenario_id=scenario_id,
                overview_path=overview_path_str,
            )
            _update_metrics_payload_overview_path(
                _train_metrics_path(paths, loss_name, state=state),
                state=state,
                scenario_id=scenario_id,
                overview_path=overview_path_str,
            )
    return generated_paths


def refresh_existing_scenario_artifacts(
    paths: PathsConfig,
    *,
    run_evaluation_fn: Callable[..., dict[str, Any]],
    device_name: str = "auto",
    evaluation_config: EvaluationConfig | None = None,
    loss_name: str | None = None,
) -> list[dict[str, Any]]:
    ensure_output_dirs(paths)
    prediction_paths = sorted(paths.predictions_dir.glob("*/*_prediction.json"))
    candidate_prediction_paths: list[Path] = []
    refreshed_payloads: list[dict[str, Any]] = []
    scheduled_keys: set[tuple[str, str]] = set()

    for prediction_path in prediction_paths:
        payload = evaluation_shared.PersistedArtifactLoader.load_json_object(prediction_path)
        if payload.get("artifact_type") != "holdout_scenario_prediction":
            continue
        candidate_prediction_paths.append(prediction_path)
        payload_loss_name = str(payload["loss_name"]).lower()
        payload_state = str(payload["state"]).lower()
        if loss_name is not None and payload_loss_name != loss_name.lower():
            continue
        schedule_key = (payload_state, payload_loss_name)
        if schedule_key in scheduled_keys:
            continue
        scheduled_keys.add(schedule_key)

        source_path = Path(str(payload["source_path"]))
        checkpoint_path = artifact_paths.train_best_checkpoint_path(
            paths,
            payload_loss_name,
            state=payload_state,
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "Refresh requires an existing checkpoint for each loss. "
                f"Missing checkpoint: {checkpoint_path}"
            )

        data_config = _build_refresh_data_config(
            source_path=source_path,
            train_batch_size=payload.get("train_config", {}).get(
                "train_batch_size",
                payload.get("train_config", {}).get("scenario_batch_size"),
            ),
            train_metrics_metadata=_load_refresh_train_metrics_metadata(
                paths=paths,
                loss_name=payload_loss_name,
                source_path=source_path,
                state=payload_state,
            ),
        )
        try:
            refreshed_payloads.append(
                run_evaluation_fn(
                    data_config=data_config,
                    paths=paths,
                    checkpoint_path=checkpoint_path,
                    device_name=device_name,
                    evaluation_config=evaluation_config,
                    loss_name=payload_loss_name,
                )
            )
        except Exception as exc:
            print(
                (
                    "Skipping refresh for existing scenario artifacts "
                    f"state={payload_state} loss={payload_loss_name}: {exc}"
                ),
                file=sys.stderr,
            )

    if candidate_prediction_paths and not refreshed_payloads:
        raise RuntimeError("Refresh could not rebuild any existing scenario artifacts.")
    return refreshed_payloads


def backfill_monitoring_holdout_backtest_overviews(
    paths: PathsConfig,
    *,
    output_dirs: list[Path] | None = None,
    state: str | None = None,
    epoch: int | None = None,
    loss_order: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    ensure_output_dirs(paths)
    resolved_output_dirs = output_dirs
    if resolved_output_dirs is None and state is None:
        resolved_output_dirs = sorted(
            path for path in paths.predictions_dir.glob("*/*_holdout_backtest") if path.is_dir()
        )
    return rebuild_monitoring_holdout_backtest_overviews(
        paths,
        state=state,
        epoch=epoch,
        output_dirs=resolved_output_dirs,
        loss_order=loss_order,
    )


def _format_monitoring_overview_backfill_terminal_summary(generated_paths: list[str]) -> str:
    lines = [f"rebuild_overview_count: {len(generated_paths)}"]
    lines.extend(f"rebuild: {path}" for path in generated_paths)
    return "\n".join(lines)

