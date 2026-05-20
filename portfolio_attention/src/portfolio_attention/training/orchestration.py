"""Training orchestration for a single training run."""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any

import torch
from torch.utils.data import Dataset

from ..artifact import paths as artifact_paths
from ..artifact import run_metadata
from ..common.utils import (
    append_log,
    ensure_output_dirs,
    resolve_device,
    save_runtime_config_artifact,
)
from ..config import DataConfig, EvaluationConfig, ModelConfig, PathsConfig, TrainConfig
from ..config.validation import (
    validate_train_config_against_data_config,
    validated_data_config,
    validated_model_config,
    validated_train_config,
)
from ..data.dataset import PortfolioPanelDataset
from .engine import (
    _append_dataset_split_summary,
    _build_validation_rolling_metadata,
    _log_reproducibility_status,
    _prepare_training_runtime,
    _run_training_epoch,
    _run_validation_epoch,
    build_dataset_bundle,
)
from .finalization import (
    _epoch_candidate_checkpoint_path,
    _finalize_training_outputs,
    _normalize_best_epoch_selection_window,
    _save_training_checkpoint,
    _select_best_epoch_record,
)
from .monitoring import (
    _run_monitoring_holdout_backtest_epoch,
    _should_run_monitoring_holdout_backtest,
    _update_running_epoch_status,
)
from .status import (
    TrainingStatusReporter,
    log_path_for_loss,
    write_training_status,
)


def _run_epoch_training_with_datasets(
    *,
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    paths: PathsConfig,
    device: torch.device,
    log_path: Path,
    dataset: PortfolioPanelDataset,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    test_dataset: Dataset,
    dataset_ready_message: str,
    initialization_lock=None,
) -> dict[str, Any]:
    if len(train_dataset) == 0 or len(validation_dataset) == 0 or len(test_dataset) == 0:
        raise RuntimeError("Scenario training requires non-empty train, validation, and holdout test splits.")
    evaluation_config = EvaluationConfig()

    status_reporter = TrainingStatusReporter(
        paths=paths,
        loss_name=train_config.loss_name,
        base_status={
            "device": str(device),
            "epoch": 0,
            "num_epochs": train_config.num_epochs,
            "progress_ratio": 0.0,
        },
    )
    runtime = _prepare_training_runtime(
        status_reporter=status_reporter,
        initialization_lock=initialization_lock,
        paths=paths,
        data_config=data_config,
        model_config=model_config,
        train_config=train_config,
        dataset=dataset,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        device=device,
    )
    model = runtime.model
    optimizer = runtime.optimizer
    lr_scheduler = runtime.lr_scheduler
    train_loader = runtime.train_loader
    validation_loader = runtime.validation_loader
    resolved_shuffle_seed = runtime.resolved_shuffle_seed
    train_batch_size = runtime.train_batch_size
    initialization_state = runtime.runtime_state

    append_log(
        log_path,
        (
            "Loaded feature columns successfully: "
            f"stock={dataset.loaded_stock_feature_columns} "
            f"market={dataset.loaded_market_feature_columns}"
        ),
    )
    _append_dataset_split_summary(log_path, dataset)

    selection_window = _normalize_best_epoch_selection_window(
        train_config.select_best_from_last_x_epochs
    )
    append_log(
        log_path,
        (
            "Running scenario-based training with "
            f"train_scenarios={dataset.metadata.num_train_scenarios} "
            f"train_samples={len(train_dataset)} "
            f"validation_scenarios={len(validation_dataset)} "
            f"holdout_test_scenarios={len(test_dataset)} "
            f"train_batch_size={train_batch_size} "
            f"shuffle_scenario_splits={bool(data_config.shuffle_scenario_splits)} "
            f"scenario_split_seed={int(data_config.scenario_split_seed)} "
            f"shuffle_train_scenarios={bool(data_config.shuffle_train_scenarios)} "
            f"shuffle_train_scenarios_seed={resolved_shuffle_seed} "
            f"num_epochs={train_config.num_epochs} "
            f"select_best_from_last_x_epochs={train_config.select_best_from_last_x_epochs} "
            f"holdout_backtest_interval_epochs={train_config.holdout_backtest_interval_epochs} "
            f"enable_fixed_epoch_holdout_backtests="
            f"{train_config.enable_fixed_epoch_holdout_backtests} "
            f"normalized_best_epoch_selection_window={selection_window}."
        ),
    )

    global_best_val_loss = (
        float(initialization_state["global_best_val_loss"])
        if initialization_state is not None
        else float("inf")
    )
    epochs_without_improvement = (
        int(initialization_state["epochs_without_improvement"])
        if initialization_state is not None
        else 0
    )
    epochs_completed = (
        int(initialization_state["checkpoint_epoch"])
        if initialization_state is not None
        else 0
    )
    initial_best_epoch = (
        int(initialization_state["current_window_best_epoch"])
        if initialization_state is not None
        else None
    )
    initial_best_val_loss = (
        float(initialization_state["current_window_best_val_loss"])
        if initialization_state is not None
        else None
    )
    starting_epoch = int(initialization_state["next_epoch"]) if initialization_state is not None else 1
    epoch_selection_records: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = (
        list(initialization_state["history"]) if initialization_state is not None else []
    )
    shape_logged = False
    completed_epoch_seconds_total = 0.0
    validation_runtime_metadata = _build_validation_rolling_metadata(
        lookback_days=int(dataset.metadata.lookback_days),
    )
    latest_validation_num_rolling_windows_total = validation_runtime_metadata[
        "validation_num_rolling_windows_total"
    ]

    append_log(
        log_path,
        (
            "Validation rolling evaluation enabled: "
            f"mode={validation_runtime_metadata['validation_evaluation_mode']} "
            f"lookback_days={validation_runtime_metadata['validation_rolling_window_lookback_days']} "
            f"context_num_time_steps={validation_runtime_metadata['validation_context_num_time_steps']} "
            f"stride_days={validation_runtime_metadata['validation_rolling_window_stride_days']}."
        ),
    )

    status_reporter.set_base(
        num_epochs=train_config.num_epochs,
        select_best_from_last_x_epochs=selection_window,
        **validation_runtime_metadata,
    )
    current_phase = "training"
    status_reporter.update(
        "RUNNING",
        epoch=epochs_completed,
        progress_ratio=(epochs_completed / train_config.num_epochs) if train_config.num_epochs else 0.0,
        avg_epoch_seconds=None,
        eta_seconds=None,
        best_epoch=initial_best_epoch,
        best_val_loss=initial_best_val_loss,
        global_best_val_loss=global_best_val_loss if math.isfinite(global_best_val_loss) else None,
        epochs_without_improvement=epochs_without_improvement,
        phase=current_phase,
        message=dataset_ready_message,
    )

    checkpoint_state = data_config.state
    last_checkpoint_path = artifact_paths.train_last_checkpoint_path(
        paths,
        train_config.loss_name,
        state=checkpoint_state,
    )

    try:
        for epoch in range(starting_epoch, train_config.num_epochs + 1):
            epoch_started_at = time.time()
            train_loss, train_mean_final_return, num_train_batches, shape_logged = _run_training_epoch(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                train_loader=train_loader,
                device=device,
                loss_name=train_config.loss_name,
                turnover_penalty=train_config.turnover_penalty,
                transaction_cost_rate=train_config.transaction_cost_rate,
                turnover_penalty_norm=train_config.turnover_penalty_norm,
                grad_clip_norm=train_config.grad_clip_norm,
                epoch=epoch,
                num_epochs=train_config.num_epochs,
                epoch_started_at=epoch_started_at,
                status_reporter=status_reporter,
                log_path=log_path,
                shape_logged=shape_logged,
            )

            current_phase = "validation"
            val_loss, val_mean_final_return, validation_epoch_metadata = _run_validation_epoch(
                model=model,
                dataset=dataset,
                validation_loader=validation_loader,
                device=device,
                loss_name=train_config.loss_name,
                lookback_days=int(dataset.metadata.lookback_days),
                evaluation_transaction_cost_rate=float(
                    evaluation_config.evaluation_transaction_cost_rate
                ),
                epoch=epoch,
                num_epochs=train_config.num_epochs,
                num_train_batches=num_train_batches,
                epoch_started_at=epoch_started_at,
                status_reporter=status_reporter,
            )
            current_phase = "training"
            latest_validation_num_rolling_windows_total = validation_epoch_metadata[
                "validation_num_rolling_windows_total"
            ]
            status_reporter.set_base(**validation_epoch_metadata)

            epoch_metrics = run_metadata.create_epoch_metrics(
                epoch=epoch,
                train_loss=train_loss,
                train_mean_final_return=train_mean_final_return,
                val_loss=val_loss,
                val_mean_final_return=val_mean_final_return,
                validation_epoch_metadata=validation_epoch_metadata,
            )
            epochs_completed = epoch
            global_best_checkpoint_updated = False
            epoch_duration_seconds = time.time() - epoch_started_at
            completed_epoch_seconds_total += epoch_duration_seconds
            avg_epoch_seconds = completed_epoch_seconds_total / epochs_completed
            remaining_epochs = max(0, train_config.num_epochs - epochs_completed)
            eta_seconds = avg_epoch_seconds * remaining_epochs

            append_log(
                log_path,
                (
                    f"epoch={epoch} train_loss={train_loss:.8f} "
                    f"train_mean_final_return={train_mean_final_return:.8f} "
                    f"val_loss={val_loss:.8f} val_mean_final_return={val_mean_final_return:.8f} "
                    f"validation_num_rolling_windows_total="
                    f"{validation_epoch_metadata['validation_num_rolling_windows_total']} "
                    f"epoch_duration_seconds={epoch_duration_seconds:.4f} "
                    f"avg_epoch_seconds={avg_epoch_seconds:.4f} "
                    f"eta_seconds={eta_seconds:.4f}"
                ),
            )
            append_log(log_path, f"Aggregated validation loss at epoch {epoch}: {val_loss:.8f}")

            if val_loss < global_best_val_loss:
                global_best_val_loss = val_loss
                epochs_without_improvement = 0
                global_best_checkpoint_updated = True
            else:
                epochs_without_improvement += 1

            candidate_checkpoint_metrics = dict(epoch_metrics)
            candidate_checkpoint_metrics[run_metadata.KEY_GLOBAL_BEST_CHECKPOINT_UPDATED] = (
                global_best_checkpoint_updated
            )
            candidate_checkpoint_metrics[run_metadata.KEY_EPOCHS_WITHOUT_IMPROVEMENT] = (
                epochs_without_improvement
            )
            candidate_checkpoint_path = _epoch_candidate_checkpoint_path(paths, train_config.loss_name, epoch)
            _save_training_checkpoint(
                candidate_checkpoint_path,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                model_config=model_config,
                data_config=data_config,
                train_config=train_config,
                dataset=dataset,
                epoch=epoch,
                best_val_loss=val_loss,
                extra_metrics=candidate_checkpoint_metrics,
            )
            epoch_selection_records.append(
                {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "checkpoint_path": str(candidate_checkpoint_path),
                }
            )
            if len(epoch_selection_records) > selection_window:
                stale_record = epoch_selection_records.pop(0)
                try:
                    Path(str(stale_record["checkpoint_path"])).unlink()
                except FileNotFoundError:
                    pass

            current_window_best_record = _select_best_epoch_record(
                epoch_selection_records,
                selection_window,
            )
            current_window_best_epoch = int(current_window_best_record["epoch"])
            current_window_best_val_loss = float(current_window_best_record["val_loss"])
            last_checkpoint_metrics = dict(epoch_metrics)
            run_metadata.inject_best_state_fields(
                last_checkpoint_metrics,
                current_window_best_epoch=current_window_best_epoch,
                current_window_best_val_loss=current_window_best_val_loss,
                global_best_val_loss=global_best_val_loss,
                global_best_checkpoint_updated=global_best_checkpoint_updated,
                epochs_without_improvement=epochs_without_improvement,
            )
            _save_training_checkpoint(
                last_checkpoint_path,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                model_config=model_config,
                data_config=data_config,
                train_config=train_config,
                dataset=dataset,
                epoch=epoch,
                best_val_loss=current_window_best_val_loss,
                extra_metrics=last_checkpoint_metrics,
            )

            epoch_status = {
                "epoch": epoch,
                "progress_ratio": epoch / train_config.num_epochs,
                "train_loss": train_loss,
                "train_mean_final_return": train_mean_final_return,
                "val_loss": val_loss,
                "val_mean_final_return": val_mean_final_return,
                "best_epoch": current_window_best_epoch,
                "best_val_loss": current_window_best_val_loss,
                "global_best_val_loss": global_best_val_loss,
                "epochs_without_improvement": epochs_without_improvement,
                "avg_epoch_seconds": avg_epoch_seconds,
                "eta_seconds": eta_seconds,
            }
            _update_running_epoch_status(
                status_reporter,
                epoch_status,
                phase=current_phase,
                message="Running optimizer and validation steps.",
            )

            if _should_run_monitoring_holdout_backtest(epoch, train_config):
                _run_monitoring_holdout_backtest_epoch(
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    model_config=model_config,
                    data_config=data_config,
                    train_config=train_config,
                    dataset=dataset,
                    test_dataset=test_dataset,
                    paths=paths,
                    device=device,
                    log_path=log_path,
                    checkpoint_state=checkpoint_state,
                    epoch=epoch,
                    current_window_best_val_loss=current_window_best_val_loss,
                    current_window_best_epoch=current_window_best_epoch,
                    global_best_val_loss=global_best_val_loss,
                    global_best_checkpoint_updated=global_best_checkpoint_updated,
                    epochs_without_improvement=epochs_without_improvement,
                    epoch_metrics=epoch_metrics,
                    epoch_status=epoch_status,
                    status_reporter=status_reporter,
                )

            history.append(dict(epoch_metrics))
            if epochs_without_improvement >= train_config.early_stopping_patience:
                append_log(
                    log_path,
                    (
                        "Early stopping triggered with "
                        f"patience={train_config.early_stopping_patience} at epoch={epoch}."
                    ),
                )
                break
    except Exception as exc:
        status_reporter.update(
            "FAILED",
            error_message=str(exc),
            epoch=epochs_completed,
            progress_ratio=(epochs_completed / train_config.num_epochs) if train_config.num_epochs else 0.0,
            phase=current_phase,
            **_build_validation_rolling_metadata(
                lookback_days=int(dataset.metadata.lookback_days),
                num_rolling_windows_total=latest_validation_num_rolling_windows_total,
            ),
            message="Training worker failed.",
        )
        raise

    return _finalize_training_outputs(
        data_config=data_config,
        train_config=train_config,
        paths=paths,
        dataset=dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        device=device,
        log_path=log_path,
        status_reporter=status_reporter,
        epoch_selection_records=epoch_selection_records,
        selection_window=selection_window,
        epochs_completed=epochs_completed,
        completed_epoch_seconds_total=completed_epoch_seconds_total,
        latest_validation_num_rolling_windows_total=latest_validation_num_rolling_windows_total,
        resolved_shuffle_seed=resolved_shuffle_seed,
        train_batch_size=train_batch_size,
        history=history,
    )


def run_epoch_training(
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    paths: PathsConfig,
) -> dict[str, Any]:
    data_config = validated_data_config(data_config)
    model_config = validated_model_config(model_config)
    train_config = validated_train_config(train_config)
    if train_config.post_train_from is not None:
        raise ValueError(
            "TrainConfig.post_train_from is only supported by portfolio_attention.cli.lightning_train."
        )
    validate_train_config_against_data_config(train_config, data_config)
    ensure_output_dirs(paths)
    save_runtime_config_artifact(
        paths=paths,
        data_config=data_config,
        model_config=model_config,
        train_config=train_config,
        evaluation_config=EvaluationConfig(),
    )
    device = resolve_device(train_config.device)
    log_path = log_path_for_loss(paths, train_config.loss_name, state=data_config.state)
    _log_reproducibility_status(log_path, train_config, device)
    current_phase = "building_dataset"
    write_training_status(
        paths,
        train_config.loss_name,
        "PREPARING_DATA",
        device=str(device),
        epoch=0,
        num_epochs=train_config.num_epochs,
        progress_ratio=0.0,
        phase=current_phase,
        message="Building dataset and scenario splits.",
    )

    dataset_bundle = build_dataset_bundle(
        data_config=data_config,
        paths=paths,
        loss_name=train_config.loss_name,
        device=device,
        num_epochs=train_config.num_epochs,
        log_path=log_path,
    )
    return _run_epoch_training_with_datasets(
        data_config=data_config,
        model_config=model_config,
        train_config=train_config,
        paths=paths,
        device=device,
        log_path=log_path,
        dataset=dataset_bundle.dataset,
        train_dataset=dataset_bundle.train_dataset,
        validation_dataset=dataset_bundle.validation_dataset,
        test_dataset=dataset_bundle.test_dataset,
        dataset_ready_message="Dataset ready; waiting for first optimizer step.",
    )


run_training = run_epoch_training
