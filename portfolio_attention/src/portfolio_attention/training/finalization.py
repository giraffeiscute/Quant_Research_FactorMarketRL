"""Training artifacts and finalization helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
from typing import Any

import torch
from torch.utils.data import Dataset

from ..artifact import paths as artifact_paths
from ..config import DataConfig, ModelConfig, PathsConfig, TrainConfig
from ..data.dataset import PortfolioPanelDataset
from ..evaluation.pipeline import run_evaluation
from .engine import _build_validation_rolling_metadata
from .status import TrainingStatusReporter
from ..common.utils import append_log, save_json


@dataclass
class FinalCheckpointSelection:
    best_epoch: int
    best_val_loss: float
    selected_best_checkpoint_path: Path
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    effective_selection_window: int


def _serialize_config(config: object) -> dict[str, Any]:
    serialized = asdict(config)  # type: ignore[arg-type]
    for key, value in list(serialized.items()):
        if isinstance(value, Path):
            serialized[key] = str(value)
    return serialized


def _build_checkpoint_payload(
    *,
    model,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    model_config: ModelConfig,
    data_config: DataConfig,
    train_config: TrainConfig,
    dataset: PortfolioPanelDataset,
    epoch: int | None,
    best_val_loss: float | None,
    extra_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scaler_state = {
        "stock_mean": (
            None if dataset.stock_scaler.mean is None else dataset.stock_scaler.mean.tolist()
        ),
        "stock_std": None if dataset.stock_scaler.std is None else dataset.stock_scaler.std.tolist(),
        "market_mean": (
            None if dataset.market_scaler.mean is None else dataset.market_scaler.mean.tolist()
        ),
        "market_std": (
            None if dataset.market_scaler.std is None else dataset.market_scaler.std.tolist()
        ),
    }
    payload: dict[str, Any] = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model_config.as_dict(),
        "max_lookback": model.max_lookback,
        "data_config": _serialize_config(data_config),
        "train_config": _serialize_config(train_config),
        "metadata": dataset.metadata.as_dict(),
        "scaler_state": scaler_state,
        "selected_stock_ids": list(dataset.selected_stock_ids),
    }
    if lr_scheduler is not None:
        payload["lr_scheduler_state_dict"] = lr_scheduler.state_dict()
    if extra_metrics:
        payload["metrics"] = extra_metrics
    return payload


def _save_training_checkpoint(
    checkpoint_path: Path,
    *,
    model,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    model_config: ModelConfig,
    data_config: DataConfig,
    train_config: TrainConfig,
    dataset: PortfolioPanelDataset,
    epoch: int | None,
    best_val_loss: float | None,
    checkpoint_kind: str | None = None,
    extra_metrics: dict[str, Any] | None = None,
) -> None:
    metrics = dict(extra_metrics or {})
    if checkpoint_kind is not None and "checkpoint_kind" not in metrics:
        metrics["checkpoint_kind"] = checkpoint_kind
    torch.save(
        _build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            model_config=model_config,
            data_config=data_config,
            train_config=train_config,
            dataset=dataset,
            epoch=epoch,
            best_val_loss=best_val_loss,
            extra_metrics=(metrics or None),
        ),
        checkpoint_path,
    )


def _train_metrics_path(
    paths: PathsConfig,
    loss_name: str,
    *,
    state: str | None = None,
) -> Path:
    return artifact_paths.train_metrics_path(paths, loss_name, state=state)


def _normalize_best_epoch_selection_window(select_best_from_last_x_epochs: int) -> int:
    return max(1, int(select_best_from_last_x_epochs))


def _epoch_candidate_checkpoint_path(paths: PathsConfig, loss_name: str, epoch: int) -> Path:
    return artifact_paths.epoch_candidate_checkpoint_path(paths, loss_name, epoch)


def _monitoring_epoch_checkpoint_path(
    paths: PathsConfig,
    loss_name: str,
    epoch: int,
    state: str | None = None,
) -> Path:
    return artifact_paths.monitoring_epoch_checkpoint_path(
        paths,
        loss_name,
        epoch,
        state=state,
    )


def _select_best_epoch_record(
    epoch_records: list[dict[str, Any]],
    select_best_from_last_x_epochs: int,
) -> dict[str, Any]:
    if not epoch_records:
        raise RuntimeError("No epoch records were collected for best-epoch selection.")
    candidate_records = epoch_records[-_normalize_best_epoch_selection_window(select_best_from_last_x_epochs) :]
    return min(candidate_records, key=lambda record: (float(record["val_loss"]), int(record["epoch"])))


def _cleanup_temp_epoch_checkpoints(epoch_records: list[dict[str, Any]]) -> None:
    seen_paths: set[Path] = set()
    for record in epoch_records:
        checkpoint_path = record.get("checkpoint_path")
        if checkpoint_path is None:
            continue
        path = Path(str(checkpoint_path))
        if path in seen_paths:
            continue
        seen_paths.add(path)
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def select_final_best_checkpoint(
    *,
    epoch_selection_records: list[dict[str, Any]],
    selection_window: int,
    epochs_completed: int,
    data_config: DataConfig,
    train_config: TrainConfig,
    paths: PathsConfig,
) -> FinalCheckpointSelection:
    selected_best_record = _select_best_epoch_record(epoch_selection_records, selection_window)
    best_epoch = int(selected_best_record["epoch"])
    best_val_loss = float(selected_best_record["val_loss"])
    selected_best_checkpoint_path = Path(str(selected_best_record["checkpoint_path"]))
    checkpoint_state = data_config.state
    return FinalCheckpointSelection(
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        selected_best_checkpoint_path=selected_best_checkpoint_path,
        best_checkpoint_path=artifact_paths.train_best_checkpoint_path(
            paths,
            train_config.loss_name,
            state=checkpoint_state,
        ),
        last_checkpoint_path=artifact_paths.train_last_checkpoint_path(
            paths,
            train_config.loss_name,
            state=checkpoint_state,
        ),
        effective_selection_window=min(selection_window, epochs_completed),
    )


def promote_best_checkpoint(
    *,
    selection: FinalCheckpointSelection,
    epoch_selection_records: list[dict[str, Any]],
    log_path: Path,
    train_config: TrainConfig,
    selection_window: int,
) -> None:
    append_log(
        log_path,
        (
            "Selecting final best checkpoint from trailing validation window: "
            f"configured_window={train_config.select_best_from_last_x_epochs} "
            f"normalized_window={selection_window} "
            f"effective_window={selection.effective_selection_window} "
            f"selected_best_epoch={selection.best_epoch} "
            f"selected_best_val_loss={selection.best_val_loss:.8f}."
        ),
    )
    shutil.copy2(selection.selected_best_checkpoint_path, selection.best_checkpoint_path)
    _cleanup_temp_epoch_checkpoints(epoch_selection_records)


def run_final_holdout_evaluation(
    *,
    data_config: DataConfig,
    train_config: TrainConfig,
    paths: PathsConfig,
    dataset: PortfolioPanelDataset,
    test_dataset: Dataset,
    best_checkpoint_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    append_log(log_path, f"Loading best checkpoint for final held-out evaluation: {best_checkpoint_path}.")
    final_backtest = run_evaluation(
        data_config=data_config,
        paths=paths,
        checkpoint_path=best_checkpoint_path,
        device_name=train_config.device,
        dataset=dataset,
        holdout_dataset=test_dataset,
    )
    append_log(
        log_path,
        (
            "Final held-out aggregate metrics: "
            f"mean_final_return={final_backtest['mean_final_return']:.8f} "
            f"std_final_return={final_backtest['std_final_return']:.8f} "
            f"median_final_return={final_backtest['median_final_return']:.8f} "
            f"worst_scenario_final_return={final_backtest['worst_scenario_final_return']:.8f}"
        ),
    )
    return final_backtest


def build_final_training_metrics(
    *,
    train_config: TrainConfig,
    dataset: PortfolioPanelDataset,
    validation_dataset: Dataset,
    test_dataset: Dataset,
    device: torch.device,
    resolved_shuffle_seed: int,
    train_batch_size: int,
    history: list[dict[str, Any]],
    epochs_completed: int,
    selection_window: int,
    selection: FinalCheckpointSelection,
    final_backtest: dict[str, Any],
    latest_validation_num_rolling_windows_total: int | None,
) -> dict[str, Any]:
    return {
        "mode": "train",
        "device": str(device),
        "loaded_feature_columns": {
            "stock": dataset.loaded_stock_feature_columns,
            "market": dataset.loaded_market_feature_columns,
        },
        "loss_name": train_config.loss_name,
        "seed": train_config.seed,
        "shuffle_train_scenarios_seed": resolved_shuffle_seed,
        "train_batch_size": train_batch_size,
        "num_epochs_requested": train_config.num_epochs,
        "epochs_completed": epochs_completed,
        "history": history,
        **_build_validation_rolling_metadata(
            lookback_days=int(dataset.metadata.lookback_days),
            num_rolling_windows_total=latest_validation_num_rolling_windows_total,
        ),
        "best_epoch": selection.best_epoch,
        "best_val_loss": selection.best_val_loss,
        "select_best_from_last_x_epochs": train_config.select_best_from_last_x_epochs,
        "holdout_backtest_interval_epochs": train_config.holdout_backtest_interval_epochs,
        "enable_fixed_epoch_holdout_backtests": train_config.enable_fixed_epoch_holdout_backtests,
        "normalized_best_epoch_selection_window": selection_window,
        "effective_best_epoch_selection_window": selection.effective_selection_window,
        "train_scenario_count": dataset.metadata.num_train_scenarios,
        "train_window_count": dataset.metadata.train_window_count,
        "validation_scenario_count": len(validation_dataset),
        "holdout_test_scenario_count": len(test_dataset),
        "early_stopping_patience": train_config.early_stopping_patience,
        "stopped_early": epochs_completed < train_config.num_epochs,
        "best_checkpoint_path": str(selection.best_checkpoint_path),
        "last_checkpoint_path": str(selection.last_checkpoint_path),
        "final_backtest": final_backtest,
        "metadata": dataset.metadata.as_dict(),
    }


def persist_final_training_metrics(
    *,
    metrics: dict[str, Any],
    paths: PathsConfig,
    loss_name: str,
    state: str,
) -> None:
    save_json(metrics, _train_metrics_path(paths, loss_name, state=state))


def mark_training_completed(
    *,
    status_reporter: TrainingStatusReporter,
    epochs_completed: int,
    num_epochs: int,
    best_epoch: int,
    best_val_loss: float,
    completed_epoch_seconds_total: float,
) -> None:
    status_reporter.update(
        "DONE",
        epoch=epochs_completed,
        progress_ratio=1.0,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        stopped_early=epochs_completed < num_epochs,
        avg_epoch_seconds=(
            completed_epoch_seconds_total / epochs_completed if epochs_completed > 0 else None
        ),
        eta_seconds=None,
        phase="completed",
        message="Training and evaluation finished successfully.",
    )


def _finalize_training_outputs(
    *,
    data_config: DataConfig,
    train_config: TrainConfig,
    paths: PathsConfig,
    dataset: PortfolioPanelDataset,
    validation_dataset: Dataset,
    test_dataset: Dataset,
    device: torch.device,
    log_path: Path,
    status_reporter: TrainingStatusReporter,
    epoch_selection_records: list[dict[str, Any]],
    selection_window: int,
    epochs_completed: int,
    completed_epoch_seconds_total: float,
    latest_validation_num_rolling_windows_total: int | None,
    resolved_shuffle_seed: int,
    train_batch_size: int,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    if not epoch_selection_records:
        raise RuntimeError("Train loop did not record any epoch candidates for best selection.")

    selection = select_final_best_checkpoint(
        epoch_selection_records=epoch_selection_records,
        selection_window=selection_window,
        epochs_completed=epochs_completed,
        data_config=data_config,
        train_config=train_config,
        paths=paths,
    )
    promote_best_checkpoint(
        selection=selection,
        epoch_selection_records=epoch_selection_records,
        log_path=log_path,
        train_config=train_config,
        selection_window=selection_window,
    )

    status_reporter.set_base(
        **_build_validation_rolling_metadata(
            lookback_days=int(dataset.metadata.lookback_days),
            num_rolling_windows_total=latest_validation_num_rolling_windows_total,
        )
    )
    status_reporter.update(
        "RUNNING",
        epoch=epochs_completed,
        progress_ratio=1.0,
        best_epoch=selection.best_epoch,
        best_val_loss=selection.best_val_loss,
        avg_epoch_seconds=(
            completed_epoch_seconds_total / epochs_completed if epochs_completed > 0 else None
        ),
        eta_seconds=None,
        phase="evaluating",
        message="Running final held-out evaluation on the best checkpoint.",
    )
    final_backtest = run_final_holdout_evaluation(
        data_config=data_config,
        train_config=train_config,
        paths=paths,
        dataset=dataset,
        test_dataset=test_dataset,
        best_checkpoint_path=selection.best_checkpoint_path,
        log_path=log_path,
    )

    metrics = build_final_training_metrics(
        train_config=train_config,
        dataset=dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        device=device,
        resolved_shuffle_seed=resolved_shuffle_seed,
        train_batch_size=train_batch_size,
        history=history,
        epochs_completed=epochs_completed,
        selection_window=selection_window,
        selection=selection,
        final_backtest=final_backtest,
        latest_validation_num_rolling_windows_total=latest_validation_num_rolling_windows_total,
    )
    persist_final_training_metrics(
        metrics=metrics,
        paths=paths,
        loss_name=train_config.loss_name,
        state=data_config.state,
    )
    mark_training_completed(
        status_reporter=status_reporter,
        epochs_completed=epochs_completed,
        num_epochs=train_config.num_epochs,
        best_epoch=selection.best_epoch,
        best_val_loss=selection.best_val_loss,
        completed_epoch_seconds_total=completed_epoch_seconds_total,
    )
    return metrics
