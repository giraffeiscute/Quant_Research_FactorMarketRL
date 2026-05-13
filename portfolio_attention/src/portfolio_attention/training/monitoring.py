"""Training-time monitoring and observability helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ..artifact import run_metadata
from ..config import DataConfig, ModelConfig, PathsConfig, TrainConfig
from ..data.dataset import PortfolioPanelDataset
from ..evaluation.monitoring import run_monitoring_holdout_backtest
from .finalization import _monitoring_epoch_checkpoint_path, _save_training_checkpoint
from .status import TrainingStatusReporter
from ..common.utils import append_log


FIXED_EPOCH_MONITORING_HOLDOUT_BACKTEST_EPOCHS = frozenset({50, 100})


def resolve_monitoring_holdout_backtest_epochs(
    train_config: TrainConfig,
    *,
    max_epoch: int,
) -> tuple[int, ...]:
    """Resolve sorted unique 1-based epochs selected by existing monitoring config rules."""
    resolved_max_epoch = int(max_epoch)
    if resolved_max_epoch <= 0:
        return ()

    scheduled_epochs: set[int] = set()
    interval_epochs = int(train_config.holdout_backtest_interval_epochs)
    if interval_epochs > 0:
        scheduled_epochs.update(range(interval_epochs, resolved_max_epoch + 1, interval_epochs))

    if train_config.enable_fixed_epoch_holdout_backtests:
        scheduled_epochs.update(
            epoch
            for epoch in FIXED_EPOCH_MONITORING_HOLDOUT_BACKTEST_EPOCHS
            if 1 <= int(epoch) <= resolved_max_epoch
        )

    return tuple(sorted(int(epoch) for epoch in scheduled_epochs))


def _should_run_monitoring_holdout_backtest(epoch: int, train_config: TrainConfig) -> bool:
    resolved_epoch = int(epoch)
    if resolved_epoch <= 0:
        return False
    return resolved_epoch in resolve_monitoring_holdout_backtest_epochs(
        train_config,
        max_epoch=resolved_epoch,
    )


def _update_running_epoch_status(
    status_reporter: TrainingStatusReporter,
    epoch_status: dict[str, Any],
    *,
    phase: str,
    message: str,
    **overrides: Any,
) -> None:
    status_reporter.update(
        "RUNNING",
        **epoch_status,
        phase=phase,
        message=message,
        **overrides,
    )


def compute_monitoring_holdout_backtest(
    *,
    model,
    dataset: PortfolioPanelDataset,
    test_dataset: Dataset,
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    loss_name: str,
    epoch: int,
    paths: PathsConfig,
    device: torch.device,
) -> dict[str, Any]:
    return run_monitoring_holdout_backtest(
        model=model,
        dataset=dataset,
        holdout_dataset=test_dataset,
        loss_name=loss_name,
        epoch=epoch,
        paths=paths,
        device=device,
        data_config=data_config,
        model_config=model_config,
        train_config=train_config,
    )


def update_epoch_metrics_with_monitoring_backtest(
    *,
    epoch_metrics: dict[str, Any],
    epoch: int,
    monitoring_backtest: dict[str, Any],
) -> None:
    run_metadata.apply_monitoring_backtest_to_epoch_metrics(
        epoch_metrics,
        epoch=epoch,
        monitoring_backtest=monitoring_backtest,
    )


def save_monitoring_checkpoint(
    *,
    paths: PathsConfig,
    loss_name: str,
    state: str | None,
    epoch: int,
    model,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    model_config: ModelConfig,
    data_config: DataConfig,
    train_config: TrainConfig,
    dataset: PortfolioPanelDataset,
    current_window_best_val_loss: float,
    current_window_best_epoch: int,
    global_best_val_loss: float,
    global_best_checkpoint_updated: bool,
    epochs_without_improvement: int,
    epoch_metrics: dict[str, Any],
) -> Path:
    checkpoint_metrics = dict(epoch_metrics)
    run_metadata.inject_best_state_fields(
        checkpoint_metrics,
        current_window_best_epoch=current_window_best_epoch,
        current_window_best_val_loss=current_window_best_val_loss,
        global_best_val_loss=global_best_val_loss,
        global_best_checkpoint_updated=global_best_checkpoint_updated,
        epochs_without_improvement=epochs_without_improvement,
    )
    monitoring_checkpoint_path = _monitoring_epoch_checkpoint_path(
        paths,
        loss_name,
        epoch,
        state=state,
    )
    _save_training_checkpoint(
        monitoring_checkpoint_path,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        model_config=model_config,
        data_config=data_config,
        train_config=train_config,
        dataset=dataset,
        epoch=epoch,
        best_val_loss=current_window_best_val_loss,
        checkpoint_kind="monitoring_interval",
        extra_metrics=checkpoint_metrics,
    )
    return monitoring_checkpoint_path


def log_monitoring_backtest_summary(
    *,
    log_path: Path,
    epoch: int,
    monitoring_backtest: dict[str, Any],
) -> None:
    append_log(
        log_path,
        (
            f"Monitoring holdout backtest at epoch {epoch}: "
            f"holdout_backtest_loss={monitoring_backtest['holdout_backtest_loss']:.8f} "
            f"mean_final_return={monitoring_backtest['mean_final_return']:.8f} "
            f"std_final_return={monitoring_backtest['std_final_return']:.8f} "
            f"median_final_return={monitoring_backtest['median_final_return']:.8f} "
            f"best_scenario_id={monitoring_backtest['best_scenario_id']} "
            f"output_dir={monitoring_backtest['holdout_backtest_output_dir']} "
            f"overview_charts={len(monitoring_backtest['holdout_backtest_overview_paths'])}"
        ),
    )


def apply_monitoring_holdout_backtest_outputs(
    *,
    epoch_metrics: dict[str, Any],
    epoch: int,
    monitoring_backtest: dict[str, Any],
    paths: PathsConfig,
    loss_name: str,
    checkpoint_state: str | None,
    model,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    model_config: ModelConfig,
    data_config: DataConfig,
    train_config: TrainConfig,
    dataset: PortfolioPanelDataset,
    current_window_best_val_loss: float,
    current_window_best_epoch: int,
    global_best_val_loss: float,
    global_best_checkpoint_updated: bool,
    epochs_without_improvement: int,
    log_path: Path,
    status_reporter: TrainingStatusReporter,
    epoch_status: dict[str, Any],
) -> None:
    update_epoch_metrics_with_monitoring_backtest(
        epoch_metrics=epoch_metrics,
        epoch=epoch,
        monitoring_backtest=monitoring_backtest,
    )
    monitoring_checkpoint_path = save_monitoring_checkpoint(
        paths=paths,
        loss_name=loss_name,
        state=checkpoint_state,
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        model_config=model_config,
        data_config=data_config,
        train_config=train_config,
        dataset=dataset,
        current_window_best_val_loss=current_window_best_val_loss,
        current_window_best_epoch=current_window_best_epoch,
        global_best_val_loss=global_best_val_loss,
        global_best_checkpoint_updated=global_best_checkpoint_updated,
        epochs_without_improvement=epochs_without_improvement,
        epoch_metrics=epoch_metrics,
    )
    run_metadata.set_monitoring_checkpoint_path(epoch_metrics, monitoring_checkpoint_path)
    log_monitoring_backtest_summary(
        log_path=log_path,
        epoch=epoch,
        monitoring_backtest=monitoring_backtest,
    )
    _update_running_epoch_status(
        status_reporter,
        epoch_status,
        phase="training",
        message="Running optimizer and validation steps.",
        holdout_backtest_loss=monitoring_backtest["holdout_backtest_loss"],
        holdout_backtest_mean_final_return=monitoring_backtest["mean_final_return"],
    )


def _run_monitoring_holdout_backtest_epoch(
    *,
    model,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    model_config: ModelConfig,
    data_config: DataConfig,
    train_config: TrainConfig,
    dataset: PortfolioPanelDataset,
    test_dataset: Dataset,
    paths: PathsConfig,
    device: torch.device,
    log_path: Path,
    checkpoint_state: str | None,
    epoch: int,
    current_window_best_val_loss: float,
    current_window_best_epoch: int,
    global_best_val_loss: float,
    global_best_checkpoint_updated: bool,
    epochs_without_improvement: int,
    epoch_metrics: dict[str, Any],
    epoch_status: dict[str, Any],
    status_reporter: TrainingStatusReporter,
) -> None:
    _update_running_epoch_status(
        status_reporter,
        epoch_status,
        phase="monitoring_holdout_backtest",
        message=f"Running monitoring holdout backtest at epoch {epoch}.",
    )
    monitoring_backtest = compute_monitoring_holdout_backtest(
        model=model,
        dataset=dataset,
        test_dataset=test_dataset,
        data_config=data_config,
        model_config=model_config,
        train_config=train_config,
        loss_name=train_config.loss_name,
        epoch=epoch,
        paths=paths,
        device=device,
    )
    apply_monitoring_holdout_backtest_outputs(
        epoch_metrics=epoch_metrics,
        epoch=epoch,
        monitoring_backtest=monitoring_backtest,
        paths=paths,
        loss_name=train_config.loss_name,
        checkpoint_state=checkpoint_state,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        model_config=model_config,
        data_config=data_config,
        train_config=train_config,
        dataset=dataset,
        current_window_best_val_loss=current_window_best_val_loss,
        current_window_best_epoch=current_window_best_epoch,
        global_best_val_loss=global_best_val_loss,
        global_best_checkpoint_updated=global_best_checkpoint_updated,
        epochs_without_improvement=epochs_without_improvement,
        log_path=log_path,
        status_reporter=status_reporter,
        epoch_status=epoch_status,
    )
