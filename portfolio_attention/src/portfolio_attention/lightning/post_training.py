"""Post-fit holdout orchestration for Lightning training."""

from __future__ import annotations

from typing import Callable

import pytorch_lightning as pl

from ..config import DataConfig, ModelConfig, PathsConfig, TrainConfig
from .callbacks import ConfigEpochCheckpointCallback
from .datamodule import LightningTrainDataModule
from .distributed import _state_transition_barrier, _sync_bool_flag_across_ranks
from .run_safety import _INTERRUPT_CONTROLLER, _emit_lightning_console_message


def run_post_training_holdout_after_fit(
    *,
    trainer: pl.Trainer,
    checkpoint_callback: ConfigEpochCheckpointCallback,
    paths: PathsConfig,
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    datamodule: LightningTrainDataModule,
    holdout_runner: Callable[..., list[tuple[int, str]]] | None = None,
) -> None:
    completed_epochs = int(checkpoint_callback.latest_completed_epoch)
    if completed_epochs <= 0:
        if trainer.is_global_zero:
            _emit_lightning_console_message("Skipping post-training holdout: no completed epochs were detected.")
        return

    resolved_world_size = int(getattr(trainer, "world_size", 1) or 1)
    # When torch.distributed is initialized (e.g., DDP fit), the holdout path performs
    # collective ops (DistributedSampler/all_gather_object), so every rank must participate.
    should_run_on_this_rank = bool(getattr(trainer, "is_global_zero", False)) or resolved_world_size > 1
    if not should_run_on_this_rank:
        return

    if holdout_runner is None:
        from .holdout_test import run_post_training_holdout as holdout_runner_impl
    else:
        holdout_runner_impl = holdout_runner

    if trainer.is_global_zero:
        _emit_lightning_console_message(
            f"Starting post-training holdout evaluation up to completed_epoch={completed_epochs}."
        )
    holdout_kwargs = {
        "paths": paths,
        "data_config": data_config,
        "model_config": model_config,
        "train_config": train_config,
        "max_epoch": completed_epochs,
        "devices": resolved_world_size,
        "datamodule": datamodule,
        "interrupt_checker": _INTERRUPT_CONTROLLER.raise_if_interrupted,
    }
    try:
        holdout_runner_impl(**holdout_kwargs)
    except TypeError as exc:
        if "devices" not in str(exc):
            raise
        holdout_kwargs.pop("devices", None)
        holdout_runner_impl(**holdout_kwargs)


def run_post_training_holdout_after_fit_with_barriers(
    *,
    trainer: pl.Trainer,
    checkpoint_callback: ConfigEpochCheckpointCallback,
    paths: PathsConfig,
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    datamodule: LightningTrainDataModule,
    holdout_runner: Callable[..., list[tuple[int, str]]] | None = None,
) -> None:
    _state_transition_barrier(
        trainer=trainer,
        barrier_name="before_post_training_holdout",
    )
    holdout_exc: BaseException | None = None
    try:
        run_post_training_holdout_after_fit(
            trainer=trainer,
            checkpoint_callback=checkpoint_callback,
            paths=paths,
            data_config=data_config,
            model_config=model_config,
            train_config=train_config,
            datamodule=datamodule,
            holdout_runner=holdout_runner,
        )
    except BaseException as exc:  # noqa: BLE001
        holdout_exc = exc
    finally:
        _state_transition_barrier(
            trainer=trainer,
            barrier_name="after_post_training_holdout",
        )

    holdout_failed = _sync_bool_flag_across_ranks(
        trainer=trainer,
        flag=(holdout_exc is not None),
    )
    if holdout_failed:
        if holdout_exc is not None:
            raise holdout_exc
        raise RuntimeError("Post-training holdout failed on another DDP rank.")


_run_post_training_holdout_after_fit = run_post_training_holdout_after_fit
_run_post_training_holdout_after_fit_with_barriers = run_post_training_holdout_after_fit_with_barriers
