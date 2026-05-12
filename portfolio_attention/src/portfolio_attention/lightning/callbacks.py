"""Lightning callbacks for scenario training."""

from __future__ import annotations

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from ..artifact import paths as artifact_paths
from ..config import PathsConfig, TrainConfig
from ..training.monitoring import resolve_monitoring_holdout_backtest_epochs
from .run_safety import _emit_lightning_console_message


class ConfigEpochCheckpointCallback(Callback):
    """Persist Lightning checkpoints for config-selected 1-based epochs."""

    def __init__(self, *, paths: PathsConfig, state: str, train_config: TrainConfig) -> None:
        super().__init__()
        self.paths = paths
        self.state = state
        self.train_config = train_config
        self.latest_completed_epoch: int = 0

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del pl_module
        if trainer.sanity_checking:
            return
        completed_epoch = int(trainer.current_epoch) + 1
        self.latest_completed_epoch = max(self.latest_completed_epoch, completed_epoch)
        configured_epochs = resolve_monitoring_holdout_backtest_epochs(
            self.train_config,
            max_epoch=completed_epoch,
        )
        if completed_epoch not in configured_epochs:
            return

        checkpoint_path = artifact_paths.lightning_epoch_checkpoint_path(
            self.paths,
            str(getattr(self.train_config, "loss_name", "")),
            completed_epoch,
            state=self.state,
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        # Trainer.save_checkpoint must run on all ranks for distributed strategies.
        trainer.save_checkpoint(str(checkpoint_path), weights_only=False)
        if trainer.is_global_zero:
            _emit_lightning_console_message(
                f"Saved configured epoch checkpoint for post-training holdout: epoch={completed_epoch} path={checkpoint_path}"
            )
