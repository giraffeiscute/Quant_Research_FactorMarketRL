"""PyTorch Lightning training entrypoint for single-loss scenario training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Callable

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from portfolio_attention.lightning_run_safety import (
        _INTERRUPT_CONTROLLER,
        _configure_warning_routing,
        _destroy_distributed_process_group_if_initialized,
        _emit_lightning_console_message,
        _is_global_rank_zero,
    )
else:
    from .lightning_run_safety import (
        _INTERRUPT_CONTROLLER,
        _configure_warning_routing,
        _destroy_distributed_process_group_if_initialized,
        _emit_lightning_console_message,
        _is_global_rank_zero,
    )

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Metric

if __package__ is None or __package__ == "":
    from portfolio_attention import artifact_paths
    from portfolio_attention.config import DataConfig, EvaluationConfig, ModelConfig, PathsConfig, TrainConfig
    from portfolio_attention.dataset import PortfolioPanelDataset
    from portfolio_attention.evaluation_runtime import _collect_single_scenario_rolling_one_step_outputs
    from portfolio_attention.losses import build_loss
    from portfolio_attention.train_cli import (
        _parse_states_args,
        build_arg_parser,
        resolve_model_config_from_args,
        resolve_paths_config_from_args,
        resolve_runtime_configs_from_args,
    )
    from portfolio_attention.train_engine import _run_loss_step, build_training_model
    from portfolio_attention.train_monitoring import resolve_monitoring_holdout_backtest_epochs
    from portfolio_attention.utils import save_runtime_config_artifact, set_seed
else:
    from . import artifact_paths
    from .config import DataConfig, EvaluationConfig, ModelConfig, PathsConfig, TrainConfig
    from .dataset import PortfolioPanelDataset
    from .evaluation_runtime import _collect_single_scenario_rolling_one_step_outputs
    from .losses import build_loss
    from .train_cli import (
        _parse_states_args,
        build_arg_parser,
        resolve_model_config_from_args,
        resolve_paths_config_from_args,
        resolve_runtime_configs_from_args,
    )
    from .train_engine import _run_loss_step, build_training_model
    from .train_monitoring import resolve_monitoring_holdout_backtest_epochs
    from .utils import save_runtime_config_artifact, set_seed


class LightningTrainDataModule(pl.LightningDataModule):
    """Thin DataModule wrapper around the existing scenario dataset stack."""

    def __init__(
        self,
        *,
        data_config: DataConfig,
        num_workers: int = 0,
        interrupt_checker: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.num_workers = int(num_workers)
        self.interrupt_checker = interrupt_checker

        self.dataset: PortfolioPanelDataset | None = None
        self.train_dataset: Dataset | None = None
        self.validation_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    def _raise_if_interrupted(self) -> None:
        if self.interrupt_checker is None:
            return
        self.interrupt_checker()

    def build_datasets(self) -> None:
        self._raise_if_interrupted()
        if self.dataset is not None:
            return
        dataset = PortfolioPanelDataset(
            self.data_config,
            interrupt_checker=self.interrupt_checker,
        )
        train_dataset, validation_dataset, test_dataset = dataset.build_train_validation_test_datasets()
        self.dataset = dataset
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self._raise_if_interrupted()

    def validate_validation_divisibility(self, world_size: int) -> None:
        self.build_datasets()
        if self.validation_dataset is None:
            raise RuntimeError("validation_dataset is unavailable before divisibility validation.")
        resolved_world_size = int(world_size)
        if resolved_world_size <= 0 or (len(self.validation_dataset) % resolved_world_size) != 0:
            raise ValueError(
                "validation dataset size must be divisible by world size to avoid duplicated validation scenarios under DistributedSampler"
            )

    def setup(self, stage: str | None = None) -> None:
        if stage not in (None, "fit", "validate", "test"):
            return
        self.build_datasets()

    def _build_dataloader(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is unavailable before building the train DataLoader.")
        return self._build_dataloader(
            self.train_dataset,
            batch_size=self.data_config.train_batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.validation_dataset is None:
            raise RuntimeError(
                "validation_dataset is unavailable before building the validation DataLoader."
            )
        return self._build_dataloader(
            self.validation_dataset,
            batch_size=1,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is unavailable before building the test DataLoader.")
        return self._build_dataloader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
        )


def _compute_selected_stock_count_from_weights(
    stock_weights: torch.Tensor,
    *,
    threshold: float,
    min_active_days: int,
) -> int:
    if stock_weights.ndim != 2:
        raise ValueError("stock_weights must have shape [T, N].")
    resolved_min_active_days = int(min_active_days)
    if resolved_min_active_days <= 0:
        raise ValueError(f"min_active_days must be positive, received {min_active_days}.")
    if int(stock_weights.shape[0]) <= 0:
        raise ValueError("stock_weights must include at least one scored day.")

    threshold_tensor = torch.full_like(stock_weights, float(threshold))
    above_threshold = (stock_weights > threshold_tensor) & (
        ~torch.isclose(stock_weights, threshold_tensor, rtol=0.0, atol=1e-9)
    )
    effective_min_active_days = min(resolved_min_active_days, int(stock_weights.shape[0]))
    selected_stock_count = int((above_threshold.sum(dim=0) >= effective_min_active_days).sum().item())
    return selected_stock_count


class ScenarioRollingValidationMetric(Metric):
    """Aggregate scenario-level validation outputs across DDP workers."""

    full_state_update = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state("loss_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("final_return_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("scenario_count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state(
            "rolling_window_count",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state("selected_stock_count_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        *,
        loss_value: torch.Tensor | float,
        scenario_final_return: torch.Tensor | float,
        num_rolling_windows: torch.Tensor | int,
        selected_stock_count: torch.Tensor | int | float,
        scenario_count: torch.Tensor | int = 1,
    ) -> None:
        self.loss_sum += torch.as_tensor(loss_value, device=self.loss_sum.device, dtype=self.loss_sum.dtype)
        self.final_return_sum += torch.as_tensor(
            scenario_final_return,
            device=self.final_return_sum.device,
            dtype=self.final_return_sum.dtype,
        )
        self.scenario_count += torch.as_tensor(
            scenario_count,
            device=self.scenario_count.device,
            dtype=self.scenario_count.dtype,
        )
        self.rolling_window_count += torch.as_tensor(
            num_rolling_windows,
            device=self.rolling_window_count.device,
            dtype=self.rolling_window_count.dtype,
        )
        self.selected_stock_count_sum += torch.as_tensor(
            selected_stock_count,
            device=self.selected_stock_count_sum.device,
            dtype=self.selected_stock_count_sum.dtype,
        )

    def compute(self) -> dict[str, torch.Tensor]:
        if int(self.scenario_count.item()) <= 0:
            zero = self.loss_sum.new_zeros(())
            zero_count = self.rolling_window_count.new_zeros(())
            return {
                "val_loss": zero,
                "val_mean_final_return": zero,
                "validation_num_rolling_windows_total": zero_count,
                "validation_stocks_bought": zero,
            }

        scenario_count = self.scenario_count.to(dtype=self.loss_sum.dtype)
        return {
            "val_loss": self.loss_sum / scenario_count,
            "val_mean_final_return": self.final_return_sum / scenario_count,
            "validation_num_rolling_windows_total": self.rolling_window_count.clone(),
            "validation_stocks_bought": self.selected_stock_count_sum / scenario_count,
        }


class PortfolioLightningModule(pl.LightningModule):
    """LightningModule that reuses the repo's model/loss/validation helpers."""

    def __init__(
        self,
        *,
        data_config: DataConfig,
        model_config: ModelConfig,
        train_config: TrainConfig,
        dataset: PortfolioPanelDataset,
        stock_count_weight_threshold: float,
        stock_count_min_active_days: int,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.model_config = model_config
        self.train_config = train_config
        self.dataset = dataset
        self.stock_count_weight_threshold = float(stock_count_weight_threshold)
        self.stock_count_min_active_days = int(stock_count_min_active_days)

        self.model = build_training_model(
            model_config=model_config,
            dataset=dataset,
            data_config=data_config,
            device=torch.device("cpu"),
        )
        self.val_metric = ScenarioRollingValidationMetric()

    def forward(
        self,
        x_stock: torch.Tensor,
        x_market: torch.Tensor,
        stock_indices: torch.Tensor,
        target_returns: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        return self.model(
            x_stock,
            x_market,
            stock_indices,
            target_returns=target_returns,
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        loss, _, summary = _run_loss_step(
            self.model,
            batch,
            self.train_config.loss_name,
        )
        train_mean_final_return = summary["scenario_final_returns"].mean()
        batch_size = int(summary["scenario_final_returns"].numel())

        self._log_train_epoch_metric("train_loss", loss, prog_bar=True, batch_size=batch_size)
        self._log_train_epoch_metric(
            "train_mean_final_return",
            train_mean_final_return,
            prog_bar=False,
            batch_size=batch_size,
        )
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_metric.reset()

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        del batch_idx
        rolling_outputs = _collect_single_scenario_rolling_one_step_outputs(
            model=self.model,
            dataset=self.dataset,
            raw_batch=batch,
            device=self.device,
            lookback_days=int(self.dataset.metadata.lookback_days),
            evaluation_label="Lightning validation rolling evaluation",
            collect_weights=True,
        )
        scored_returns = rolling_outputs["portfolio_returns"].unsqueeze(0)
        loss = build_loss(self.train_config.loss_name, scored_returns)
        scenario_final_return = (torch.prod(1.0 + scored_returns, dim=1) - 1.0).mean()
        selected_stock_count = _compute_selected_stock_count_from_weights(
            rolling_outputs["stock_weights"],
            threshold=self.stock_count_weight_threshold,
            min_active_days=self.stock_count_min_active_days,
        )

        self.val_metric.update(
            loss_value=loss.detach(),
            scenario_final_return=scenario_final_return.detach(),
            num_rolling_windows=int(rolling_outputs["num_rolling_windows"]),
            selected_stock_count=selected_stock_count,
        )

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metric.compute()
        self._log_validation_epoch_metric("val_loss", metrics["val_loss"], prog_bar=True)
        self._log_validation_epoch_metric(
            "val_mean_final_return",
            metrics["val_mean_final_return"],
            prog_bar=True,
        )
        validation_num_rolling_windows_total = metrics["validation_num_rolling_windows_total"].to(
            dtype=torch.float32
        )
        self._log_validation_epoch_metric(
            "validation_num_rolling_windows_total",
            validation_num_rolling_windows_total,
            prog_bar=False,
        )
        self._log_validation_epoch_metric(
            "validation_stocks_bought",
            metrics["validation_stocks_bought"],
            prog_bar=False,
        )

    def _log_train_epoch_metric(
        self,
        name: str,
        value: torch.Tensor | float,
        *,
        prog_bar: bool,
        batch_size: int,
    ) -> None:
        self.log(
            name,
            value,
            on_step=False,
            on_epoch=True,
            prog_bar=prog_bar,
            logger=True,
            sync_dist=True,
            batch_size=int(batch_size),
        )

    def _log_validation_epoch_metric(
        self,
        name: str,
        value: torch.Tensor | float,
        *,
        prog_bar: bool,
    ) -> None:
        self.log(
            name,
            value,
            on_step=False,
            on_epoch=True,
            prog_bar=prog_bar,
            logger=True,
            sync_dist=False,
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.train_config.learning_rate),
            weight_decay=float(self.train_config.weight_decay),
        )


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
        trainer.save_checkpoint(str(checkpoint_path))
        if trainer.is_global_zero:
            _emit_lightning_console_message(
                f"Saved configured epoch checkpoint for post-training holdout: epoch={completed_epoch} path={checkpoint_path}"
            )


def _trainer_was_interrupted(trainer: pl.Trainer) -> bool:
    if bool(getattr(trainer, "interrupted", False)):
        return True
    state = getattr(trainer, "state", None)
    status = getattr(state, "status", None)
    if status is None:
        return False
    status_name = getattr(status, "name", None)
    if isinstance(status_name, str):
        return status_name.upper() == "INTERRUPTED"
    return str(status).upper().endswith("INTERRUPTED")


def _exception_represents_interrupt(exc: BaseException) -> bool:
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return True
    text = " ".join(str(exc).split()).lower()
    if "keyboardinterrupt" in text:
        return True
    if "terminated with code 130" in text:
        return True
    return False


def _run_post_training_holdout_after_fit(
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
    if not trainer.is_global_zero:
        return

    completed_epochs = int(checkpoint_callback.latest_completed_epoch)
    if completed_epochs <= 0:
        _emit_lightning_console_message("Skipping post-training holdout: no completed epochs were detected.")
        return

    if holdout_runner is None:
        if __package__ is None or __package__ == "":
            from portfolio_attention.lightning_holdout_test import run_post_training_holdout as holdout_runner_impl
        else:
            from .lightning_holdout_test import run_post_training_holdout as holdout_runner_impl
    else:
        holdout_runner_impl = holdout_runner

    _emit_lightning_console_message(
        f"Starting post-training holdout evaluation up to completed_epoch={completed_epochs}."
    )
    holdout_runner_impl(
        paths=paths,
        data_config=data_config,
        model_config=model_config,
        train_config=train_config,
        max_epoch=completed_epochs,
        datamodule=datamodule,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser()
    parser.description = "Run single-loss Lightning training for portfolio_attention."
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of local GPUs to use on this machine.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count for train/validation loaders.",
    )
    return parser


def _validate_cli_args(args: argparse.Namespace) -> None:
    args_dict = vars(args)

    if int(args_dict.get("parallel", 1)) != 1:
        raise ValueError("lightning_train.py only supports a single loss per invocation; --parallel must be 1.")

    unsupported_losses = args_dict.get("losses")
    if unsupported_losses is not None:
        raise ValueError("lightning_train.py only supports --loss, not --losses.")

    legacy_resume_checkpoints = args_dict.get("resume_checkpoints")
    if legacy_resume_checkpoints is not None:
        raise ValueError(
            "lightning_train.py does not support --resume-checkpoints; use a single Lightning .ckpt with --resume-from."
        )

    requested_device = args_dict.get("device")
    if requested_device is not None:
        normalized_device = str(requested_device).strip().lower()
        if normalized_device not in {"auto", "cuda"}:
            raise ValueError(
                "lightning_train.py always runs Lightning with accelerator='gpu'; --device must be 'auto' or 'cuda'."
            )

    if int(getattr(args, "devices", 0)) <= 0:
        raise ValueError("--devices must be positive.")


def _build_state_args(args: argparse.Namespace, state: str) -> argparse.Namespace:
    state_args_dict = vars(args).copy()
    state_args_dict["state"] = state
    return argparse.Namespace(**state_args_dict)


def _resolve_single_state_runtime(
    args: argparse.Namespace,
) -> tuple[PathsConfig, DataConfig, TrainConfig, EvaluationConfig, ModelConfig]:
    _INTERRUPT_CONTROLLER.raise_if_interrupted()
    paths = resolve_paths_config_from_args(args)
    data_config, train_config = resolve_runtime_configs_from_args(args)
    evaluation_config = EvaluationConfig()
    _configure_warning_routing(state=data_config.state, paths=paths)
    model_config = resolve_model_config_from_args(args)

    if not train_config.loss_name:
        raise ValueError("lightning_train.py requires a single --loss.")
    if train_config.resume_from is not None and Path(train_config.resume_from).suffix != ".ckpt":
        raise ValueError(
            "lightning_train.py expects --resume-from to point to a Lightning .ckpt checkpoint in this MVP."
        )
    save_runtime_config_artifact(
        paths=paths,
        data_config=data_config,
        model_config=model_config,
        train_config=train_config,
    )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "lightning_train.py requires CUDA because the Trainer is configured with accelerator='gpu'."
        )
    available_gpus = torch.cuda.device_count()
    if int(args.devices) > available_gpus:
        raise ValueError(
            f"Requested devices={int(args.devices)}, but only {available_gpus} CUDA device(s) are available."
        )

    return paths, data_config, train_config, evaluation_config, model_config


def _prepare_single_state_datamodule(
    *,
    args: argparse.Namespace,
    data_config: DataConfig,
) -> LightningTrainDataModule:
    datamodule = LightningTrainDataModule(
        data_config=data_config,
        num_workers=int(args.num_workers),
        interrupt_checker=_INTERRUPT_CONTROLLER.raise_if_interrupted,
    )
    _emit_lightning_console_message("Starting build datasets.")
    _emit_lightning_console_message(
        "Starting scenario splitting and dataset materialization.",
    )
    _INTERRUPT_CONTROLLER.raise_if_interrupted()
    datamodule.build_datasets()
    _INTERRUPT_CONTROLLER.raise_if_interrupted()
    _emit_lightning_console_message(
        "Finished scenario splitting and dataset materialization.",
    )

    _emit_lightning_console_message("Starting validation divisibility check.")
    _INTERRUPT_CONTROLLER.raise_if_interrupted()
    datamodule.validate_validation_divisibility(int(args.devices))
    _INTERRUPT_CONTROLLER.raise_if_interrupted()

    if datamodule.dataset is None:
        raise RuntimeError("Dataset build completed without populating datamodule.dataset.")
    if datamodule.train_dataset is None or datamodule.validation_dataset is None or datamodule.test_dataset is None:
        raise RuntimeError("Dataset build completed without all split datasets.")
    if (
        len(datamodule.train_dataset) == 0
        or len(datamodule.validation_dataset) == 0
        or len(datamodule.test_dataset) == 0
    ):
        raise RuntimeError(
            "Scenario training requires non-empty train, validation, and holdout test splits."
        )
    return datamodule


def _build_single_state_training_stack(
    *,
    args: argparse.Namespace,
    paths: PathsConfig,
    data_config: DataConfig,
    train_config: TrainConfig,
    model_config: ModelConfig,
    evaluation_config: EvaluationConfig,
    datamodule: LightningTrainDataModule,
) -> tuple[PortfolioLightningModule, pl.Trainer, ConfigEpochCheckpointCallback]:
    model = PortfolioLightningModule(
        data_config=data_config,
        model_config=model_config,
        train_config=train_config,
        dataset=datamodule.dataset,
        stock_count_weight_threshold=float(evaluation_config.stock_count_weight_threshold),
        stock_count_min_active_days=int(evaluation_config.stock_count_min_active_days),
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(paths.checkpoints_dir),
        filename=f"{data_config.state}_{train_config.loss_name}" + "-epoch{epoch:03d}-val{val_loss:.8f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=int(train_config.early_stopping_patience),
    )
    config_epoch_checkpoint_callback = ConfigEpochCheckpointCallback(
        paths=paths,
        state=data_config.state,
        train_config=train_config,
    )
    csv_logger = CSVLogger(
        save_dir=str(paths.outputs_dir),
        name="lightning_logs",
        version=f"{data_config.state}_{train_config.loss_name}",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=int(args.devices),
        num_nodes=1,
        strategy="ddp" if int(args.devices) > 1 else "auto",
        max_epochs=int(train_config.num_epochs),
        gradient_clip_val=float(train_config.grad_clip_norm),
        callbacks=[checkpoint_callback, early_stopping_callback, config_epoch_checkpoint_callback],
        logger=csv_logger,
        default_root_dir=str(paths.outputs_dir),
        enable_progress_bar=True,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )
    return model, trainer, config_epoch_checkpoint_callback


def _run_single_state(args: argparse.Namespace) -> None:
    paths, data_config, train_config, evaluation_config, model_config = _resolve_single_state_runtime(args)

    set_seed(int(train_config.seed))
    pl.seed_everything(int(train_config.seed), workers=True)

    datamodule = _prepare_single_state_datamodule(
        args=args,
        data_config=data_config,
    )
    model, trainer, config_epoch_checkpoint_callback = _build_single_state_training_stack(
        args=args,
        paths=paths,
        data_config=data_config,
        train_config=train_config,
        model_config=model_config,
        evaluation_config=evaluation_config,
        datamodule=datamodule,
    )
    _emit_lightning_console_message("Starting trainer.fit().")
    _INTERRUPT_CONTROLLER.raise_if_interrupted()
    try:
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=(str(train_config.resume_from) if train_config.resume_from is not None else None),
        )
    except BaseException as exc:
        if _INTERRUPT_CONTROLLER.interrupted or _exception_represents_interrupt(exc):
            raise KeyboardInterrupt("Trainer interrupted by user signal.") from exc
        raise
    if _trainer_was_interrupted(trainer):
        raise KeyboardInterrupt("Trainer interrupted by user signal.")
    _INTERRUPT_CONTROLLER.raise_if_interrupted()
    _run_post_training_holdout_after_fit(
        trainer=trainer,
        checkpoint_callback=config_epoch_checkpoint_callback,
        paths=paths,
        data_config=data_config,
        model_config=model_config,
        train_config=train_config,
        datamodule=datamodule,
    )


def _run_states_sequentially(args: argparse.Namespace, states_to_run: list[str]) -> list[str]:
    failed_states: list[str] = []
    total_states = len(states_to_run)
    for index, state in enumerate(states_to_run, start=1):
        _INTERRUPT_CONTROLLER.raise_if_interrupted()
        state_args = _build_state_args(args, state)
        if total_states > 1 and _is_global_rank_zero():
            print(f"\n=== Running state {index}/{total_states}: {state} ===", flush=True)
        try:
            _run_single_state(state_args)
        except KeyboardInterrupt:
            if _is_global_rank_zero():
                print("Interrupted; stopping remaining states.", flush=True)
            raise
        except Exception as exc:
            if total_states <= 1:
                raise
            failed_states.append(state)
            if _is_global_rank_zero():
                print(f"ERROR: State '{state}' failed: {exc}", flush=True)
    return failed_states


def main() -> None:
    _INTERRUPT_CONTROLLER.install()
    try:
        parser = _build_parser()
        args = parser.parse_args()
        _validate_cli_args(args)

        states_to_run = _parse_states_args(args)
        if len(states_to_run) > 1 and getattr(args, "resume_from", None) is not None:
            raise ValueError(
                "Multi-state training does not support --resume-from. "
                "Resume one state at a time."
            )

        failed_states = _run_states_sequentially(args, states_to_run)
        if failed_states:
            if _is_global_rank_zero():
                print(f"ERROR: Some states failed: {failed_states}", flush=True)
            sys.exit(1)
    except KeyboardInterrupt:
        _destroy_distributed_process_group_if_initialized()
        if _is_global_rank_zero():
            print("Interrupted by user signal. Exiting gracefully.", flush=True)
        return
    finally:
        _INTERRUPT_CONTROLLER.restore()


if __name__ == "__main__":
    main()
