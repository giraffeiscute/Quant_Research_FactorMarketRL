"""PyTorch Lightning training entrypoint for single-loss scenario training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from portfolio_attention.lightning.run_safety import (
        _INTERRUPT_CONTROLLER,
        _configure_warning_routing,
        _destroy_distributed_process_group_if_initialized,
        _emit_lightning_console_message,
        _exception_represents_interrupt,
        _is_global_rank_zero,
        _trainer_was_interrupted,
    )
else:
    from .run_safety import (
        _INTERRUPT_CONTROLLER,
        _configure_warning_routing,
        _destroy_distributed_process_group_if_initialized,
        _emit_lightning_console_message,
        _exception_represents_interrupt,
        _is_global_rank_zero,
        _trainer_was_interrupted,
    )

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import torch

if __package__ is None or __package__ == "":
    from portfolio_attention.config import DataConfig, EvaluationConfig, ModelConfig, PathsConfig, TrainConfig
    from portfolio_attention.evaluation.metrics import (
        compute_average_turnover_from_weights as _compute_average_turnover_from_weights,
        compute_selected_stock_count_from_weights as _compute_selected_stock_count_from_weights,
    )
    from portfolio_attention.evaluation.runtime import (
        _collect_single_scenario_rolling_one_step_outputs,
        _rebuild_evaluation_window_x_stock,
        _slice_single_scenario_rolling_window_batch,
    )
    from portfolio_attention.lightning import module as _lightning_module
    from portfolio_attention.lightning import post_training as _lightning_post_training
    from portfolio_attention.lightning import validation as _lightning_validation
    from portfolio_attention.lightning.callbacks import ConfigEpochCheckpointCallback
    from portfolio_attention.lightning.datamodule import LightningTrainDataModule
    from portfolio_attention.lightning.distributed import (
        sync_bool_flag_across_ranks,
        state_transition_barrier,
    )
    from portfolio_attention.lightning.logging import (
        CSV_METRIC_DECIMAL_PLACES,
        RL_PREFERRED_METRIC_KEY_ORDER,
        RoundedCSVLogger,
        RoundedMetricsExperimentWriter,
    )
    from portfolio_attention.lightning.gradient_diagnostics import gradient_diagnostics_path
    from portfolio_attention.lightning.module import PortfolioLightningModule as _BasePortfolioLightningModule
    from portfolio_attention.lightning.validation import (
        ScenarioRollingValidationMetric,
        compute_validation_scenario_metrics,
        compute_validation_window_objective_loss,
    )
    from portfolio_attention.model.losses import build_loss
    from portfolio_attention.cli.train import (
        _parse_states_args,
        build_arg_parser,
        resolve_model_config_from_args,
        resolve_paths_config_from_args,
        resolve_evaluation_config_from_args,
        resolve_runtime_configs_from_args,
    )
    from portfolio_attention.training.engine import _run_loss_step, build_training_model
    from portfolio_attention.training.rl_engine import run_rl_policy_step
    from portfolio_attention.common.utils import save_runtime_config_artifact, set_seed
else:
    from ..config import DataConfig, EvaluationConfig, ModelConfig, PathsConfig, TrainConfig
    from ..evaluation.metrics import (
        compute_average_turnover_from_weights as _compute_average_turnover_from_weights,
        compute_selected_stock_count_from_weights as _compute_selected_stock_count_from_weights,
    )
    from ..evaluation.runtime import (
        _collect_single_scenario_rolling_one_step_outputs,
        _rebuild_evaluation_window_x_stock,
        _slice_single_scenario_rolling_window_batch,
    )
    from . import module as _lightning_module
    from . import post_training as _lightning_post_training
    from . import validation as _lightning_validation
    from .callbacks import ConfigEpochCheckpointCallback
    from .datamodule import LightningTrainDataModule
    from .distributed import (
        sync_bool_flag_across_ranks,
        state_transition_barrier,
    )
    from .logging import (
        CSV_METRIC_DECIMAL_PLACES,
        RL_PREFERRED_METRIC_KEY_ORDER,
        RoundedCSVLogger,
        RoundedMetricsExperimentWriter,
    )
    from .gradient_diagnostics import gradient_diagnostics_path
    from .module import PortfolioLightningModule as _BasePortfolioLightningModule
    from .validation import (
        ScenarioRollingValidationMetric,
        compute_validation_scenario_metrics,
        compute_validation_window_objective_loss,
    )
    from ..model.losses import build_loss
    from ..cli.train import (
        _parse_states_args,
        build_arg_parser,
        resolve_model_config_from_args,
        resolve_paths_config_from_args,
        resolve_evaluation_config_from_args,
        resolve_runtime_configs_from_args,
    )
    from ..training.engine import _run_loss_step, build_training_model
    from ..training.rl_engine import run_rl_policy_step
    from ..common.utils import save_runtime_config_artifact, set_seed


def _sync_legacy_validation_hooks() -> None:
    _lightning_validation._rebuild_evaluation_window_x_stock = _rebuild_evaluation_window_x_stock
    _lightning_validation._slice_single_scenario_rolling_window_batch = (
        _slice_single_scenario_rolling_window_batch
    )
    _lightning_validation._run_loss_step = _run_loss_step
    _lightning_validation._collect_validation_rolling_outputs = (
        _collect_single_scenario_rolling_one_step_outputs
    )
    _lightning_validation._compute_validation_window_objective_loss = (
        _compute_validation_window_objective_loss
    )
    _lightning_validation.build_loss = build_loss


def _compute_validation_window_objective_loss(**kwargs):
    _sync_legacy_validation_hooks()
    return compute_validation_window_objective_loss(**kwargs)


def _compute_validation_scenario_metrics(**kwargs):
    _sync_legacy_validation_hooks()
    return compute_validation_scenario_metrics(**kwargs)


def _sync_legacy_module_hooks() -> None:
    _sync_legacy_validation_hooks()
    _lightning_module.build_training_model = build_training_model
    _lightning_module._run_loss_step = _run_loss_step
    _lightning_module._collect_single_scenario_rolling_one_step_outputs = (
        _collect_single_scenario_rolling_one_step_outputs
    )
    _lightning_module.build_loss = build_loss
    _lightning_module.compute_validation_window_objective_loss = _compute_validation_window_objective_loss
    _lightning_module.compute_validation_scenario_metrics = _compute_validation_scenario_metrics
    _lightning_module.compute_selected_stock_count_from_weights = _compute_selected_stock_count_from_weights
    _lightning_module.compute_average_turnover_from_weights = _compute_average_turnover_from_weights
    _lightning_module.run_rl_policy_step = run_rl_policy_step


class PortfolioLightningModule(_BasePortfolioLightningModule):
    """Backward-compatible facade for the split LightningModule implementation."""

    def __init__(self, *args, **kwargs) -> None:
        _sync_legacy_module_hooks()
        super().__init__(*args, **kwargs)

    def training_step(self, *args, **kwargs):
        _sync_legacy_module_hooks()
        return super().training_step(*args, **kwargs)

    def validation_step(self, *args, **kwargs):
        _sync_legacy_module_hooks()
        return super().validation_step(*args, **kwargs)


def _state_transition_barrier(*, trainer: pl.Trainer, barrier_name: str) -> None:
    state_transition_barrier(trainer=trainer, barrier_name=barrier_name)


def _sync_bool_flag_across_ranks(*, trainer: pl.Trainer, flag: bool) -> bool:
    return sync_bool_flag_across_ranks(trainer=trainer, flag=flag)


def _sync_legacy_post_training_hooks() -> None:
    _lightning_post_training._state_transition_barrier = _state_transition_barrier
    _lightning_post_training._sync_bool_flag_across_ranks = globals()["_sync_bool_flag_across_ranks"]
    _lightning_post_training._emit_lightning_console_message = _emit_lightning_console_message
    _lightning_post_training._INTERRUPT_CONTROLLER = _INTERRUPT_CONTROLLER


def _run_post_training_holdout_after_fit(**kwargs) -> None:
    _sync_legacy_post_training_hooks()
    _lightning_post_training.run_post_training_holdout_after_fit(**kwargs)


def _run_post_training_holdout_after_fit_with_barriers(**kwargs) -> None:
    _sync_legacy_post_training_hooks()
    _lightning_post_training.run_post_training_holdout_after_fit_with_barriers(**kwargs)


def _build_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser()
    parser.description = "Run single-loss Lightning training for portfolio_attention."
    parser.prog = "python -m portfolio_attention.cli.lightning_train"
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
        raise ValueError(
            "portfolio_attention.cli.lightning_train only supports a single loss per invocation; --parallel must be 1."
        )

    unsupported_losses = args_dict.get("losses")
    if unsupported_losses is not None:
        raise ValueError("portfolio_attention.cli.lightning_train only supports --loss, not --losses.")

    requested_device = args_dict.get("device")
    if requested_device is not None:
        normalized_device = str(requested_device).strip().lower()
        if normalized_device not in {"auto", "cuda"}:
            raise ValueError(
                "portfolio_attention.cli.lightning_train always runs Lightning with accelerator='gpu'; --device must be 'auto' or 'cuda'."
            )

    if int(getattr(args, "devices", 0)) <= 0:
        raise ValueError("--devices must be positive.")


def _build_state_args(args: argparse.Namespace, state: str) -> argparse.Namespace:
    state_args_dict = vars(args).copy()
    state_args_dict["state"] = state
    return argparse.Namespace(**state_args_dict)


def _strip_wrapping_quotes(value: str) -> str:
    return value.strip().strip("\"'“”‘’")


def _parse_post_train_from_entries(raw_value: object) -> list[str]:
    if isinstance(raw_value, (list, tuple)):
        candidates = [str(item) for item in raw_value]
    else:
        candidates = [str(raw_value)]

    parsed: list[str] = []
    for candidate in candidates:
        for token in candidate.split(","):
            normalized = _strip_wrapping_quotes(token)
            if normalized:
                parsed.append(normalized)

    return parsed


def _resolve_post_train_from_overrides_by_state(
    args: argparse.Namespace,
    states_to_run: list[str],
) -> dict[str, str] | None:
    args_dict = vars(args)
    if "post_train_from" not in args_dict:
        return None

    parsed_entries = _parse_post_train_from_entries(args_dict["post_train_from"])
    if not parsed_entries:
        raise ValueError(
            "--post-train-from was provided but no valid checkpoint paths were parsed."
        )

    if len(parsed_entries) == 1:
        shared_path = parsed_entries[0]
        return {state: shared_path for state in states_to_run}

    if len(parsed_entries) != len(states_to_run):
        raise ValueError(
            "When multiple --post-train-from checkpoints are provided, the number of "
            f"checkpoints must match the number of states. states={states_to_run}, "
            f"num_states={len(states_to_run)}, num_checkpoints={len(parsed_entries)}."
        )

    return {
        state: checkpoint_path
        for state, checkpoint_path in zip(states_to_run, parsed_entries)
    }


def _resolve_single_state_runtime(
    args: argparse.Namespace,
) -> tuple[PathsConfig, DataConfig, TrainConfig, EvaluationConfig, ModelConfig]:
    _INTERRUPT_CONTROLLER.raise_if_interrupted()
    paths = resolve_paths_config_from_args(args)
    data_config, train_config = resolve_runtime_configs_from_args(args)
    evaluation_config = resolve_evaluation_config_from_args(args)
    _configure_warning_routing(state=data_config.state, paths=paths)
    model_config = resolve_model_config_from_args(args)

    if not train_config.loss_name:
        raise ValueError("portfolio_attention.cli.lightning_train requires a single --loss.")
    save_runtime_config_artifact(
        paths=paths,
        data_config=data_config,
        model_config=model_config,
        train_config=train_config,
    )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "portfolio_attention.cli.lightning_train requires CUDA because the Trainer is configured with accelerator='gpu'."
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
        evaluation_transaction_cost_rate=float(evaluation_config.evaluation_transaction_cost_rate),
        gradient_diagnostics_path=gradient_diagnostics_path(
            outputs_dir=paths.outputs_dir,
            state=data_config.state,
            loss_name=train_config.loss_name,
        ),
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(paths.checkpoints_dir),
        filename=(
            f"{data_config.state}_{train_config.loss_name}"
            + "-epoch{epoch:03d}-val_window{val_loss_window:.8f}"
        ),
        monitor="val_loss_window",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss_window",
        mode="min",
        patience=int(train_config.early_stopping_patience),
        check_on_train_epoch_end=False,
    )
    config_epoch_checkpoint_callback = ConfigEpochCheckpointCallback(
        paths=paths,
        state=data_config.state,
        train_config=train_config,
    )
    callbacks = [checkpoint_callback, early_stopping_callback, config_epoch_checkpoint_callback]
    if bool(train_config.enable_lr_warmup_decay):
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    rl_enabled = bool(train_config.rl_training.enabled)
    csv_logger = RoundedCSVLogger(
        save_dir=str(paths.outputs_dir),
        name="lightning_logs",
        version=f"{data_config.state}_{train_config.loss_name}",
        metrics_filename=("RL_metrics.csv" if rl_enabled else "metrics.csv"),
        preferred_metric_key_order=(
            RL_PREFERRED_METRIC_KEY_ORDER if rl_enabled else None
        ),
        excluded_metric_keys={"val_loss_window"} if rl_enabled else None,
    )
    trainer_logger = csv_logger
    if rl_enabled:
        wandb_logger = WandbLogger(
            save_dir=str(paths.outputs_dir),
            project="portfolio-attention-rl",
            name=f"{data_config.state}_{train_config.loss_name}",
            offline=False,
            log_model=False,
        )
        wandb_logger.experiment.define_metric("*", step_metric="epoch")
        trainer_logger = [csv_logger, wandb_logger]

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=int(args.devices),
        num_nodes=1,
        strategy="ddp" if int(args.devices) > 1 else "auto",
        max_epochs=int(train_config.num_epochs),
        gradient_clip_val=float(train_config.grad_clip_norm),
        callbacks=callbacks,
        logger=trainer_logger,
        default_root_dir=str(paths.outputs_dir),
        enable_progress_bar=True,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )
    return model, trainer, config_epoch_checkpoint_callback


def _finish_wandb_run_if_needed(train_config: TrainConfig) -> None:
    if not bool(train_config.rl_training.enabled):
        return
    import wandb

    if wandb.run is not None:
        wandb.finish()


def _resolve_post_train_checkpoint_path(train_config: TrainConfig) -> Path | None:
    if train_config.post_train_from is None:
        return None
    return Path(train_config.post_train_from).expanduser().resolve()


def _extract_model_state_dict_from_lightning_checkpoint(
    checkpoint: dict[str, object],
    *,
    checkpoint_path: Path,
) -> dict[str, torch.Tensor]:
    raw_state_dict = checkpoint.get("state_dict")
    if not isinstance(raw_state_dict, dict):
        raise ValueError(
            "post_train_from checkpoint is missing dict payload 'state_dict': "
            f"{checkpoint_path}"
        )

    model_state_dict: dict[str, torch.Tensor] = {}
    for key, value in raw_state_dict.items():
        if not isinstance(key, str) or not key.startswith("model."):
            continue
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                "post_train_from checkpoint contains non-tensor model parameter "
                f"for key {key!r}: {type(value).__name__}."
            )
        model_state_dict[key[len("model.") :]] = value

    if not model_state_dict:
        raise ValueError(
            "post_train_from checkpoint did not contain any model-prefixed parameters "
            f"under 'state_dict': {checkpoint_path}"
        )
    return model_state_dict


def _load_post_training_model_weights(
    *,
    model: PortfolioLightningModule,
    checkpoint_path: Path,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(
            "post_train_from checkpoint must contain a dict payload: "
            f"{checkpoint_path}"
        )
    model_state_dict = _extract_model_state_dict_from_lightning_checkpoint(
        checkpoint,
        checkpoint_path=checkpoint_path,
    )
    try:
        model.model.load_state_dict(model_state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load post_train_from model weights with strict=True from "
            f"{checkpoint_path}: {exc}"
        ) from exc


def _run_single_state(args: argparse.Namespace) -> None:
    paths, data_config, train_config, evaluation_config, model_config = _resolve_single_state_runtime(args)

    try:
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
        resolved_post_train_checkpoint = _resolve_post_train_checkpoint_path(train_config)
        if resolved_post_train_checkpoint is not None:
            _emit_lightning_console_message(
                "Loading post-training model weights: "
                f"post_train_from={train_config.post_train_from} "
                f"resolved_post_train_checkpoint={resolved_post_train_checkpoint}"
            )
            _INTERRUPT_CONTROLLER.raise_if_interrupted()
            _load_post_training_model_weights(
                model=model,
                checkpoint_path=resolved_post_train_checkpoint,
            )
        _emit_lightning_console_message("Starting trainer.fit().")
        _INTERRUPT_CONTROLLER.raise_if_interrupted()
        try:
            trainer.fit(
                model=model,
                datamodule=datamodule,
            )
        except BaseException as exc:
            if _INTERRUPT_CONTROLLER.interrupted or _exception_represents_interrupt(exc):
                raise KeyboardInterrupt("Trainer interrupted by user signal.") from exc
            raise
        if _trainer_was_interrupted(trainer):
            raise KeyboardInterrupt("Trainer interrupted by user signal.")
        _INTERRUPT_CONTROLLER.raise_if_interrupted()
        _run_post_training_holdout_after_fit_with_barriers(
            trainer=trainer,
            checkpoint_callback=config_epoch_checkpoint_callback,
            paths=paths,
            data_config=data_config,
            model_config=model_config,
            train_config=train_config,
            datamodule=datamodule,
        )
    finally:
        _finish_wandb_run_if_needed(train_config)


def _run_states_sequentially(args: argparse.Namespace, states_to_run: list[str]) -> list[str]:
    failed_states: list[str] = []
    total_states = len(states_to_run)
    post_train_from_overrides = _resolve_post_train_from_overrides_by_state(
        args,
        states_to_run,
    )
    for index, state in enumerate(states_to_run, start=1):
        _INTERRUPT_CONTROLLER.raise_if_interrupted()
        state_args = _build_state_args(args, state)
        if post_train_from_overrides is not None:
            setattr(state_args, "post_train_from", post_train_from_overrides[state])
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
