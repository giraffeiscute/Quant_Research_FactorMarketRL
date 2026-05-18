"""Post-training holdout monitoring entrypoint for Lightning checkpoints."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
from typing import Any, Callable

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from portfolio_attention.artifact import paths as artifact_paths
    from portfolio_attention.config import EvaluationConfig
    from portfolio_attention.evaluation.artifacts import build_per_scenario_payload
    from portfolio_attention.evaluation.monitoring import (
        run_monitoring_holdout_backtest as _run_monitoring_holdout_backtest_impl,
        run_monitoring_holdout_backtest_from_per_scenario_payloads as _run_monitoring_holdout_backtest_from_per_scenario_payloads_impl,
    )
    from portfolio_attention.evaluation.runtime import (
        EVALUATION_PRICE_ANCHOR_MODE_PER_WINDOW,
        ROLLING_ONE_STEP_EVALUATION_MODE,
        ROLLING_ONE_STEP_HORIZON_DAYS,
        ROLLING_ONE_STEP_STRIDE_DAYS,
        _collect_single_scenario_rolling_one_step_outputs,
    )
    from portfolio_attention.lightning.train import (
        LightningTrainDataModule,
        PortfolioLightningModule,
        _configure_warning_routing,
    )
    from portfolio_attention.lightning.distributed import sync_bool_flag_across_initialized_ranks
    from portfolio_attention.lightning.run_safety import (
        GracefulInterruptController,
        _destroy_distributed_process_group_if_initialized,
        _exception_represents_interrupt,
    )
    from portfolio_attention.cli.cuda_devices import resolve_holdout_cuda_gpu_ids
    from portfolio_attention.training.monitoring import resolve_monitoring_holdout_backtest_epochs
else:
    from ..artifact import paths as artifact_paths
    from ..config import EvaluationConfig
    from ..evaluation.artifacts import build_per_scenario_payload
    from ..evaluation.monitoring import (
        run_monitoring_holdout_backtest as _run_monitoring_holdout_backtest_impl,
        run_monitoring_holdout_backtest_from_per_scenario_payloads as _run_monitoring_holdout_backtest_from_per_scenario_payloads_impl,
    )
    from ..evaluation.runtime import (
        EVALUATION_PRICE_ANCHOR_MODE_PER_WINDOW,
        ROLLING_ONE_STEP_EVALUATION_MODE,
        ROLLING_ONE_STEP_HORIZON_DAYS,
        ROLLING_ONE_STEP_STRIDE_DAYS,
        _collect_single_scenario_rolling_one_step_outputs,
    )
    from .train import (
        LightningTrainDataModule,
        PortfolioLightningModule,
        _configure_warning_routing,
    )
    from .distributed import sync_bool_flag_across_initialized_ranks
    from .run_safety import (
        GracefulInterruptController,
        _destroy_distributed_process_group_if_initialized,
        _exception_represents_interrupt,
    )
    from ..cli.cuda_devices import resolve_holdout_cuda_gpu_ids
    from ..training.monitoring import resolve_monitoring_holdout_backtest_epochs

def _resolve_env_global_rank() -> int | None:
    raw_rank = os.environ.get("RANK")
    if raw_rank is not None:
        try:
            return int(raw_rank)
        except ValueError:
            return None
    raw_local_rank = os.environ.get("LOCAL_RANK")
    if raw_local_rank is not None:
        try:
            return int(raw_local_rank)
        except ValueError:
            return None
    return None


def _is_global_rank_zero_process() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank()) == 0
    env_rank = _resolve_env_global_rank()
    if env_rank is None:
        return True
    return int(env_rank) == 0


def _emit_holdout_console_message(message: str, *, rank_zero_only: bool = True) -> None:
    if rank_zero_only and not _is_global_rank_zero_process():
        return
    print(f"[portfolio_attention.cli.holdout_test] {message}", flush=True)


_INTERRUPT_CONTROLLER = GracefulInterruptController()
_LEGACY_STOCK_FFN_MISSING_KEYS = frozenset(
    {
        "model.stock_ffn.0.weight",
        "model.stock_ffn.0.bias",
        "model.stock_ffn.3.weight",
        "model.stock_ffn.3.bias",
    }
)


def _extract_missing_state_dict_keys_from_error(error_text: str) -> set[str]:
    match = re.search(
        r"Missing key\(s\) in state_dict:\s*(.*?)(?:\n\s*Unexpected key\(s\) in state_dict:|\n\s*size mismatch for|\Z)",
        error_text,
        flags=re.DOTALL,
    )
    if match is None:
        return set()
    return {value.strip() for value in re.findall(r'"([^"]+)"', match.group(1))}


def _is_legacy_stock_ffn_only_missing_error(error: Exception) -> bool:
    message = str(error)
    if "Missing key(s) in state_dict" not in message:
        return False
    if "Unexpected key(s) in state_dict" in message:
        return False
    if "size mismatch for" in message:
        return False
    missing_keys = _extract_missing_state_dict_keys_from_error(message)
    return missing_keys == _LEGACY_STOCK_FFN_MISSING_KEYS


class HoldoutPredictionModule(pl.LightningModule):
    """Prediction-only wrapper that emits legacy per-scenario payloads."""

    def __init__(
        self,
        *,
        base_lightning_module: PortfolioLightningModule,
        dataset,
        checkpoint: dict[str, Any],
        loss_name: str,
        evaluation_config: EvaluationConfig,
        interrupt_checker: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self.model = base_lightning_module.model
        self.dataset = dataset
        self.checkpoint = checkpoint
        self.loss_name = str(loss_name)
        self.evaluation_config = evaluation_config
        self.interrupt_checker = interrupt_checker

    def _raise_if_interrupted(self) -> None:
        if self.interrupt_checker is None:
            return
        self.interrupt_checker()

    def predict_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Any]:
        del batch_idx, dataloader_idx
        self._raise_if_interrupted()
        rolling_outputs = _collect_single_scenario_rolling_one_step_outputs(
            model=self.model,
            dataset=self.dataset,
            raw_batch=batch,
            device=self.device,
            lookback_days=int(self.dataset.metadata.lookback_days),
            evaluation_label="Holdout distributed rolling evaluation",
            collect_weights=True,
            interrupt_checker=self.interrupt_checker,
        )
        payload = build_per_scenario_payload(
            scenario_id=str(rolling_outputs["scenario_id"]),
            source_path=Path(str(rolling_outputs["source_path"])),
            loss_name=self.loss_name,
            checkpoint=self.checkpoint,
            context_target_time_indices=rolling_outputs["context_target_time_indices"],
            target_time_indices=rolling_outputs["scored_target_time_indices"],
            portfolio_returns=rolling_outputs["portfolio_returns"],
            turnover=rolling_outputs["turnover"],
            stock_weights=rolling_outputs["stock_weights"],
            cash_weights=rolling_outputs["cash_weights"],
            dataset=self.dataset,
            evaluation_config=self.evaluation_config,
            warmup_time_steps=int(rolling_outputs["lookback_days"]),
            evaluation_mode=ROLLING_ONE_STEP_EVALUATION_MODE,
            rolling_window_lookback_days=int(rolling_outputs["lookback_days"]),
            rolling_window_horizon_days=ROLLING_ONE_STEP_HORIZON_DAYS,
            rolling_window_stride_days=ROLLING_ONE_STEP_STRIDE_DAYS,
            num_rolling_windows=int(rolling_outputs["num_rolling_windows"]),
            evaluation_price_anchor_mode=EVALUATION_PRICE_ANCHOR_MODE_PER_WINDOW,
        )
        self._raise_if_interrupted()
        return payload


def _resolve_requested_gpu_ids(
    devices: str | int | list[int] | tuple[int, ...] | None,
    *,
    int_mode: str = "count",
) -> list[int]:
    return resolve_holdout_cuda_gpu_ids(devices, int_mode=int_mode)


def _build_prediction_trainer(
    *,
    train_config,
    requested_gpu_ids: list[int],
) -> pl.Trainer:
    requested_device_name = str(getattr(train_config, "device", "auto")).strip().lower()
    prefers_gpu = requested_device_name in {"auto", "cuda"} and torch.cuda.is_available()
    dist_initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
    dist_world_size = int(torch.distributed.get_world_size()) if dist_initialized else 1

    accelerator = "cpu"
    strategy: str = "auto"
    trainer_devices: int | list[int] = 1

    if prefers_gpu:
        accelerator = "gpu"
        available_gpus = int(torch.cuda.device_count())
        if dist_initialized and dist_world_size > 1:
            # Reuse externally launched distributed ranks (e.g., post-fit holdout in existing DDP workers).
            # Avoid re-initializing nested DDP process groups in this mode.
            trainer_devices = [int(torch.cuda.current_device())]
            strategy = "auto"
        else:
            if any(gpu_id >= available_gpus for gpu_id in requested_gpu_ids):
                raise ValueError(
                    f"Requested GPU ids={requested_gpu_ids}, but only {available_gpus} CUDA device(s) are available."
                )
            trainer_devices = list(requested_gpu_ids)
            strategy = "ddp" if len(requested_gpu_ids) > 1 else "auto"

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=trainer_devices,
        strategy=strategy,
        logger=False,
        enable_checkpointing=False,
        inference_mode=True,
        enable_progress_bar=True,
    )
    return trainer


def _flatten_prediction_outputs(predictions: Any) -> list[dict[str, Any]]:
    if predictions is None:
        return []
    if isinstance(predictions, dict):
        return [predictions]
    if isinstance(predictions, (list, tuple)):
        flattened: list[dict[str, Any]] = []
        for item in predictions:
            flattened.extend(_flatten_prediction_outputs(item))
        return flattened
    raise RuntimeError(
        "Distributed prediction returned an unexpected payload container type: "
        f"{type(predictions)!r}."
    )


def _gather_prediction_payloads(
    *,
    local_payloads: list[dict[str, Any]],
    trainer: pl.Trainer,
) -> list[dict[str, Any]]:
    dist_initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
    world_size = int(torch.distributed.get_world_size()) if dist_initialized else int(
        getattr(trainer, "world_size", 1) or 1
    )
    if world_size <= 1:
        return list(local_payloads)
    if not dist_initialized:
        if _is_global_rank_zero_process():
            return list(local_payloads)
        return []

    gathered_objects: list[Any] = [None for _ in range(world_size)]
    try:
        torch.distributed.all_gather_object(gathered_objects, list(local_payloads))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Distributed prediction result gathering failed.") from exc

    merged_payloads: list[dict[str, Any]] = []
    for rank_payloads in gathered_objects:
        if rank_payloads is None:
            continue
        if not isinstance(rank_payloads, list):
            raise RuntimeError("Distributed prediction gather produced a non-list rank payload.")
        for payload in rank_payloads:
            if not isinstance(payload, dict):
                raise RuntimeError("Distributed prediction gather produced a non-dict scenario payload.")
            merged_payloads.append(payload)
    return merged_payloads


def _order_and_validate_prediction_payloads(
    *,
    gathered_payloads: list[dict[str, Any]],
    dataset,
    expected_scenario_count: int,
) -> list[dict[str, Any]]:
    deduped_by_scenario_id: dict[str, dict[str, Any]] = {}
    for payload in gathered_payloads:
        raw_scenario_id = payload.get("scenario_id")
        if raw_scenario_id in {None, ""}:
            raise RuntimeError("Distributed prediction payload is missing scenario_id.")
        scenario_id = str(raw_scenario_id)
        if scenario_id not in deduped_by_scenario_id:
            deduped_by_scenario_id[scenario_id] = payload

    expected_order = [str(item) for item in list(dataset.metadata.test_scenarios)]
    missing_ids = [scenario_id for scenario_id in expected_order if scenario_id not in deduped_by_scenario_id]
    if missing_ids:
        raise RuntimeError(
            "Distributed prediction did not produce payloads for every holdout scenario. "
            f"Missing scenario_ids={missing_ids}."
        )

    extra_ids = sorted(
        scenario_id for scenario_id in deduped_by_scenario_id if scenario_id not in set(expected_order)
    )
    if extra_ids:
        raise RuntimeError(
            "Distributed prediction produced unexpected holdout scenario payloads. "
            f"Unexpected scenario_ids={extra_ids}."
        )

    ordered_payloads = [deduped_by_scenario_id[scenario_id] for scenario_id in expected_order]
    if len(ordered_payloads) != int(expected_scenario_count):
        raise RuntimeError(
            "Distributed prediction payload count mismatch after deduplication. "
            f"expected={int(expected_scenario_count)} actual={len(ordered_payloads)}."
        )
    return ordered_payloads


def _sync_bool_flag_across_ranks(flag: bool) -> bool:
    return sync_bool_flag_across_initialized_ranks(flag)


def _build_prediction_dataloader(datamodule: LightningTrainDataModule) -> DataLoader:
    if datamodule.test_dataset is None:
        raise RuntimeError("test_dataset is unavailable before building holdout prediction dataloader.")
    sampler = None
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = int(torch.distributed.get_world_size())
        rank = int(torch.distributed.get_rank())
        sampler = DistributedSampler(
            datamodule.test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    return DataLoader(
        datamodule.test_dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=int(getattr(datamodule, "num_workers", 0)),
        pin_memory=True,
        persistent_workers=False,
    )


def run_monitoring_holdout_backtest(
    *,
    model,
    dataset,
    holdout_dataset,
    loss_name: str,
    epoch: int,
    paths,
    device: torch.device,
    evaluation_config: EvaluationConfig | None = None,
    data_config=None,
    model_config=None,
    train_config=None,
    interrupt_checker: Callable[[], None] | None = None,
    per_scenario_payloads: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compatibility shim for monkeypatch targets during migration."""
    if per_scenario_payloads is not None:
        return _run_monitoring_holdout_backtest_from_per_scenario_payloads_impl(
            per_scenario_payloads=per_scenario_payloads,
            dataset=dataset,
            loss_name=loss_name,
            epoch=int(epoch),
            paths=paths,
            evaluation_config=evaluation_config,
            data_config=data_config,
            model_config=model_config,
            train_config=train_config,
            interrupt_checker=interrupt_checker,
        )
    return _run_monitoring_holdout_backtest_impl(
        model=model,
        dataset=dataset,
        holdout_dataset=holdout_dataset,
        loss_name=loss_name,
        epoch=int(epoch),
        paths=paths,
        device=device,
        evaluation_config=evaluation_config,
        data_config=data_config,
        model_config=model_config,
        train_config=train_config,
        interrupt_checker=interrupt_checker,
    )


def _build_parser() -> argparse.ArgumentParser:
    from portfolio_attention.cli.holdout_test import _build_parser as _cli_build_parser

    return _cli_build_parser()


def _validate_cli_args(args: argparse.Namespace) -> None:
    from portfolio_attention.cli.holdout_test import _validate_cli_args as _cli_validate_cli_args

    _cli_validate_cli_args(args)


def run_post_training_holdout(
    *,
    paths,
    data_config,
    model_config,
    train_config,
    max_epoch: int,
    devices: str | int | list[int] | tuple[int, ...] | None = None,
    evaluation_config: EvaluationConfig | None = None,
    datamodule: LightningTrainDataModule | None = None,
    interrupt_checker: Callable[[], None] | None = None,
) -> list[tuple[int, str]]:
    resolved_interrupt_checker = interrupt_checker or _INTERRUPT_CONTROLLER.raise_if_interrupted
    resolved_interrupt_checker()
    if not train_config.loss_name:
        raise ValueError("portfolio_attention.cli.holdout_test requires a single --loss.")

    configured_epochs = resolve_monitoring_holdout_backtest_epochs(
        train_config,
        max_epoch=int(max_epoch),
    )
    if not configured_epochs:
        _emit_holdout_console_message(
            "No post-training holdout epochs were selected by holdout_backtest_interval_epochs/fixed-epoch rules."
        )
        return []

    resolved_datamodule = datamodule or LightningTrainDataModule(
        data_config=data_config,
        num_workers=0,
        interrupt_checker=resolved_interrupt_checker,
    )
    _emit_holdout_console_message("Starting scenario splitting and dataset materialization.")
    resolved_interrupt_checker()
    resolved_datamodule.build_datasets()
    resolved_interrupt_checker()
    _emit_holdout_console_message("Finished scenario splitting and dataset materialization.")

    if resolved_datamodule.dataset is None:
        raise RuntimeError("Dataset build completed without populating datamodule.dataset.")
    if resolved_datamodule.test_dataset is None:
        raise RuntimeError("Dataset build completed without a holdout test split.")
    if len(resolved_datamodule.test_dataset) == 0:
        raise RuntimeError("Holdout evaluation requires a non-empty holdout test split.")

    evaluation_config = evaluation_config or EvaluationConfig()
    dataset_metadata = getattr(resolved_datamodule.dataset, "metadata", None)
    use_distributed_prediction = bool(
        dataset_metadata is not None and hasattr(dataset_metadata, "train_batch_size")
    )
    trainer = None
    holdout_dataloader = None
    checkpoint_metadata = None
    if use_distributed_prediction:
        requested_gpu_ids = _resolve_requested_gpu_ids(devices, int_mode="count")
        trainer = _build_prediction_trainer(
            train_config=train_config,
            requested_gpu_ids=requested_gpu_ids,
        )
        resolved_strategy = str(getattr(trainer, "strategy", "auto"))
        requested_device_count = len(requested_gpu_ids)
        local_num_devices = int(getattr(trainer, "num_devices", requested_device_count) or requested_device_count)
        _emit_holdout_console_message(
            "Configured distributed prediction runtime: "
            f"accelerator={trainer.accelerator.__class__.__name__} "
            f"requested_gpu_ids={requested_gpu_ids} "
            f"local_devices={local_num_devices} "
            f"strategy={resolved_strategy}"
        )
        holdout_dataloader = _build_prediction_dataloader(resolved_datamodule)
        checkpoint_metadata = {
            "train_config": {"loss_name": str(train_config.loss_name)},
            "data_config": {
                "train_batch_size": int(dataset_metadata.train_batch_size),
            },
        }
    else:
        _emit_holdout_console_message(
            "Dataset metadata unavailable; using compatibility holdout runtime without distributed prediction."
        )
    _emit_holdout_console_message(
        f"Selected holdout epochs for evaluation: {list(configured_epochs)}."
    )

    rank_is_global_zero = _is_global_rank_zero_process()
    completed_runs: list[tuple[int, str]] = []
    for epoch in configured_epochs:
        resolved_interrupt_checker()
        checkpoint_path = artifact_paths.lightning_epoch_checkpoint_path(
            paths,
            train_config.loss_name,
            int(epoch),
            state=data_config.state,
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "Configured post-training Lightning checkpoint is missing. "
                f"epoch={int(epoch)} expected_path={checkpoint_path}"
            )

        _emit_holdout_console_message(
            f"Starting distributed holdout prediction for epoch {int(epoch)}."
        )
        load_kwargs = dict(
            map_location=torch.device("cpu"),
            data_config=data_config,
            model_config=model_config,
            train_config=train_config,
            dataset=resolved_datamodule.dataset,
            stock_count_weight_threshold=float(evaluation_config.stock_count_weight_threshold),
            stock_count_min_active_days=int(evaluation_config.stock_count_min_active_days),
            evaluation_transaction_cost_rate=float(evaluation_config.evaluation_transaction_cost_rate),
        )
        try:
            lightning_module = PortfolioLightningModule.load_from_checkpoint(
                str(checkpoint_path),
                weights_only=False,
                **load_kwargs,
            )
        except Exception as exc:
            if _is_legacy_stock_ffn_only_missing_error(exc):
                _emit_holdout_console_message(
                    "Detected legacy checkpoint without model.stock_ffn weights; retrying with strict=False."
                )
                try:
                    lightning_module = PortfolioLightningModule.load_from_checkpoint(
                        str(checkpoint_path),
                        strict=False,
                        weights_only=False,
                        **load_kwargs,
                    )
                    if hasattr(lightning_module, "model") and hasattr(
                        lightning_module.model, "enable_legacy_stock_ffn_noop_for_inference"
                    ):
                        lightning_module.model.enable_legacy_stock_ffn_noop_for_inference()
                        _emit_holdout_console_message(
                            "Enabled legacy inference compatibility: treating stock_ffn as no-op."
                        )
                except Exception as fallback_exc:
                    raise RuntimeError(
                        f"Failed to load Lightning checkpoint for epoch {int(epoch)} at {checkpoint_path}. "
                        f"Original error: {fallback_exc}"
                    ) from fallback_exc
            else:
                raise RuntimeError(
                    f"Failed to load Lightning checkpoint for epoch {int(epoch)} at {checkpoint_path}. "
                    f"Original error: {exc}"
                ) from exc

        if not use_distributed_prediction:
            if not rank_is_global_zero:
                continue
            resolved_interrupt_checker()
            monitoring_backtest = run_monitoring_holdout_backtest(
                model=lightning_module.model,
                dataset=resolved_datamodule.dataset,
                holdout_dataset=resolved_datamodule.test_dataset,
                loss_name=str(train_config.loss_name),
                epoch=int(epoch),
                paths=paths,
                device=torch.device("cpu"),
                evaluation_config=evaluation_config,
                data_config=data_config,
                model_config=model_config,
                train_config=train_config,
                interrupt_checker=resolved_interrupt_checker,
            )
            output_dir = str(monitoring_backtest["holdout_backtest_output_dir"])
            completed_runs.append((int(epoch), output_dir))
            _emit_holdout_console_message(
                f"Completed holdout monitoring for epoch {int(epoch)}. output_dir={output_dir}"
            )
            continue

        if trainer is None or holdout_dataloader is None or checkpoint_metadata is None:
            raise RuntimeError("Distributed prediction runtime was not initialized.")

        epoch_exception: BaseException | None = None
        ordered_payloads: list[dict[str, Any]] | None = None
        try:
            prediction_module = HoldoutPredictionModule(
                base_lightning_module=lightning_module,
                dataset=resolved_datamodule.dataset,
                checkpoint=checkpoint_metadata,
                loss_name=str(train_config.loss_name),
                evaluation_config=evaluation_config,
                interrupt_checker=resolved_interrupt_checker,
            )
            prediction_outputs = trainer.predict(
                model=prediction_module,
                dataloaders=holdout_dataloader,
                return_predictions=True,
            )
            resolved_interrupt_checker()
            local_payloads = _flatten_prediction_outputs(prediction_outputs)
            gathered_payloads = _gather_prediction_payloads(
                local_payloads=local_payloads,
                trainer=trainer,
            )
            if rank_is_global_zero:
                _emit_holdout_console_message(
                    f"Prediction complete for epoch {int(epoch)}. "
                    f"gathered_payloads={len(gathered_payloads)}",
                )
                ordered_payloads = _order_and_validate_prediction_payloads(
                    gathered_payloads=gathered_payloads,
                    dataset=resolved_datamodule.dataset,
                    expected_scenario_count=len(resolved_datamodule.test_dataset),
                )
        except BaseException as exc:  # noqa: BLE001
            if _exception_represents_interrupt(exc):
                raise KeyboardInterrupt("Holdout monitoring interrupted by user signal.") from exc
            epoch_exception = exc

        epoch_failed = _sync_bool_flag_across_ranks(epoch_exception is not None)
        if epoch_failed:
            if epoch_exception is not None:
                raise epoch_exception
            raise RuntimeError(
                f"Distributed holdout prediction failed on another rank for epoch {int(epoch)}."
            )

        if rank_is_global_zero:
            if ordered_payloads is None:
                raise RuntimeError(
                    f"Rank 0 did not produce gathered holdout payloads for epoch {int(epoch)}."
                )
            resolved_interrupt_checker()
            _emit_holdout_console_message(
                f"Starting rank0 backtest/output writing for epoch {int(epoch)}."
            )
            monitoring_backtest = run_monitoring_holdout_backtest(
                model=lightning_module.model,
                dataset=resolved_datamodule.dataset,
                holdout_dataset=resolved_datamodule.test_dataset,
                loss_name=str(train_config.loss_name),
                epoch=int(epoch),
                paths=paths,
                device=torch.device("cpu"),
                evaluation_config=evaluation_config,
                data_config=data_config,
                model_config=model_config,
                train_config=train_config,
                interrupt_checker=resolved_interrupt_checker,
                per_scenario_payloads=ordered_payloads,
            )
            output_dir = str(monitoring_backtest["holdout_backtest_output_dir"])
            completed_runs.append((int(epoch), output_dir))
            _emit_holdout_console_message(
                f"Completed holdout monitoring for epoch {int(epoch)}. output_dir={output_dir}"
            )
    if not rank_is_global_zero:
        return []
    _emit_holdout_console_message(
        "Post-training holdout evaluation summary: "
        f"requested_epochs={len(configured_epochs)} completed_runs={len(completed_runs)}"
    )
    for epoch, output_dir in completed_runs:
        _emit_holdout_console_message(f"epoch={epoch} output_dir={output_dir}")
    return completed_runs


def main() -> None:
    from portfolio_attention.cli.holdout_test import main as _cli_main

    _cli_main()


if __name__ == "__main__":
    main()
