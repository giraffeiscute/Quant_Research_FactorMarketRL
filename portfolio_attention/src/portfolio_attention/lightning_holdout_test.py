"""Post-training holdout monitoring entrypoint for Lightning checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
from typing import Any

import pytorch_lightning as pl
import torch

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from portfolio_attention import artifact_paths
    from portfolio_attention.config import EvaluationConfig
    from portfolio_attention.evaluation_monitoring import run_monitoring_holdout_backtest
    from portfolio_attention.lightning_train import (
        LightningTrainDataModule,
        PortfolioLightningModule,
        _configure_warning_routing,
    )
    from portfolio_attention.train_cli import (
        build_arg_parser,
        resolve_model_config_from_args,
        resolve_paths_config_from_args,
        resolve_runtime_configs_from_args,
    )
    from portfolio_attention.train_monitoring import resolve_monitoring_holdout_backtest_epochs
    from portfolio_attention.utils import resolve_device, set_seed
else:
    from . import artifact_paths
    from .config import EvaluationConfig
    from .evaluation_monitoring import run_monitoring_holdout_backtest
    from .lightning_train import (
        LightningTrainDataModule,
        PortfolioLightningModule,
        _configure_warning_routing,
    )
    from .train_cli import (
        build_arg_parser,
        resolve_model_config_from_args,
        resolve_paths_config_from_args,
        resolve_runtime_configs_from_args,
    )
    from .train_monitoring import resolve_monitoring_holdout_backtest_epochs
    from .utils import resolve_device, set_seed

def _emit_holdout_console_message(message: str) -> None:
    print(f"[lightning_holdout_test] {message}", flush=True)


class GracefulInterruptController:
    """Capture SIGINT/SIGTERM and expose interruption checks."""

    def __init__(self) -> None:
        self._interrupted = False
        self._installed = False
        self._previous_handlers: dict[int, Any] = {}

    def install(self) -> None:
        if self._installed:
            return
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle_signal)
        self._installed = True

    def restore(self) -> None:
        if not self._installed:
            return
        for signum, previous_handler in self._previous_handlers.items():
            signal.signal(signum, previous_handler)
        self._previous_handlers.clear()
        self._installed = False

    def raise_if_interrupted(self) -> None:
        if self._interrupted:
            raise KeyboardInterrupt("Interrupt requested.")

    def _handle_signal(self, signum: int, frame: Any | None) -> None:
        del frame
        self._interrupted = True
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = str(signum)
        raise KeyboardInterrupt(f"Received {signal_name}.")


_INTERRUPT_CONTROLLER = GracefulInterruptController()


def _destroy_distributed_process_group_if_initialized() -> None:
    if not torch.distributed.is_available():
        return
    if not torch.distributed.is_initialized():
        return
    try:
        torch.distributed.destroy_process_group()
    except Exception:
        return


def _build_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser()
    parser.description = "Run post-training holdout monitoring from config-selected Lightning checkpoints."
    return parser


def _validate_cli_args(args: argparse.Namespace) -> None:
    args_dict = vars(args)

    if int(args_dict.get("parallel", 1)) != 1:
        raise ValueError(
            "lightning_holdout_test.py only supports a single loss per invocation; --parallel must be 1."
        )

    if args_dict.get("losses") is not None:
        raise ValueError("lightning_holdout_test.py only supports --loss, not --losses.")

    if args_dict.get("states") is not None:
        raise ValueError("lightning_holdout_test.py only supports --state, not --states.")

    if args_dict.get("resume_checkpoints") is not None:
        raise ValueError("lightning_holdout_test.py does not support --resume-checkpoints.")


def run_post_training_holdout(
    *,
    paths,
    data_config,
    model_config,
    train_config,
    max_epoch: int,
    datamodule: LightningTrainDataModule | None = None,
) -> list[tuple[int, str]]:
    _INTERRUPT_CONTROLLER.raise_if_interrupted()
    if not train_config.loss_name:
        raise ValueError("lightning_holdout_test.py requires a single --loss.")

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
        interrupt_checker=_INTERRUPT_CONTROLLER.raise_if_interrupted,
    )
    _emit_holdout_console_message("Starting scenario splitting and dataset materialization.")
    _INTERRUPT_CONTROLLER.raise_if_interrupted()
    resolved_datamodule.build_datasets()
    _INTERRUPT_CONTROLLER.raise_if_interrupted()
    _emit_holdout_console_message("Finished scenario splitting and dataset materialization.")

    if resolved_datamodule.dataset is None:
        raise RuntimeError("Dataset build completed without populating datamodule.dataset.")
    if resolved_datamodule.test_dataset is None:
        raise RuntimeError("Dataset build completed without a holdout test split.")
    if len(resolved_datamodule.test_dataset) == 0:
        raise RuntimeError("Holdout evaluation requires a non-empty holdout test split.")

    evaluation_config = EvaluationConfig()
    device = resolve_device(str(train_config.device))
    _emit_holdout_console_message(f"Using runtime device: {device}")

    completed_runs: list[tuple[int, str]] = []
    for epoch in configured_epochs:
        _INTERRUPT_CONTROLLER.raise_if_interrupted()
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

        _emit_holdout_console_message(f"Running holdout monitoring for epoch {int(epoch)}.")
        try:
            lightning_module = PortfolioLightningModule.load_from_checkpoint(
                str(checkpoint_path),
                map_location=device,
                data_config=data_config,
                model_config=model_config,
                train_config=train_config,
                dataset=resolved_datamodule.dataset,
                stock_count_weight_threshold=float(evaluation_config.stock_count_weight_threshold),
                stock_count_min_active_days=int(evaluation_config.stock_count_min_active_days),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Lightning checkpoint for epoch {int(epoch)} at {checkpoint_path}. "
                f"Original error: {exc}"
            ) from exc

        lightning_module.to(device)
        lightning_module.eval()

        with torch.no_grad():
            monitoring_backtest = run_monitoring_holdout_backtest(
                model=lightning_module.model,
                dataset=resolved_datamodule.dataset,
                holdout_dataset=resolved_datamodule.test_dataset,
                loss_name=train_config.loss_name,
                epoch=int(epoch),
                paths=paths,
                device=device,
                data_config=data_config,
                model_config=model_config,
                train_config=train_config,
            )
        _INTERRUPT_CONTROLLER.raise_if_interrupted()
        output_dir = str(monitoring_backtest["holdout_backtest_output_dir"])
        completed_runs.append((int(epoch), output_dir))
        _emit_holdout_console_message(
            f"Completed holdout monitoring for epoch {int(epoch)}. output_dir={output_dir}"
        )

    _emit_holdout_console_message(
        "Post-training holdout evaluation summary: "
        f"requested_epochs={len(configured_epochs)} completed_runs={len(completed_runs)}"
    )
    for epoch, output_dir in completed_runs:
        _emit_holdout_console_message(f"epoch={epoch} output_dir={output_dir}")
    return completed_runs


def main() -> None:
    _INTERRUPT_CONTROLLER.install()
    try:
        parser = _build_parser()
        args = parser.parse_args()
        _validate_cli_args(args)

        paths = resolve_paths_config_from_args(args)
        data_config, train_config = resolve_runtime_configs_from_args(args)
        _configure_warning_routing(state=data_config.state, paths=paths)
        model_config = resolve_model_config_from_args(args)

        set_seed(int(train_config.seed))
        pl.seed_everything(int(train_config.seed), workers=True)
        run_post_training_holdout(
            paths=paths,
            data_config=data_config,
            model_config=model_config,
            train_config=train_config,
            max_epoch=int(train_config.num_epochs),
        )
    except KeyboardInterrupt:
        _destroy_distributed_process_group_if_initialized()
        _emit_holdout_console_message("Interrupted by user signal. Exiting gracefully.")
        return
    finally:
        _INTERRUPT_CONTROLLER.restore()


if __name__ == "__main__":
    main()
