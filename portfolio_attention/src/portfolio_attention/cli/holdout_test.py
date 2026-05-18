"""Canonical CLI entrypoint for Lightning holdout testing."""

from __future__ import annotations

import argparse

import lightning.pytorch as pl

from portfolio_attention.cli.cuda_devices import resolve_holdout_cuda_gpu_ids
from portfolio_attention.cli.train import (
    build_arg_parser,
    resolve_model_config_from_args,
    resolve_paths_config_from_args,
    resolve_runtime_configs_from_args,
)
from portfolio_attention.common.utils import set_seed
from portfolio_attention.config import EvaluationConfig
import portfolio_attention.lightning.holdout_test as lightning_holdout_test


def _build_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser()
    parser.description = "Run post-training holdout monitoring from config-selected Lightning checkpoints."
    parser.prog = "python -m portfolio_attention.cli.holdout_test"
    if not any(getattr(action, "dest", None) == "devices" for action in parser._actions):
        parser.add_argument(
            "--devices",
            type=str,
            default="0",
            help="GPU ids for holdout prediction (for example: '0' or '0,1').",
        )
    return parser


def _validate_cli_args(args: argparse.Namespace) -> None:
    args_dict = vars(args)

    if int(args_dict.get("parallel", 1)) != 1:
        raise ValueError(
            "portfolio_attention.cli.holdout_test only supports a single loss per invocation; --parallel must be 1."
        )

    if args_dict.get("losses") is not None:
        raise ValueError("portfolio_attention.cli.holdout_test only supports --loss, not --losses.")

    if args_dict.get("states") is not None:
        raise ValueError("portfolio_attention.cli.holdout_test only supports --state, not --states.")

    resolve_holdout_cuda_gpu_ids(args_dict.get("devices", "0"), int_mode="gpu_id")


def main() -> None:
    lightning_holdout_test._INTERRUPT_CONTROLLER.install()
    try:
        parser = _build_parser()
        args = parser.parse_args()
        _validate_cli_args(args)

        paths = resolve_paths_config_from_args(args)
        data_config, train_config = resolve_runtime_configs_from_args(args)
        evaluation_config = EvaluationConfig()
        lightning_holdout_test._configure_warning_routing(state=data_config.state, paths=paths)
        model_config = resolve_model_config_from_args(args)

        set_seed(int(train_config.seed))
        pl.seed_everything(int(train_config.seed), workers=True)
        lightning_holdout_test.run_post_training_holdout(
            paths=paths,
            data_config=data_config,
            model_config=model_config,
            train_config=train_config,
            evaluation_config=evaluation_config,
            max_epoch=int(train_config.num_epochs),
            devices=str(args.devices),
            interrupt_checker=lightning_holdout_test._INTERRUPT_CONTROLLER.raise_if_interrupted,
        )
    except KeyboardInterrupt:
        lightning_holdout_test._destroy_distributed_process_group_if_initialized()
        lightning_holdout_test._emit_holdout_console_message("Interrupted by user signal. Exiting gracefully.")
        return
    finally:
        lightning_holdout_test._INTERRUPT_CONTROLLER.restore()


if __name__ == "__main__":
    main()
