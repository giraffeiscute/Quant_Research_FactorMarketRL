"""Canonical CLI entrypoint for Lightning training."""

from __future__ import annotations

import argparse
import sys

import portfolio_attention.lightning.train as lightning_train
from portfolio_attention.cli.cuda_devices import resolve_lightning_cuda_devices
from portfolio_attention.cli.train import _parse_states_args, build_arg_parser


def _build_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser()
    parser.description = "Run single-loss Lightning training for portfolio_attention."
    parser.prog = "python -m portfolio_attention.cli.lightning_train"
    parser.add_argument(
        "--devices",
        type=str,
        default="0",
        help="Local GPU ids for Lightning training (for example: '0' or '0,1').",
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

    resolve_lightning_cuda_devices(getattr(args, "devices", "1"))


def main() -> None:
    lightning_train._INTERRUPT_CONTROLLER.install()
    try:
        parser = _build_parser()
        args = parser.parse_args()
        _validate_cli_args(args)

        states_to_run = _parse_states_args(args)

        failed_states = lightning_train._run_states_sequentially(args, states_to_run)
        if failed_states:
            if lightning_train._is_global_rank_zero():
                print(f"ERROR: Some states failed: {failed_states}", flush=True)
            sys.exit(1)
    except KeyboardInterrupt:
        lightning_train._destroy_distributed_process_group_if_initialized()
        if lightning_train._is_global_rank_zero():
            print("Interrupted by user signal. Exiting gracefully.", flush=True)
        return
    finally:
        lightning_train._INTERRUPT_CONTROLLER.restore()


if __name__ == "__main__":
    main()
