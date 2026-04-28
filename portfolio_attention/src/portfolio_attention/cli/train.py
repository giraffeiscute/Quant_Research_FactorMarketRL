"""Canonical training CLI entrypoint and launch orchestration."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

import torch
from torch.utils.data import Dataset

from ..config import DataConfig, ModelConfig, PathsConfig, TrainConfig, default_scenario_dir
from ..config.validation import (
    validated_data_config,
    validated_model_config,
    validated_train_config,
)
from ..data.dataset import PortfolioPanelDataset
from portfolio_attention.cli.evaluate_rebuild import (
    cleanup_monitoring_holdout_backtest_artifacts,
    cleanup_multi_loss_weight_trajectory_overviews,
    rebuild_monitoring_holdout_backtest_overviews,
    rebuild_multi_loss_weight_trajectory_overviews,
)
from ..evaluation.shared import SCENARIO_FILENAME_PATTERN
from ..training.engine import _TRAINING_INITIALIZATION_LOCK, _log_reproducibility_status
from ..training.orchestration import _run_epoch_training_with_datasets, run_training
from ..training.status import (
    SHARED_DASHBOARD_REFRESH_INTERVAL_SECONDS,
    build_dataset_progress_callback,
    build_failure_summary,
    clear_cached_training_status,
    console_log_path_for_loss,
    dataset_progress_message,
    load_training_status,
    log_path_for_loss,
    monitor_multi_loss_dashboard,
    status_path_for_loss,
    write_training_status,
)
from ..common.utils import ensure_output_dirs, resolve_device


TERMINAL_SUMMARY_KEYS = [
    "state",
    "loss_name",
    "mean_final_return",
    "std_final_return",
    "median_final_return",
    "worst_scenario_final_return",
    "best_scenario_final_return",
    "best_scenario_id",
    "best_epoch",
    "best_val_loss",
]
VALID_STATES = ("bear", "neutral", "bull")
DEFAULT_LOSSES = ["return", "sharpe", "dsr", "sortino", "mdd", "cvar"]


def _build_terminal_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    final_backtest = payload.get("final_backtest")
    if isinstance(final_backtest, dict):
        for key in TERMINAL_SUMMARY_KEYS:
            if key in final_backtest:
                summary[key] = final_backtest[key]
    for key in ("best_epoch", "best_val_loss"):
        if key in payload:
            summary[key] = payload[key]
    if "loss_name" in payload:
        summary.setdefault("loss_name", payload["loss_name"])
    return summary


def _format_terminal_summary(payload: dict[str, Any]) -> str:
    summary = _build_terminal_summary(payload)
    return "\n".join(
        f"{key}: {summary[key]}"
        for key in TERMINAL_SUMMARY_KEYS + ["best_epoch", "best_val_loss"]
        if key in summary
    )


def _format_shared_dataset_summary(
    dataset: PortfolioPanelDataset,
    *,
    train_samples: int,
    validation_scenarios: int,
    holdout_test_scenarios: int,
) -> str:
    metadata = dataset.metadata
    return "\n".join(
        [
            "Shared Dataset Summary",
            (
                "  "
                f"train_scenarios={metadata.num_train_scenarios} | "
                f"train_samples={train_samples} | "
                f"windows_per_scenario={metadata.train_windows_per_scenario}"
            ),
            (
                "  "
                f"validation_scenarios={validation_scenarios} | "
                f"holdout_test_scenarios={holdout_test_scenarios}"
            ),
            (
                "  "
                f"train_window_count={metadata.train_window_count} | "
                f"train_dataset_is_lazy_rolling={metadata.train_dataset_is_lazy_rolling} | "
                f"rolling_train_dataset_mode={metadata.rolling_train_dataset_mode}"
            ),
        ]
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run training for portfolio_attention.",
        prog="python -m portfolio_attention.cli.train",
    )
    parser.add_argument("--output-root", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--state", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--states", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--num-stocks", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--stock-id-representation-type",
        choices=["learning", "gaussian"],
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--stock-embedding-type",
        choices=["concat", "pre_temporal"],
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--stock-temporal-encoder-type",
        choices=["running_summary", "causal_self_attention"],
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--stock-cross-sectional-encoder-type",
        choices=["mlp", "self_attention"],
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--initial-allocation-mode",
        choices=["equal_weight", "random_dirichlet"],
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--initial-random-concentration",
        type=float,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--device", default=argparse.SUPPRESS)
    parser.add_argument(
        "--loss",
        default=argparse.SUPPRESS,
        choices=["return", "terminal_return", "sharpe", "dsr", "sortino", "mdd", "cvar"],
    )
    parser.add_argument("--losses", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num-epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--weight-decay", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--turnover-penalty", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--turnover-penalty-norm",
        choices=["l1", "l2"],
        default=None,
        help=(
            "Norm used for turnover regularization: "
            "l1 keeps turnover.mean(); l2 uses turnover.pow(2).mean()."
        ),
    )
    parser.add_argument("--transaction-cost-rate", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--grad-clip-norm", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--early-stopping-patience", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--resume-from", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--resume-checkpoints", type=str, default=argparse.SUPPRESS)
    parser.add_argument(
        "--select-best-from-last-x-epochs",
        type=int,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--holdout-backtest-interval-epochs",
        type=int,
        default=argparse.SUPPRESS,
    )
    return parser


def resolve_paths_config_from_args(
    args: argparse.Namespace,
    *,
    paths: PathsConfig | None = None,
) -> PathsConfig:
    resolved_paths = paths or PathsConfig()
    args_dict = vars(args)

    if "output_root" in args_dict:
        return PathsConfig(
            project_dir=resolved_paths.project_dir,
            output_root=args_dict["output_root"],
        )
    return resolved_paths


def resolve_runtime_configs_from_args(
    args: argparse.Namespace,
    *,
    data_config: DataConfig | None = None,
    train_config: TrainConfig | None = None,
) -> tuple[DataConfig, TrainConfig]:
    resolved_data_config = validated_data_config(data_config or DataConfig())
    resolved_train_config = validated_train_config(train_config or TrainConfig())
    args_dict = vars(args)

    data_overrides: dict[str, Any] = {}
    if "state" in args_dict:
        normalized_state = str(args_dict["state"]).strip().lower()
        if normalized_state not in VALID_STATES:
            raise ValueError(
                "Invalid state: "
                f"{args_dict['state']!r}. Must be one of {list(VALID_STATES)}."
            )
        data_overrides["state"] = normalized_state
        data_overrides["scenario_dir"] = default_scenario_dir(normalized_state)
    if "num_stocks" in args_dict:
        data_overrides["num_stocks"] = args_dict["num_stocks"]
    if data_overrides:
        resolved_data_config = replace(resolved_data_config, **data_overrides)

    train_overrides: dict[str, Any] = {}
    if "device" in args_dict:
        train_overrides["device"] = args_dict["device"]
    if "seed" in args_dict:
        train_overrides["seed"] = args_dict["seed"]
    if "loss" in args_dict:
        train_overrides["loss_name"] = args_dict["loss"]
    if "num_epochs" in args_dict:
        train_overrides["num_epochs"] = args_dict["num_epochs"]
    if "weight_decay" in args_dict:
        train_overrides["weight_decay"] = args_dict["weight_decay"]
    if "turnover_penalty" in args_dict:
        train_overrides["turnover_penalty"] = args_dict["turnover_penalty"]
    if args_dict.get("turnover_penalty_norm") is not None:
        train_overrides["turnover_penalty_norm"] = args_dict["turnover_penalty_norm"]
    if "transaction_cost_rate" in args_dict:
        train_overrides["transaction_cost_rate"] = args_dict["transaction_cost_rate"]
    if "grad_clip_norm" in args_dict:
        train_overrides["grad_clip_norm"] = args_dict["grad_clip_norm"]
    if "early_stopping_patience" in args_dict:
        train_overrides["early_stopping_patience"] = args_dict["early_stopping_patience"]
    if "resume_from" in args_dict:
        train_overrides["resume_from"] = args_dict["resume_from"]
    if "select_best_from_last_x_epochs" in args_dict:
        train_overrides["select_best_from_last_x_epochs"] = args_dict[
            "select_best_from_last_x_epochs"
        ]
    if "holdout_backtest_interval_epochs" in args_dict:
        train_overrides["holdout_backtest_interval_epochs"] = args_dict[
            "holdout_backtest_interval_epochs"
        ]
    if train_overrides:
        resolved_train_config = replace(resolved_train_config, **train_overrides)

    return (
        validated_data_config(resolved_data_config),
        validated_train_config(resolved_train_config),
    )


def resolve_model_config_from_args(
    args: argparse.Namespace,
    *,
    model_config: ModelConfig | None = None,
) -> ModelConfig:
    resolved_model_config = validated_model_config(model_config or ModelConfig())
    args_dict = vars(args)

    model_overrides: dict[str, Any] = {}
    if "stock_id_representation_type" in args_dict:
        model_overrides["stock_id_representation_type"] = args_dict[
            "stock_id_representation_type"
        ]
    if "stock_embedding_type" in args_dict:
        model_overrides["stock_embedding_type"] = args_dict["stock_embedding_type"]
    if "stock_temporal_encoder_type" in args_dict:
        model_overrides["stock_temporal_encoder_type"] = args_dict[
            "stock_temporal_encoder_type"
        ]
    if "stock_cross_sectional_encoder_type" in args_dict:
        model_overrides["stock_cross_sectional_encoder_type"] = args_dict[
            "stock_cross_sectional_encoder_type"
        ]
    if "initial_allocation_mode" in args_dict:
        model_overrides["initial_allocation_mode"] = args_dict["initial_allocation_mode"]
    if "initial_random_concentration" in args_dict:
        model_overrides["initial_random_concentration"] = args_dict["initial_random_concentration"]
    if model_overrides:
        resolved_model_config = replace(resolved_model_config, **model_overrides)
    return validated_model_config(resolved_model_config)


def _normalize_losses(raw_losses: list[str]) -> list[str]:
    valid_losses = {"return", "sharpe", "dsr", "sortino", "mdd", "cvar"}
    result: list[str] = []
    seen: set[str] = set()
    for loss in raw_losses:
        normalized = loss.strip()
        if not normalized:
            continue
        if normalized == "terminal_return":
            normalized = "return"
        if normalized not in valid_losses:
            raise ValueError(f"Invalid loss: '{normalized}'. Must be one of {valid_losses} or 'terminal_return'")
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _parse_losses_args(args: argparse.Namespace) -> list[str]:
    args_dict = vars(args)
    if "loss" in args_dict:
        return _normalize_losses([args_dict["loss"]])
    if "losses" in args_dict:
        raw = args_dict["losses"]
        if not raw or not raw.strip():
            raise ValueError("--losses cannot be empty string")
        return _normalize_losses(raw.split(","))
    return list(DEFAULT_LOSSES)


def _normalize_states(raw_states: list[str]) -> list[str]:
    valid_states = set(VALID_STATES)
    result: list[str] = []
    seen: set[str] = set()
    for raw_state in raw_states:
        normalized = str(raw_state).strip().lower()
        if not normalized:
            continue
        if normalized not in valid_states:
            raise ValueError(f"Invalid state: '{normalized}'. Must be one of {sorted(valid_states)}.")
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _parse_states_args(args: argparse.Namespace) -> list[str]:
    args_dict = vars(args)
    if "state" in args_dict:
        parsed = _normalize_states([str(args_dict["state"])])
        if not parsed:
            raise ValueError("--state cannot be empty string")
        return parsed
    if "states" in args_dict:
        raw = str(args_dict["states"])
        if not raw or not raw.strip():
            raise ValueError("--states cannot be empty string")
        parsed = _normalize_states(raw.split(","))
        if not parsed:
            raise ValueError("--states must include at least one valid state")
        return parsed
    return [DataConfig().state]


def _parse_resume_checkpoints_arg(args: argparse.Namespace) -> dict[str, Path] | None:
    args_dict = vars(args)
    if "resume_checkpoints" not in args_dict:
        return None

    raw_value = str(args_dict["resume_checkpoints"]).strip()
    if not raw_value:
        raise ValueError("--resume-checkpoints cannot be empty string")

    resume_checkpoints_by_loss: dict[str, Path] = {}
    for chunk in raw_value.split(","):
        entry = chunk.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(
                "--resume-checkpoints entries must use the format loss=path. "
                f"Received {entry!r}."
            )
        raw_loss_name, raw_path = entry.split("=", 1)
        normalized_losses = _normalize_losses([raw_loss_name])
        if len(normalized_losses) != 1:
            raise ValueError(f"Could not parse resume checkpoint loss name: {raw_loss_name!r}")
        checkpoint_path = raw_path.strip()
        if not checkpoint_path:
            raise ValueError(
                "--resume-checkpoints entries must provide a non-empty checkpoint path. "
                f"Received {entry!r}."
            )
        loss_name = normalized_losses[0]
        if loss_name in resume_checkpoints_by_loss:
            raise ValueError(f"Duplicate resume checkpoint entry for loss {loss_name!r}.")
        resume_checkpoints_by_loss[loss_name] = Path(checkpoint_path)

    if not resume_checkpoints_by_loss:
        raise ValueError("--resume-checkpoints cannot be empty.")
    return resume_checkpoints_by_loss


def _resolve_round_robin_gpu_ids(parallel: int) -> list[int]:
    if parallel <= 0:
        raise ValueError("parallel must be positive")
    if not torch.cuda.is_available():
        return []
    gpu_count = torch.cuda.device_count()
    if gpu_count <= 0:
        return []
    return list(range(min(4, gpu_count)))


def _build_subprocess_cmd(loss: str, state: str, device: str | None = None) -> list[str]:
    cmd = [sys.executable, "-m", "portfolio_attention.cli.train"]
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg.startswith("--losses=") or arg.startswith("--parallel="):
            continue
        if arg.startswith("--states=") or arg.startswith("--state="):
            continue
        if arg.startswith("--resume-checkpoints="):
            continue
        if arg in {"--losses", "--parallel", "--resume-checkpoints", "--states", "--state"}:
            skip_next = True
            continue
        if arg.startswith("--loss="):
            continue
        if arg == "--loss":
            skip_next = True
            continue
        if arg.startswith("--device="):
            continue
        if arg == "--device":
            skip_next = True
            continue
        cmd.append(arg)
    cmd.extend(["--state", state])
    cmd.extend(["--loss", loss])
    if device is not None:
        cmd.extend(["--device", device])
    return cmd


def _is_worker_mode() -> bool:
    return os.environ.get("PORTFOLIO_ATTENTION_CHILD") == "1"


def _preflight_runtime_config(
    data_config: DataConfig,
    train_config: TrainConfig,
    losses: list[str],
    *,
    parallel: int,
    resume_checkpoints_by_loss: dict[str, Path] | None = None,
) -> None:
    if not Path(data_config.scenario_dir).exists():
        raise FileNotFoundError(f"Scenario directory not found: {data_config.scenario_dir}")
    if train_config.num_epochs <= 0:
        raise ValueError(f"num_epochs must be positive, received {train_config.num_epochs}.")
    if not losses:
        raise ValueError("At least one loss must be requested.")
    if train_config.resume_from is not None and resume_checkpoints_by_loss:
        raise ValueError("--resume-from and --resume-checkpoints cannot be used together.")
    if train_config.resume_from is not None:
        if len(losses) != 1:
            raise ValueError("--resume-from requires running exactly one loss.")
        if not Path(train_config.resume_from).exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {train_config.resume_from}")
    if resume_checkpoints_by_loss:
        if len(losses) <= 1 or parallel <= 1:
            raise ValueError("--resume-checkpoints requires shared multi-loss mode with --parallel > 1.")
        expected_losses = set(losses)
        provided_losses = set(resume_checkpoints_by_loss)
        if provided_losses != expected_losses:
            details: list[str] = []
            missing_losses = sorted(expected_losses - provided_losses)
            unexpected_losses = sorted(provided_losses - expected_losses)
            if missing_losses:
                details.append(f"missing={missing_losses}")
            if unexpected_losses:
                details.append(f"unexpected={unexpected_losses}")
            raise ValueError(
                "--resume-checkpoints must provide exactly one checkpoint per requested loss. "
                + " ".join(details)
            )
        for loss_name, checkpoint_path in resume_checkpoints_by_loss.items():
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Resume checkpoint for loss {loss_name!r} not found: {checkpoint_path}"
                )


def _build_parallel_rolling_window_warning(data_config: DataConfig, parallel: int) -> str | None:
    if parallel <= 1:
        return None
    scenario_dir = Path(data_config.scenario_dir)
    scenario_glob = data_config.resolved_scenario_glob
    scenario_paths = sorted(scenario_dir.glob(scenario_glob))
    if not scenario_paths:
        return (
            "WARNING: rolling_window mode with parallel > 1 will build one dataset per loss worker. "
            "This can substantially increase build time and RAM usage. Start with --parallel 1 or 2."
        )
    match = SCENARIO_FILENAME_PATTERN.fullmatch(scenario_paths[0].name)
    if match is None:
        return (
            "WARNING: rolling_window mode with parallel > 1 will build one dataset per loss worker. "
            "Start with --parallel 1 or 2 if building_dataset becomes slow or memory-heavy."
        )
    total_time_steps = int(match.group("num_time_steps"))
    num_stocks = int(match.group("num_stocks"))
    train_time_steps = total_time_steps - 1
    context_time_steps = int(data_config.lookback_days) + int(data_config.rolling_horizon_days)
    if train_time_steps < context_time_steps:
        return None
    windows_per_scenario = ((train_time_steps - context_time_steps) // int(data_config.rolling_stride_days)) + 1
    train_window_count = windows_per_scenario * int(data_config.num_train_scenarios)
    approx_bytes_per_window = context_time_steps * num_stocks * ((4 * 4) + 4)
    approx_worker_gb = (approx_bytes_per_window * train_window_count) / float(1024 ** 3)
    return (
        "WARNING: rolling_window mode with parallel multi-loss training can be memory-heavy. "
        f"Estimated train_windows_per_scenario={windows_per_scenario}, "
        f"train_window_count={train_window_count}, parallel={parallel}, "
        f"approx_train_window_storage_per_worker={approx_worker_gb:.2f} GB "
        "(x_stock + r_stock only, rough estimate). "
        "If building_dataset stalls or RAM is tight, start with --parallel 1 or 2."
    )


def _should_build_multi_loss_weight_trajectory_overviews(losses: list[str]) -> bool:
    return len(losses) == 4 and len(set(losses)) == 4


def finalize_multi_loss_weight_trajectory_overviews(
    paths: PathsConfig,
    *,
    data_config: DataConfig,
    losses: list[str],
) -> list[str]:
    if not _should_build_multi_loss_weight_trajectory_overviews(losses):
        return []
    state = data_config.state
    if state is None:
        return []
    return rebuild_multi_loss_weight_trajectory_overviews(paths, state=state, loss_order=losses)


def finalize_monitoring_holdout_backtest_overviews(
    paths: PathsConfig,
    *,
    data_config: DataConfig,
    losses: list[str],
) -> list[str]:
    if not _should_build_multi_loss_weight_trajectory_overviews(losses):
        return []
    state = data_config.state
    if state is None:
        return []
    return rebuild_monitoring_holdout_backtest_overviews(paths, state=state, loss_order=losses)


def _cleanup_previous_multi_loss_artifacts(paths: PathsConfig, losses: list[str]) -> None:
    for loss in losses:
        clear_cached_training_status(paths, loss)
    if paths.status_dir.exists():
        for loss in losses:
            try:
                status_path_for_loss(paths, loss).unlink()
            except FileNotFoundError:
                pass


def _validate_shared_multi_gpu_request(
    train_config: TrainConfig,
    losses: list[str],
    parallel: int,
) -> None:
    if parallel != len(losses):
        raise ValueError(
            "Shared multi-loss GPU mode requires --parallel to match the number of requested losses. "
            f"Received parallel={parallel}, losses={len(losses)}."
        )
    requested_device = str(train_config.device).strip().lower()
    if requested_device.startswith("cpu") or requested_device.startswith("mps"):
        raise ValueError(
            "Shared multi-loss GPU mode requires CUDA devices. "
            f"Received device={train_config.device!r}."
        )
    if not torch.cuda.is_available():
        raise ValueError(
            "Shared multi-loss GPU mode requires CUDA, but torch.cuda.is_available() is False."
        )
    available_gpus = int(torch.cuda.device_count())
    if available_gpus < len(losses):
        raise ValueError(
            "Shared multi-loss GPU mode requires one GPU per loss. "
            f"Requested losses={len(losses)}, available_gpus={available_gpus}."
        )


def _resolve_shared_multi_gpu_devices(losses: list[str]) -> dict[str, str]:
    return {loss: f"cuda:{index}" for index, loss in enumerate(losses)}


def _run_shared_dataset_worker(
    *,
    loss_name: str,
    device_name: str,
    data_config: DataConfig,
    model_config: ModelConfig,
    base_train_config: TrainConfig,
    paths: PathsConfig,
    dataset: PortfolioPanelDataset,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    test_dataset: Dataset,
    resume_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    worker_train_config = replace(
        base_train_config,
        loss_name=loss_name,
        device=device_name,
        resume_from=resume_checkpoint_path,
    )
    log_path = log_path_for_loss(paths, loss_name, state=data_config.state)
    device = resolve_device(device_name)
    _log_reproducibility_status(log_path, worker_train_config, device)
    write_training_status(
        paths,
        loss_name,
        "PREPARING_DATA",
        device=str(device),
        epoch=0,
        num_epochs=worker_train_config.num_epochs,
        progress_ratio=0.0,
        phase="attaching_shared_dataset",
        message="Attaching to dataset and caches built once by the parent process.",
    )
    return _run_epoch_training_with_datasets(
        data_config=data_config,
        model_config=model_config,
        train_config=worker_train_config,
        paths=paths,
        device=device,
        log_path=log_path,
        dataset=dataset,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        dataset_ready_message="Shared dataset attached; waiting for first optimizer step.",
        initialization_lock=_TRAINING_INITIALIZATION_LOCK,
    )


def run_multi_loss_training_shared_dataset(
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    paths: PathsConfig,
    losses: list[str],
    *,
    parallel: int,
    resume_checkpoints_by_loss: dict[str, Path] | None = None,
) -> dict[str, dict[str, Any]]:
    _validate_shared_multi_gpu_request(train_config, losses, parallel)
    ensure_output_dirs(paths)
    for loss_name in losses:
        write_training_status(
            paths,
            loss_name,
            "PREPARING_DATA",
            device="shared-dataset",
            epoch=0,
            num_epochs=train_config.num_epochs,
            progress_ratio=0.0,
            phase="building_dataset",
            message=dataset_progress_message("Building shared dataset and scenario splits."),
        )
    shared_dataset_progress_callback = build_dataset_progress_callback(
        paths=paths,
        loss_names=losses,
        device="shared-dataset",
        num_epochs=train_config.num_epochs,
        phase="building_dataset",
        print_to_stdout=True,
    )
    dataset = PortfolioPanelDataset(
        data_config,
        progress_callback=shared_dataset_progress_callback,
    )
    train_dataset, validation_dataset, test_dataset = dataset.build_shared_train_validation_test_datasets()
    if len(train_dataset) == 0 or len(validation_dataset) == 0 or len(test_dataset) == 0:
        raise RuntimeError("Scenario training requires non-empty train, validation, and holdout test splits.")
    device_map = _resolve_shared_multi_gpu_devices(losses)
    print(
        _format_shared_dataset_summary(
            dataset,
            train_samples=len(train_dataset),
            validation_scenarios=len(validation_dataset),
            holdout_test_scenarios=len(test_dataset),
        ),
        flush=True,
    )
    print(
        "Worker to GPU mapping: "
        + ", ".join(f"{loss}->{device_name}" for loss, device_name in device_map.items()),
        flush=True,
    )
    metrics_by_loss: dict[str, dict[str, Any]] = {}
    futures_by_loss: dict[str, Future[dict[str, Any]]] = {}
    failure_summaries: list[str] = []
    dashboard_stop_event = threading.Event()
    dashboard_errors: list[BaseException] = []

    def _shared_dashboard_runner() -> None:
        try:
            monitor_multi_loss_dashboard(
                paths=paths,
                losses=losses,
                stop_event=dashboard_stop_event,
                prefer_cache=True,
            )
        except BaseException as exc:  # pragma: no cover
            dashboard_errors.append(exc)
            dashboard_stop_event.set()

    dashboard_thread = threading.Thread(
        target=_shared_dashboard_runner,
        name="shared-dashboard-monitor",
        daemon=True,
    )
    dashboard_thread.start()

    try:
        with ThreadPoolExecutor(max_workers=len(losses), thread_name_prefix="portfolio-loss") as executor:
            for loss_name in losses:
                futures_by_loss[loss_name] = executor.submit(
                    _run_shared_dataset_worker,
                    loss_name=loss_name,
                    device_name=device_map[loss_name],
                    data_config=data_config,
                    model_config=model_config,
                    base_train_config=train_config,
                    paths=paths,
                    dataset=dataset,
                    train_dataset=train_dataset,
                    validation_dataset=validation_dataset,
                    test_dataset=test_dataset,
                    resume_checkpoint_path=(
                        None
                        if resume_checkpoints_by_loss is None
                        else resume_checkpoints_by_loss.get(loss_name)
                    ),
                )
            pending_losses = set(losses)
            while pending_losses:
                completed_losses: list[str] = []
                for loss_name in list(pending_losses):
                    future = futures_by_loss[loss_name]
                    if not future.done():
                        continue
                    try:
                        metrics_by_loss[loss_name] = future.result()
                    except Exception:
                        failure_summaries.append(
                            build_failure_summary(
                                paths,
                                loss_name,
                                None,
                                state=data_config.state,
                            )
                        )
                    completed_losses.append(loss_name)
                for loss_name in completed_losses:
                    pending_losses.discard(loss_name)
                if pending_losses:
                    time.sleep(SHARED_DASHBOARD_REFRESH_INTERVAL_SECONDS)
    finally:
        dashboard_stop_event.set()
        dashboard_thread.join()

    if dashboard_errors:
        raise RuntimeError(f"Shared dashboard monitor failed: {dashboard_errors[0]}")
    if failure_summaries:
        raise RuntimeError("\n\n".join(failure_summaries))
    return metrics_by_loss


def _build_state_args(args: argparse.Namespace, state: str) -> argparse.Namespace:
    state_args_dict = vars(args).copy()
    state_args_dict["state"] = state
    return argparse.Namespace(**state_args_dict)


def _run_single_state_cli(
    args: argparse.Namespace,
    *,
    paths: PathsConfig,
    parallel: int,
    losses_to_run: list[str],
    resume_checkpoints_by_loss: dict[str, Path] | None,
    worker_mode: bool,
    model_config: ModelConfig,
) -> None:
    args_dict = vars(args)
    data_config, base_train_config = resolve_runtime_configs_from_args(args)

    if "loss" in args_dict:
        loss = losses_to_run[0]
        train_config = replace(base_train_config, loss_name=loss)
        _preflight_runtime_config(
            data_config,
            train_config,
            [loss],
            parallel=parallel,
            resume_checkpoints_by_loss=resume_checkpoints_by_loss,
        )
        if not worker_mode:
            print(
                f"\n>>> Running training with state: {data_config.state} | loss: {loss}",
                flush=True,
            )
        metrics = run_training(data_config, model_config, train_config, paths)
        if not worker_mode:
            print(f"--- Results for state: {data_config.state} | loss: {loss} ---", flush=True)
            print(_format_terminal_summary(metrics), flush=True)
        return

    _preflight_runtime_config(
        data_config,
        base_train_config,
        losses_to_run,
        parallel=parallel,
        resume_checkpoints_by_loss=resume_checkpoints_by_loss,
    )
    if parallel > 1:
        ensure_output_dirs(paths)
        _cleanup_previous_multi_loss_artifacts(paths, losses_to_run)
        if _should_build_multi_loss_weight_trajectory_overviews(losses_to_run):
            cleanup_multi_loss_weight_trajectory_overviews(paths, state=data_config.state)
            cleanup_monitoring_holdout_backtest_artifacts(paths, state=data_config.state)
        run_multi_loss_training_shared_dataset(
            data_config,
            model_config,
            base_train_config,
            paths,
            losses_to_run,
            parallel=parallel,
            resume_checkpoints_by_loss=resume_checkpoints_by_loss,
        )
        overview_paths = finalize_multi_loss_weight_trajectory_overviews(
            paths,
            data_config=data_config,
            losses=losses_to_run,
        )
        if overview_paths:
            print(
                f"Generated multi-loss weight trajectory overview charts: {len(overview_paths)}",
                flush=True,
            )
        monitoring_overview_paths = finalize_monitoring_holdout_backtest_overviews(
            paths,
            data_config=data_config,
            losses=losses_to_run,
        )
        if monitoring_overview_paths:
            print(
                f"Generated monitoring multi-loss weight trajectory overview charts: {len(monitoring_overview_paths)}",
                flush=True,
            )
        return

    gpu_ids = _resolve_round_robin_gpu_ids(parallel)
    ensure_output_dirs(paths)
    _cleanup_previous_multi_loss_artifacts(paths, losses_to_run)
    rolling_warning = _build_parallel_rolling_window_warning(data_config, parallel)
    if rolling_warning is not None:
        print(rolling_warning, flush=True)
    if _should_build_multi_loss_weight_trajectory_overviews(losses_to_run):
        cleanup_multi_loss_weight_trajectory_overviews(paths, state=data_config.state)
        cleanup_monitoring_holdout_backtest_artifacts(paths, state=data_config.state)
    active_processes: list[dict[str, Any]] = []
    pending_losses = list(losses_to_run)
    failed_losses: list[str] = []
    failure_summaries: list[str] = []
    launch_index = 0
    env_base = os.environ.copy()
    env_base["PORTFOLIO_ATTENTION_CHILD"] = "1"
    dashboard_stop_event = threading.Event()
    dashboard_errors: list[BaseException] = []

    def _subprocess_dashboard_runner() -> None:
        try:
            monitor_multi_loss_dashboard(
                paths=paths,
                losses=losses_to_run,
                stop_event=dashboard_stop_event,
                prefer_cache=False,
            )
        except BaseException as exc:  # pragma: no cover
            dashboard_errors.append(exc)
            dashboard_stop_event.set()

    dashboard_thread = threading.Thread(
        target=_subprocess_dashboard_runner,
        name="subprocess-dashboard-monitor",
        daemon=True,
    )
    dashboard_thread.start()
    try:
        while pending_losses or active_processes:
            while pending_losses and len(active_processes) < parallel:
                loss = pending_losses.pop(0)
                gpu_id: int | None = None
                device_arg: str | None = None
                if gpu_ids:
                    gpu_id = gpu_ids[launch_index % len(gpu_ids)]
                    device_arg = f"cuda:{gpu_id}"
                cmd = _build_subprocess_cmd(loss, data_config.state, device=device_arg)
                env = env_base.copy()
                console_log_path = console_log_path_for_loss(paths, loss, state=data_config.state)
                console_log_path.parent.mkdir(parents=True, exist_ok=True)
                console_handle = console_log_path.open("a", encoding="utf-8")
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=console_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                write_training_status(
                    paths,
                    loss,
                    "STARTING",
                    pid=process.pid,
                    device=device_arg or "cpu",
                    epoch=0,
                    num_epochs=base_train_config.num_epochs,
                    progress_ratio=0.0,
                    phase="spawned",
                    message="Worker started; waiting for dataset setup.",
                )
                active_processes.append(
                    {
                        "loss": loss,
                        "gpu_id": gpu_id,
                        "process": process,
                        "console_handle": console_handle,
                        "console_log_path": console_log_path,
                    }
                )
                launch_index += 1
                time.sleep(0.2)
            remaining_processes: list[dict[str, Any]] = []
            for item in active_processes:
                process = item["process"]
                returncode = process.poll()
                if returncode is None:
                    remaining_processes.append(item)
                    continue
                item["console_handle"].close()
                if returncode not in (0, None):
                    failed_losses.append(str(item["loss"]))
                    status_data = load_training_status(paths, str(item["loss"]))
                    if status_data.get("status") != "FAILED":
                        write_training_status(
                            paths,
                            str(item["loss"]),
                            "FAILED",
                            pid=process.pid,
                            device=status_data.get("device", item.get("device", "-")),
                            epoch=int(status_data.get("epoch", 0)),
                            num_epochs=int(status_data.get("num_epochs", base_train_config.num_epochs)),
                            progress_ratio=float(status_data.get("progress_ratio", 0.0)),
                            phase=str(status_data.get("phase", "worker_exit")),
                            message="Worker process exited with a non-zero code.",
                            error_message=f"Worker exited with code {returncode}.",
                        )
                    failure_summaries.append(
                        build_failure_summary(
                            paths,
                            str(item["loss"]),
                            int(returncode),
                            state=data_config.state,
                        )
                    )
            active_processes = remaining_processes
            time.sleep(0.5)
    finally:
        dashboard_stop_event.set()
        dashboard_thread.join()
        for item in active_processes:
            try:
                item["console_handle"].close()
            except Exception:
                pass
    if dashboard_errors:
        raise RuntimeError(f"Subprocess dashboard monitor failed: {dashboard_errors[0]}")
    if failed_losses:
        raise RuntimeError(
            "\n\n".join(failure_summaries)
            if failure_summaries
            else f"Some losses failed: {sorted(set(failed_losses))}"
        )
    overview_paths = finalize_multi_loss_weight_trajectory_overviews(
        paths,
        data_config=data_config,
        losses=losses_to_run,
    )
    if overview_paths:
        print(
            f"Generated multi-loss weight trajectory overview charts: {len(overview_paths)}",
            flush=True,
        )
    monitoring_overview_paths = finalize_monitoring_holdout_backtest_overviews(
        paths,
        data_config=data_config,
        losses=losses_to_run,
    )
    if monitoring_overview_paths:
        print(
            f"Generated monitoring multi-loss weight trajectory overview charts: {len(monitoring_overview_paths)}",
            flush=True,
        )


def _run_states_sequentially(
    args: argparse.Namespace,
    *,
    paths: PathsConfig,
    parallel: int,
    losses_to_run: list[str],
    resume_checkpoints_by_loss: dict[str, Path] | None,
    worker_mode: bool,
    model_config: ModelConfig,
    states_to_run: list[str],
) -> list[str]:
    failed_states: list[str] = []
    total_states = len(states_to_run)
    for index, state in enumerate(states_to_run, start=1):
        state_args = _build_state_args(args, state)
        if total_states > 1 and not worker_mode:
            print(f"\n=== Running state {index}/{total_states}: {state} ===", flush=True)
        try:
            _run_single_state_cli(
                state_args,
                paths=paths,
                parallel=parallel,
                losses_to_run=losses_to_run,
                resume_checkpoints_by_loss=resume_checkpoints_by_loss,
                worker_mode=worker_mode,
                model_config=model_config,
            )
        except Exception as exc:
            if total_states <= 1 or worker_mode:
                raise
            failed_states.append(state)
            print(f"ERROR: State '{state}' failed: {exc}", flush=True)
    return failed_states


def main() -> None:
    args = build_arg_parser().parse_args()
    paths = resolve_paths_config_from_args(args)
    args_dict = vars(args)
    parallel = args_dict.get("parallel", 1)
    if parallel < 1:
        raise ValueError("--parallel must be >= 1")
    losses_to_run = _parse_losses_args(args)
    states_to_run = _parse_states_args(args)
    resume_checkpoints_by_loss = _parse_resume_checkpoints_arg(args)
    worker_mode = _is_worker_mode()
    base_model_config = resolve_model_config_from_args(args)
    if len(states_to_run) > 1 and (
        "resume_from" in args_dict or resume_checkpoints_by_loss is not None
    ):
        raise ValueError(
            "Multi-state training does not support --resume-from or --resume-checkpoints. "
            "Resume one state at a time."
        )
    try:
        failed_states = _run_states_sequentially(
            args,
            paths=paths,
            parallel=parallel,
            losses_to_run=losses_to_run,
            resume_checkpoints_by_loss=resume_checkpoints_by_loss,
            worker_mode=worker_mode,
            model_config=base_model_config,
            states_to_run=states_to_run,
        )
    except Exception as exc:
        if worker_mode:
            raise
        print(f"ERROR: {exc}", flush=True)
        sys.exit(1)
    if failed_states:
        print(f"ERROR: Some states failed: {failed_states}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
