"""Shared artifact path and filename rules."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import PathsConfig


MONITORING_HOLDOUT_BACKTEST_MANIFEST_SUFFIX = "_monitoring_holdout_backtest.json"
WEIGHT_TRAJECTORY_OVERVIEW_FILENAME_SUFFIX = "_weight_trajectory_overview.png"


def metrics_dir_for_state(paths: PathsConfig, state: str | None) -> Path:
    if state is None:
        return paths.metrics_dir
    return paths.get_state_metrics_dir(state)


def train_metrics_path(
    paths: PathsConfig,
    loss_name: str,
    *,
    state: str | None = None,
) -> Path:
    return metrics_dir_for_state(paths, state) / f"train_metrics_{loss_name}.json"


def evaluation_metrics_path(
    paths: PathsConfig,
    loss_name: str,
    *,
    state: str | None = None,
) -> Path:
    return metrics_dir_for_state(paths, state) / f"evaluation_metrics_{loss_name}.json"


def evaluation_per_scenario_metrics_path(
    paths: PathsConfig,
    loss_name: str,
    *,
    state: str | None = None,
) -> Path:
    return metrics_dir_for_state(paths, state) / f"evaluation_metrics_{loss_name}_per_scenario.csv"


def candidate_train_metrics_paths(
    paths: PathsConfig,
    loss_name: str,
    *,
    state: str | None = None,
    checkpoint_path: Path | None = None,
) -> list[Path]:
    candidates: list[Path] = []
    if state is not None:
        candidates.append(train_metrics_path(paths, loss_name, state=state))
    candidates.append(train_metrics_path(paths, loss_name))
    if checkpoint_path is not None and checkpoint_path.parent.name == "checkpoints":
        checkpoint_metrics_dir = checkpoint_path.parent.parent / "metrics"
        if state is not None:
            candidates.append(checkpoint_metrics_dir / state / f"train_metrics_{loss_name}.json")
        candidates.append(checkpoint_metrics_dir / f"train_metrics_{loss_name}.json")
    return candidates


def monitoring_manifest_path(output_dir: Path, loss_name: str) -> Path:
    return output_dir / f"{loss_name}{MONITORING_HOLDOUT_BACKTEST_MANIFEST_SUFFIX}"


def monitoring_overview_path(output_dir: Path, scenario_id: str) -> Path:
    return output_dir / f"{scenario_id}{WEIGHT_TRAJECTORY_OVERVIEW_FILENAME_SUFFIX}"


def _checkpoint_name(stem: str, loss_name: str, *, state: str | None = None) -> str:
    prefix = f"{state}_" if state else ""
    if loss_name:
        return f"{prefix}{stem}_{loss_name}.pt"
    return f"{prefix}{stem}.pt"


def train_best_checkpoint_name(loss_name: str, *, state: str | None = None) -> str:
    return _checkpoint_name("train_best", loss_name, state=state)


def train_last_checkpoint_name(loss_name: str, *, state: str | None = None) -> str:
    return _checkpoint_name("train_last", loss_name, state=state)


def train_best_checkpoint_path(
    paths: PathsConfig,
    loss_name: str,
    *,
    state: str | None = None,
) -> Path:
    return paths.checkpoints_dir / train_best_checkpoint_name(loss_name, state=state)


def train_last_checkpoint_path(
    paths: PathsConfig,
    loss_name: str,
    *,
    state: str | None = None,
) -> Path:
    return paths.checkpoints_dir / train_last_checkpoint_name(loss_name, state=state)


def epoch_candidate_checkpoint_path(paths: PathsConfig, loss_name: str, epoch: int) -> Path:
    return paths.checkpoints_dir / f"train_candidate_{loss_name}_epoch_{epoch}.pt"


def monitoring_epoch_checkpoint_path(
    paths: PathsConfig,
    loss_name: str,
    epoch: int,
    *,
    state: str | None = None,
) -> Path:
    prefix = f"{state}_" if state else ""
    return paths.checkpoints_dir / f"{prefix}train_monitoring_{loss_name}_epoch_{epoch}.pt"


def lightning_epoch_checkpoints_dir(paths: PathsConfig) -> Path:
    return paths.checkpoints_dir / "lightning_epoch_checkpoints"


def lightning_epoch_checkpoint_name(
    loss_name: str,
    epoch: int,
    *,
    state: str | None = None,
) -> str:
    if not loss_name:
        raise ValueError("loss_name is required for Lightning epoch checkpoint naming.")
    resolved_epoch = int(epoch)
    if resolved_epoch <= 0:
        raise ValueError(f"epoch must be positive for Lightning epoch checkpoint naming, received {resolved_epoch}.")
    prefix = f"{state}_" if state else ""
    return f"{prefix}{loss_name}-epoch{resolved_epoch:03d}.ckpt"


def lightning_epoch_checkpoint_path(
    paths: PathsConfig,
    loss_name: str,
    epoch: int,
    *,
    state: str | None = None,
) -> Path:
    return lightning_epoch_checkpoints_dir(paths) / lightning_epoch_checkpoint_name(
        loss_name,
        epoch,
        state=state,
    )
