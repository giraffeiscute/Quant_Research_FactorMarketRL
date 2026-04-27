"""Path helpers for portfolio_attention config objects."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import PathsConfig


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def repo_root() -> Path:
    return project_root().parent


def default_scenario_dir(state: str = "bear") -> Path:
    return repo_root() / "toy_ff_generator" / "outputs" / "data v3" / str(state)


def outputs_dir(paths: PathsConfig) -> Path:
    return paths.output_root or paths.project_dir / "outputs"


def checkpoints_dir(paths: PathsConfig) -> Path:
    return outputs_dir(paths) / "checkpoints"


def metrics_dir(paths: PathsConfig) -> Path:
    return outputs_dir(paths) / "metrics"


def state_metrics_dir(paths: PathsConfig, state: str) -> Path:
    return metrics_dir(paths) / str(state)


def logs_dir(paths: PathsConfig) -> Path:
    return outputs_dir(paths) / "logs"


def state_logs_dir(paths: PathsConfig, state: str) -> Path:
    return logs_dir(paths) / str(state)


def predictions_dir(paths: PathsConfig) -> Path:
    return outputs_dir(paths) / "predictions"


def status_dir(paths: PathsConfig) -> Path:
    return outputs_dir(paths) / "status"


def state_predictions_dir(paths: PathsConfig, state: str) -> Path:
    return predictions_dir(paths) / state


def scenario_predictions_dir(paths: PathsConfig, state_id: str) -> Path:
    return state_predictions_dir(paths, state_id.split("_")[0])
