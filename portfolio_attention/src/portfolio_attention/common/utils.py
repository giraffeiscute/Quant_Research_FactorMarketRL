"""Utility helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import DataConfig, EvaluationConfig, ModelConfig, PathsConfig, TrainConfig


DEFAULT_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python / NumPy / PyTorch and enable deterministic execution when requested."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic and not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = DEFAULT_CUBLAS_WORKSPACE_CONFIG

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False

        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_determinism_status(device: torch.device | None = None, seed: int | None = None) -> dict[str, Any]:
    """Return a structured snapshot of reproducibility-related runtime settings."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    status: dict[str, Any] = {
        "seed": seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        status["cuda_matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
    if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
        status["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)

    if torch.cuda.is_available():
        current_index = device.index if device.type == "cuda" and device.index is not None else torch.cuda.current_device()
        status["cuda_device_count"] = torch.cuda.device_count()
        status["cuda_current_device"] = int(current_index)
        status["cuda_device_name"] = torch.cuda.get_device_name(current_index)
    else:
        status["cuda_device_count"] = 0
        status["cuda_current_device"] = None
        status["cuda_device_name"] = None

    return status


def format_determinism_status(status: dict[str, Any]) -> str:
    ordered_keys = [
        "seed",
        "device",
        "torch_version",
        "cuda_available",
        "cuda_version",
        "cuda_device_count",
        "cuda_current_device",
        "cuda_device_name",
        "cudnn_version",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "deterministic_algorithms",
        "warn_only",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
        "pythonhashseed",
        "cublas_workspace_config",
    ]
    parts = []
    for key in ordered_keys:
        if key in status:
            parts.append(f"{key}={status[key]}")
    return "Determinism status | " + " | ".join(parts)


def resolve_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def ensure_output_dirs(paths: PathsConfig) -> None:
    for path in (
        paths.outputs_dir,
        paths.checkpoints_dir,
        paths.metrics_dir,
        paths.logs_dir,
        paths.predictions_dir,
        paths.status_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def serialize_config_dataclass(config: object) -> dict[str, Any]:
    if not is_dataclass(config):
        raise TypeError("serialize_config_dataclass requires a dataclass instance.")
    serialized = asdict(config)
    for key, value in list(serialized.items()):
        if isinstance(value, Path):
            serialized[key] = str(value)
    return serialized


def runtime_config_output_path(paths: PathsConfig, state: str, loss_name: str) -> Path:
    normalized_loss = str(loss_name).strip().lower() or "unknown"
    return paths.get_state_predictions_dir(state) / f"runtime_config_{normalized_loss}.json"


def save_runtime_config_artifact(
    *,
    paths: PathsConfig,
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    evaluation_config: EvaluationConfig | None = None,
) -> Path:
    payload = {
        "data_config": serialize_config_dataclass(data_config),
        "model_config": serialize_config_dataclass(model_config),
        "train_config": serialize_config_dataclass(train_config),
        "evaluation_config": serialize_config_dataclass(evaluation_config or EvaluationConfig()),
    }
    output_path = runtime_config_output_path(paths, data_config.state, train_config.loss_name)
    save_json(payload, output_path)
    return output_path


def apply_score_mask(values: torch.Tensor, score_mask: torch.Tensor) -> torch.Tensor:
    if values.ndim < 2:
        raise ValueError("values must have shape [batch, time, ...].")
    if score_mask.ndim != 2:
        raise ValueError("score_mask must have shape [batch, time].")
    if tuple(score_mask.shape) != tuple(values.shape[:2]):
        raise ValueError(
            "score_mask must match the leading [batch, time] dimensions of values. "
            f"Received values={tuple(values.shape)} score_mask={tuple(score_mask.shape)}."
        )

    score_counts = score_mask.sum(dim=1)
    if score_counts.numel() == 0:
        raise ValueError("score_mask must contain at least one scenario.")
    if not torch.all(score_counts == score_counts[0]):
        raise ValueError("All scenarios in a batch must share the same number of scored time steps.")

    scored_steps = int(score_counts[0].item())
    if scored_steps <= 0:
        raise ValueError("score_mask must select at least one time step.")

    expanded_mask = score_mask
    while expanded_mask.ndim < values.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)
    expanded_mask = expanded_mask.expand_as(values)

    tail_shape = tuple(values.shape[2:])
    masked = torch.masked_select(values, expanded_mask)
    return masked.view(values.shape[0], scored_steps, *tail_shape)


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")
