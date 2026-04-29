"""Resume/history helpers for scenario-mode training."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ..artifact import paths as artifact_paths
from ..artifact import run_metadata
from ..config import (
    DataConfig,
    ModelConfig,
    PathsConfig,
    TrainConfig,
)
from ..config.validation import (
    normalize_model_config_dict,
    raise_if_checkpoint_uses_legacy_stock_id_representation_type,
)
from ..data.dataset import PortfolioPanelDataset


def _serialize_config(config: object) -> dict[str, Any]:
    serialized = asdict(config)  # type: ignore[arg-type]
    for key, value in list(serialized.items()):
        if isinstance(value, Path):
            serialized[key] = str(value)
    return serialized


def _train_metrics_path(paths: PathsConfig, loss_name: str, *, state: str | None = None) -> Path:
    return artifact_paths.train_metrics_path(paths, loss_name, state=state)


def _normalize_best_epoch_selection_window(select_best_from_last_x_epochs: int) -> int:
    return max(1, int(select_best_from_last_x_epochs))


def _coerce_history_epoch(history_item: dict[str, Any]) -> int | None:
    try:
        return int(history_item.get("epoch"))
    except (TypeError, ValueError):
        return None


def _coerce_history_val_loss(history_item: dict[str, Any]) -> float | None:
    try:
        return float(history_item.get("val_loss"))
    except (TypeError, ValueError):
        return None


def _load_history_from_metrics_path(metrics_path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []
    history = payload.get("history")
    if not isinstance(history, list):
        return []
    return [dict(item) for item in history if isinstance(item, dict)]


def _align_resume_history_to_checkpoint(
    history: list[dict[str, Any]],
    *,
    checkpoint_epoch: int,
    checkpoint_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    aligned_history: list[dict[str, Any]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        epoch_value = _coerce_history_epoch(item)
        if epoch_value is None or int(epoch_value) > checkpoint_epoch:
            continue
        aligned_history.append(dict(item))
    if not aligned_history:
        return []

    last_history_item = aligned_history[-1]
    history_epoch = _coerce_history_epoch(last_history_item)
    if history_epoch != checkpoint_epoch:
        return []

    if run_metadata.resume_history_item_matches_checkpoint(
        last_history_item,
        checkpoint_metrics,
        history_epoch=history_epoch,
    ):
        return aligned_history
    return []


def _history_to_epoch_val_loss_records(history: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    records: list[dict[str, float | int]] = []
    for item in history:
        epoch_value = _coerce_history_epoch(item)
        val_loss_value = _coerce_history_val_loss(item)
        if epoch_value is None or val_loss_value is None:
            continue
        records.append(
            {
                "epoch": int(epoch_value),
                "val_loss": float(val_loss_value),
            }
        )
    return records


def _derive_resume_best_state(
    history: list[dict[str, Any]],
    *,
    checkpoint_epoch: int,
    selection_window: int,
    fallback_val_loss: float,
) -> tuple[int, float, float]:
    epoch_records = _history_to_epoch_val_loss_records(history)
    if not epoch_records:
        return checkpoint_epoch, fallback_val_loss, fallback_val_loss

    current_window_records = epoch_records[-_normalize_best_epoch_selection_window(selection_window) :]
    current_window_best = min(current_window_records, key=lambda record: (record["val_loss"], record["epoch"]))
    global_best = min(epoch_records, key=lambda record: (record["val_loss"], record["epoch"]))
    return (
        int(current_window_best["epoch"]),
        float(current_window_best["val_loss"]),
        float(global_best["val_loss"]),
    )


def _recompute_epochs_without_improvement(history: list[dict[str, Any]]) -> int:
    epoch_records = _history_to_epoch_val_loss_records(history)
    if not epoch_records:
        return 0

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    for record in epoch_records:
        if record["val_loss"] < best_val_loss:
            best_val_loss = record["val_loss"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
    return epochs_without_improvement


def _load_resume_history(
    *,
    paths: PathsConfig,
    loss_name: str,
    state: str | None,
    checkpoint_path: Path,
    checkpoint_epoch: int,
    checkpoint_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    metrics_paths = artifact_paths.candidate_train_metrics_paths(
        paths,
        loss_name,
        state=state,
        checkpoint_path=checkpoint_path,
    )
    seen_paths: set[Path] = set()
    for metrics_path in metrics_paths:
        resolved_path = metrics_path.resolve()
        if resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)
        history = _load_history_from_metrics_path(metrics_path)
        aligned_history = _align_resume_history_to_checkpoint(
            history,
            checkpoint_epoch=checkpoint_epoch,
            checkpoint_metrics=checkpoint_metrics,
        )
        if aligned_history:
            return aligned_history

    if not checkpoint_metrics:
        return []
    history_item = dict(checkpoint_metrics)
    history_item["epoch"] = checkpoint_epoch
    return [history_item]


def _validate_resume_checkpoint(
    *,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    dataset: PortfolioPanelDataset,
) -> None:
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Resume checkpoint is missing model_state_dict: {checkpoint_path}")
    if "optimizer_state_dict" not in checkpoint:
        raise ValueError(f"Resume checkpoint is missing optimizer_state_dict: {checkpoint_path}")

    checkpoint_train_config = checkpoint.get("train_config", {})
    if not isinstance(checkpoint_train_config, dict):
        raise ValueError(f"Resume checkpoint has invalid train_config payload: {checkpoint_path}")
    checkpoint_loss_name = str(checkpoint_train_config.get("loss_name", "")).strip().lower()
    expected_loss_name = str(train_config.loss_name).strip().lower()
    if checkpoint_loss_name != expected_loss_name:
        raise ValueError(
            "Resume checkpoint loss_name does not match the requested training loss. "
            f"checkpoint={checkpoint_loss_name!r} requested={expected_loss_name!r}"
        )

    checkpoint_model_config = checkpoint.get("model_config", {})
    if not isinstance(checkpoint_model_config, dict):
        raise ValueError(f"Resume checkpoint has invalid model_config payload: {checkpoint_path}")
    raise_if_checkpoint_uses_legacy_stock_id_representation_type(
        checkpoint_model_config,
        context=f"Resume checkpoint {checkpoint_path}",
    )
    normalized_checkpoint_model_config = normalize_model_config_dict(checkpoint_model_config)
    if normalized_checkpoint_model_config != model_config.as_dict():
        raise ValueError("Resume checkpoint model_config does not match the current model configuration.")

    checkpoint_data_config = checkpoint.get("data_config", {})
    if not isinstance(checkpoint_data_config, dict):
        raise ValueError(f"Resume checkpoint has invalid data_config payload: {checkpoint_path}")
    if "num_stocks" in checkpoint_data_config:
        raise ValueError(
            "Resume checkpoint data_config contains legacy key 'num_stocks'. "
            "Use DataConfig.sample_num_stocks for training sampling; full stock universe size "
            "is inferred from scenario data and cannot be manually specified."
        )
    current_data_config = _serialize_config(data_config)
    for key, expected_value in current_data_config.items():
        if checkpoint_data_config.get(key) != expected_value:
            raise ValueError(
                "Resume checkpoint data_config does not match the current dataset configuration. "
                f"Mismatch for {key!r}: checkpoint={checkpoint_data_config.get(key)!r} "
                f"current={expected_value!r}"
            )

    current_train_config = _serialize_config(train_config)
    ignored_train_keys = {"num_epochs", "device", "resume_from"}
    legacy_missing_train_config_defaults = {
        "enable_fixed_epoch_holdout_backtests": False,
        "turnover_penalty": 0.0,
        "turnover_penalty_norm": "l1",
        "transaction_cost_rate": 0.0,
    }
    for key, expected_value in current_train_config.items():
        if key in ignored_train_keys:
            continue
        checkpoint_value = checkpoint_train_config.get(key)
        if key not in checkpoint_train_config and key in legacy_missing_train_config_defaults:
            checkpoint_value = legacy_missing_train_config_defaults[key]
        if checkpoint_value != expected_value:
            raise ValueError(
                "Resume checkpoint train_config does not match the current training configuration. "
                f"Mismatch for {key!r}: checkpoint={checkpoint_value!r} "
                f"current={expected_value!r}"
            )

    checkpoint_metadata = checkpoint.get("metadata", {})
    if not isinstance(checkpoint_metadata, dict):
        raise ValueError(f"Resume checkpoint has invalid metadata payload: {checkpoint_path}")
    checkpoint_num_stocks = checkpoint_metadata.get("selected_num_stocks")
    if checkpoint_num_stocks is not None and int(checkpoint_num_stocks) != dataset.num_stocks:
        raise ValueError(
            "Resume checkpoint selected_num_stocks does not match the current dataset. "
            f"checkpoint={checkpoint_num_stocks} current={dataset.num_stocks}"
        )
    checkpoint_max_lookback = checkpoint.get("max_lookback")
    if checkpoint_max_lookback is None:
        checkpoint_max_lookback = checkpoint_metadata.get("max_context_time_steps")
    if checkpoint_max_lookback is not None and int(checkpoint_max_lookback) != dataset.max_time_steps:
        raise ValueError(
            "Resume checkpoint max_lookback does not match the current dataset context length. "
            f"checkpoint={checkpoint_max_lookback} current={dataset.max_time_steps}"
        )


def advance_train_loader_generator(
    *,
    generator: torch.Generator,
    train_dataset: Dataset,
    completed_epochs: int,
    shuffle_enabled: bool,
) -> None:
    if not shuffle_enabled or completed_epochs <= 0:
        return
    train_dataset_size = len(train_dataset)
    if train_dataset_size <= 1:
        return
    for _ in range(completed_epochs):
        torch.randperm(train_dataset_size, generator=generator)


def load_resume_training_state(
    *,
    paths: PathsConfig,
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    dataset: PortfolioPanelDataset,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any] | None:
    if train_config.resume_from is None:
        return None

    checkpoint_path = Path(train_config.resume_from)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Resume checkpoint must contain a dict payload: {checkpoint_path}")

    _validate_resume_checkpoint(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        data_config=data_config,
        model_config=model_config,
        train_config=train_config,
        dataset=dataset,
    )

    checkpoint_epoch = int(checkpoint.get("epoch") or 0)
    if checkpoint_epoch < 0:
        raise ValueError(f"Resume checkpoint epoch must be non-negative, received {checkpoint_epoch}.")
    if checkpoint_epoch >= int(train_config.num_epochs):
        raise ValueError(
            "Resume checkpoint epoch already reached or exceeded the requested total num_epochs. "
            f"checkpoint_epoch={checkpoint_epoch} requested_num_epochs={train_config.num_epochs}"
        )

    checkpoint_metrics = checkpoint.get("metrics", {})
    if not isinstance(checkpoint_metrics, dict):
        checkpoint_metrics = {}

    checkpoint_best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    fallback_resume_val_loss = _coerce_history_val_loss(checkpoint_metrics)
    if fallback_resume_val_loss is None:
        fallback_resume_val_loss = checkpoint_best_val_loss

    resume_history = _load_resume_history(
        paths=paths,
        loss_name=train_config.loss_name,
        state=data_config.state,
        checkpoint_path=checkpoint_path,
        checkpoint_epoch=checkpoint_epoch,
        checkpoint_metrics=checkpoint_metrics,
    )
    current_window_best_epoch, current_window_best_val_loss, global_best_val_loss = _derive_resume_best_state(
        resume_history,
        checkpoint_epoch=checkpoint_epoch,
        selection_window=train_config.select_best_from_last_x_epochs,
        fallback_val_loss=fallback_resume_val_loss,
    )
    try:
        epochs_without_improvement = int(checkpoint_metrics["epochs_without_improvement"])
    except (KeyError, TypeError, ValueError):
        epochs_without_improvement = _recompute_epochs_without_improvement(resume_history)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": checkpoint_epoch,
        "next_epoch": checkpoint_epoch + 1,
        "best_val_loss": current_window_best_val_loss,
        "current_window_best_epoch": current_window_best_epoch,
        "current_window_best_val_loss": current_window_best_val_loss,
        "global_best_val_loss": global_best_val_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "history": resume_history,
    }
