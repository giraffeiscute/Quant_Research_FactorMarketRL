"""General evaluation workflow orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from . import artifact_paths
from .config import (
    DataConfig,
    EvaluationConfig,
    ModelConfig,
    PathsConfig,
)
from .config_validation import (
    normalize_model_config_dict,
    raise_if_checkpoint_uses_legacy_stock_id_representation_type,
    validated_data_config,
    validated_evaluation_config,
    validated_model_config,
)
from .dataset import PortfolioPanelDataset
from .evaluation_artifacts import (
    build_holdout_summary_payload,
    cleanup_stale_prediction_artifacts,
    export_scenario_payload,
    extract_exported_train_config,
    strip_transient_scenario_tensor_fields,
)
from .evaluation_runtime import _collect_holdout_per_scenario_payloads
from .model import PortfolioAttentionModel
from .utils import ensure_output_dirs, resolve_device, save_json, set_seed


def _resolve_checkpoint_state(data_config: DataConfig) -> str | None:
    return data_config.state


def _resolve_checkpoint_path(
    *,
    paths: PathsConfig,
    data_config: DataConfig,
    checkpoint_path: Path | None,
    loss_name: str | None,
) -> Path:
    return checkpoint_path or artifact_paths.train_best_checkpoint_path(
        paths,
        loss_name or "dsr",
        state=_resolve_checkpoint_state(data_config),
    )


def _build_model_config_from_checkpoint(checkpoint: dict[str, Any]) -> ModelConfig:
    checkpoint_model_config = checkpoint.get("model_config", {})
    if not isinstance(checkpoint_model_config, dict):
        raise ValueError("Checkpoint model_config payload must be a dictionary.")
    raise_if_checkpoint_uses_legacy_stock_id_representation_type(
        checkpoint_model_config,
        context="Checkpoint model_config",
    )
    if "stock_temporal_encoder_type" not in checkpoint_model_config:
        raise ValueError(
            "Checkpoint model_config is missing 'stock_temporal_encoder_type'. "
            "This checkpoint was saved with an older architecture and is not compatible with the current model."
        )
    normalized_model_config = normalize_model_config_dict(checkpoint_model_config)
    filtered_config_dict = {
        key: value
        for key, value in normalized_model_config.items()
        if key in ModelConfig.__dataclass_fields__
    }
    return validated_model_config(ModelConfig(**filtered_config_dict))


def _build_data_config_from_checkpoint(
    checkpoint: dict[str, Any],
    *,
    fallback_data_config: DataConfig,
) -> DataConfig:
    checkpoint_data_config = checkpoint.get("data_config", {})
    if not isinstance(checkpoint_data_config, dict):
        return fallback_data_config
    filtered_config_dict = {
        key: value
        for key, value in checkpoint_data_config.items()
        if key in DataConfig.__dataclass_fields__
    }
    if not filtered_config_dict:
        return validated_data_config(fallback_data_config)

    fallback_dict = fallback_data_config.__dict__.copy()
    fallback_dict.update(filtered_config_dict)
    return validated_data_config(DataConfig(**fallback_dict))


def _validate_checkpoint_metadata(checkpoint: dict[str, Any], dataset: PortfolioPanelDataset) -> None:
    checkpoint_metadata = checkpoint.get("metadata", {})
    checkpoint_num_stocks = checkpoint_metadata.get("selected_num_stocks")
    if checkpoint_num_stocks is not None and int(checkpoint_num_stocks) != dataset.num_stocks:
        raise ValueError(
            f"Checkpoint expects selected_num_stocks={checkpoint_num_stocks}, "
            f"but the evaluation dataset provides {dataset.num_stocks} stocks."
        )


def run_evaluation(
    data_config: DataConfig,
    paths: PathsConfig,
    checkpoint_path: Path | None = None,
    device_name: str = "auto",
    evaluation_config: EvaluationConfig | None = None,
    loss_name: str | None = None,
    dataset: PortfolioPanelDataset | None = None,
    holdout_dataset: Dataset | None = None,
) -> dict[str, Any]:
    data_config = validated_data_config(data_config)
    ensure_output_dirs(paths)
    device = resolve_device(device_name)
    resolved_evaluation_config = validated_evaluation_config(evaluation_config or EvaluationConfig())
    resolved_checkpoint = _resolve_checkpoint_path(
        paths=paths,
        data_config=data_config,
        checkpoint_path=checkpoint_path,
        loss_name=loss_name,
    )
    checkpoint = torch.load(resolved_checkpoint, map_location=device, weights_only=False)
    checkpoint_train_config = checkpoint.get("train_config", {})
    checkpoint_seed = (
        checkpoint_train_config.get("seed")
        if isinstance(checkpoint_train_config, dict)
        else None
    )
    if checkpoint_seed is not None:
        set_seed(int(checkpoint_seed))
    resolved_data_config = _build_data_config_from_checkpoint(
        checkpoint,
        fallback_data_config=data_config,
    )
    resolved_dataset = dataset or PortfolioPanelDataset(resolved_data_config)
    resolved_holdout_dataset = holdout_dataset or resolved_dataset.get_split_dataset("test")
    holdout_loader = DataLoader(
        resolved_holdout_dataset,
        batch_size=1,
        shuffle=False,
    )
    _validate_checkpoint_metadata(checkpoint, resolved_dataset)

    max_lookback = checkpoint.get("max_lookback")
    if max_lookback is None:
        max_lookback = checkpoint.get("metadata", {}).get("max_context_time_steps")
    if max_lookback is None:
        max_lookback = resolved_dataset.max_time_steps
    model_config = _build_model_config_from_checkpoint(checkpoint)
    model = PortfolioAttentionModel(
        model_config,
        num_stocks=resolved_dataset.num_stocks,
        max_lookback=int(max_lookback),
        stock_temporal_attention_window=int(resolved_data_config.lookback_days),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    checkpoint_loss_name = str(checkpoint_train_config.get("loss_name", loss_name or "unknown")).lower()
    state_predictions_dir = paths.get_state_predictions_dir(resolved_dataset.state)
    state_predictions_dir.mkdir(parents=True, exist_ok=True)
    cleanup_stale_prediction_artifacts(state_predictions_dir, checkpoint_loss_name)
    legacy_holdout_dir = state_predictions_dir / "holdout_test"
    if legacy_holdout_dir.exists():
        cleanup_stale_prediction_artifacts(legacy_holdout_dir, checkpoint_loss_name)

    per_scenario_payloads = _collect_holdout_per_scenario_payloads(
        model=model,
        holdout_loader=holdout_loader,
        device=device,
        dataset=resolved_dataset,
        loss_name=checkpoint_loss_name,
        evaluation_config=resolved_evaluation_config,
        checkpoint=checkpoint,
    )
    if len(per_scenario_payloads) != len(resolved_holdout_dataset):
        raise RuntimeError(
            "Holdout evaluation did not produce a per-scenario payload for every holdout scenario."
        )

    scenario_artifacts = [
        export_scenario_payload(
            scenario_payload=item,
            checkpoint=checkpoint,
            dataset=resolved_dataset,
            output_dir=state_predictions_dir,
            evaluation_config=resolved_evaluation_config,
            loss_name=checkpoint_loss_name,
            checkpoint_path=resolved_checkpoint,
        )
        for item in per_scenario_payloads
    ]
    scenario_artifacts = sorted(
        scenario_artifacts,
        key=lambda artifact: float(artifact["final_return"]),
        reverse=True,
    )

    per_scenario_rows = [
        {
            "scenario_id": item["scenario_id"],
            "source_path": item["source_path"],
            "final_return": item["final_return"],
            "mean_step_return": item["mean_step_return"],
            "std_step_return": item["std_step_return"],
            "average_turnover": item["average_turnover"],
            "final_cash_weight": item["final_cash_weight"],
            "mean_cash_weight": item["mean_cash_weight"],
        }
        for item in per_scenario_payloads
    ]
    per_scenario_csv_path = artifact_paths.evaluation_per_scenario_metrics_path(
        paths,
        checkpoint_loss_name,
        state=resolved_dataset.state,
    )
    per_scenario_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_scenario_rows).to_csv(per_scenario_csv_path, index=False)

    aggregate_payload: dict[str, Any] = {
        **build_holdout_summary_payload(
            per_scenario_payloads,
            dataset=resolved_dataset,
            loss_name=checkpoint_loss_name,
            evaluation_split="holdout_test",
        ),
        "per_scenario_metrics_csv": str(per_scenario_csv_path),
        "scenario_artifacts": scenario_artifacts,
        "train_config": extract_exported_train_config(checkpoint),
        "metadata": resolved_dataset.metadata.as_dict(),
    }
    save_json(
        aggregate_payload,
        artifact_paths.evaluation_metrics_path(paths, checkpoint_loss_name, state=resolved_dataset.state),
    )

    strip_transient_scenario_tensor_fields(per_scenario_payloads)

    return aggregate_payload
