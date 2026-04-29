"""Checkpoint metadata/configuration helpers for evaluation workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..artifact import paths as artifact_paths
from ..config import DataConfig, ModelConfig, PathsConfig
from ..config.validation import (
    normalize_model_config_dict,
    raise_if_checkpoint_uses_legacy_stock_id_representation_type,
    validated_data_config,
    validated_model_config,
)


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


def _resolve_checkpoint_metadata_dict(
    checkpoint: dict[str, Any],
    key: str,
) -> Any:
    checkpoint_payload = checkpoint.get(key)
    if checkpoint_payload:
        return checkpoint_payload
    checkpoint_metadata = checkpoint.get("portfolio_attention_metadata", {})
    if not isinstance(checkpoint_metadata, dict):
        return checkpoint_payload
    return checkpoint_metadata.get(key, checkpoint_payload)


def _build_model_config_from_checkpoint(checkpoint: dict[str, Any]) -> ModelConfig:
    checkpoint_model_config = _resolve_checkpoint_metadata_dict(checkpoint, "model_config")
    if checkpoint_model_config is None:
        checkpoint_model_config = {}
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
    checkpoint_data_config = _resolve_checkpoint_metadata_dict(checkpoint, "data_config")
    if not isinstance(checkpoint_data_config, dict):
        return fallback_data_config
    if "num_stocks" in checkpoint_data_config:
        raise ValueError(
            "Checkpoint data_config contains legacy key 'num_stocks'. "
            "Use DataConfig.sample_num_stocks for training sampling; full stock universe size "
            "is inferred from scenario data and cannot be manually specified."
        )
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


def _validate_requested_runtime_configs_against_checkpoint(
    *,
    requested_data_config: DataConfig,
    requested_model_config: ModelConfig,
    checkpoint: dict[str, Any],
    args_dict: dict[str, Any],
) -> None:
    checkpoint_data_config = _build_data_config_from_checkpoint(
        checkpoint,
        fallback_data_config=requested_data_config,
    )
    checkpoint_model_config = _build_model_config_from_checkpoint(checkpoint)

    if (
        "sample_num_stocks" in args_dict
        and checkpoint_data_config.sample_num_stocks != requested_data_config.sample_num_stocks
    ):
        raise ValueError(
            "Requested sample_num_stocks does not match the checkpoint data configuration. "
            f"checkpoint={checkpoint_data_config.sample_num_stocks} "
            f"requested={requested_data_config.sample_num_stocks}"
        )
    if (
        "stock_id_representation_type" in args_dict
        and checkpoint_model_config.stock_id_representation_type
        != requested_model_config.stock_id_representation_type
    ):
        raise ValueError(
            "Requested stock_id_representation_type does not match the checkpoint model configuration. "
            f"checkpoint={checkpoint_model_config.stock_id_representation_type!r} "
            f"requested={requested_model_config.stock_id_representation_type!r}"
        )
    if (
        "stock_embedding_type" in args_dict
        and checkpoint_model_config.stock_embedding_type
        != requested_model_config.stock_embedding_type
    ):
        raise ValueError(
            "Requested stock_embedding_type does not match the checkpoint model configuration. "
            f"checkpoint={checkpoint_model_config.stock_embedding_type!r} "
            f"requested={requested_model_config.stock_embedding_type!r}"
        )
    if (
        "stock_temporal_encoder_type" in args_dict
        and checkpoint_model_config.stock_temporal_encoder_type
        != requested_model_config.stock_temporal_encoder_type
    ):
        raise ValueError(
            "Requested stock_temporal_encoder_type does not match the checkpoint model configuration. "
            f"checkpoint={checkpoint_model_config.stock_temporal_encoder_type!r} "
            f"requested={requested_model_config.stock_temporal_encoder_type!r}"
        )
    if (
        "stock_cross_sectional_encoder_type" in args_dict
        and checkpoint_model_config.stock_cross_sectional_encoder_type
        != requested_model_config.stock_cross_sectional_encoder_type
    ):
        raise ValueError(
            "Requested stock_cross_sectional_encoder_type does not match the checkpoint model configuration. "
            f"checkpoint={checkpoint_model_config.stock_cross_sectional_encoder_type!r} "
            f"requested={requested_model_config.stock_cross_sectional_encoder_type!r}"
        )
