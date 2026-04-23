"""Validation helpers for portfolio_attention config dataclasses."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .config_paths import default_scenario_dir

if TYPE_CHECKING:
    from .config import DataConfig, EvaluationConfig, ModelConfig, TrainConfig


VALID_DATA_STATES = ("bear", "neutral", "bull")
# Retained for legacy checkpoint metadata handling during evaluation/analysis refresh.
LOOKBACK_MODE_ROLLING_WINDOW = "rolling_window"
LEGACY_LOOKBACK_MODES = frozenset({"full_history", "bounded"})


def normalize_lookback_mode(value: object) -> str:
    return str(value).strip().lower()


def validate_data_config(config: DataConfig) -> None:
    config.state = str(config.state).strip().lower()
    if not config.state:
        raise ValueError("DataConfig.state must be one of ['bear', 'bull', 'neutral'], received empty value.")
    if config.state not in VALID_DATA_STATES:
        raise ValueError(
            "DataConfig.state must be one of "
            f"{sorted(VALID_DATA_STATES)}, received {config.state!r}."
        )

    if config.scenario_dir is None:
        config.scenario_dir = default_scenario_dir(config.state)
    config.scenario_dir = Path(config.scenario_dir)
    if config.scenario_dir.name and config.scenario_dir.name.lower() != config.state:
        raise ValueError(
            "DataConfig.state must match the scenario_dir leaf directory name. "
            f"Received state={config.state!r}, scenario_dir={config.scenario_dir}."
        )

    config.price_normalization_mode = str(config.price_normalization_mode).strip().lower()
    valid_price_normalization_modes = {"none", "relative_to_anchor"}
    if config.price_normalization_mode not in valid_price_normalization_modes:
        raise ValueError(
            "DataConfig.price_normalization_mode must be one of "
            f"{sorted(valid_price_normalization_modes)}, "
            f"received {config.price_normalization_mode!r}."
        )

    config.lookback_days = int(config.lookback_days)
    if config.lookback_days <= 0:
        raise ValueError(
            "DataConfig.lookback_days must be positive, "
            f"received {config.lookback_days}."
        )

    config.rolling_horizon_days = int(config.rolling_horizon_days)
    if config.rolling_horizon_days <= 0:
        raise ValueError(
            "DataConfig.rolling_horizon_days must be positive, "
            f"received {config.rolling_horizon_days}."
        )

    config.rolling_stride_days = int(config.rolling_stride_days)
    if config.rolling_stride_days <= 0:
        raise ValueError(
            "DataConfig.rolling_stride_days must be positive, "
            f"received {config.rolling_stride_days}."
        )

    valid_rolling_train_dataset_modes = {"lazy", "eager"}
    if config.rolling_train_dataset_mode not in valid_rolling_train_dataset_modes:
        raise ValueError(
            "DataConfig.rolling_train_dataset_mode must be one of "
            f"{sorted(valid_rolling_train_dataset_modes)}, "
            f"received {config.rolling_train_dataset_mode!r}."
        )

    scenario_counts = {
        "num_train_scenarios": int(config.num_train_scenarios),
        "num_validation_scenarios": int(config.num_validation_scenarios),
        "num_test_scenarios": int(config.num_test_scenarios),
    }
    for name, value in scenario_counts.items():
        if value <= 0:
            raise ValueError(f"DataConfig.{name} must be positive, received {value}.")

    if int(config.train_batch_size) <= 0:
        raise ValueError(
            "DataConfig.train_batch_size must be positive, "
            f"received {config.train_batch_size}."
        )

    split_seed_fields = {
        "scenario_split_seed": config.scenario_split_seed,
    }
    for name, value in split_seed_fields.items():
        resolved_seed = int(value)
        if resolved_seed < 0:
            raise ValueError(
                f"DataConfig.{name} must be non-negative, "
                f"received {value}."
            )
        setattr(config, name, resolved_seed)

    if config.shuffle_train_scenarios_seed is not None:
        resolved_shuffle_seed = int(config.shuffle_train_scenarios_seed)
        if resolved_shuffle_seed < 0:
            raise ValueError(
                "DataConfig.shuffle_train_scenarios_seed must be non-negative, "
                f"received {config.shuffle_train_scenarios_seed}."
            )
        config.shuffle_train_scenarios_seed = resolved_shuffle_seed

    if config.expected_total_scenarios <= 0:
        raise ValueError(
            "Expected total scenarios must be positive, "
            f"received {config.expected_total_scenarios}."
        )


def validate_model_config(config: ModelConfig) -> None:
    config.stock_id_representation_type = str(config.stock_id_representation_type).strip().lower()
    valid_stock_id_representation_types = {"learning", "gaussian"}
    if config.stock_id_representation_type not in valid_stock_id_representation_types:
        raise ValueError(
            "ModelConfig.stock_id_representation_type must be one of "
            f"{sorted(valid_stock_id_representation_types)}, "
            f"received {config.stock_id_representation_type!r}."
        )
    config.stock_id_embedding_dim = int(config.stock_id_embedding_dim)
    if config.stock_id_embedding_dim <= 0:
        raise ValueError(
            "ModelConfig.stock_id_embedding_dim must be positive, "
            f"received {config.stock_id_embedding_dim}."
        )
    config.stock_temporal_dim = int(config.stock_temporal_dim)
    if config.stock_temporal_dim <= 0:
        raise ValueError(
            "ModelConfig.stock_temporal_dim must be positive, "
            f"received {config.stock_temporal_dim}."
        )
    config.stock_embedding_type = str(config.stock_embedding_type).strip().lower()
    valid_stock_embedding_types = {"concat", "pre_temporal"}
    if config.stock_embedding_type not in valid_stock_embedding_types:
        raise ValueError(
            "ModelConfig.stock_embedding_type must be one of "
            f"{sorted(valid_stock_embedding_types)}, "
            f"received {config.stock_embedding_type!r}."
        )
    if (
        config.stock_embedding_type == "pre_temporal"
        and config.stock_id_embedding_dim != config.stock_temporal_dim
    ):
        raise ValueError(
            "ModelConfig.stock_id_embedding_dim must equal ModelConfig.stock_temporal_dim "
            "when stock_embedding_type='pre_temporal'. "
            f"received stock_id_embedding_dim={config.stock_id_embedding_dim} "
            f"stock_temporal_dim={config.stock_temporal_dim}."
        )

    valid_stock_temporal_encoder_types = {"running_summary", "causal_self_attention"}
    if config.stock_temporal_encoder_type not in valid_stock_temporal_encoder_types:
        raise ValueError(
            "ModelConfig.stock_temporal_encoder_type must be one of "
            f"{sorted(valid_stock_temporal_encoder_types)}, "
            f"received {config.stock_temporal_encoder_type!r}."
        )
    valid_stock_cross_sectional_encoder_types = {"mlp", "self_attention"}
    if config.stock_cross_sectional_encoder_type not in valid_stock_cross_sectional_encoder_types:
        raise ValueError(
            "ModelConfig.stock_cross_sectional_encoder_type must be one of "
            f"{sorted(valid_stock_cross_sectional_encoder_types)}, "
            f"received {config.stock_cross_sectional_encoder_type!r}."
        )
    config.time_positional_encoding_type = str(config.time_positional_encoding_type).strip().lower()
    if config.time_positional_encoding_type == "running_mean":
        config.time_positional_encoding_type = "none"
    valid_time_positional_encoding_types = {"none", "sinusoidal"}
    if config.time_positional_encoding_type not in valid_time_positional_encoding_types:
        raise ValueError(
            "ModelConfig.time_positional_encoding_type must be one of "
            f"{sorted(valid_time_positional_encoding_types)}, "
            f"received {config.time_positional_encoding_type!r}."
        )


def validate_train_config(config: TrainConfig) -> None:
    config.holdout_backtest_interval_epochs = int(config.holdout_backtest_interval_epochs)
    if config.holdout_backtest_interval_epochs < 0:
        raise ValueError(
            "TrainConfig.holdout_backtest_interval_epochs must be non-negative, "
            f"received {config.holdout_backtest_interval_epochs}."
        )

    if not isinstance(config.enable_fixed_epoch_holdout_backtests, bool):
        raise ValueError(
            "TrainConfig.enable_fixed_epoch_holdout_backtests must be a bool, "
            f"received {config.enable_fixed_epoch_holdout_backtests!r}."
        )

    if config.resume_from is not None:
        config.resume_from = Path(config.resume_from)


def validate_evaluation_config(config: EvaluationConfig) -> None:
    config.stock_count_weight_threshold = float(config.stock_count_weight_threshold)
    if config.stock_count_weight_threshold < 0.0:
        raise ValueError(
            "EvaluationConfig.stock_count_weight_threshold must be non-negative, "
            f"received {config.stock_count_weight_threshold}."
        )

    config.stock_count_min_active_days = int(config.stock_count_min_active_days)
    if config.stock_count_min_active_days <= 0:
        raise ValueError(
            "EvaluationConfig.stock_count_min_active_days must be positive, "
            f"received {config.stock_count_min_active_days}."
        )
