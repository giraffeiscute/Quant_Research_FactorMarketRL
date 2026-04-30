"""Validation helpers for portfolio_attention config dataclasses."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .paths import default_scenario_dir

if TYPE_CHECKING:
    from .schema import DataConfig, EvaluationConfig, ModelConfig, TrainConfig


VALID_DATA_STATES = ("bear", "neutral", "bull")
# Retained for legacy checkpoint metadata handling during evaluation/analysis refresh.
LOOKBACK_MODE_ROLLING_WINDOW = "rolling_window"
LEGACY_LOOKBACK_MODES = frozenset({"full_history", "bounded"})


def normalize_lookback_mode(value: object) -> str:
    return str(value).strip().lower()


def normalize_model_config_dict(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Backfill model-config fields that were previously implicit.

    Older checkpoints stored a single `cross_sectional_dim` that also acted as
    the stock temporal hidden dimension. The current architecture splits that
    into `stock_temporal_dim` and `cross_sectional_dim`, so we infer the former
    from the legacy field when it is missing. Older checkpoints also predate
    explicit stock-id representation and placement selection, so they
    implicitly use learnable stock-id lookups with post-temporal concat.
    """

    normalized = dict(config_dict)
    if "stock_temporal_dim" not in normalized and "cross_sectional_dim" in normalized:
        normalized["stock_temporal_dim"] = normalized["cross_sectional_dim"]
    if "stock_id_representation_type" not in normalized:
        normalized["stock_id_representation_type"] = "learning"
    if "stock_embedding_type" not in normalized:
        normalized["stock_embedding_type"] = "concat"
    if normalized.get("time_positional_encoding_type") == "running_mean":
        normalized["time_positional_encoding_type"] = "none"
    if "allocation_smoothing_alpha" not in normalized:
        normalized["allocation_smoothing_alpha"] = 1.0
    if "initial_allocation_mode" not in normalized:
        normalized["initial_allocation_mode"] = "equal_weight"
    if "initial_random_concentration" not in normalized:
        normalized["initial_random_concentration"] = 1.0
    if "allocation_distribution_type" not in normalized:
        normalized["allocation_distribution_type"] = "softmax"
    if "dirichlet_alpha_offset" not in normalized:
        normalized["dirichlet_alpha_offset"] = 0.1
    if "detach_prev_weight" not in normalized:
        normalized["detach_prev_weight"] = False
    if "use_prev_weight_feature" not in normalized:
        normalized["use_prev_weight_feature"] = False
    if "dropout" not in normalized:
        normalized["dropout"] = 0.1
    return normalized


def raise_if_checkpoint_uses_legacy_stock_id_representation_type(
    checkpoint_model_config: dict[str, Any],
    *,
    context: str,
) -> None:
    raw_value = checkpoint_model_config.get("stock_id_representation_type")
    if raw_value is None:
        return

    normalized_value = str(raw_value).strip().lower()
    if normalized_value in {"embedding", "one_hot"}:
        raise ValueError(
            f"{context} uses unsupported legacy stock_id_representation_type="
            f"{normalized_value!r}. Supported values are ['gaussian', 'learning']. "
            "Only checkpoints missing stock_id_representation_type can be backfilled to "
            "'learning'."
        )


def validated_data_config(config: DataConfig) -> DataConfig:
    resolved = replace(config)
    validate_data_config(resolved)
    return resolved


def validated_model_config(config: ModelConfig) -> ModelConfig:
    resolved = replace(config)
    validate_model_config(resolved)
    return resolved


def validated_train_config(config: TrainConfig) -> TrainConfig:
    resolved = replace(config)
    validate_train_config(resolved)
    return resolved


def validated_evaluation_config(config: EvaluationConfig) -> EvaluationConfig:
    resolved = replace(config)
    validate_evaluation_config(resolved)
    return resolved


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
    config.sample_num_stocks = int(config.sample_num_stocks)
    if config.sample_num_stocks <= 0:
        raise ValueError(
            "DataConfig.sample_num_stocks must be positive, "
            f"received {config.sample_num_stocks}."
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

    config.dropout = float(config.dropout)
    if not 0.0 <= config.dropout <= 1.0:
        raise ValueError(
            "ModelConfig.dropout must be in [0.0, 1.0], "
            f"received {config.dropout}."
        )

    config.allocation_smoothing_alpha = float(config.allocation_smoothing_alpha)
    if not 0.0 <= config.allocation_smoothing_alpha <= 1.0:
        raise ValueError(
            "ModelConfig.allocation_smoothing_alpha must be in [0.0, 1.0], "
            f"received {config.allocation_smoothing_alpha}."
        )

    config.initial_allocation_mode = str(config.initial_allocation_mode).strip().lower()
    valid_initial_allocation_modes = {"equal_weight", "random_dirichlet"}
    if config.initial_allocation_mode not in valid_initial_allocation_modes:
        raise ValueError(
            "ModelConfig.initial_allocation_mode must be one of "
            f"{sorted(valid_initial_allocation_modes)}, "
            f"received {config.initial_allocation_mode!r}."
        )

    config.initial_random_concentration = float(config.initial_random_concentration)
    if config.initial_random_concentration <= 0.0:
        raise ValueError(
            "ModelConfig.initial_random_concentration must be > 0.0, "
            f"received {config.initial_random_concentration}."
        )

    config.allocation_distribution_type = str(config.allocation_distribution_type).strip().lower()
    valid_allocation_distribution_types = {"softmax", "dirichlet"}
    if config.allocation_distribution_type not in valid_allocation_distribution_types:
        raise ValueError(
            "ModelConfig.allocation_distribution_type must be one of "
            f"{sorted(valid_allocation_distribution_types)}, "
            f"received {config.allocation_distribution_type!r}."
        )

    config.dirichlet_alpha_offset = float(config.dirichlet_alpha_offset)
    if config.dirichlet_alpha_offset <= 0.0:
        raise ValueError(
            "ModelConfig.dirichlet_alpha_offset must be > 0.0, "
            f"received {config.dirichlet_alpha_offset}."
        )

    if not isinstance(config.detach_prev_weight, bool):
        raise ValueError(
            "ModelConfig.detach_prev_weight must be a bool, "
            f"received {config.detach_prev_weight!r}."
        )
    if not isinstance(config.use_prev_weight_feature, bool):
        raise ValueError(
            "ModelConfig.use_prev_weight_feature must be a bool, "
            f"received {config.use_prev_weight_feature!r}."
        )
    if not config.use_prev_weight_feature and config.detach_prev_weight:
        raise ValueError(
            "ModelConfig.detach_prev_weight must be False when "
            "ModelConfig.use_prev_weight_feature is False."
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

    config.turnover_penalty = float(config.turnover_penalty)
    if config.turnover_penalty < 0.0:
        raise ValueError(
            "TrainConfig.turnover_penalty must be non-negative, "
            f"received {config.turnover_penalty}."
        )

    config.turnover_penalty_norm = str(config.turnover_penalty_norm).strip().lower()
    valid_turnover_penalty_norms = {"l1", "l2"}
    if config.turnover_penalty_norm not in valid_turnover_penalty_norms:
        raise ValueError(
            "TrainConfig.turnover_penalty_norm must be one of "
            f"{sorted(valid_turnover_penalty_norms)}, "
            f"received {config.turnover_penalty_norm!r}."
        )

    config.transaction_cost_rate = float(config.transaction_cost_rate)
    if config.transaction_cost_rate < 0.0:
        raise ValueError(
            "TrainConfig.transaction_cost_rate must be non-negative, "
            f"received {config.transaction_cost_rate}."
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

    config.evaluation_transaction_cost_rate = float(config.evaluation_transaction_cost_rate)
    if config.evaluation_transaction_cost_rate < 0.0:
        raise ValueError(
            "EvaluationConfig.evaluation_transaction_cost_rate must be non-negative, "
            f"received {config.evaluation_transaction_cost_rate}."
        )
