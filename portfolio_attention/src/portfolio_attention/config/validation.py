"""Validation helpers for portfolio_attention config dataclasses."""

from __future__ import annotations

from dataclasses import replace
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .paths import default_scenario_dir
from .schema import GRPOTrainingConfig, PPOTrainingConfig, RLTrainingConfig, SACTrainingConfig

if TYPE_CHECKING:
    from .schema import DataConfig, EvaluationConfig, ModelConfig, TrainConfig


VALID_DATA_STATES = ("bear", "neutral", "bull")
# Retained for legacy checkpoint metadata handling during evaluation/analysis refresh.
LOOKBACK_MODE_ROLLING_WINDOW = "rolling_window"
LEGACY_LOOKBACK_MODES = frozenset({"full_history", "bounded"})
POST_TRAIN_FROM_SUFFIX = ".ckpt"


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
    if "inference_allocation_mode" not in normalized:
        normalized["inference_allocation_mode"] = "softmax"
    if "initial_random_concentration" not in normalized:
        normalized["initial_random_concentration"] = 1.0
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


def validate_train_config_against_data_config(
    train_config: TrainConfig,
    data_config: DataConfig,
) -> None:
    if not bool(train_config.rl_training.enabled):
        return
    rolling_horizon_days = int(data_config.rolling_horizon_days)
    if rolling_horizon_days < 2:
        raise ValueError(
            "DataConfig.rolling_horizon_days must be >= 2 when TrainConfig.rl_training.enabled is True, "
            f"received {rolling_horizon_days}."
        )


def validate_train_config_against_model_config(
    train_config: TrainConfig,
    model_config: ModelConfig,
) -> None:
    rl_config = train_config.rl_training
    algorithm = str(getattr(rl_config, "algorithm", "")).strip().lower()
    if (
        bool(getattr(rl_config, "enabled", False))
        and algorithm == "rollout_ppo"
        and bool(getattr(model_config, "use_prev_weight_feature", False))
    ):
        raise ValueError(
            "rollout_ppo currently requires ModelConfig.use_prev_weight_feature=False "
            "so the frozen PPO rollout state is not conditioned on previous weights."
        )


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

    config.rebalance_interval_days = int(config.rebalance_interval_days)
    if config.rebalance_interval_days <= 0:
        raise ValueError(
            "DataConfig.rebalance_interval_days must be positive, "
            f"received {config.rebalance_interval_days}."
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
    config.dirichlet_logit_scale = float(config.dirichlet_logit_scale)
    if config.dirichlet_logit_scale <= 0.0:
        raise ValueError(
            "ModelConfig.dirichlet_logit_scale must be > 0.0, "
            f"received {config.dirichlet_logit_scale}."
        )

    config.initial_allocation_mode = str(config.initial_allocation_mode).strip().lower()
    valid_initial_allocation_modes = {"equal_weight", "random_dirichlet"}
    if config.initial_allocation_mode not in valid_initial_allocation_modes:
        raise ValueError(
            "ModelConfig.initial_allocation_mode must be one of "
            f"{sorted(valid_initial_allocation_modes)}, "
            f"received {config.initial_allocation_mode!r}."
        )

    config.inference_allocation_mode = str(config.inference_allocation_mode).strip().lower()
    valid_inference_allocation_modes = {"softmax", "dirichlet_mean"}
    if config.inference_allocation_mode not in valid_inference_allocation_modes:
        raise ValueError(
            "ModelConfig.inference_allocation_mode must be one of "
            f"{sorted(valid_inference_allocation_modes)}, "
            f"received {config.inference_allocation_mode!r}."
        )

    config.initial_random_concentration = float(config.initial_random_concentration)
    if config.initial_random_concentration <= 0.0:
        raise ValueError(
            "ModelConfig.initial_random_concentration must be > 0.0, "
            f"received {config.initial_random_concentration}."
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
    if not isinstance(config.enable_lr_warmup_decay, bool):
        raise ValueError(
            "TrainConfig.enable_lr_warmup_decay must be a bool, "
            f"received {config.enable_lr_warmup_decay!r}."
        )

    config.lr_warmup_fraction = float(config.lr_warmup_fraction)
    if not 0.0 <= config.lr_warmup_fraction < 1.0:
        raise ValueError(
            "TrainConfig.lr_warmup_fraction must be in [0.0, 1.0), "
            f"received {config.lr_warmup_fraction}."
        )

    config.lr_min_factor = float(config.lr_min_factor)
    if not 0.0 <= config.lr_min_factor <= 1.0:
        raise ValueError(
            "TrainConfig.lr_min_factor must be in [0.0, 1.0], "
            f"received {config.lr_min_factor}."
        )

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

    config.grad_monitor_interval_steps = int(config.grad_monitor_interval_steps)
    if config.grad_monitor_interval_steps < 0:
        raise ValueError(
            "TrainConfig.grad_monitor_interval_steps must be non-negative, "
            f"received {config.grad_monitor_interval_steps}."
        )

    if not isinstance(config.grad_monitor_fail_fast, bool):
        raise ValueError(
            "TrainConfig.grad_monitor_fail_fast must be a bool, "
            f"received {config.grad_monitor_fail_fast!r}."
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

    if config.post_train_from is not None:
        config.post_train_from = Path(config.post_train_from)
        if not config.post_train_from.exists():
            raise FileNotFoundError(
                f"TrainConfig.post_train_from checkpoint is missing: {config.post_train_from}"
            )
        if not config.post_train_from.is_file():
            raise ValueError(
                "TrainConfig.post_train_from must be a checkpoint file path, "
                f"received directory/non-file path: {config.post_train_from}"
            )
        if config.post_train_from.suffix.lower() != POST_TRAIN_FROM_SUFFIX:
            raise ValueError(
                "TrainConfig.post_train_from must point to a Lightning checkpoint file ending in "
                f"{POST_TRAIN_FROM_SUFFIX!r}. Received {config.post_train_from}."
            )

    config.rl_training = _validated_rl_training_config(config.rl_training)
    if (
        bool(config.enable_lr_warmup_decay)
        and bool(config.rl_training.enabled)
        and str(config.rl_training.algorithm).strip().lower() == "sac"
    ):
        raise ValueError(
            "SAC training currently does not support TrainConfig.enable_lr_warmup_decay=True; "
            "disable the scheduler or add explicit SAC manual scheduler stepping."
        )


def _validate_nested_rl_config_keys(
    value: dict[object, object],
    key: str,
    config_type: type[GRPOTrainingConfig] | type[PPOTrainingConfig] | type[SACTrainingConfig],
) -> None:
    nested_value = value.get(key)
    if nested_value is None or isinstance(nested_value, config_type):
        return
    if not isinstance(nested_value, dict):
        raise ValueError(
            f"TrainConfig.rl_training.{key} must be a {config_type.__name__} or mapping, "
            f"received {type(nested_value).__name__}."
        )
    legal_keys = {field_info.name for field_info in fields(config_type)}
    if config_type is SACTrainingConfig:
        legal_keys |= {"alpha_init", "alpha_lr"}
    unknown_keys = sorted(str(nested_key) for nested_key in nested_value if nested_key not in legal_keys)
    if unknown_keys:
        raise ValueError(
            f"TrainConfig.rl_training.{key} contains unknown keys {unknown_keys}. "
            f"Legal keys: {sorted(legal_keys)}."
        )


def _validated_rl_training_config(value: object) -> RLTrainingConfig:
    if isinstance(value, RLTrainingConfig):
        config = RLTrainingConfig(
            enabled=value.enabled,
            algorithm=value.algorithm,
            reward_type=value.reward_type,
            reward_scale=value.reward_scale,
            alpha_min=value.alpha_min,
            alpha_max=value.alpha_max,
            rl_post_train_evidence_scale=value.rl_post_train_evidence_scale,
            grpo=value.grpo,
            ppo=value.ppo,
            sac=value.sac,
        )
    elif isinstance(value, dict):
        legacy_keys = {
            "group_size",
            "warmup_allocation_mode",
            "dsr_var_eps",
            "reward_clip",
            "entropy_coef",
            "ppo_clip_range",
            "value_loss_coef",
            "ppo_num_epochs",
            "ppo_gamma",
            "normalize_rollout_advantages",
            "sac_gamma",
            "sac_tau",
            "sac_buffer_size",
            "sac_batch_size",
            "sac_updates_per_batch",
            "sac_warmup_steps",
            "sac_context_window_steps",
            "sac_target_entropy",
            "sac_temp_init",
            "sac_temp_lr",
            "sac_alpha_init",
            "sac_alpha_lr",
            "sac_auto_entropy",
            "sac_reward_clip",
        }
        legal_keys = {field_info.name for field_info in fields(RLTrainingConfig)} | legacy_keys
        unknown_keys = sorted(str(key) for key in value.keys() if key not in legal_keys)
        if unknown_keys:
            raise ValueError(
                "TrainConfig.rl_training contains unknown keys "
                f"{unknown_keys}. Legal keys: {sorted(legal_keys)}."
            )
        _validate_nested_rl_config_keys(value, "grpo", GRPOTrainingConfig)
        _validate_nested_rl_config_keys(value, "ppo", PPOTrainingConfig)
        _validate_nested_rl_config_keys(value, "sac", SACTrainingConfig)
        config = RLTrainingConfig(**value)
    else:
        raise ValueError(
            "TrainConfig.rl_training must be an RLTrainingConfig or mapping, "
            f"received {type(value).__name__}."
        )

    if not isinstance(config.enabled, bool):
        raise ValueError(
            "TrainConfig.rl_training.enabled must be a bool, "
            f"received {config.enabled!r}."
        )
    config.algorithm = str(config.algorithm).strip().lower()  # type: ignore[assignment]
    valid_algorithms = {
        "grpo_like",
        "rollout_ppo",
        "sac",
    }
    if config.algorithm not in valid_algorithms:
        raise ValueError(
            "TrainConfig.rl_training.algorithm must be one of "
            f"{sorted(valid_algorithms)}, "
            f"received {config.algorithm!r}."
        )
    config.reward_type = str(config.reward_type).strip().lower()  # type: ignore[assignment]
    valid_reward_types = {"dsr_day_last", "rolling_sharpe", "return", "win_rate"}
    if config.reward_type not in valid_reward_types:
        raise ValueError(
            "TrainConfig.rl_training.reward_type must be one of "
            f"{sorted(valid_reward_types)}, "
            f"received {config.reward_type!r}."
        )
    if config.algorithm == "rollout_ppo" and config.reward_type != "return":
        raise ValueError(
            "rollout_ppo currently requires reward_type='return'. "
            f"Received {config.reward_type!r}."
        )

    config.grpo.warmup_allocation_mode = str(config.grpo.warmup_allocation_mode).strip().lower()  # type: ignore[assignment]
    if config.grpo.warmup_allocation_mode != "deterministic_mean":
        raise ValueError(
            "TrainConfig.rl_training.warmup_allocation_mode must be 'deterministic_mean', "
            f"received {config.grpo.warmup_allocation_mode!r}."
        )

    config.grpo.group_size = int(config.grpo.group_size)
    if config.grpo.group_size <= 0:
        raise ValueError(
            "TrainConfig.rl_training.group_size must be positive, "
            f"received {config.grpo.group_size}."
        )

    config.grpo.dsr_var_eps = float(config.grpo.dsr_var_eps)
    if config.grpo.dsr_var_eps <= 0.0:
        raise ValueError(
            "TrainConfig.rl_training.dsr_var_eps must be > 0, "
            f"received {config.grpo.dsr_var_eps}."
        )

    config.reward_scale = float(config.reward_scale)
    if config.reward_scale <= 0.0:
        raise ValueError(
            "TrainConfig.rl_training.reward_scale must be > 0, "
            f"received {config.reward_scale}."
        )

    config.grpo.reward_clip = float(config.grpo.reward_clip)
    if config.grpo.reward_clip <= 0.0:
        raise ValueError(
            "TrainConfig.rl_training.reward_clip must be > 0, "
            f"received {config.grpo.reward_clip}."
        )
    config.ppo.reward_clip = float(config.ppo.reward_clip)
    if config.ppo.reward_clip <= 0.0:
        raise ValueError(
            "TrainConfig.rl_training.ppo.reward_clip must be > 0, "
            f"received {config.ppo.reward_clip}."
        )

    config.grpo.entropy_coef = float(config.grpo.entropy_coef)
    if config.grpo.entropy_coef < 0.0:
        raise ValueError(
            "TrainConfig.rl_training.entropy_coef must be non-negative, "
            f"received {config.grpo.entropy_coef}."
        )
    config.ppo.entropy_coef = float(config.ppo.entropy_coef)
    if config.ppo.entropy_coef < 0.0:
        raise ValueError(
            "TrainConfig.rl_training.ppo.entropy_coef must be non-negative, "
            f"received {config.ppo.entropy_coef}."
        )

    config.ppo.clip_range = float(config.ppo.clip_range)
    if config.ppo.clip_range <= 0.0:
        raise ValueError(
            "TrainConfig.rl_training.ppo_clip_range must be > 0, "
            f"received {config.ppo.clip_range}."
        )

    config.ppo.value_loss_coef = float(config.ppo.value_loss_coef)
    if config.ppo.value_loss_coef < 0.0:
        raise ValueError(
            "TrainConfig.rl_training.value_loss_coef must be non-negative, "
            f"received {config.ppo.value_loss_coef}."
        )

    config.ppo.num_epochs = int(config.ppo.num_epochs)
    if config.ppo.num_epochs < 1:
        raise ValueError(
            "TrainConfig.rl_training.ppo_num_epochs must be >= 1. "
            f"Received {config.ppo.num_epochs}."
        )

    config.ppo.gamma = float(config.ppo.gamma)
    if config.ppo.gamma < 0.0 or config.ppo.gamma > 1.0:
        raise ValueError(
            "TrainConfig.rl_training.ppo_gamma must be in [0, 1], "
            f"received {config.ppo.gamma}."
        )

    if not isinstance(config.ppo.normalize_advantages, bool):
        raise ValueError(
            "TrainConfig.rl_training.normalize_rollout_advantages must be a bool, "
            f"received {config.ppo.normalize_advantages!r}."
        )

    config.sac.gamma = float(config.sac.gamma)
    if config.sac.gamma < 0.0 or config.sac.gamma > 1.0:
        raise ValueError(
            "TrainConfig.rl_training.sac_gamma must be in [0, 1], "
            f"received {config.sac.gamma}."
        )

    config.sac.tau = float(config.sac.tau)
    if config.sac.tau <= 0.0 or config.sac.tau > 1.0:
        raise ValueError(
            "TrainConfig.rl_training.sac_tau must be in (0, 1], "
            f"received {config.sac.tau}."
        )

    config.sac.buffer_size = int(config.sac.buffer_size)
    if config.sac.buffer_size <= 0:
        raise ValueError(
            "TrainConfig.rl_training.sac_buffer_size must be positive, "
            f"received {config.sac.buffer_size}."
        )

    config.sac.batch_size = int(config.sac.batch_size)
    if config.sac.batch_size <= 0:
        raise ValueError(
            "TrainConfig.rl_training.sac_batch_size must be positive, "
            f"received {config.sac.batch_size}."
        )

    config.sac.updates_per_batch = int(config.sac.updates_per_batch)
    if config.sac.updates_per_batch <= 0:
        raise ValueError(
            "TrainConfig.rl_training.sac_updates_per_batch must be positive, "
            f"received {config.sac.updates_per_batch}."
        )

    config.sac.warmup_steps = int(config.sac.warmup_steps)
    if config.sac.warmup_steps < 0:
        raise ValueError(
            "TrainConfig.rl_training.sac_warmup_steps must be non-negative, "
            f"received {config.sac.warmup_steps}."
        )

    if config.sac.context_window_steps is not None:
        config.sac.context_window_steps = int(config.sac.context_window_steps)
        if config.sac.context_window_steps <= 0:
            raise ValueError(
                "TrainConfig.rl_training.sac_context_window_steps must be positive when set, "
                f"received {config.sac.context_window_steps}."
            )

    if config.sac.target_entropy is not None:
        config.sac.target_entropy = float(config.sac.target_entropy)

    config.sac.temp_init = float(config.sac.temp_init)
    if config.sac.temp_init <= 0.0:
        raise ValueError(
            "TrainConfig.rl_training.sac_temp_init must be > 0, "
            f"received {config.sac.temp_init}."
        )

    config.sac.temp_lr = float(config.sac.temp_lr)
    if config.sac.temp_lr <= 0.0:
        raise ValueError(
            "TrainConfig.rl_training.sac_temp_lr must be > 0, "
            f"received {config.sac.temp_lr}."
        )

    if not isinstance(config.sac.auto_entropy, bool):
        raise ValueError(
            "TrainConfig.rl_training.sac_auto_entropy must be a bool, "
            f"received {config.sac.auto_entropy!r}."
        )

    if config.sac.reward_clip is not None:
        config.sac.reward_clip = float(config.sac.reward_clip)
        if config.sac.reward_clip <= 0.0:
            config.sac.reward_clip = None

    config.alpha_min = float(config.alpha_min)
    config.alpha_max = float(config.alpha_max)
    if config.alpha_min <= 0.0:
        raise ValueError(
            "TrainConfig.rl_training.alpha_min must be > 0, "
            f"received {config.alpha_min}."
        )
    if config.alpha_max <= 0.0:
        raise ValueError(
            "TrainConfig.rl_training.alpha_max must be > 0, "
            f"received {config.alpha_max}."
        )
    if config.alpha_min > config.alpha_max:
        raise ValueError(
            "TrainConfig.rl_training.alpha_min must be <= alpha_max, "
            f"received alpha_min={config.alpha_min} alpha_max={config.alpha_max}."
        )

    config.rl_post_train_evidence_scale = float(config.rl_post_train_evidence_scale)
    if config.rl_post_train_evidence_scale <= 0.0:
        raise ValueError(
            "TrainConfig.rl_training.rl_post_train_evidence_scale must be > 0, "
            f"received {config.rl_post_train_evidence_scale}."
        )

    return config


def validate_evaluation_config(config: EvaluationConfig) -> None:
    config.reward_baseline = str(config.reward_baseline).strip().lower()  # type: ignore[assignment]
    valid_reward_baselines = {"cash", "uniform"}
    if config.reward_baseline not in valid_reward_baselines:
        raise ValueError(
            "EvaluationConfig.reward_baseline must be one of "
            f"{sorted(valid_reward_baselines)}, "
            f"received {config.reward_baseline!r}."
        )

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
