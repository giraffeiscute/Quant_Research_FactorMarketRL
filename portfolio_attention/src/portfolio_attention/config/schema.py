"""Configuration objects for portfolio_attention."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from ..artifact import paths as artifact_paths
from . import paths as config_paths
from .paths import default_scenario_dir, project_root


@dataclass
class PathsConfig:
    project_dir: Path = field(default_factory=project_root)
    output_root: Path | None = None

    @property
    def outputs_dir(self) -> Path:
        return config_paths.outputs_dir(self)

    @property
    def checkpoints_dir(self) -> Path:
        return config_paths.checkpoints_dir(self)

    @property
    def metrics_dir(self) -> Path:
        return config_paths.metrics_dir(self)

    def get_state_metrics_dir(self, state: str) -> Path:
        return config_paths.state_metrics_dir(self, state)

    @property
    def logs_dir(self) -> Path:
        return config_paths.logs_dir(self)

    def get_state_logs_dir(self, state: str) -> Path:
        return config_paths.state_logs_dir(self, state)

    @property
    def predictions_dir(self) -> Path:
        return config_paths.predictions_dir(self)

    @property
    def status_dir(self) -> Path:
        return config_paths.status_dir(self)

    def get_state_predictions_dir(self, state: str) -> Path:
        return config_paths.state_predictions_dir(self, state)

    def get_scenario_predictions_dir(self, state_id: str) -> Path:
        """Backward-compatible helper for paths keyed by a scenario/state id."""
        return config_paths.scenario_predictions_dir(self, state_id)


@dataclass
class DataConfig:
    """Scenario-aware dataset construction settings.

    The project is scenario-only. One scenario file corresponds to one scenario path.
    The loader optionally shuffles scenario files into train / validation /
    holdout-test groups using a dedicated split seed. Each scenario then uses
    its full time range: train builds rolling windows across the whole scenario,
    while validation and holdout backtests score the whole scenario after a
    lookback warmup prefix.
    """

    # Dataset source
    state: Literal["bear", "neutral", "bull"] = "bear"
    scenario_dir: Path | None = None
    scenario_glob: str = "{state}_*_PL_*.parquet"

    # Scenario counts
    num_train_scenarios: int = 74
    num_validation_scenarios: int = 8
    num_test_scenarios: int = 6

    # Shuffle / seed
    train_batch_size: int = 10
    shuffle_scenario_splits: bool = True
    scenario_split_seed: int = 456
    shuffle_train_scenarios: bool = True
    shuffle_train_scenarios_seed: int = 42

    # Rolling window
    lookback_days: int = 50
    rolling_horizon_days: int = 30
    rolling_stride_days: int = 1
    rolling_train_dataset_mode: Literal["lazy", "eager"] = "lazy"
    price_normalization_mode: Literal["none", "relative_to_anchor"] = "relative_to_anchor"

    # Number of stocks sampled per training rolling window.
    sample_num_stocks: int = 400

    @property
    def resolved_scenario_dir(self) -> Path:
        if self.scenario_dir is None:
            return default_scenario_dir(str(self.state))
        return Path(self.scenario_dir)

    @property
    def resolved_scenario_glob(self) -> str:
        return self.scenario_glob.format(state=self.state)

    @property
    def expected_total_scenarios(self) -> int:
        return (
            int(self.num_train_scenarios)
            + int(self.num_validation_scenarios)
            + int(self.num_test_scenarios)
        )


@dataclass
class ModelConfig:
    """Scenario-mode model settings.

    The defaults are intentionally lightweight enough to preserve a full
    `[scenario, time, stock]` layout during training on large stock universes.
    """

    stock_feature_dim: int = 4
    market_feature_dim: int = 3
    market_temporal_dim: int = 16
    stock_temporal_dim: int = 32
    cross_sectional_dim: int = 32
    dropout: float = 0.1
    stock_id_representation_type: Literal["learning", "gaussian"] = "learning"
    stock_id_embedding_dim: int = 32
    stock_embedding_type: Literal["concat", "pre_temporal"] = "pre_temporal"
    stock_temporal_encoder_type: Literal["running_summary", "causal_self_attention"] = "causal_self_attention"
    stock_cross_sectional_encoder_type: Literal["mlp", "self_attention"] = "self_attention"
    time_positional_encoding_type: Literal["none", "sinusoidal"] = "sinusoidal"
    allocation_smoothing_alpha: float = 0.9
    dirichlet_logit_scale: float = 3.0
    initial_allocation_mode: Literal["equal_weight", "random_dirichlet"] = "random_dirichlet"
    inference_allocation_mode: Literal["softmax", "dirichlet_mean"] = "softmax"
    initial_random_concentration: float = 1
    detach_prev_weight: bool = False
    use_prev_weight_feature: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    """Training settings for scenario-mode optimization."""

    seed: int = 42
    learning_rate: float = 1e-4
    num_epochs: int = 150
    weight_decay: float = 3e-4
    enable_lr_warmup_decay: bool = False
    lr_warmup_fraction: float = 0.05
    lr_min_factor: float = 0.1
    grad_clip_norm: float = 1.0
    grad_monitor_interval_steps: int = 0
    grad_monitor_fail_fast: bool = True
    early_stopping_patience: int = 80
    select_best_from_last_x_epochs: int = 1
    holdout_backtest_interval_epochs: int = 4
    enable_fixed_epoch_holdout_backtests: bool = False
    turnover_penalty: float = 5000
    turnover_penalty_norm: Literal["l1", "l2"] = "l2"
    transaction_cost_rate: float = 0
    loss_name: Literal["", "return", "sharpe", "sortino", "mdd", "cvar"] = ""
    device: str = "cuda"
    post_train_from: Path | None = None
    rl_training: "RLTrainingConfig" = field(default_factory=lambda: RLTrainingConfig())
    def _checkpoint_name(self, stem: str, state: str | None = None) -> str:
        prefix = f"{state}_" if state else ""
        if self.loss_name:
            return f"{prefix}{stem}_{self.loss_name}.pt"
        return f"{prefix}{stem}.pt"

    def train_best_checkpoint_name_for_state(self, state: str | None = None) -> str:
        return artifact_paths.train_best_checkpoint_name(self.loss_name, state=state)

    def train_last_checkpoint_name_for_state(self, state: str | None = None) -> str:
        return artifact_paths.train_last_checkpoint_name(self.loss_name, state=state)

    @property
    def train_best_checkpoint_name(self) -> str:
        return self.train_best_checkpoint_name_for_state()

    @property
    def train_last_checkpoint_name(self) -> str:
        return self.train_last_checkpoint_name_for_state()


@dataclass
class EvaluationConfig:
    """Evaluation and visualization settings."""

    allocation_group_top_n: int = 7
    stock_count_weight_threshold: float = 0.001
    stock_count_min_active_days: int = 2
    evaluation_transaction_cost_rate: float = 0.001
    reward_baseline: Literal["cash", "uniform"] = "cash"


@dataclass
class GRPOTrainingConfig:
    group_size: int = 128
    warmup_allocation_mode: Literal["deterministic_mean"] = "deterministic_mean"
    dsr_var_eps: float = 1e-8
    reward_clip: float = 5.0
    entropy_coef: float = 0.001


@dataclass
class PPOTrainingConfig:
    """Rollout PPO knobs.

    ``rollout_ppo`` collects one rollout batch and reuses the frozen rollout
    for ``num_epochs`` PPO optimizer updates. Setting ``num_epochs=1``
    preserves the single-update PPO behavior.
    """

    reward_clip: float = 5.0
    entropy_coef: float = 0.001
    clip_range: float = 0.2
    value_loss_coef: float = 0.5
    num_epochs: int = 1
    gamma: float = 0.99
    normalize_advantages: bool = True


@dataclass(init=False)
class SACTrainingConfig:
    gamma: float = 0.99
    tau: float = 0.005
    buffer_size: int = 1_000_000
    batch_size: int = 256
    updates_per_batch: int = 1
    warmup_steps: int = 1_000
    context_window_steps: int | None = None
    target_entropy: float | None = None
    # SAC entropy temperature, separate from PPO/GRPO entropy_coef.
    temp_init: float = 0.2
    temp_lr: float = 1e-4
    auto_entropy: bool = True
    # SAC reward clipping is opt-in; None disables clipping by default.
    reward_clip: float | None = None

    def __init__(
        self,
        *,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        updates_per_batch: int = 1,
        warmup_steps: int = 1_000,
        context_window_steps: int | None = None,
        target_entropy: float | None = None,
        temp_init: float = 0.2,
        temp_lr: float = 1e-4,
        auto_entropy: bool = True,
        reward_clip: float | None = None,
        alpha_init: float | None = None,
        alpha_lr: float | None = None,
    ) -> None:
        if alpha_init is not None:
            if float(temp_init) != 0.2 and float(temp_init) != float(alpha_init):
                raise TypeError(
                    "SACTrainingConfig received both temp_init and alpha_init "
                    "with different values."
                )
            temp_init = alpha_init
        if alpha_lr is not None:
            if float(temp_lr) != 1e-4 and float(temp_lr) != float(alpha_lr):
                raise TypeError(
                    "SACTrainingConfig received both temp_lr and alpha_lr "
                    "with different values."
                )
            temp_lr = alpha_lr
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.updates_per_batch = updates_per_batch
        self.warmup_steps = warmup_steps
        self.context_window_steps = context_window_steps
        self.target_entropy = target_entropy
        self.temp_init = temp_init
        self.temp_lr = temp_lr
        self.auto_entropy = auto_entropy
        self.reward_clip = reward_clip


@dataclass(init=False)
class RLTrainingConfig:
    """RL training knobs grouped into shared and algorithm-specific configs."""

    enabled: bool = False
    algorithm: Literal["grpo_like", "rollout_ppo", "sac"] = "grpo_like"
    reward_type: Literal["dsr_day_last", "rolling_sharpe", "return", "win_rate"] = "dsr_day_last"
    reward_scale: float = 1.0
    alpha_min: float = 0.05
    alpha_max: float = 50.0
    rl_post_train_evidence_scale: float = 0.3
    grpo: GRPOTrainingConfig = field(default_factory=GRPOTrainingConfig)
    ppo: PPOTrainingConfig = field(default_factory=PPOTrainingConfig)
    sac: SACTrainingConfig = field(default_factory=SACTrainingConfig)

    def __init__(
        self,
        *,
        enabled: bool = False,
        algorithm: Literal["grpo_like", "rollout_ppo", "sac"] = "grpo_like",
        reward_type: Literal["dsr_day_last", "rolling_sharpe", "return", "win_rate"] = "dsr_day_last",
        reward_scale: float = 1.0,
        alpha_min: float = 0.05,
        alpha_max: float = 50.0,
        rl_post_train_evidence_scale: float = 0.3,
        grpo: GRPOTrainingConfig | dict[str, Any] | None = None,
        ppo: PPOTrainingConfig | dict[str, Any] | None = None,
        sac: SACTrainingConfig | dict[str, Any] | None = None,
        **legacy_algorithm_fields: Any,
    ) -> None:
        self.enabled = enabled
        self.algorithm = algorithm
        self.reward_type = reward_type
        self.reward_scale = reward_scale
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.rl_post_train_evidence_scale = rl_post_train_evidence_scale
        self.grpo = self._coerce_algorithm_config(grpo, GRPOTrainingConfig)
        self.ppo = self._coerce_algorithm_config(ppo, PPOTrainingConfig)
        self.sac = self._coerce_algorithm_config(sac, SACTrainingConfig)
        self._apply_legacy_algorithm_fields(legacy_algorithm_fields)

    @staticmethod
    def _coerce_algorithm_config(
        value: object,
        config_type: type[GRPOTrainingConfig] | type[PPOTrainingConfig] | type[SACTrainingConfig],
    ) -> GRPOTrainingConfig | PPOTrainingConfig | SACTrainingConfig:
        if value is None:
            return config_type()
        if isinstance(value, config_type):
            return replace(value)
        if isinstance(value, dict):
            return config_type(**value)
        raise TypeError(
            f"{config_type.__name__} override must be a {config_type.__name__} or mapping, "
            f"received {type(value).__name__}."
        )

    def _apply_legacy_algorithm_fields(self, values: dict[str, Any]) -> None:
        legacy_routes = {
            "group_size": (self.grpo, "group_size"),
            "warmup_allocation_mode": (self.grpo, "warmup_allocation_mode"),
            "dsr_var_eps": (self.grpo, "dsr_var_eps"),
            "reward_clip": (self.grpo, "reward_clip", self.ppo, "reward_clip"),
            "entropy_coef": (self.grpo, "entropy_coef", self.ppo, "entropy_coef"),
            "ppo_clip_range": (self.ppo, "clip_range"),
            "value_loss_coef": (self.ppo, "value_loss_coef"),
            "ppo_num_epochs": (self.ppo, "num_epochs"),
            "ppo_gamma": (self.ppo, "gamma"),
            "normalize_rollout_advantages": (self.ppo, "normalize_advantages"),
            "sac_gamma": (self.sac, "gamma"),
            "sac_tau": (self.sac, "tau"),
            "sac_buffer_size": (self.sac, "buffer_size"),
            "sac_batch_size": (self.sac, "batch_size"),
            "sac_updates_per_batch": (self.sac, "updates_per_batch"),
            "sac_warmup_steps": (self.sac, "warmup_steps"),
            "sac_context_window_steps": (self.sac, "context_window_steps"),
            "sac_target_entropy": (self.sac, "target_entropy"),
            "sac_temp_init": (self.sac, "temp_init"),
            "sac_temp_lr": (self.sac, "temp_lr"),
            "sac_alpha_init": (self.sac, "temp_init"),
            "sac_alpha_lr": (self.sac, "temp_lr"),
            "sac_auto_entropy": (self.sac, "auto_entropy"),
            "sac_reward_clip": (self.sac, "reward_clip"),
        }
        unknown_keys = sorted(key for key in values if key not in legacy_routes)
        if unknown_keys:
            raise TypeError(f"Unknown RLTrainingConfig fields: {unknown_keys}.")
        for key, value in values.items():
            route = legacy_routes[key]
            if len(route) == 2:
                target, field_name = route
                setattr(target, field_name, value)
            else:
                first_target, first_field, second_target, second_field = route
                setattr(first_target, first_field, value)
                setattr(second_target, second_field, value)

    @property
    def group_size(self) -> int:
        return self.grpo.group_size

    @group_size.setter
    def group_size(self, value: int) -> None:
        self.grpo.group_size = value

    @property
    def warmup_allocation_mode(self) -> Literal["deterministic_mean"]:
        return self.grpo.warmup_allocation_mode

    @warmup_allocation_mode.setter
    def warmup_allocation_mode(self, value: Literal["deterministic_mean"]) -> None:
        self.grpo.warmup_allocation_mode = value

    @property
    def dsr_var_eps(self) -> float:
        return self.grpo.dsr_var_eps

    @dsr_var_eps.setter
    def dsr_var_eps(self, value: float) -> None:
        self.grpo.dsr_var_eps = value

    @property
    def reward_clip(self) -> float:
        if str(self.algorithm).strip().lower() == "rollout_ppo":
            return self.ppo.reward_clip
        return self.grpo.reward_clip

    @reward_clip.setter
    def reward_clip(self, value: float) -> None:
        self.grpo.reward_clip = value
        self.ppo.reward_clip = value

    @property
    def entropy_coef(self) -> float:
        if str(self.algorithm).strip().lower() == "rollout_ppo":
            return self.ppo.entropy_coef
        return self.grpo.entropy_coef

    @entropy_coef.setter
    def entropy_coef(self, value: float) -> None:
        self.grpo.entropy_coef = value
        self.ppo.entropy_coef = value

    @property
    def ppo_clip_range(self) -> float:
        return self.ppo.clip_range

    @ppo_clip_range.setter
    def ppo_clip_range(self, value: float) -> None:
        self.ppo.clip_range = value

    @property
    def value_loss_coef(self) -> float:
        return self.ppo.value_loss_coef

    @value_loss_coef.setter
    def value_loss_coef(self, value: float) -> None:
        self.ppo.value_loss_coef = value

    @property
    def ppo_num_epochs(self) -> int:
        return self.ppo.num_epochs

    @ppo_num_epochs.setter
    def ppo_num_epochs(self, value: int) -> None:
        self.ppo.num_epochs = value

    @property
    def ppo_gamma(self) -> float:
        return self.ppo.gamma

    @ppo_gamma.setter
    def ppo_gamma(self, value: float) -> None:
        self.ppo.gamma = value

    @property
    def normalize_rollout_advantages(self) -> bool:
        return self.ppo.normalize_advantages

    @normalize_rollout_advantages.setter
    def normalize_rollout_advantages(self, value: bool) -> None:
        self.ppo.normalize_advantages = value

    @property
    def sac_gamma(self) -> float:
        return self.sac.gamma

    @sac_gamma.setter
    def sac_gamma(self, value: float) -> None:
        self.sac.gamma = value

    @property
    def sac_tau(self) -> float:
        return self.sac.tau

    @sac_tau.setter
    def sac_tau(self, value: float) -> None:
        self.sac.tau = value

    @property
    def sac_buffer_size(self) -> int:
        return self.sac.buffer_size

    @sac_buffer_size.setter
    def sac_buffer_size(self, value: int) -> None:
        self.sac.buffer_size = value

    @property
    def sac_batch_size(self) -> int:
        return self.sac.batch_size

    @sac_batch_size.setter
    def sac_batch_size(self, value: int) -> None:
        self.sac.batch_size = value

    @property
    def sac_updates_per_batch(self) -> int:
        return self.sac.updates_per_batch

    @sac_updates_per_batch.setter
    def sac_updates_per_batch(self, value: int) -> None:
        self.sac.updates_per_batch = value

    @property
    def sac_warmup_steps(self) -> int:
        return self.sac.warmup_steps

    @sac_warmup_steps.setter
    def sac_warmup_steps(self, value: int) -> None:
        self.sac.warmup_steps = value

    @property
    def sac_context_window_steps(self) -> int | None:
        return self.sac.context_window_steps

    @sac_context_window_steps.setter
    def sac_context_window_steps(self, value: int | None) -> None:
        self.sac.context_window_steps = value

    @property
    def sac_target_entropy(self) -> float | None:
        return self.sac.target_entropy

    @sac_target_entropy.setter
    def sac_target_entropy(self, value: float | None) -> None:
        self.sac.target_entropy = value

    @property
    def sac_temp_init(self) -> float:
        return self.sac.temp_init

    @sac_temp_init.setter
    def sac_temp_init(self, value: float) -> None:
        self.sac.temp_init = value

    @property
    def sac_temp_lr(self) -> float:
        return self.sac.temp_lr

    @sac_temp_lr.setter
    def sac_temp_lr(self, value: float) -> None:
        self.sac.temp_lr = value

    @property
    def sac_alpha_init(self) -> float:
        return self.sac_temp_init

    @sac_alpha_init.setter
    def sac_alpha_init(self, value: float) -> None:
        self.sac_temp_init = value

    @property
    def sac_alpha_lr(self) -> float:
        return self.sac_temp_lr

    @sac_alpha_lr.setter
    def sac_alpha_lr(self, value: float) -> None:
        self.sac_temp_lr = value

    @property
    def sac_auto_entropy(self) -> bool:
        return self.sac.auto_entropy

    @sac_auto_entropy.setter
    def sac_auto_entropy(self, value: bool) -> None:
        self.sac.auto_entropy = value

    @property
    def sac_reward_clip(self) -> float | None:
        return self.sac.reward_clip

    @sac_reward_clip.setter
    def sac_reward_clip(self, value: float | None) -> None:
        self.sac.reward_clip = value

    @staticmethod
    def warmup_days_for_horizon(rolling_horizon_days: int) -> int:
        return int(rolling_horizon_days) - 1

    @staticmethod
    def reward_day_for_horizon(rolling_horizon_days: int) -> int:
        return int(rolling_horizon_days)


# Backward-compatible re-export for legacy script imports.
from .validation import normalize_model_config_dict
