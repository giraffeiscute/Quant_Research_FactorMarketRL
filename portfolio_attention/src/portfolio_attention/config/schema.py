"""Configuration objects for portfolio_attention."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

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
    rolling_stride_days: int = 2
    rolling_train_dataset_mode: Literal["lazy", "eager"] = "lazy"
    price_normalization_mode: Literal["none", "relative_to_anchor"] = "relative_to_anchor"

    # Number of stocks sampled per training rolling window.
    sample_num_stocks: int = 1000

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
    stock_temporal_dim: int = 16
    cross_sectional_dim: int = 16
    dropout: float = 0.1
    stock_id_representation_type: Literal["learning", "gaussian"] = "learning"
    stock_id_embedding_dim: int = 16
    stock_embedding_type: Literal["concat", "pre_temporal"] = "pre_temporal"
    stock_temporal_encoder_type: Literal["running_summary", "causal_self_attention"] = "causal_self_attention"
    stock_cross_sectional_encoder_type: Literal["mlp", "self_attention"] = "self_attention"
    time_positional_encoding_type: Literal["none", "sinusoidal"] = "sinusoidal"
    allocation_smoothing_alpha: float = 0.9
    initial_allocation_mode: Literal["equal_weight", "random_dirichlet"] = "random_dirichlet"
    initial_random_concentration: float = 5
    detach_prev_weight: bool = False
    use_prev_weight_feature: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    """Training settings for scenario-mode optimization."""

    seed: int = 42
    learning_rate: float = 3e-4
    num_epochs: int = 30
    weight_decay: float = 3e-4
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 7
    select_best_from_last_x_epochs: int = 1
    holdout_backtest_interval_epochs: int = 4
    enable_fixed_epoch_holdout_backtests: bool = False
    turnover_penalty: float = 0.5
    turnover_penalty_norm: Literal["l1", "l2"] = "l1"
    transaction_cost_rate: float = 0.001
    loss_name: Literal["", "return", "sharpe", "dsr", "sortino", "mdd", "cvar"] = ""
    device: str = "auto"
    resume_from: Path | None = None
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


# Backward-compatible re-export for legacy script imports.
from .validation import normalize_model_config_dict
