"""Lightning logging helpers."""

from __future__ import annotations

import os
from typing import Any

from lightning_fabric.loggers.logger import rank_zero_experiment
from lightning.pytorch.loggers.csv_logs import CSVLogger, ExperimentWriter
import torch


CSV_METRIC_DECIMAL_PLACES = 8
VAL_STOCK_METRIC_DECIMAL_PLACES = 1
VAL_STOCK_METRIC_KEYS = {"val_stock"}
PREFERRED_METRIC_KEY_ORDER = [
    "epoch",
    "step",
    "train_loss",
    "train_weight_loss",
    "train_TO",
    "train_return",
    "train_SR",
    "val_loss",
    "val_loss_window",
    "val_return",
    "val_SR",
    "val_TO",
    "val_stock",
    "val_cash",
    "val_win_rate",
]
RL_PREFERRED_METRIC_KEY_ORDER = [
    "epoch",
    "step",
    "train_policy_loss",
    "train_entropy_loss",
    "train_entropy_per_dim",
    "train_alpha_min",
    "train_alpha_max",
    "train_alpha_mean",
    "train_reward_base",
    "train_reward_final",
    "train_reward_TO_penalty",
    "train_advantage_mean",
    "train_advantage_std",
    "train_log_prob_mean",
    "train_log_prob_std",
    "train_rollout_value_loss",
    "train_rollout_target_mean",
    "train_rollout_target_std",
    "train_rollout_advantage_mean",
    "train_rollout_advantage_std",
    "train_rollout_ppo_ratio_mean",
    "train_rollout_ppo_clip_fraction",
    "train_rollout_entropy_per_dim",
    "train_rollout_total_loss",
    "train_rollout_reward_base",
    "train_rollout_reward_final",
    "train_rollout_reward_TO_penalty",
    "train_rollout_return",
    "train_rollout_TO",
    "train_rollout_final_returns",
    "train_TO",
    "train_return",
    "train_SR",
    "val_loss",
    "val_return",
    "val_SR",
    "val_TO",
    "val_stock",
    "val_cash",
    "val_win_rate",
]


class RoundedMetricsExperimentWriter(ExperimentWriter):
    """CSV experiment writer that renders floating-point metrics with fixed precision."""

    def __init__(
        self,
        log_dir: str,
        *,
        decimal_places: int = CSV_METRIC_DECIMAL_PLACES,
        metrics_filename: str = "metrics.csv",
        preferred_metric_key_order: list[str] | None = None,
        excluded_metric_keys: set[str] | None = None,
    ) -> None:
        super().__init__(log_dir=log_dir)
        self.decimal_places = int(decimal_places)
        self.metrics_filename = str(metrics_filename)
        self.metrics_file_path = os.path.join(self.log_dir, self.metrics_filename)
        self.excluded_metric_keys = set(excluded_metric_keys or set())
        self.preferred_metric_key_order = (
            list(preferred_metric_key_order)
            if preferred_metric_key_order is not None
            else list(PREFERRED_METRIC_KEY_ORDER)
        )

    def _format_metric_value(self, key: str, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            value = value.item()
        if isinstance(value, float):
            decimal_places = (
                VAL_STOCK_METRIC_DECIMAL_PLACES
                if key in VAL_STOCK_METRIC_KEYS
                else self.decimal_places
            )
            return f"{value:.{decimal_places}f}"
        return value

    def log_metrics(self, metrics_dict: dict[str, float], step: int | None = None) -> None:
        if step is None:
            step = len(self.metrics)

        metrics = {
            key: self._format_metric_value(key, value)
            for key, value in metrics_dict.items()
            if key not in self.excluded_metric_keys
        }
        if not metrics:
            return
        metrics["step"] = step
        self.metrics.append(metrics)

    def _record_new_keys(self) -> set[str]:
        """Record new keys and keep selected metrics in a stable relative order."""
        current_keys = set().union(*self.metrics)
        new_keys = current_keys - set(self.metrics_keys)
        self.metrics_keys.extend(new_keys)
        self.metrics_keys.sort()
        preferred_present = [key for key in self.preferred_metric_key_order if key in self.metrics_keys]
        remaining = [key for key in self.metrics_keys if key not in preferred_present]
        self.metrics_keys = preferred_present + remaining
        return new_keys


class RoundedCSVLogger(CSVLogger):
    """CSVLogger that writes floating-point metric values rounded to 8 decimal places."""

    def __init__(
        self,
        *args: Any,
        decimal_places: int = CSV_METRIC_DECIMAL_PLACES,
        metrics_filename: str = "metrics.csv",
        preferred_metric_key_order: list[str] | None = None,
        excluded_metric_keys: set[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.decimal_places = int(decimal_places)
        self.metrics_filename = str(metrics_filename)
        self.preferred_metric_key_order = (
            list(preferred_metric_key_order)
            if preferred_metric_key_order is not None
            else None
        )
        self.excluded_metric_keys = set(excluded_metric_keys or set())

    @property
    @rank_zero_experiment
    def experiment(self) -> RoundedMetricsExperimentWriter:
        if self._experiment is not None:
            return self._experiment

        self._fs.makedirs(self.root_dir, exist_ok=True)
        metrics_path = os.path.join(self.log_dir, self.metrics_filename)
        if os.path.exists(metrics_path):
            os.remove(metrics_path)
        self._experiment = RoundedMetricsExperimentWriter(
            log_dir=self.log_dir,
            decimal_places=self.decimal_places,
            metrics_filename=self.metrics_filename,
            preferred_metric_key_order=self.preferred_metric_key_order,
            excluded_metric_keys=self.excluded_metric_keys,
        )
        return self._experiment
