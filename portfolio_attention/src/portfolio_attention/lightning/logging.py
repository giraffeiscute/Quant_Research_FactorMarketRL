"""Lightning logging helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from lightning_fabric.loggers.logger import rank_zero_experiment
from lightning.pytorch.loggers.csv_logs import CSVLogger, ExperimentWriter
from lightning.pytorch.loggers.logger import Logger
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
    "lr",
]
GRPO_PREFERRED_METRIC_KEY_ORDER = [
    "epoch",
    "step",
    "train_loss",
    "train_weight_loss",
    "train_TO",
    "train_return",
    "val_loss",
    "val_loss_window",
    "val_return",
    "val_TO",
    "val_stock",
    "val_cash",
    "val_win_rate",
    "lr",
]
GRPO_INCLUDED_METRIC_KEYS = set(GRPO_PREFERRED_METRIC_KEY_ORDER)
RL_PREFERRED_METRIC_KEY_ORDER = [
    "epoch",
    "step",
    "train_rollout_total_loss",
    "train_policy_loss",
    "train_rollout_value_loss",
    "train_rollout_ppo_ratio_mean",
    "train_rollout_ppo_clip_fraction",
    "train_rollout_ppo_approx_kl",
    "train_reward_final",
    "train_return",
    "train_TO",
    "val_loss",
    "val_return",
    "val_SR",
    "val_TO",
    "val_win_rate",
    "train_rollout_ppo_epoch",
    "val_stock",
    "val_cash",
    "lr",
]
RL_INCLUDED_METRIC_KEYS = set(RL_PREFERRED_METRIC_KEY_ORDER)


@dataclass(frozen=True)
class MetricFilterConfig:
    """Metric selection shared by CSV and external loggers."""

    preferred_metric_key_order: list[str] | None = None
    excluded_metric_keys: set[str] | None = None
    included_metric_keys: set[str] | None = None

    def filter(self, metrics_dict: dict[str, Any]) -> dict[str, Any]:
        return filter_metric_keys(
            metrics_dict,
            excluded_metric_keys=self.excluded_metric_keys,
            included_metric_keys=self.included_metric_keys,
        )


def filter_metric_keys(
    metrics_dict: dict[str, Any],
    *,
    excluded_metric_keys: set[str] | None = None,
    included_metric_keys: set[str] | None = None,
) -> dict[str, Any]:
    excluded = set(excluded_metric_keys or set())
    included = set(included_metric_keys) if included_metric_keys is not None else None
    return {
        key: value
        for key, value in metrics_dict.items()
        if key not in excluded and (included is None or key in included)
    }


class MetricFilteringLogger(Logger):
    """Logger adapter that keeps metric filtering out of training code."""

    def __init__(
        self,
        logger: Logger,
        *,
        metric_filter: MetricFilterConfig | None = None,
    ) -> None:
        super().__init__()
        self.logger = logger
        self.metric_filter = metric_filter or MetricFilterConfig()

    @property
    def name(self) -> str | None:
        return getattr(self.logger, "name", None)

    @property
    def version(self) -> str | int | None:
        return getattr(self.logger, "version", None)

    @property
    @rank_zero_experiment
    def experiment(self) -> Any:
        return self.logger.experiment

    def log_hyperparams(self, params: Any, *args: Any, **kwargs: Any) -> None:
        self.logger.log_hyperparams(params, *args, **kwargs)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        filtered = self.metric_filter.filter(metrics)
        if filtered:
            self.logger.log_metrics(filtered, step=step)

    @property
    def excluded_metric_keys(self) -> set[str]:
        return set(self.metric_filter.excluded_metric_keys or set())

    @property
    def included_metric_keys(self) -> set[str] | None:
        if self.metric_filter.included_metric_keys is None:
            return None
        return set(self.metric_filter.included_metric_keys)

    def save(self) -> None:
        self.logger.save()

    def finalize(self, status: str) -> None:
        self.logger.finalize(status)

    def after_save_checkpoint(self, checkpoint_callback: Any) -> None:
        hook = getattr(self.logger, "after_save_checkpoint", None)
        if hook is not None:
            hook(checkpoint_callback)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.logger, name)


class RoundedMetricsExperimentWriter(ExperimentWriter):
    """CSV experiment writer that renders floating-point metrics with fixed precision."""

    def __init__(
        self,
        log_dir: str,
        *,
        decimal_places: int = CSV_METRIC_DECIMAL_PLACES,
        metrics_filename: str = "metrics.csv",
        metric_filter: MetricFilterConfig | None = None,
    ) -> None:
        super().__init__(log_dir=log_dir)
        self._fs.makedirs(self.log_dir, exist_ok=True)
        self.decimal_places = int(decimal_places)
        self.metrics_filename = str(metrics_filename)
        self.metrics_file_path = os.path.join(self.log_dir, self.metrics_filename)
        self.metric_filter = metric_filter or MetricFilterConfig()
        self.preferred_metric_key_order = (
            list(self.metric_filter.preferred_metric_key_order)
            if self.metric_filter.preferred_metric_key_order is not None
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

        filtered_metrics = self.metric_filter.filter(metrics_dict)
        metrics = {
            key: self._format_metric_value(key, value)
            for key, value in filtered_metrics.items()
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

    def save(self) -> None:
        self._fs.makedirs(self.log_dir, exist_ok=True)
        super().save()


class RoundedCSVLogger(CSVLogger):
    """CSVLogger that writes floating-point metric values rounded to 8 decimal places."""

    def __init__(
        self,
        *args: Any,
        decimal_places: int = CSV_METRIC_DECIMAL_PLACES,
        metrics_filename: str = "metrics.csv",
        preferred_metric_key_order: list[str] | None = None,
        excluded_metric_keys: set[str] | None = None,
        included_metric_keys: set[str] | None = None,
        metric_filter: MetricFilterConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.decimal_places = int(decimal_places)
        self.metrics_filename = str(metrics_filename)
        self.metric_filter = metric_filter or MetricFilterConfig(
            preferred_metric_key_order=(
                list(preferred_metric_key_order)
                if preferred_metric_key_order is not None
                else None
            ),
            excluded_metric_keys=set(excluded_metric_keys or set()),
            included_metric_keys=(
                set(included_metric_keys)
                if included_metric_keys is not None
                else None
            ),
        )

    @property
    def preferred_metric_key_order(self) -> list[str] | None:
        if self.metric_filter.preferred_metric_key_order is None:
            return None
        return list(self.metric_filter.preferred_metric_key_order)

    @property
    def excluded_metric_keys(self) -> set[str]:
        return set(self.metric_filter.excluded_metric_keys or set())

    @property
    def included_metric_keys(self) -> set[str] | None:
        if self.metric_filter.included_metric_keys is None:
            return None
        return set(self.metric_filter.included_metric_keys)

    @property
    @rank_zero_experiment
    def experiment(self) -> RoundedMetricsExperimentWriter:
        if self._experiment is not None:
            return self._experiment

        self._fs.makedirs(self.log_dir, exist_ok=True)
        metrics_path = os.path.join(self.log_dir, self.metrics_filename)
        if os.path.exists(metrics_path):
            os.remove(metrics_path)
        self._experiment = RoundedMetricsExperimentWriter(
            log_dir=self.log_dir,
            decimal_places=self.decimal_places,
            metrics_filename=self.metrics_filename,
            metric_filter=self.metric_filter,
        )
        return self._experiment
