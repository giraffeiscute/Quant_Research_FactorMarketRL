"""Lightning logging helpers."""

from __future__ import annotations

from typing import Any

from lightning_fabric.loggers.logger import rank_zero_experiment
from pytorch_lightning.loggers.csv_logs import CSVLogger, ExperimentWriter
import torch


CSV_METRIC_DECIMAL_PLACES = 8
VALIDATION_STOCKS_BOUGHT_DECIMAL_PLACES = 1
VAL_STOCK_METRIC_KEY = "val_stock"
PREFERRED_METRIC_KEY_ORDER = [
    "epoch",
    "step",
    "train_loss",
    "train_weight_loss",
    "train_OT",
    "train_return",
    "val_loss",
    "val_loss_window",
    "val_return",
    "val_OT",
    "val_stock",
    "val_cash",
]


class RoundedMetricsExperimentWriter(ExperimentWriter):
    """CSV experiment writer that renders floating-point metrics with fixed precision."""

    def __init__(self, log_dir: str, *, decimal_places: int = CSV_METRIC_DECIMAL_PLACES) -> None:
        super().__init__(log_dir=log_dir)
        self.decimal_places = int(decimal_places)

    def _format_metric_value(self, key: str, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            value = value.item()
        if isinstance(value, float):
            decimal_places = (
                VALIDATION_STOCKS_BOUGHT_DECIMAL_PLACES
                if key == VAL_STOCK_METRIC_KEY
                else self.decimal_places
            )
            return f"{value:.{decimal_places}f}"
        return value

    def log_metrics(self, metrics_dict: dict[str, float], step: int | None = None) -> None:
        if step is None:
            step = len(self.metrics)

        metrics = {key: self._format_metric_value(key, value) for key, value in metrics_dict.items()}
        metrics["step"] = step
        self.metrics.append(metrics)

    def _record_new_keys(self) -> set[str]:
        """Record new keys and keep selected metrics in a stable relative order."""
        current_keys = set().union(*self.metrics)
        new_keys = current_keys - set(self.metrics_keys)
        self.metrics_keys.extend(new_keys)
        self.metrics_keys.sort()
        preferred_present = [key for key in PREFERRED_METRIC_KEY_ORDER if key in self.metrics_keys]
        remaining = [key for key in self.metrics_keys if key not in preferred_present]
        self.metrics_keys = preferred_present + remaining
        return new_keys


class RoundedCSVLogger(CSVLogger):
    """CSVLogger that writes floating-point metric values rounded to 8 decimal places."""

    def __init__(
        self,
        *args: Any,
        decimal_places: int = CSV_METRIC_DECIMAL_PLACES,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.decimal_places = int(decimal_places)

    @property
    @rank_zero_experiment
    def experiment(self) -> RoundedMetricsExperimentWriter:
        if self._experiment is not None:
            return self._experiment

        self._fs.makedirs(self.root_dir, exist_ok=True)
        self._experiment = RoundedMetricsExperimentWriter(
            log_dir=self.log_dir,
            decimal_places=self.decimal_places,
        )
        return self._experiment
