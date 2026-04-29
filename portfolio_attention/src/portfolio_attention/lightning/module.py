"""LightningModule implementation for portfolio attention training."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch

from ..config import DataConfig, ModelConfig, TrainConfig
from ..data.dataset import PortfolioPanelDataset
from ..evaluation.metrics import (
    compute_average_turnover_from_weights,
    compute_selected_stock_count_from_weights,
)
from ..evaluation.runtime import _collect_single_scenario_rolling_one_step_outputs
from ..model.losses import build_loss
from ..training.engine import _run_loss_step, build_training_model
from .validation import ScenarioRollingValidationMetric, compute_validation_window_objective_loss


class PortfolioLightningModule(pl.LightningModule):
    """LightningModule that reuses the repo's model/loss/validation helpers."""

    def __init__(
        self,
        *,
        data_config: DataConfig,
        model_config: ModelConfig,
        train_config: TrainConfig,
        dataset: PortfolioPanelDataset,
        stock_count_weight_threshold: float,
        stock_count_min_active_days: int,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.model_config = model_config
        self.train_config = train_config
        self.dataset = dataset
        self.stock_count_weight_threshold = float(stock_count_weight_threshold)
        self.stock_count_min_active_days = int(stock_count_min_active_days)

        self.model = build_training_model(
            model_config=model_config,
            dataset=dataset,
            data_config=data_config,
            device=torch.device("cpu"),
        )
        self.val_metric = ScenarioRollingValidationMetric()

    def forward(
        self,
        x_stock: torch.Tensor,
        x_market: torch.Tensor,
        stock_indices: torch.Tensor,
        target_returns: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        return self.model(
            x_stock,
            x_market,
            stock_indices,
            target_returns=target_returns,
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        loss, _, summary = _run_loss_step(
            self.model,
            batch,
            self.train_config.loss_name,
            turnover_penalty=self.train_config.turnover_penalty,
            transaction_cost_rate=self.train_config.transaction_cost_rate,
            turnover_penalty_norm=self.train_config.turnover_penalty_norm,
        )
        train_mean_final_return = summary["scenario_final_returns"].mean()
        train_weight_loss = summary.get("weight_loss", loss.new_zeros(()))
        train_ot = summary.get("mean_turnover", loss.new_zeros(()))
        batch_size = int(summary["scenario_final_returns"].numel())

        self._log_train_epoch_metric("train_loss", loss, prog_bar=True, batch_size=batch_size)
        self._log_train_epoch_metric(
            "train_weight_loss",
            train_weight_loss,
            prog_bar=False,
            batch_size=batch_size,
        )
        self._log_train_epoch_metric(
            "train_OT",
            train_ot,
            prog_bar=False,
            batch_size=batch_size,
        )
        self._log_train_epoch_metric(
            "train_mean_final_return",
            train_mean_final_return,
            prog_bar=False,
            batch_size=batch_size,
        )
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_metric.reset()

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        del batch_idx
        rolling_outputs = _collect_single_scenario_rolling_one_step_outputs(
            model=self.model,
            dataset=self.dataset,
            raw_batch=batch,
            device=self.device,
            lookback_days=int(self.dataset.metadata.lookback_days),
            evaluation_label="Lightning validation rolling evaluation",
            collect_weights=True,
        )
        scored_returns = rolling_outputs["portfolio_returns"].unsqueeze(0)
        loss = build_loss(self.train_config.loss_name, scored_returns)
        window_loss, window_count = compute_validation_window_objective_loss(
            model=self.model,
            dataset=self.dataset,
            raw_batch=batch,
            device=self.device,
            lookback_days=int(self.dataset.metadata.lookback_days),
            rolling_horizon_days=int(self.dataset.metadata.rolling_horizon_days),
            rolling_stride_days=int(self.dataset.metadata.rolling_stride_days),
            loss_name=self.train_config.loss_name,
            turnover_penalty=float(self.train_config.turnover_penalty),
            transaction_cost_rate=float(self.train_config.transaction_cost_rate),
            turnover_penalty_norm=str(self.train_config.turnover_penalty_norm),
        )
        scenario_final_return = (torch.prod(1.0 + scored_returns, dim=1) - 1.0).mean()
        selected_stock_count = compute_selected_stock_count_from_weights(
            rolling_outputs["stock_weights"],
            threshold=self.stock_count_weight_threshold,
            min_active_days=self.stock_count_min_active_days,
        )
        average_turnover = compute_average_turnover_from_weights(
            rolling_outputs["stock_weights"],
            rolling_outputs["cash_weights"],
        )
        mean_cash_weight = rolling_outputs["cash_weights"].mean()

        self.val_metric.update(
            loss_value=loss.detach(),
            window_loss_value=window_loss.detach(),
            window_count=window_count,
            scenario_final_return=scenario_final_return.detach(),
            selected_stock_count=selected_stock_count,
            average_turnover=average_turnover,
            mean_cash_weight=mean_cash_weight,
        )

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metric.compute()
        self._log_validation_epoch_metric("val_loss", metrics["val_loss"], prog_bar=True)
        self._log_validation_epoch_metric("val_loss_window", metrics["val_loss_window"], prog_bar=False)
        self._log_validation_epoch_metric(
            "val_mean_final_return",
            metrics["val_mean_final_return"],
            prog_bar=True,
        )
        self._log_validation_epoch_metric(
            "validation_stocks_bought",
            metrics["validation_stocks_bought"],
            prog_bar=False,
        )
        self._log_validation_epoch_metric(
            "validation_average_turnover",
            metrics["validation_average_turnover"],
            prog_bar=False,
        )
        self._log_validation_epoch_metric(
            "validation_mean_cash_weight",
            metrics["validation_mean_cash_weight"],
            prog_bar=False,
        )

    def _log_train_epoch_metric(
        self,
        name: str,
        value: torch.Tensor | float,
        *,
        prog_bar: bool,
        batch_size: int,
    ) -> None:
        self.log(
            name,
            value,
            on_step=False,
            on_epoch=True,
            prog_bar=prog_bar,
            logger=True,
            sync_dist=True,
            batch_size=int(batch_size),
        )

    def _log_validation_epoch_metric(
        self,
        name: str,
        value: torch.Tensor | float,
        *,
        prog_bar: bool,
    ) -> None:
        self.log(
            name,
            value,
            on_step=False,
            on_epoch=True,
            prog_bar=prog_bar,
            logger=True,
            sync_dist=False,
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.train_config.learning_rate),
            weight_decay=float(self.train_config.weight_decay),
        )
