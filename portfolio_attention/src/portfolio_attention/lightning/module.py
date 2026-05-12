"""LightningModule implementation for portfolio attention training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch

from ..config import DataConfig, ModelConfig, TrainConfig
from ..config.validation import validated_train_config
from ..data.dataset import PortfolioPanelDataset
from ..training.engine import _run_loss_step, build_training_model
from ..training.rl_engine import run_rl_policy_step
from .gradient_diagnostics import (
    GradientDiagnosticsCSVWriter,
    compute_gradient_diagnostics,
)
from .validation import (
    ScenarioRollingValidationMetric,
    compute_validation_scenario_metrics,
)


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
        evaluation_transaction_cost_rate: float = 0.0,
        gradient_diagnostics_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.model_config = model_config
        self.train_config = validated_train_config(train_config)
        self.dataset = dataset
        self.stock_count_weight_threshold = float(stock_count_weight_threshold)
        self.stock_count_min_active_days = int(stock_count_min_active_days)
        self.evaluation_transaction_cost_rate = float(evaluation_transaction_cost_rate)
        self.gradient_diagnostics_writer = (
            GradientDiagnosticsCSVWriter(Path(gradient_diagnostics_path))
            if gradient_diagnostics_path is not None
            else None
        )
        self._latest_train_diagnostics: dict[str, Any] = {}

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
        if bool(self.train_config.rl_training.enabled):
            return self._training_step_rl(batch, batch_idx)

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
        self._latest_train_diagnostics = self._build_train_diagnostics(
            loss=loss,
            summary=summary,
            batch_idx=batch_idx,
        )

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
            "train_return",
            train_mean_final_return,
            prog_bar=False,
            batch_size=batch_size,
        )
        return loss

    def _training_step_rl(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        result = run_rl_policy_step(
            self.model,
            batch,
            data_config=self.data_config,
            train_config=self.train_config,
        )
        self._latest_train_diagnostics = self._build_train_diagnostics(
            loss=result.policy_loss,
            summary=result.summary,
            batch_idx=batch_idx,
        )
        for metric_name, metric_value in result.metrics.items():
            self._log_train_epoch_metric(
                metric_name,
                metric_value,
                prog_bar=(metric_name == "train_policy_loss"),
                batch_size=result.batch_size,
            )
        return result.policy_loss

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer
        diagnostics = compute_gradient_diagnostics(self.model.named_parameters())
        row = {
            "event": "gradient",
            "epoch": int(self.current_epoch),
            "global_step": int(self.global_step),
            **self._latest_train_diagnostics,
            "grad_norm": diagnostics.grad_norm,
            "grad_abs_max": diagnostics.grad_abs_max,
            "grad_nonfinite_count": diagnostics.grad_nonfinite_count,
            "first_nonfinite_param": diagnostics.first_nonfinite_param,
        }

        if diagnostics.grad_nonfinite_count > 0:
            row["event"] = "nonfinite_gradient"
            self._write_gradient_diagnostics(row)
            if bool(self.train_config.grad_monitor_fail_fast):
                raise FloatingPointError(
                    "Non-finite gradient detected before optimizer step: "
                    f"global_step={int(self.global_step)} "
                    f"epoch={int(self.current_epoch)} "
                    f"first_nonfinite_param={diagnostics.first_nonfinite_param!r} "
                    f"grad_nonfinite_count={diagnostics.grad_nonfinite_count}."
                )
            return

        interval = int(self.train_config.grad_monitor_interval_steps)
        if interval > 0 and int(self.global_step) % interval == 0:
            self._write_gradient_diagnostics(row)

    def on_validation_epoch_start(self) -> None:
        self.val_metric.reset()

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        del batch_idx
        metrics = compute_validation_scenario_metrics(
            model=self.model,
            dataset=self.dataset,
            raw_batch=batch,
            device=self.device,
            loss_name=self.train_config.loss_name,
            turnover_penalty=float(self.train_config.turnover_penalty),
            evaluation_transaction_cost_rate=float(self.evaluation_transaction_cost_rate),
            turnover_penalty_norm=str(self.train_config.turnover_penalty_norm),
            stock_count_weight_threshold=self.stock_count_weight_threshold,
            stock_count_min_active_days=self.stock_count_min_active_days,
        )

        self.val_metric.update(
            loss_value=metrics["loss"],
            window_loss_value=metrics["window_loss"],
            window_count=metrics["window_count"],
            scenario_final_return=metrics["scenario_final_return"],
            selected_stock_count=metrics["selected_stock_count"],
            average_turnover=metrics["average_turnover"],
            mean_cash_weight=metrics["mean_cash_weight"],
        )

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metric.compute()
        self._log_validation_epoch_metric("val_loss", metrics["val_loss"], prog_bar=True)
        self._log_validation_epoch_metric("val_loss_window", metrics["val_loss_window"], prog_bar=False)
        self._log_validation_epoch_metric(
            "val_return",
            metrics["val_return"],
            prog_bar=True,
        )
        self._log_validation_epoch_metric(
            "val_stock",
            metrics["val_stock"],
            prog_bar=False,
        )
        self._log_validation_epoch_metric(
            "val_OT",
            metrics["val_OT"],
            prog_bar=False,
        )
        self._log_validation_epoch_metric(
            "val_cash",
            metrics["val_cash"],
            prog_bar=False,
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

    def _build_train_diagnostics(
        self,
        *,
        loss: torch.Tensor,
        summary: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, Any]:
        keys = (
            "return_mean_min",
            "return_mean_max",
            "return_std_min",
            "return_std_max",
            "allocation_logits_abs_max",
            "raw_allocation_min",
            "raw_allocation_max",
        )
        diagnostics: dict[str, Any] = {
            "batch_idx": int(batch_idx),
            "train_loss": self._to_diagnostic_value(loss),
        }
        for key in keys:
            diagnostics[key] = self._to_diagnostic_value(summary.get(key))
        return diagnostics

    def _write_gradient_diagnostics(self, row: dict[str, Any]) -> None:
        if self.gradient_diagnostics_writer is None:
            return
        try:
            trainer = self.trainer
        except RuntimeError:
            trainer = None
        if trainer is not None and not bool(getattr(trainer, "is_global_zero", True)):
            return
        self.gradient_diagnostics_writer.write_row(row)

    @staticmethod
    def _to_diagnostic_value(value: Any) -> float | str:
        if value is None:
            return ""
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        if isinstance(value, (int, float)):
            return float(value)
        return str(value)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.train_config.learning_rate),
            weight_decay=float(self.train_config.weight_decay),
        )
