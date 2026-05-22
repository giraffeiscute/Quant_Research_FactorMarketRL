"""LightningModule implementation for portfolio attention training."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch

from ..config import DataConfig, EvaluationConfig, ModelConfig, TrainConfig
from ..config.validation import validated_evaluation_config, validated_train_config
from ..data.dataset import PortfolioPanelDataset
from ..evaluation.metrics import compute_portfolio_sr
from ..training.engine import _run_loss_step, build_training_model
from ..training.lr_scheduler import (
    build_lr_warmup_decay_scheduler,
    resolve_total_optimizer_steps,
)
from ..rl.ppo import build_rollout_ppo_update_metrics
from ..training.rl_engine import (
    collect_rollout_ppo_training_batch,
    run_rl_policy_step,
    run_rollout_ppo_update,
)
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
        evaluation_config: EvaluationConfig | None = None,
        gradient_diagnostics_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.model_config = model_config
        self.train_config = validated_train_config(train_config)
        self.evaluation_config = validated_evaluation_config(evaluation_config or EvaluationConfig())
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
        self.automatic_optimization = not self._uses_manual_rollout_ppo()

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
        compute_value_prediction: bool = False,
    ) -> dict[str, Any]:
        return self.model(
            x_stock,
            x_market,
            stock_indices,
            target_returns=target_returns,
            compute_value_prediction=compute_value_prediction,
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        if bool(self.train_config.rl_training.enabled):
            return self._training_step_rl(batch, batch_idx)

        loss, net_scored_returns, summary = _run_loss_step(
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
        train_sr = compute_portfolio_sr(net_scored_returns)
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
            "train_TO",
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
        self._log_train_epoch_metric(
            "train_SR",
            train_sr,
            prog_bar=False,
            batch_size=batch_size,
        )
        return loss

    def _training_step_rl(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        if self._uses_manual_rollout_ppo():
            return self._training_step_multi_epoch_rollout_ppo(batch, batch_idx)

        result = run_rl_policy_step(
            self.model,
            batch,
            data_config=self.data_config,
            model_config=self.model_config,
            train_config=self.train_config,
            evaluation_config=self.evaluation_config,
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

    def _training_step_multi_epoch_rollout_ppo(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        optimizer.zero_grad()
        collected = collect_rollout_ppo_training_batch(
            self.model,
            batch,
            data_config=self.data_config,
            model_config=self.model_config,
            train_config=self.train_config,
        )
        last_loss: torch.Tensor | None = None
        ppo_num_epochs = int(self.train_config.rl_training.ppo_num_epochs)
        for _ppo_epoch in range(ppo_num_epochs):
            optimizer.zero_grad()
            ppo_update = run_rollout_ppo_update(
                self.model,
                batch,
                collected.ppo_batch,
                data_config=self.data_config,
                model_config=self.model_config,
                train_config=self.train_config,
            )
            self._latest_train_diagnostics = self._build_train_diagnostics(
                loss=ppo_update.policy_loss,
                summary=collected.summary,
                batch_idx=batch_idx,
            )
            self.manual_backward(ppo_update.policy_loss)
            if float(self.train_config.grad_clip_norm) > 0.0:
                self.clip_gradients(
                    optimizer,
                    gradient_clip_val=float(self.train_config.grad_clip_norm),
                    gradient_clip_algorithm="norm",
                )
            optimizer.step()
            self._step_manual_scheduler(scheduler)

            metrics = build_rollout_ppo_update_metrics(collected.ppo_batch, ppo_update)
            for metric_name, metric_value in metrics.items():
                self._log_train_epoch_metric(
                    metric_name,
                    metric_value,
                    prog_bar=(metric_name == "train_policy_loss"),
                    batch_size=collected.batch_size,
                )
            last_loss = ppo_update.policy_loss

        if last_loss is None:
            raise RuntimeError("multi_epoch_rollout_ppo requires ppo_num_epochs >= 1.")
        return last_loss.detach()

    @staticmethod
    def _step_manual_scheduler(scheduler: Any) -> None:
        if scheduler is None:
            return
        if isinstance(scheduler, (list, tuple)):
            for item in scheduler:
                PortfolioLightningModule._step_manual_scheduler(item)
            return
        scheduler.step()

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
            reward_baseline=str(self.evaluation_config.reward_baseline),
        )

        self.val_metric.update(
            loss_value=metrics["loss"],
            window_loss_value=metrics["window_loss"],
            window_count=metrics["window_count"],
            scenario_sr=metrics["scenario_sr"],
            scenario_final_return=metrics["scenario_final_return"],
            selected_stock_count=metrics["selected_stock_count"],
            average_turnover=metrics["average_turnover"],
            mean_cash_weight=metrics["mean_cash_weight"],
            win_count=metrics["win_count"],
            win_rate_window_count=metrics["win_rate_window_count"],
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
            "val_SR",
            metrics["val_SR"],
            prog_bar=False,
        )
        self._log_validation_epoch_metric(
            "val_stock",
            metrics["val_stock"],
            prog_bar=False,
        )
        self._log_validation_epoch_metric(
            "val_TO",
            metrics["val_TO"],
            prog_bar=False,
        )
        self._log_validation_epoch_metric(
            "val_cash",
            metrics["val_cash"],
            prog_bar=False,
        )
        self._log_validation_epoch_metric(
            "val_win_rate",
            metrics["val_win_rate"],
            prog_bar=True,
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

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.train_config.learning_rate),
            weight_decay=float(self.train_config.weight_decay),
        )
        scheduler = build_lr_warmup_decay_scheduler(
            optimizer=optimizer,
            train_config=self.train_config,
            total_steps=self._resolve_total_optimizer_steps(),
        )
        if scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "lr",
            },
        }

    def _resolve_total_optimizer_steps(self) -> int:
        multiplier = self._optimizer_steps_per_training_batch()
        try:
            estimated_steps = int(self.trainer.estimated_stepping_batches)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            estimated_steps = 0
        if estimated_steps > 0:
            return estimated_steps * multiplier

        metadata = getattr(self.dataset, "metadata", None)
        train_samples = int(getattr(metadata, "train_window_count", 0) or 0)
        if train_samples <= 0:
            train_samples = int(getattr(metadata, "num_train_scenarios", 0) or 0)
        train_batch_size = max(1, int(self.data_config.train_batch_size))
        train_batches_per_epoch = int(math.ceil(train_samples / train_batch_size))
        base_steps = resolve_total_optimizer_steps(
            num_epochs=int(self.train_config.num_epochs),
            train_batches_per_epoch=train_batches_per_epoch,
        )
        return base_steps * multiplier

    def _uses_manual_rollout_ppo(self) -> bool:
        rl_config = self.train_config.rl_training
        algorithm = str(getattr(rl_config, "algorithm", "")).strip().lower()
        return bool(getattr(rl_config, "enabled", False)) and algorithm == "multi_epoch_rollout_ppo"

    def _optimizer_steps_per_training_batch(self) -> int:
        if self._uses_manual_rollout_ppo():
            return max(1, int(self.train_config.rl_training.ppo_num_epochs))
        return 1
