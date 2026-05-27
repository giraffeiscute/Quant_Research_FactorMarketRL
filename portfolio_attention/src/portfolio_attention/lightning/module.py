"""LightningModule implementation for portfolio attention training."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch

from ..config import DataConfig, EvaluationConfig, ModelConfig, TrainConfig
from ..config.validation import (
    validate_train_config_against_model_config,
    validated_evaluation_config,
    validated_train_config,
)
from ..data.dataset import PortfolioPanelDataset
from ..evaluation.metrics import compute_portfolio_sr
from ..model import TwinPortfolioQCritic, clone_target_q_critic
from ..rl.replay import SACReplayBuffer
from ..rl.sac import (
    compute_sac_actor_alpha_loss_from_replay_batch,
    compute_sac_q_loss_from_replay_batch,
    soft_update_targets,
)
from ..training.engine import _run_loss_step, build_training_model
from ..training.lr_scheduler import (
    build_lr_warmup_decay_scheduler,
    resolve_total_optimizer_steps,
)
from ..rl.ppo import build_rollout_ppo_update_metrics
from ..training.ppo_engine import (
    collect_rollout_ppo_training_batch,
    run_rollout_ppo_update,
)
from ..training.rl_engine import run_rl_policy_step
from ..training.sac_engine import collect_sac_training_batch
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
        self._validate_rollout_ppo_model_config()
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
        self.automatic_optimization = not self._uses_manual_rl_optimization()

        self.model = build_training_model(
            model_config=model_config,
            dataset=dataset,
            data_config=data_config,
            device=torch.device("cpu"),
        )
        self.sac_q_critic: TwinPortfolioQCritic | None = None
        self.sac_target_q_critic: TwinPortfolioQCritic | None = None
        self.sac_replay_buffer: SACReplayBuffer | None = None
        self.sac_log_alpha: torch.nn.Parameter | None = None
        self._sac_update_count = 0
        if self._uses_sac():
            sac_action_dim = self._resolve_sac_action_dim()
            self.sac_q_critic = TwinPortfolioQCritic(
                stock_temporal_dim=int(model_config.stock_temporal_dim),
                market_temporal_dim=int(model_config.market_temporal_dim),
                action_dim=sac_action_dim,
                hidden_dim=int(model_config.cross_sectional_dim),
            )
            self.sac_target_q_critic = clone_target_q_critic(self.sac_q_critic)
            self.sac_replay_buffer = SACReplayBuffer(
                capacity=int(self.train_config.rl_training.sac.buffer_size)
            )
            if bool(self.train_config.rl_training.sac.auto_entropy):
                temp_init = float(self.train_config.rl_training.sac.temp_init)
                self.sac_log_alpha = torch.nn.Parameter(
                    torch.log(torch.tensor(temp_init, dtype=torch.float32))
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
        if self._uses_sac():
            return self._training_step_sac(batch, batch_idx)
        if self._uses_manual_rollout_ppo():
            return self._training_step_rollout_ppo(batch, batch_idx)

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
                prog_bar=(metric_name == "train_total_loss"),
                batch_size=result.batch_size,
            )
        self._log_current_learning_rate(batch_size=result.batch_size)
        return result.policy_loss

    def _training_step_sac(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        actor_optimizer, critic_optimizer, alpha_optimizer = self._sac_optimizers()
        sac_config = self.train_config.rl_training.sac
        collected = collect_sac_training_batch(
            self.model,
            batch,
            model_config=self.model_config,
            train_config=self.train_config,
        )
        replay_buffer = self._require_sac_replay_buffer()
        replay_buffer.push(collected.transitions)
        replay_size = len(replay_buffer)
        batch_size = int(collected.batch_size)
        self._latest_train_diagnostics = self._build_train_diagnostics(
            loss=collected.dummy_loss,
            summary=collected.summary,
            batch_idx=batch_idx,
        )

        base_metrics: dict[str, torch.Tensor | float] = {
            "train_sac_replay_size": float(replay_size),
            "train_sac_collected_transitions": float(batch_size),
            "train_sac_context_window_steps": float(collected.context_window_steps),
            "train_sac_update_count": float(self._sac_update_count),
            "train_sac_temp": self._current_sac_temp_tensor(),
            "train_sac_sampled_reward_mean": collected.summary[
                "sampled_action_reward_mean"
            ],
            "train_sac_sampled_return": collected.summary[
                "scenario_final_returns"
            ].mean(),
            "train_sac_sampled_turnover": collected.summary["mean_turnover"],
        }
        if replay_size < int(sac_config.warmup_steps):
            for metric_name, metric_value in base_metrics.items():
                self._log_train_epoch_metric(
                    metric_name,
                    metric_value,
                    prog_bar=False,
                    batch_size=batch_size,
            )
            self._log_current_learning_rate(batch_size=batch_size)
            return collected.dummy_loss.detach()

        q_losses: list[torch.Tensor] = []
        actor_losses: list[torch.Tensor] = []
        alpha_losses: list[torch.Tensor] = []
        target_q_means: list[torch.Tensor] = []
        log_prob_means: list[torch.Tensor] = []
        updates_per_batch = max(1, int(sac_config.updates_per_batch))
        for _ in range(updates_per_batch):
            replay_batch = replay_buffer.sample(
                int(sac_config.batch_size),
                device=self.device,
            )

            critic_optimizer.zero_grad()
            q_result = compute_sac_q_loss_from_replay_batch(
                actor_model=self.model,
                q_critic=self._require_sac_q_critic(),
                target_q_critic=self._require_sac_target_q_critic(),
                replay_batch=replay_batch,
                model_config=self.model_config,
                train_config=self.train_config,
                alpha=self._fixed_sac_alpha_or_none(),
                log_alpha=self.sac_log_alpha,
            )
            self.manual_backward(q_result.q_loss)
            self._clip_manual_optimizer_gradients(critic_optimizer)
            critic_optimizer.step()
            critic_optimizer.zero_grad()

            actor_optimizer.zero_grad()
            actor_result = compute_sac_actor_alpha_loss_from_replay_batch(
                actor_model=self.model,
                q_critic=self._require_sac_q_critic(),
                replay_batch=replay_batch,
                model_config=self.model_config,
                train_config=self.train_config,
                alpha=self._fixed_sac_alpha_or_none(),
                log_alpha=self.sac_log_alpha,
                target_entropy=sac_config.target_entropy,
            )
            self.manual_backward(actor_result.actor_loss)
            self._clip_manual_optimizer_gradients(actor_optimizer)
            actor_optimizer.step()
            actor_optimizer.zero_grad()

            if alpha_optimizer is not None and actor_result.alpha_loss is not None:
                alpha_optimizer.zero_grad()
                self.manual_backward(actor_result.alpha_loss)
                alpha_optimizer.step()
                alpha_optimizer.zero_grad()
                alpha_losses.append(actor_result.alpha_loss.detach())

            soft_update_targets(
                self._require_sac_q_critic(),
                self._require_sac_target_q_critic(),
                tau=float(sac_config.tau),
            )
            self._sac_update_count += 1
            q_losses.append(q_result.q_loss.detach())
            actor_losses.append(actor_result.actor_loss.detach())
            target_q_means.append(q_result.target_q.detach().mean())
            log_prob_means.append(actor_result.policy_log_prob.detach().mean())

        metrics = {
            **base_metrics,
            "train_sac_q_loss": torch.stack(q_losses).mean(),
            "train_sac_actor_loss": torch.stack(actor_losses).mean(),
            "train_sac_target_q_mean": torch.stack(target_q_means).mean(),
            "train_sac_log_prob_mean": torch.stack(log_prob_means).mean(),
            "train_sac_update_count": float(self._sac_update_count),
        }
        if alpha_losses:
            metrics["train_sac_temp_loss"] = torch.stack(alpha_losses).mean()
        metrics["train_sac_temp"] = self._current_sac_temp_tensor()
        for metric_name, metric_value in metrics.items():
            self._log_train_epoch_metric(
                metric_name,
                metric_value,
                prog_bar=metric_name in {"train_sac_q_loss", "train_sac_actor_loss"},
                batch_size=batch_size,
            )
        self._log_current_learning_rate(batch_size=batch_size)
        return metrics["train_sac_actor_loss"].detach()

    def _training_step_rollout_ppo(
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
        ppo_num_epochs = int(self.train_config.rl_training.ppo.num_epochs)
        for ppo_epoch in range(1, ppo_num_epochs + 1):
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
            optimizer.zero_grad()

            metrics = build_rollout_ppo_update_metrics(
                collected.ppo_batch,
                ppo_update,
                ppo_epoch=ppo_epoch,
            )
            for metric_name, metric_value in metrics.items():
                self._log_train_epoch_metric(
                    metric_name,
                    metric_value,
                    prog_bar=(metric_name == "train_total_loss"),
                    batch_size=collected.batch_size,
                )
            self._log_current_learning_rate(batch_size=collected.batch_size)
            last_loss = ppo_update.policy_loss

        if last_loss is None:
            raise RuntimeError("rollout_ppo requires ppo_num_epochs >= 1.")
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

    def _log_current_learning_rate(self, *, batch_size: int) -> None:
        lr_value = self._current_learning_rate()
        if lr_value is None:
            return
        self._log_train_epoch_metric(
            "lr",
            lr_value,
            prog_bar=False,
            batch_size=batch_size,
        )

    def _current_learning_rate(self) -> float | None:
        try:
            optimizers = getattr(self.trainer, "optimizers", None)
        except RuntimeError:
            optimizers = None
        if optimizers:
            optimizer = optimizers[0]
            param_groups = getattr(optimizer, "param_groups", None)
            if param_groups:
                return float(param_groups[0]["lr"])
        return float(self.train_config.learning_rate)

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
        if self._uses_sac():
            optimizers: list[torch.optim.Optimizer] = [
                torch.optim.Adam(
                    self.model.parameters(),
                    lr=float(self.train_config.learning_rate),
                    weight_decay=float(self.train_config.weight_decay),
                ),
                torch.optim.Adam(
                    self._require_sac_q_critic().parameters(),
                    lr=float(self.train_config.learning_rate),
                    weight_decay=float(self.train_config.weight_decay),
                ),
            ]
            if self.sac_log_alpha is not None:
                optimizers.append(
                    torch.optim.Adam(
                        [self.sac_log_alpha],
                        lr=float(self.train_config.rl_training.sac.temp_lr),
                    )
                )
            return optimizers
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
        return (
            bool(getattr(rl_config, "enabled", False))
            and algorithm == "rollout_ppo"
            and int(rl_config.ppo.num_epochs) > 1
        )

    def _uses_sac(self) -> bool:
        rl_config = self.train_config.rl_training
        algorithm = str(getattr(rl_config, "algorithm", "")).strip().lower()
        return bool(getattr(rl_config, "enabled", False)) and algorithm == "sac"

    def _uses_manual_rl_optimization(self) -> bool:
        return self._uses_sac() or self._uses_manual_rollout_ppo()

    def _optimizer_steps_per_training_batch(self) -> int:
        if self._uses_sac():
            return max(1, int(self.train_config.rl_training.sac.updates_per_batch))
        if self._uses_manual_rollout_ppo():
            return max(1, int(self.train_config.rl_training.ppo.num_epochs))
        return 1

    def _validate_rollout_ppo_model_config(self) -> None:
        validate_train_config_against_model_config(self.train_config, self.model_config)

    def _sac_optimizers(
        self,
    ) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer | None]:
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        expected_count = 3 if self.sac_log_alpha is not None else 2
        if len(optimizers) != expected_count:
            raise RuntimeError(
                f"SAC expected {expected_count} optimizers, received {len(optimizers)}."
            )
        alpha_optimizer = optimizers[2] if expected_count == 3 else None
        return optimizers[0], optimizers[1], alpha_optimizer

    def _clip_manual_optimizer_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        if float(self.train_config.grad_clip_norm) <= 0.0:
            return
        self.clip_gradients(
            optimizer,
            gradient_clip_val=float(self.train_config.grad_clip_norm),
            gradient_clip_algorithm="norm",
        )

    def _resolve_sac_action_dim(self) -> int:
        dataset_num_stocks = int(getattr(self.dataset, "num_stocks", 0) or 0)
        sample_num_stocks = int(getattr(self.data_config, "sample_num_stocks", dataset_num_stocks))
        if dataset_num_stocks <= 0:
            return sample_num_stocks + 1
        return min(sample_num_stocks, dataset_num_stocks) + 1

    def _fixed_sac_alpha_or_none(self) -> float | None:
        if self.sac_log_alpha is not None:
            return None
        return float(self.train_config.rl_training.sac.temp_init)

    def _current_sac_temp_tensor(self) -> torch.Tensor:
        if self.sac_log_alpha is not None:
            return self.sac_log_alpha.detach().exp()
        reference = next(self.model.parameters())
        return reference.detach().new_tensor(float(self.train_config.rl_training.sac.temp_init))

    def _require_sac_replay_buffer(self) -> SACReplayBuffer:
        if self.sac_replay_buffer is None:
            raise RuntimeError("SAC replay buffer is not initialized.")
        return self.sac_replay_buffer

    def _require_sac_q_critic(self) -> TwinPortfolioQCritic:
        if self.sac_q_critic is None:
            raise RuntimeError("SAC Q critic is not initialized.")
        return self.sac_q_critic

    def _require_sac_target_q_critic(self) -> TwinPortfolioQCritic:
        if self.sac_target_q_critic is None:
            raise RuntimeError("SAC target Q critic is not initialized.")
        return self.sac_target_q_critic
