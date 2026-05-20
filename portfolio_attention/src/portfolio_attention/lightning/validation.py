"""Lightning validation helpers and distributed metrics."""

from __future__ import annotations

from typing import Any

import torch
from torchmetrics import Metric

from ..common.win_rate import compute_win_rate_metrics
from ..data.dataset import PortfolioPanelDataset
from ..evaluation.metrics import (
    apply_transaction_cost_to_returns,
    compute_average_turnover_from_weights,
    compute_selected_stock_count_from_weights,
)
from ..evaluation.runtime import (
    _rebuild_evaluation_window_x_stock,
    _slice_single_scenario_rolling_window_batch,
    collect_single_scenario_rolling_outputs,
)
from ..evaluation.types import RollingScenarioOutputs
from ..model.losses import build_loss
from ..training.engine import _run_loss_step


@torch.no_grad()
def compute_validation_window_objective_loss(
    *,
    model: torch.nn.Module,
    dataset: PortfolioPanelDataset,
    raw_batch: dict[str, Any],
    device: torch.device,
    lookback_days: int,
    rolling_horizon_days: int,
    rolling_stride_days: int,
    loss_name: str,
    turnover_penalty: float,
    transaction_cost_rate: float,
    turnover_penalty_norm: str,
) -> tuple[torch.Tensor, int]:
    resolved_lookback_days = int(lookback_days)
    resolved_horizon_days = int(rolling_horizon_days)
    resolved_stride_days = int(rolling_stride_days)
    if resolved_lookback_days <= 0:
        raise ValueError(f"lookback_days must be positive, received {lookback_days}.")
    if resolved_horizon_days <= 0:
        raise ValueError(f"rolling_horizon_days must be positive, received {rolling_horizon_days}.")
    if resolved_stride_days <= 0:
        raise ValueError(f"rolling_stride_days must be positive, received {rolling_stride_days}.")

    target_time_indices = raw_batch.get("target_time_indices")
    if not isinstance(target_time_indices, torch.Tensor):
        raise RuntimeError("Validation window loss requires tensor target_time_indices.")
    if target_time_indices.ndim != 2 or int(target_time_indices.shape[0]) != 1:
        raise RuntimeError(
            "Validation window loss expects batch_size=1 target_time_indices with shape [1, T]. "
            f"Received {tuple(target_time_indices.shape)}."
        )

    full_time_steps = int(target_time_indices.shape[1])
    context_time_steps = resolved_lookback_days + resolved_horizon_days
    if full_time_steps < context_time_steps:
        raise RuntimeError(
            "Validation window loss requires at least lookback_days + rolling_horizon_days "
            f"time steps. Received full_time_steps={full_time_steps}, "
            f"lookback_days={resolved_lookback_days}, rolling_horizon_days={resolved_horizon_days}."
        )

    loss_sum = torch.zeros((), device=device)
    window_count = 0
    for window_start in range(0, full_time_steps - context_time_steps + 1, resolved_stride_days):
        window_stop = window_start + context_time_steps
        window_batch = _slice_single_scenario_rolling_window_batch(
            raw_batch,
            window_start=window_start,
            window_stop=window_stop,
        )
        score_mask = torch.zeros((1, context_time_steps), dtype=torch.bool)
        score_mask[:, resolved_lookback_days:] = True
        window_batch["score_mask"] = score_mask
        window_batch = _rebuild_evaluation_window_x_stock(
            window_batch=window_batch,
            dataset=dataset,
            evaluation_label="Lightning validation window loss",
        )
        window_batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in window_batch.items()
        }
        loss, _, _ = _run_loss_step(
            model,
            window_batch,
            loss_name,
            turnover_penalty=turnover_penalty,
            transaction_cost_rate=transaction_cost_rate,
            turnover_penalty_norm=turnover_penalty_norm,
        )
        loss_sum = loss_sum + loss.detach()
        window_count += 1

    if window_count <= 0:
        raise RuntimeError("Validation window loss produced no windows.")
    return loss_sum / window_count, window_count


def _coerce_rolling_outputs(value: RollingScenarioOutputs | dict[str, Any]) -> RollingScenarioOutputs:
    if isinstance(value, RollingScenarioOutputs):
        return value
    portfolio_returns = value.get("portfolio_returns")
    turnover = value.get("turnover")
    stock_weights = value.get("stock_weights")
    cash_weights = value.get("cash_weights")
    previous_allocation = value.get("previous_allocation")
    if not isinstance(portfolio_returns, torch.Tensor):
        raise RuntimeError("Validation rolling outputs require portfolio_returns tensor.")
    if not isinstance(turnover, torch.Tensor):
        raise RuntimeError("Validation rolling outputs require turnover tensor.")
    scored_target_time_indices = value.get("scored_target_time_indices")
    if not isinstance(scored_target_time_indices, torch.Tensor):
        scored_target_time_indices = torch.arange(
            int(portfolio_returns.shape[0]),
            dtype=torch.long,
        )
    context_target_time_indices = value.get("context_target_time_indices")
    if not isinstance(context_target_time_indices, torch.Tensor):
        context_target_time_indices = scored_target_time_indices
    return RollingScenarioOutputs(
        scenario_id=str(value.get("scenario_id", "")),
        source_path=str(value.get("source_path", "")),
        portfolio_returns=portfolio_returns,
        turnover=turnover,
        scored_target_time_indices=scored_target_time_indices,
        context_target_time_indices=context_target_time_indices,
        lookback_days=int(value.get("lookback_days", 0)),
        context_time_steps=int(value.get("context_time_steps", int(context_target_time_indices.shape[0]))),
        num_rolling_windows=int(value.get("num_rolling_windows", int(portfolio_returns.shape[0]))),
        evaluation_price_anchor_mode=str(value.get("evaluation_price_anchor_mode", "")),
        stock_weights=stock_weights if isinstance(stock_weights, torch.Tensor) else None,
        cash_weights=cash_weights if isinstance(cash_weights, torch.Tensor) else None,
        previous_allocation=(
            previous_allocation if isinstance(previous_allocation, torch.Tensor) else None
        ),
    )


def _collect_validation_rolling_outputs(**kwargs: Any) -> RollingScenarioOutputs | dict[str, Any]:
    return collect_single_scenario_rolling_outputs(**kwargs)


def compute_validation_scenario_metrics(
    *,
    model: torch.nn.Module,
    dataset: PortfolioPanelDataset,
    raw_batch: dict[str, Any],
    device: torch.device,
    loss_name: str,
    turnover_penalty: float,
    evaluation_transaction_cost_rate: float,
    turnover_penalty_norm: str,
    stock_count_weight_threshold: float,
    stock_count_min_active_days: int,
    reward_baseline: str = "cash",
) -> dict[str, torch.Tensor | float | int]:
    rolling_outputs = _coerce_rolling_outputs(
        _collect_validation_rolling_outputs(
            model=model,
            dataset=dataset,
            raw_batch=raw_batch,
            device=device,
            lookback_days=int(dataset.metadata.lookback_days),
            evaluation_label="Lightning validation rolling evaluation",
            collect_weights=True,
        )
    )
    if (
        rolling_outputs.stock_weights is None
        or rolling_outputs.cash_weights is None
    ):
        raise RuntimeError("Lightning validation requires stock and cash weights.")
    scored_returns = apply_transaction_cost_to_returns(
        rolling_outputs.portfolio_returns,
        rolling_outputs.turnover,
        transaction_cost_rate=evaluation_transaction_cost_rate,
    ).unsqueeze(0)
    loss = build_loss(loss_name, scored_returns)
    window_loss, window_count = _compute_validation_window_objective_loss(
        model=model,
        dataset=dataset,
        raw_batch=raw_batch,
        device=device,
        lookback_days=int(dataset.metadata.lookback_days),
        rolling_horizon_days=int(dataset.metadata.rolling_horizon_days),
        rolling_stride_days=int(dataset.metadata.rolling_stride_days),
        loss_name=loss_name,
        turnover_penalty=turnover_penalty,
        transaction_cost_rate=evaluation_transaction_cost_rate,
        turnover_penalty_norm=turnover_penalty_norm,
    )
    scenario_final_return = (torch.prod(1.0 + scored_returns, dim=1) - 1.0).mean()
    selected_stock_count = compute_selected_stock_count_from_weights(
        rolling_outputs.stock_weights,
        threshold=stock_count_weight_threshold,
        min_active_days=stock_count_min_active_days,
    )
    average_turnover = compute_average_turnover_from_weights(
        rolling_outputs.stock_weights,
        rolling_outputs.cash_weights,
    )
    mean_cash_weight = rolling_outputs.cash_weights.mean()
    score_mask = raw_batch.get("score_mask")
    raw_stock_returns = raw_batch.get("r_stock")
    if not isinstance(score_mask, torch.Tensor):
        raise RuntimeError("Lightning validation win rate requires tensor score_mask.")
    if not isinstance(raw_stock_returns, torch.Tensor):
        raise RuntimeError("Lightning validation win rate requires tensor r_stock.")
    if not isinstance(rolling_outputs.previous_allocation, torch.Tensor):
        raise RuntimeError("Lightning validation win rate requires previous_allocation.")
    if score_mask.ndim != 2 or int(score_mask.shape[0]) != 1:
        raise RuntimeError(
            "Lightning validation win rate expects score_mask with shape [1, T]. "
            f"Received {tuple(score_mask.shape)}."
        )
    if raw_stock_returns.ndim != 3 or int(raw_stock_returns.shape[0]) != 1:
        raise RuntimeError(
            "Lightning validation win rate expects r_stock with shape [1, T, N]. "
            f"Received {tuple(raw_stock_returns.shape)}."
        )
    scored_stock_returns = raw_stock_returns[0, score_mask[0].to(dtype=torch.bool)]
    win_rate_metrics = compute_win_rate_metrics(
        scored_returns.squeeze(0),
        scored_stock_returns,
        rolling_outputs.previous_allocation,
        reward_baseline=reward_baseline,
        transaction_cost_rate=evaluation_transaction_cost_rate,
    )
    win_count = win_rate_metrics.win_count
    win_rate_window_count = win_rate_metrics.window_count
    return {
        "loss": loss.detach(),
        "window_loss": window_loss.detach(),
        "window_count": window_count,
        "scenario_final_return": scenario_final_return.detach(),
        "selected_stock_count": selected_stock_count,
        "average_turnover": average_turnover,
        "mean_cash_weight": mean_cash_weight,
        "win_count": win_count,
        "win_rate_window_count": win_rate_window_count,
    }


class ScenarioRollingValidationMetric(Metric):
    """Aggregate scenario-level validation outputs across DDP workers."""

    full_state_update = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state("loss_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("window_loss_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("final_return_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("scenario_count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("window_count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("selected_stock_count_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("average_turnover_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("mean_cash_weight_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("win_count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state(
            "win_rate_window_count",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        *,
        loss_value: torch.Tensor | float,
        window_loss_value: torch.Tensor | float,
        window_count: torch.Tensor | int,
        scenario_final_return: torch.Tensor | float,
        selected_stock_count: torch.Tensor | int | float,
        average_turnover: torch.Tensor | float,
        mean_cash_weight: torch.Tensor | float,
        win_count: torch.Tensor | int = 0,
        win_rate_window_count: torch.Tensor | int = 0,
        scenario_count: torch.Tensor | int = 1,
    ) -> None:
        self.loss_sum += torch.as_tensor(loss_value, device=self.loss_sum.device, dtype=self.loss_sum.dtype)
        window_count_tensor = torch.as_tensor(
            window_count,
            device=self.window_count.device,
            dtype=self.window_count.dtype,
        )
        self.window_loss_sum += torch.as_tensor(
            window_loss_value,
            device=self.window_loss_sum.device,
            dtype=self.window_loss_sum.dtype,
        ) * window_count_tensor.to(dtype=self.window_loss_sum.dtype)
        self.final_return_sum += torch.as_tensor(
            scenario_final_return,
            device=self.final_return_sum.device,
            dtype=self.final_return_sum.dtype,
        )
        self.scenario_count += torch.as_tensor(
            scenario_count,
            device=self.scenario_count.device,
            dtype=self.scenario_count.dtype,
        )
        self.window_count += window_count_tensor
        self.selected_stock_count_sum += torch.as_tensor(
            selected_stock_count,
            device=self.selected_stock_count_sum.device,
            dtype=self.selected_stock_count_sum.dtype,
        )
        self.average_turnover_sum += torch.as_tensor(
            average_turnover,
            device=self.average_turnover_sum.device,
            dtype=self.average_turnover_sum.dtype,
        )
        self.mean_cash_weight_sum += torch.as_tensor(
            mean_cash_weight,
            device=self.mean_cash_weight_sum.device,
            dtype=self.mean_cash_weight_sum.dtype,
        )
        self.win_count += torch.as_tensor(
            win_count,
            device=self.win_count.device,
            dtype=self.win_count.dtype,
        )
        self.win_rate_window_count += torch.as_tensor(
            win_rate_window_count,
            device=self.win_rate_window_count.device,
            dtype=self.win_rate_window_count.dtype,
        )

    def compute(self) -> dict[str, torch.Tensor]:
        if int(self.scenario_count.item()) <= 0:
            zero = self.loss_sum.new_zeros(())
            return {
                "val_loss": zero,
                "val_loss_window": zero,
                "val_return": zero,
                "val_mean_final_return": zero,
                "val_stock": zero,
                "val_OT": zero,
                "val_cash": zero,
                "val_win_rate": zero,
            }

        scenario_count = self.scenario_count.to(dtype=self.loss_sum.dtype)
        if int(self.window_count.item()) <= 0:
            val_loss_window = self.window_loss_sum.new_zeros(())
        else:
            val_loss_window = self.window_loss_sum / self.window_count.to(dtype=self.window_loss_sum.dtype)
        return {
            "val_loss": self.loss_sum / scenario_count,
            "val_loss_window": val_loss_window,
            "val_return": self.final_return_sum / scenario_count,
            "val_mean_final_return": self.final_return_sum / scenario_count,
            "val_stock": self.selected_stock_count_sum / scenario_count,
            "val_OT": self.average_turnover_sum / scenario_count,
            "val_cash": self.mean_cash_weight_sum / scenario_count,
            "val_win_rate": (
                self.loss_sum.new_zeros(())
                if int(self.win_rate_window_count.item()) <= 0
                else self.win_count.to(dtype=self.loss_sum.dtype)
                / self.win_rate_window_count.to(dtype=self.loss_sum.dtype)
            ),
        }


_compute_validation_window_objective_loss = compute_validation_window_objective_loss
_compute_validation_scenario_metrics = compute_validation_scenario_metrics
