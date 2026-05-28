"""Runtime execution helpers for evaluation and monitoring workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from ..config import EvaluationConfig
from ..data.dataset import PortfolioPanelDataset, scale_stock_feature_context_array
from ..rl.rebalance import build_rebalance_schedule
from .results import build_legacy_per_scenario_payload
from .types import RollingScenarioOutputs
from ..model import PortfolioAttentionModel

ROLLING_ONE_STEP_EVALUATION_MODE = "rolling_one_step"
ROLLING_ONE_STEP_HORIZON_DAYS = 1
ROLLING_ONE_STEP_STRIDE_DAYS = 1
EVALUATION_PRICE_ANCHOR_MODE_PER_WINDOW = "per_window_relative_to_anchor"


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _slice_single_scenario_rolling_window_batch(
    raw_batch: dict[str, Any],
    *,
    window_start: int,
    window_stop: int,
) -> dict[str, Any]:
    if window_start < 0 or window_stop <= window_start:
        raise ValueError(
            "Rolling one-step window bounds must satisfy 0 <= start < stop. "
            f"Received start={window_start}, stop={window_stop}."
        )

    sliced_batch: dict[str, Any] = {}
    for key in ("x_stock", "x_market", "r_stock", "feature_time_indices", "target_time_indices"):
        value = raw_batch.get(key)
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"Holdout batch is missing tensor field {key!r}.")
        if value.ndim < 2 or value.shape[0] != 1:
            raise RuntimeError(
                "Rolling one-step evaluation expects batch_size=1 tensors with a time axis. "
                f"Received {key} shape={tuple(value.shape)}."
            )
        sliced_batch[key] = value[:, window_start:window_stop]

    stock_indices = raw_batch.get("stock_indices")
    if not isinstance(stock_indices, torch.Tensor):
        raise RuntimeError("Holdout batch is missing tensor field 'stock_indices'.")
    if stock_indices.ndim != 2 or stock_indices.shape[0] != 1:
        raise RuntimeError(
            "Rolling one-step evaluation expects stock_indices with shape [1, N]. "
            f"Received {tuple(stock_indices.shape)}."
        )
    sliced_batch["stock_indices"] = stock_indices
    x_stock_raw = raw_batch.get("x_stock_raw")
    if x_stock_raw is not None:
        if not isinstance(x_stock_raw, torch.Tensor):
            raise RuntimeError("Holdout batch field 'x_stock_raw' must be a tensor when provided.")
        if x_stock_raw.ndim != 4 or x_stock_raw.shape[0] != 1:
            raise RuntimeError(
                "Rolling one-step evaluation expects x_stock_raw with shape [1, T, N, F_stock]. "
                f"Received {tuple(x_stock_raw.shape)}."
            )
        sliced_batch["x_stock_raw"] = x_stock_raw[:, window_start:window_stop]
    return sliced_batch


def _slice_single_scenario_holdout_window_batch(
    raw_batch: dict[str, Any],
    *,
    window_start: int,
    window_stop: int,
) -> dict[str, Any]:
    return _slice_single_scenario_rolling_window_batch(
        raw_batch,
        window_start=window_start,
        window_stop=window_stop,
    )


def _rebuild_evaluation_window_x_stock(
    *,
    window_batch: dict[str, Any],
    dataset: PortfolioPanelDataset,
    evaluation_label: str,
) -> dict[str, Any]:
    x_stock_raw = window_batch.get("x_stock_raw")
    if not isinstance(x_stock_raw, torch.Tensor):
        raise RuntimeError(
            f"{evaluation_label} requires validation/test batches to include x_stock_raw for per-window anchoring."
        )
    if x_stock_raw.ndim != 4 or x_stock_raw.shape[0] != 1:
        raise RuntimeError(
            f"{evaluation_label} expects x_stock_raw with shape [1, T, N, F_stock], "
            f"received {tuple(x_stock_raw.shape)}."
        )
    if dataset.stock_scaler.mean is None or dataset.stock_scaler.std is None:
        raise RuntimeError(
            f"{evaluation_label} requires fitted stock scaler statistics before per-window normalization."
        )

    rebuilt_x_stock = scale_stock_feature_context_array(
        x_stock_raw[0].detach().cpu().numpy(),
        price_normalization_mode=str(dataset.config.price_normalization_mode),
        stock_mean=dataset.stock_scaler.mean,
        stock_std=dataset.stock_scaler.std,
    )
    rebuilt_batch = dict(window_batch)
    rebuilt_batch["x_stock"] = torch.from_numpy(rebuilt_x_stock).unsqueeze(0)
    rebuilt_batch.pop("x_stock_raw", None)
    return rebuilt_batch


def collect_single_scenario_rolling_outputs(
    *,
    model: PortfolioAttentionModel,
    dataset: PortfolioPanelDataset,
    raw_batch: dict[str, Any],
    device: torch.device,
    lookback_days: int,
    evaluation_label: str,
    rebalance_interval_days: int = 1,
    collect_weights: bool = True,
    interrupt_checker: Callable[[], None] | None = None,
) -> RollingScenarioOutputs:
    scenario_ids = raw_batch.get("scenario_id")
    source_paths = raw_batch.get("source_path")
    score_mask = raw_batch.get("score_mask")
    target_time_indices = raw_batch.get("target_time_indices")

    if not isinstance(scenario_ids, (list, tuple)) or len(scenario_ids) != 1:
        raise RuntimeError(
            f"{evaluation_label} expects batch_size=1 with exactly one scenario_id."
        )
    if not isinstance(source_paths, (list, tuple)) or len(source_paths) != 1:
        raise RuntimeError(
            f"{evaluation_label} expects batch_size=1 with exactly one source_path."
        )
    if not isinstance(score_mask, torch.Tensor):
        raise RuntimeError(f"{evaluation_label} requires a tensor score_mask.")
    if not isinstance(target_time_indices, torch.Tensor):
        raise RuntimeError(f"{evaluation_label} requires tensor target_time_indices.")
    if score_mask.ndim != 2 or score_mask.shape[0] != 1:
        raise RuntimeError(
            f"{evaluation_label} expects score_mask with shape [1, T]. "
            f"Received {tuple(score_mask.shape)}."
        )
    if target_time_indices.ndim != 2 or target_time_indices.shape[0] != 1:
        raise RuntimeError(
            f"{evaluation_label} expects target_time_indices with shape [1, T]. "
            f"Received {tuple(target_time_indices.shape)}."
        )

    resolved_lookback_days = int(lookback_days)
    scenario_id = str(scenario_ids[0])
    source_path = Path(str(source_paths[0]))
    score_mask_cpu = score_mask[0].to(dtype=torch.bool).detach().cpu()
    full_target_time_indices_cpu = target_time_indices[0].detach().cpu()
    scored_positions = torch.nonzero(score_mask_cpu, as_tuple=False).flatten()
    if scored_positions.numel() <= 0:
        raise RuntimeError(
            f"{evaluation_label} requires at least one scored day for scenario {scenario_id}."
        )

    context_time_steps = resolved_lookback_days + ROLLING_ONE_STEP_HORIZON_DAYS
    full_time_steps = int(full_target_time_indices_cpu.shape[0])
    if full_time_steps <= resolved_lookback_days:
        raise RuntimeError(
            f"{evaluation_label} requires more target days than lookback_days. "
            f"Received full_time_steps={full_time_steps}, lookback_days={resolved_lookback_days}."
        )

    schedule = build_rebalance_schedule(
        horizon_steps=int(scored_positions.numel()),
        rebalance_interval_days=rebalance_interval_days,
    )
    decision_ordinals = set(schedule.starts)
    portfolio_returns_by_day: list[torch.Tensor] = []
    turnover_by_day: list[torch.Tensor] = []
    stock_weights_by_day: list[torch.Tensor] = []
    cash_weights_by_day: list[torch.Tensor] = []
    previous_allocation_by_day: list[torch.Tensor] = []
    current_allocation: torch.Tensor | None = None
    if interrupt_checker is not None:
        interrupt_checker()
    for scored_ordinal, scored_position in enumerate(scored_positions.tolist()):
        if interrupt_checker is not None:
            interrupt_checker()
        window_start = int(scored_position) - resolved_lookback_days
        window_stop = int(scored_position) + 1
        if window_start < 0:
            raise RuntimeError(
                f"{evaluation_label} encountered a scored day before the lookback warmup "
                f"for scenario {scenario_id}: position={scored_position} "
                f"lookback_days={resolved_lookback_days}."
            )
        if window_stop > full_time_steps:
            raise RuntimeError(
                f"{evaluation_label} window exceeded the available target horizon "
                f"for scenario {scenario_id}: stop={window_stop} full_time_steps={full_time_steps}."
            )

        if scored_ordinal in decision_ordinals:
            window_batch = _slice_single_scenario_rolling_window_batch(
                raw_batch,
                window_start=window_start,
                window_stop=window_stop,
            )
            window_batch = _rebuild_evaluation_window_x_stock(
                window_batch=window_batch,
                dataset=dataset,
                evaluation_label=evaluation_label,
            )
            window_batch = _move_batch_to_device(window_batch, device)
            initial_allocation_override = (
                current_allocation.unsqueeze(0).to(device)
                if current_allocation is not None
                else None
            )
            with torch.no_grad():
                try:
                    outputs = model(
                        window_batch["x_stock"],
                        window_batch["x_market"],
                        window_batch["stock_indices"],
                        target_returns=window_batch["r_stock"],
                        initial_allocation_override=initial_allocation_override,
                        return_last_step_only=True,
                    )
                except TypeError:
                    outputs = model(
                        window_batch["x_stock"],
                        window_batch["x_market"],
                        window_batch["stock_indices"],
                        target_returns=window_batch["r_stock"],
                    )

            path_returns = outputs["portfolio_return"]
            if path_returns is None:
                raise RuntimeError(f"{evaluation_label} requires target returns for every window.")
            if path_returns.shape not in {(1, 1), (1, context_time_steps)}:
                raise RuntimeError(
                    f"{evaluation_label} expected portfolio_return with shape "
                    f"(1, 1) or (1, {context_time_steps}), received {tuple(path_returns.shape)}."
                )
            stock_weights = outputs.get("stock_weights")
            cash_weights = outputs.get("cash_weight")
            previous_allocation = outputs.get("previous_allocation")
            if stock_weights is None or cash_weights is None:
                raise RuntimeError(
                    f"{evaluation_label} requires stock_weights and cash_weight "
                    "for every decision window."
                )
            allocation = torch.cat(
                [stock_weights[:, -1, :], cash_weights[:, -1:].reshape(1, 1)],
                dim=-1,
            ).detach().cpu().squeeze(0)
            previous_for_turnover = (
                current_allocation
                if current_allocation is not None
                else previous_allocation[:, -1, :].detach().cpu().squeeze(0)
                if isinstance(previous_allocation, torch.Tensor)
                else allocation
            )
            turnover = outputs.get("turnover")
            if current_allocation is None and not isinstance(previous_allocation, torch.Tensor) and isinstance(turnover, torch.Tensor):
                turnover_value = turnover[:, -1].detach().cpu().squeeze(0)
            else:
                turnover_value = 0.5 * torch.abs(allocation - previous_for_turnover).sum()
            portfolio_return_value = path_returns[:, -1].detach().cpu().squeeze(0)
            current_allocation = allocation
        else:
            if current_allocation is None:
                raise RuntimeError(
                    f"{evaluation_label} encountered a non-decision day before any allocation."
                )
            raw_stock_returns = raw_batch.get("r_stock")
            if not isinstance(raw_stock_returns, torch.Tensor):
                raise RuntimeError(f"{evaluation_label} requires tensor r_stock.")
            stock_return_t = raw_stock_returns[0, scored_position, :].detach().cpu()
            previous_for_turnover = current_allocation
            portfolio_return_value = (current_allocation[:-1] * stock_return_t).sum()
            turnover_value = torch.zeros((), dtype=portfolio_return_value.dtype)

        portfolio_returns_by_day.append(portfolio_return_value)
        turnover_by_day.append(turnover_value)
        if collect_weights:
            if current_allocation is None:
                raise RuntimeError(f"{evaluation_label} did not produce an allocation.")
            stock_weights_by_day.append(current_allocation[:-1])
            cash_weights_by_day.append(current_allocation[-1])
            previous_allocation_by_day.append(previous_for_turnover)
        if interrupt_checker is not None:
            interrupt_checker()

    scored_target_time_indices = full_target_time_indices_cpu[score_mask_cpu]
    last_scored_position = int(scored_positions[-1].item())
    context_window_start = last_scored_position - resolved_lookback_days
    context_window_stop = last_scored_position + 1
    context_target_time_indices = full_target_time_indices_cpu[
        context_window_start:context_window_stop
    ]
    if int(context_target_time_indices.shape[0]) != context_time_steps:
        raise RuntimeError(
            f"{evaluation_label} produced an unexpected context window length. "
            f"Expected {context_time_steps}, received {int(context_target_time_indices.shape[0])}."
        )

    return RollingScenarioOutputs(
        scenario_id=scenario_id,
        source_path=str(source_path),
        portfolio_returns=torch.stack(portfolio_returns_by_day, dim=0),
        turnover=torch.stack(turnover_by_day, dim=0),
        scored_target_time_indices=scored_target_time_indices,
        context_target_time_indices=context_target_time_indices,
        lookback_days=resolved_lookback_days,
        context_time_steps=context_time_steps,
        num_rolling_windows=int(scored_positions.numel()),
        evaluation_price_anchor_mode=EVALUATION_PRICE_ANCHOR_MODE_PER_WINDOW,
        stock_weights=torch.stack(stock_weights_by_day, dim=0) if collect_weights else None,
        cash_weights=torch.stack(cash_weights_by_day, dim=0) if collect_weights else None,
        previous_allocation=(
            torch.stack(previous_allocation_by_day, dim=0) if collect_weights else None
        ),
    )


def _collect_single_scenario_rolling_one_step_outputs(
    *,
    model: PortfolioAttentionModel,
    dataset: PortfolioPanelDataset,
    raw_batch: dict[str, Any],
    device: torch.device,
    lookback_days: int,
    evaluation_label: str,
    rebalance_interval_days: int | None = None,
    collect_weights: bool = True,
    interrupt_checker: Callable[[], None] | None = None,
) -> dict[str, Any]:
    resolved_rebalance_interval_days = (
        int(rebalance_interval_days)
        if rebalance_interval_days is not None
        else int(getattr(getattr(dataset, "metadata", None), "rebalance_interval_days", 1))
    )
    return collect_single_scenario_rolling_outputs(
        model=model,
        dataset=dataset,
        raw_batch=raw_batch,
        device=device,
        lookback_days=lookback_days,
        evaluation_label=evaluation_label,
        rebalance_interval_days=resolved_rebalance_interval_days,
        collect_weights=collect_weights,
        interrupt_checker=interrupt_checker,
    ).to_legacy_payload()


def _collect_single_holdout_scenario_payload(
    *,
    model: PortfolioAttentionModel,
    raw_batch: dict[str, Any],
    device: torch.device,
    dataset: PortfolioPanelDataset,
    loss_name: str,
    evaluation_config: EvaluationConfig,
    checkpoint: dict[str, Any],
    interrupt_checker: Callable[[], None] | None = None,
) -> dict[str, Any]:
    rolling_outputs = collect_single_scenario_rolling_outputs(
        model=model,
        dataset=dataset,
        raw_batch=raw_batch,
        device=device,
        lookback_days=int(dataset.metadata.lookback_days),
        rebalance_interval_days=int(getattr(dataset.metadata, "rebalance_interval_days", 1)),
        evaluation_label="Holdout rolling evaluation",
        collect_weights=True,
        interrupt_checker=interrupt_checker,
    )
    if rolling_outputs.stock_weights is None or rolling_outputs.cash_weights is None:
        raise RuntimeError("Holdout rolling evaluation requires stock and cash weights.")
    return build_legacy_per_scenario_payload(
        scenario_id=rolling_outputs.scenario_id,
        source_path=Path(rolling_outputs.source_path),
        loss_name=loss_name,
        checkpoint=checkpoint,
        context_target_time_indices=rolling_outputs.context_target_time_indices,
        target_time_indices=rolling_outputs.scored_target_time_indices,
        portfolio_returns=rolling_outputs.portfolio_returns,
        turnover=rolling_outputs.turnover,
        stock_weights=rolling_outputs.stock_weights,
        cash_weights=rolling_outputs.cash_weights,
        dataset=dataset,
        evaluation_config=evaluation_config,
        warmup_time_steps=int(rolling_outputs.lookback_days),
        evaluation_mode=ROLLING_ONE_STEP_EVALUATION_MODE,
        rolling_window_lookback_days=int(rolling_outputs.lookback_days),
        rolling_window_horizon_days=ROLLING_ONE_STEP_HORIZON_DAYS,
        rolling_window_stride_days=ROLLING_ONE_STEP_STRIDE_DAYS,
        num_rolling_windows=int(rolling_outputs.num_rolling_windows),
        evaluation_price_anchor_mode=str(rolling_outputs.evaluation_price_anchor_mode),
    )


def _collect_holdout_per_scenario_payloads(
    *,
    model: PortfolioAttentionModel,
    holdout_loader: DataLoader,
    device: torch.device,
    dataset: PortfolioPanelDataset,
    loss_name: str,
    evaluation_config: EvaluationConfig,
    checkpoint: dict[str, Any],
    interrupt_checker: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    per_scenario_payloads: list[dict[str, Any]] = []
    was_training = model.training
    model.eval()
    try:
        if interrupt_checker is not None:
            interrupt_checker()
        for raw_batch in holdout_loader:
            if interrupt_checker is not None:
                interrupt_checker()
            per_scenario_payloads.append(
                _collect_single_holdout_scenario_payload(
                    model=model,
                    raw_batch=raw_batch,
                    device=device,
                    dataset=dataset,
                    loss_name=loss_name,
                    evaluation_config=evaluation_config,
                    checkpoint=checkpoint,
                    interrupt_checker=interrupt_checker,
                )
            )
            if interrupt_checker is not None:
                interrupt_checker()
    finally:
        model.train(was_training)

    return per_scenario_payloads
