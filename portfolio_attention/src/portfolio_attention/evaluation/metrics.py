"""Shared evaluation metric helpers."""

from __future__ import annotations

import torch

from ..common.net_return import apply_transaction_cost_to_returns


def compute_selected_stock_count_from_weights(
    stock_weights: torch.Tensor,
    *,
    threshold: float,
    min_active_days: int,
) -> int:
    if stock_weights.ndim != 2:
        raise ValueError("stock_weights must have shape [T, N].")
    resolved_min_active_days = int(min_active_days)
    if resolved_min_active_days <= 0:
        raise ValueError(f"min_active_days must be positive, received {min_active_days}.")
    if int(stock_weights.shape[0]) <= 0:
        raise ValueError("stock_weights must include at least one scored day.")

    threshold_tensor = torch.full_like(stock_weights, float(threshold))
    above_threshold = (stock_weights > threshold_tensor) & (
        ~torch.isclose(stock_weights, threshold_tensor, rtol=0.0, atol=1e-9)
    )
    effective_min_active_days = min(resolved_min_active_days, int(stock_weights.shape[0]))
    selected_stock_count = int((above_threshold.sum(dim=0) >= effective_min_active_days).sum().item())
    return selected_stock_count


def compute_average_turnover_from_weights(
    stock_weights: torch.Tensor,
    cash_weights: torch.Tensor,
) -> float:
    resolved_stock_weights = stock_weights.detach().cpu()
    resolved_cash_weights = cash_weights.detach().cpu()
    if resolved_stock_weights.ndim != 2:
        raise ValueError("stock_weights must have shape [T, N].")
    if resolved_cash_weights.ndim != 1:
        raise ValueError("cash_weights must have shape [T].")
    if int(resolved_stock_weights.shape[0]) != int(resolved_cash_weights.shape[0]):
        raise ValueError("stock_weights and cash_weights must share the same time dimension.")
    if int(resolved_stock_weights.shape[0]) < 2:
        return 0.0

    allocation_weights = torch.cat(
        (resolved_stock_weights, resolved_cash_weights.unsqueeze(-1)),
        dim=1,
    )
    daily_turnover = 0.5 * torch.abs(allocation_weights[1:] - allocation_weights[:-1]).sum(dim=1)
    return float(daily_turnover.mean().item())


_compute_selected_stock_count_from_weights = compute_selected_stock_count_from_weights
_compute_average_turnover_from_weights = compute_average_turnover_from_weights
_apply_transaction_cost_to_returns = apply_transaction_cost_to_returns
