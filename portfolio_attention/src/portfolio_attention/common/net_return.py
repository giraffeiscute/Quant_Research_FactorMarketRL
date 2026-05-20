"""Shared return transforms used across training and evaluation."""

from __future__ import annotations

import torch


def apply_transaction_cost_to_returns(
    portfolio_returns: torch.Tensor,
    turnover: torch.Tensor,
    *,
    transaction_cost_rate: float,
) -> torch.Tensor:
    """Apply self-financing transaction costs before the period return is earned."""
    if portfolio_returns.shape != turnover.shape:
        raise ValueError(
            "portfolio_returns and turnover must have the same shape when applying transaction costs. "
            f"Received portfolio_returns={tuple(portfolio_returns.shape)} turnover={tuple(turnover.shape)}."
        )
    resolved_transaction_cost_rate = float(transaction_cost_rate)
    if resolved_transaction_cost_rate < 0.0:
        raise ValueError(
            "transaction_cost_rate must be non-negative, "
            f"received {resolved_transaction_cost_rate}."
        )
    if resolved_transaction_cost_rate == 0.0:
        return portfolio_returns
    cost_fraction = resolved_transaction_cost_rate * turnover
    return (1.0 - cost_fraction) * (1.0 + portfolio_returns) - 1.0
