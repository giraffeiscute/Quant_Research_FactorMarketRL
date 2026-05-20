"""Shared win-rate baseline and metric helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .net_return import apply_transaction_cost_to_returns


@dataclass(frozen=True)
class WinRateBaselineResult:
    allocation: torch.Tensor
    gross_return: torch.Tensor
    turnover: torch.Tensor
    net_return: torch.Tensor


@dataclass(frozen=True)
class WinRateMetrics:
    baseline: WinRateBaselineResult
    wins: torch.Tensor
    binary_reward: torch.Tensor
    win_count: int
    window_count: int
    win_rate: float


def compute_binary_win_rate_reward(
    portfolio_returns: torch.Tensor,
    baseline_returns: torch.Tensor,
) -> torch.Tensor:
    """Return +1 when portfolio return strictly beats baseline return, otherwise -1."""
    if portfolio_returns.numel() == 0:
        raise ValueError("portfolio_returns must not be empty.")
    if baseline_returns.numel() == 0:
        raise ValueError("baseline_returns must not be empty.")

    detached_portfolio_returns = portfolio_returns.detach()
    detached_baseline_returns = baseline_returns.detach().to(
        device=detached_portfolio_returns.device,
        dtype=detached_portfolio_returns.dtype,
    )
    portfolio_values, baseline_values = torch.broadcast_tensors(
        detached_portfolio_returns,
        detached_baseline_returns,
    )
    wins = portfolio_values > baseline_values
    return torch.where(
        wins,
        torch.ones_like(portfolio_values),
        -torch.ones_like(portfolio_values),
    )


def compute_win_rate_baseline_returns(
    stock_returns: torch.Tensor,
    previous_allocation: torch.Tensor,
    *,
    reward_baseline: str,
    transaction_cost_rate: float,
) -> WinRateBaselineResult:
    """Build cash/uniform baseline returns using portfolio turnover semantics."""
    if stock_returns.numel() == 0:
        raise ValueError("stock_returns must not be empty.")
    if previous_allocation.numel() == 0:
        raise ValueError("previous_allocation must not be empty.")
    if stock_returns.ndim < 1:
        raise ValueError("stock_returns must include a stock dimension.")
    if previous_allocation.ndim < 1:
        raise ValueError("previous_allocation must include an asset dimension.")

    resolved_previous_allocation = previous_allocation.detach()
    resolved_stock_returns = stock_returns.detach().to(
        device=resolved_previous_allocation.device,
        dtype=resolved_previous_allocation.dtype,
    )
    num_stocks = int(resolved_stock_returns.shape[-1])
    if num_stocks <= 0:
        raise ValueError("stock_returns must include at least one stock.")
    if int(resolved_previous_allocation.shape[-1]) != num_stocks + 1:
        raise ValueError(
            "previous_allocation must have one more asset than stock_returns "
            "to include cash. "
            f"Received stock_returns={tuple(resolved_stock_returns.shape)} "
            f"previous_allocation={tuple(resolved_previous_allocation.shape)}."
        )

    baseline_allocation = torch.zeros_like(resolved_previous_allocation)
    resolved_reward_baseline = str(reward_baseline).strip().lower()
    if resolved_reward_baseline == "cash":
        baseline_allocation[..., -1] = 1.0
        baseline_gross_return = torch.zeros_like(resolved_previous_allocation[..., -1])
    elif resolved_reward_baseline == "uniform":
        baseline_allocation[..., :-1] = 1.0 / float(num_stocks)
        stock_return_values, _ = torch.broadcast_tensors(
            resolved_stock_returns,
            baseline_allocation[..., :-1],
        )
        baseline_gross_return = stock_return_values.mean(dim=-1)
    else:
        raise ValueError(f"Unsupported reward_baseline: {reward_baseline!r}.")

    baseline_turnover = 0.5 * torch.abs(
        baseline_allocation - resolved_previous_allocation
    ).sum(dim=-1)
    baseline_gross_return, baseline_turnover = torch.broadcast_tensors(
        baseline_gross_return,
        baseline_turnover,
    )
    baseline_net_return = apply_transaction_cost_to_returns(
        baseline_gross_return,
        baseline_turnover,
        transaction_cost_rate=float(transaction_cost_rate),
    )
    return WinRateBaselineResult(
        allocation=baseline_allocation,
        gross_return=baseline_gross_return,
        turnover=baseline_turnover,
        net_return=baseline_net_return,
    )


def compute_win_rate_metrics(
    portfolio_returns: torch.Tensor,
    stock_returns: torch.Tensor,
    previous_allocation: torch.Tensor,
    *,
    reward_baseline: str,
    transaction_cost_rate: float,
) -> WinRateMetrics:
    """Compute strict portfolio-vs-baseline win-rate metrics and binary rewards."""
    if portfolio_returns.numel() == 0:
        raise ValueError("portfolio_returns must not be empty.")
    baseline = compute_win_rate_baseline_returns(
        stock_returns,
        previous_allocation,
        reward_baseline=reward_baseline,
        transaction_cost_rate=transaction_cost_rate,
    )
    portfolio_values = portfolio_returns.detach().to(
        device=baseline.net_return.device,
        dtype=baseline.net_return.dtype,
    )
    portfolio_values, baseline_values = torch.broadcast_tensors(
        portfolio_values,
        baseline.net_return,
    )
    binary_reward = compute_binary_win_rate_reward(
        portfolio_values,
        baseline_values,
    )
    wins = binary_reward > 0
    window_count = int(wins.numel())
    win_count = int(wins.sum().item())
    return WinRateMetrics(
        baseline=baseline,
        wins=wins,
        binary_reward=binary_reward,
        win_count=win_count,
        window_count=window_count,
        win_rate=0.0 if window_count <= 0 else win_count / window_count,
    )
