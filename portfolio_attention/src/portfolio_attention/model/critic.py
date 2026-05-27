"""Critic/value head modules for portfolio RL training."""

from __future__ import annotations

import copy

import torch
from torch import nn


class PortfolioCritic(nn.Module):
    """Predict per-timestep state values from encoded market and stock state."""

    def __init__(
        self,
        *,
        stock_temporal_dim: int,
        market_temporal_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        value_feature_width = (
            int(stock_temporal_dim)
            + int(stock_temporal_dim)
            + int(market_temporal_dim)
            + int(market_temporal_dim)
            + 4
        )
        self.value_head = nn.Sequential(
            nn.Linear(value_feature_width, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(
        self,
        *,
        stock_temporal_current: torch.Tensor,
        stock_temporal_summary: torch.Tensor,
        market_current: torch.Tensor,
        market_summary: torch.Tensor,
        previous_allocation: torch.Tensor,
    ) -> torch.Tensor:
        previous_stock_allocation = previous_allocation[..., :-1]
        previous_allocation_features = torch.cat(
            [
                previous_stock_allocation.mean(dim=-1, keepdim=True),
                previous_stock_allocation.std(dim=-1, keepdim=True, unbiased=False),
                previous_stock_allocation.amax(dim=-1, keepdim=True),
                previous_allocation[..., -1:].contiguous(),
            ],
            dim=-1,
        )
        value_features = torch.cat(
            [
                stock_temporal_current.mean(dim=2),
                stock_temporal_summary.mean(dim=2),
                market_current,
                market_summary,
                previous_allocation_features,
            ],
            dim=-1,
        )
        return self.value_head(value_features).squeeze(-1)


def _validate_positive_int(value: int, *, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, received {value}.")
    return value


def _allocation_summary(allocation: torch.Tensor) -> torch.Tensor:
    stock_allocation = allocation[..., :-1]
    return torch.cat(
        [
            stock_allocation.mean(dim=-1, keepdim=True),
            stock_allocation.std(dim=-1, keepdim=True, unbiased=False),
            stock_allocation.amax(dim=-1, keepdim=True),
            allocation[..., -1:].contiguous(),
        ],
        dim=-1,
    )


def _validate_finite_tensor(tensor: torch.Tensor, *, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite.")


def _validate_simplex_allocation(tensor: torch.Tensor, *, name: str) -> None:
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must be a floating point tensor.")
    _validate_finite_tensor(tensor, name=name)
    if (tensor < 0).any():
        raise ValueError(f"{name} must be non-negative.")
    sums = tensor.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-4, rtol=1e-4):
        raise ValueError(f"{name} must sum to 1 over the last dimension.")


class PortfolioQCritic(nn.Module):
    """Predict per-timestep Q(s, a) from encoded state and full allocation action."""

    def __init__(
        self,
        *,
        stock_temporal_dim: int,
        market_temporal_dim: int,
        action_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.stock_temporal_dim = _validate_positive_int(
            stock_temporal_dim,
            name="stock_temporal_dim",
        )
        self.market_temporal_dim = _validate_positive_int(
            market_temporal_dim,
            name="market_temporal_dim",
        )
        self.action_dim = _validate_positive_int(action_dim, name="action_dim")
        hidden_dim = _validate_positive_int(hidden_dim, name="hidden_dim")
        q_feature_width = (
            4 * self.stock_temporal_dim
            + 2 * self.market_temporal_dim
            + 4
            + 2 * self.action_dim
            + 1
        )
        self.q_head = nn.Sequential(
            nn.Linear(q_feature_width, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        *,
        stock_temporal_current: torch.Tensor,
        stock_temporal_summary: torch.Tensor,
        market_current: torch.Tensor,
        market_summary: torch.Tensor,
        previous_allocation: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        _validate_q_inputs(
            stock_temporal_current=stock_temporal_current,
            stock_temporal_summary=stock_temporal_summary,
            market_current=market_current,
            market_summary=market_summary,
            previous_allocation=previous_allocation,
            action=action,
            stock_temporal_dim=self.stock_temporal_dim,
            market_temporal_dim=self.market_temporal_dim,
            action_dim=self.action_dim,
        )
        action_stock = action[..., :-1]
        weighted_stock_current = (stock_temporal_current * action_stock.unsqueeze(-1)).sum(dim=2)
        weighted_stock_summary = (stock_temporal_summary * action_stock.unsqueeze(-1)).sum(dim=2)
        allocation_delta = action - previous_allocation
        turnover_proxy = allocation_delta.abs().sum(dim=-1, keepdim=True)
        q_features = torch.cat(
            [
                stock_temporal_current.mean(dim=2),
                stock_temporal_summary.mean(dim=2),
                weighted_stock_current,
                weighted_stock_summary,
                market_current,
                market_summary,
                _allocation_summary(previous_allocation),
                action,
                allocation_delta,
                turnover_proxy,
            ],
            dim=-1,
        )
        return self.q_head(q_features).squeeze(-1)


class TwinPortfolioQCritic(nn.Module):
    """Independent twin Q critics for clipped double-Q SAC updates."""

    def __init__(
        self,
        *,
        stock_temporal_dim: int,
        market_temporal_dim: int,
        action_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.q1 = PortfolioQCritic(
            stock_temporal_dim=stock_temporal_dim,
            market_temporal_dim=market_temporal_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.q2 = PortfolioQCritic(
            stock_temporal_dim=stock_temporal_dim,
            market_temporal_dim=market_temporal_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        *,
        stock_temporal_current: torch.Tensor,
        stock_temporal_summary: torch.Tensor,
        market_current: torch.Tensor,
        market_summary: torch.Tensor,
        previous_allocation: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_kwargs = {
            "stock_temporal_current": stock_temporal_current,
            "stock_temporal_summary": stock_temporal_summary,
            "market_current": market_current,
            "market_summary": market_summary,
            "previous_allocation": previous_allocation,
            "action": action,
        }
        return self.q1(**q_kwargs), self.q2(**q_kwargs)


def clone_target_q_critic(q_critic: TwinPortfolioQCritic) -> TwinPortfolioQCritic:
    """Create a frozen target twin-Q critic initialized from the online critic."""
    target = copy.deepcopy(q_critic)
    target.load_state_dict(q_critic.state_dict())
    target.eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    return target


def _validate_q_inputs(
    *,
    stock_temporal_current: torch.Tensor,
    stock_temporal_summary: torch.Tensor,
    market_current: torch.Tensor,
    market_summary: torch.Tensor,
    previous_allocation: torch.Tensor,
    action: torch.Tensor,
    stock_temporal_dim: int,
    market_temporal_dim: int,
    action_dim: int,
) -> None:
    if stock_temporal_current.ndim != 4:
        raise ValueError(
            "stock_temporal_current must have shape [B, T, N, D_stock], "
            f"received {tuple(stock_temporal_current.shape)}."
        )
    batch_size, time_steps, num_stocks, _ = stock_temporal_current.shape
    expected_stock_shape = (batch_size, time_steps, num_stocks, stock_temporal_dim)
    if tuple(stock_temporal_current.shape) != expected_stock_shape:
        raise ValueError(
            "stock_temporal_current must have shape [B, T, N, D_stock]. "
            f"Expected {expected_stock_shape}, received {tuple(stock_temporal_current.shape)}."
        )
    if tuple(stock_temporal_summary.shape) != expected_stock_shape:
        raise ValueError(
            "stock_temporal_summary must match stock_temporal_current shape. "
            f"Expected {expected_stock_shape}, received {tuple(stock_temporal_summary.shape)}."
        )
    expected_market_shape = (batch_size, time_steps, market_temporal_dim)
    if tuple(market_current.shape) != expected_market_shape:
        raise ValueError(
            "market_current must have shape [B, T, D_market]. "
            f"Expected {expected_market_shape}, received {tuple(market_current.shape)}."
        )
    if tuple(market_summary.shape) != expected_market_shape:
        raise ValueError(
            "market_summary must have shape [B, T, D_market]. "
            f"Expected {expected_market_shape}, received {tuple(market_summary.shape)}."
        )
    expected_action_shape = (batch_size, time_steps, action_dim)
    expected_asset_dim = num_stocks + 1
    if action_dim != expected_asset_dim:
        raise ValueError(
            "PortfolioQCritic action_dim must equal N+1 including cash. "
            f"Received action_dim={action_dim} num_stocks={num_stocks}."
        )
    if tuple(action.shape) != expected_action_shape:
        raise ValueError(
            "action must have shape [B, T, N+1] including cash. "
            f"Expected {expected_action_shape}, received {tuple(action.shape)}."
        )
    if tuple(previous_allocation.shape) != expected_action_shape:
        raise ValueError(
            "previous_allocation must have shape [B, T, N+1] including cash. "
            f"Expected {expected_action_shape}, received {tuple(previous_allocation.shape)}."
        )
    _validate_finite_tensor(stock_temporal_current, name="stock_temporal_current")
    _validate_finite_tensor(stock_temporal_summary, name="stock_temporal_summary")
    _validate_finite_tensor(market_current, name="market_current")
    _validate_finite_tensor(market_summary, name="market_summary")
    _validate_simplex_allocation(action, name="action")
    _validate_simplex_allocation(previous_allocation, name="previous_allocation")
