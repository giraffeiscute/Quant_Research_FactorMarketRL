"""Critic/value head modules for portfolio RL training."""

from __future__ import annotations

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
