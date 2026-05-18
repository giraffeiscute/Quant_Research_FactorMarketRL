"""Portfolio task heads for cross-sectional representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .allocation_distribution import (
    AllocationDistribution,
    DEFAULT_DIRICHLET_LOGIT_SCALE,
)


@dataclass
class TaskHeadResult:
    stock_logits: torch.Tensor
    cash_logit: torch.Tensor
    debug_info: dict[str, Any]
    raw_allocation: torch.Tensor | None = None
    precomputed_smoothing: tuple[torch.Tensor, ...] | None = None
    allocation_distribution_debug_info: dict[str, Any] | None = None


class MLPPortfolioHead(nn.Module):
    """Portfolio scoring head for MLP cross-sectional features."""

    def __init__(
        self,
        *,
        stock_feature_width: int,
        cash_feature_width: int,
        cross_sectional_dim: int,
        cash_hidden_dim: int,
        dropout: float,
        inference_allocation_mode: str = "softmax",
        dirichlet_logit_scale: float = DEFAULT_DIRICHLET_LOGIT_SCALE,
    ) -> None:
        super().__init__()
        self.stock_score = nn.Sequential(
            nn.Linear(stock_feature_width, cross_sectional_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cross_sectional_dim, 1),
        )
        self.cash_score = nn.Sequential(
            nn.Linear(cash_feature_width, cash_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cash_hidden_dim, 1),
        )
        self.allocation_distribution = AllocationDistribution(
            inference_allocation_mode=inference_allocation_mode,
            dirichlet_logit_scale=dirichlet_logit_scale,
        )

    def forward(
        self,
        *,
        stock_features: torch.Tensor,
        cash_features: torch.Tensor,
    ) -> TaskHeadResult:
        stock_logits = self.stock_score(stock_features).squeeze(-1)
        cash_logit = self.cash_score(cash_features).squeeze(-1)
        allocation_logits = torch.cat([stock_logits, cash_logit.unsqueeze(-1)], dim=-1)
        allocation_distribution_result = self.allocation_distribution(
            allocation_logits,
            debug_context="MLPPortfolioHead.forward",
            logits_name="allocation_logits",
        )
        raw_allocation = allocation_distribution_result.raw_allocation
        return TaskHeadResult(
            stock_logits=stock_logits,
            cash_logit=cash_logit,
            debug_info={},
            raw_allocation=raw_allocation,
            allocation_distribution_debug_info=allocation_distribution_result.debug_info,
        )


class AttentionPortfolioHead(nn.Module):
    """Portfolio scoring head for attended cross-sectional stock representations."""

    def __init__(
        self,
        *,
        stock_attention_dim: int,
        dropout: float,
        allocation_smoothing_alpha: float,
        detach_prev_weight: bool,
        use_prev_weight_feature: bool,
        inference_allocation_mode: str = "softmax",
        dirichlet_logit_scale: float = DEFAULT_DIRICHLET_LOGIT_SCALE,
    ) -> None:
        super().__init__()
        self.stock_attention_dim = stock_attention_dim
        self.allocation_smoothing_alpha = float(allocation_smoothing_alpha)
        self.detach_prev_weight = detach_prev_weight
        self.use_prev_weight_feature = use_prev_weight_feature

        self.stock_cross_attention_score = nn.Linear(stock_attention_dim, 1)
        self.cash_cross_attention_score = nn.Linear(stock_attention_dim, 1)
        self.allocation_distribution = AllocationDistribution(
            inference_allocation_mode=inference_allocation_mode,
            dirichlet_logit_scale=dirichlet_logit_scale,
        )
        self.cash_state_mlp_base = nn.Sequential(
            nn.Linear(stock_attention_dim, stock_attention_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(stock_attention_dim, stock_attention_dim),
        )
        if use_prev_weight_feature:
            self.stock_prev_weight_mlp = nn.Sequential(
                nn.Linear(stock_attention_dim + 1, stock_attention_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(stock_attention_dim, stock_attention_dim),
            )
            self.cash_prev_weight_mlp = nn.Sequential(
                nn.Linear(stock_attention_dim + 1, stock_attention_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(stock_attention_dim, stock_attention_dim),
            )
        else:
            self.stock_prev_weight_mlp = None
            self.cash_prev_weight_mlp = None

    def forward(
        self,
        *,
        attended_stock: torch.Tensor,
        initial_allocation: torch.Tensor | None = None,
    ) -> TaskHeadResult:
        num_scenarios, time_steps, num_stocks, _ = attended_stock.shape
        if not self.use_prev_weight_feature:
            stock_logits = self.stock_cross_attention_score(attended_stock).squeeze(-1)
            cash_base_input = attended_stock.mean(dim=2)
            cash_state_base = self.cash_state_mlp_base(cash_base_input)
            cash_logit = self.cash_cross_attention_score(cash_state_base).squeeze(-1)
            allocation_logits = torch.cat([stock_logits, cash_logit.unsqueeze(-1)], dim=-1)
            allocation_distribution_result = self.allocation_distribution(
                allocation_logits,
                debug_context="AttentionPortfolioHead.forward use_prev_weight_feature=False",
                logits_name="allocation_logits",
            )
            raw_allocation = allocation_distribution_result.raw_allocation
            assert stock_logits.shape == (num_scenarios, time_steps, num_stocks)
            assert cash_logit.shape == (num_scenarios, time_steps)
            assert raw_allocation.shape == (num_scenarios, time_steps, num_stocks + 1)
            return TaskHeadResult(
                stock_logits=stock_logits,
                cash_logit=cash_logit,
                debug_info={"used_prev_weight_feature": False},
                raw_allocation=raw_allocation,
                allocation_distribution_debug_info=allocation_distribution_result.debug_info,
            )

        if self.stock_prev_weight_mlp is None or self.cash_prev_weight_mlp is None:
            raise RuntimeError(
                "Self-attention prev-weight feature modules must be initialized when "
                "use_prev_weight_feature=True."
            )
        if initial_allocation is None:
            raise RuntimeError(
                "initial_allocation must be provided when use_prev_weight_feature=True."
            )
        prev_weight = initial_allocation
        if prev_weight.shape != (num_scenarios, num_stocks + 1):
            raise ValueError(
                "initial_allocation must have shape [S, N+1]. "
                f"Received {tuple(prev_weight.shape)} expected {(num_scenarios, num_stocks + 1)}."
            )

        smoothing_alpha = float(self.allocation_smoothing_alpha)
        allocation_distribution_debug_info: dict[str, Any] | None = None
        stock_logits_by_step: list[torch.Tensor] = []
        cash_logit_by_step: list[torch.Tensor] = []
        raw_allocations_by_step: list[torch.Tensor] = []
        allocations_by_step: list[torch.Tensor] = []
        turnovers_by_step: list[torch.Tensor] = []
        previous_allocations_by_step: list[torch.Tensor] = []
        for time_index in range(time_steps):
            prev_weight_step = prev_weight.detach() if self.detach_prev_weight else prev_weight
            attended_stock_t = attended_stock[:, time_index, :, :]
            stock_state_input = torch.cat(
                [attended_stock_t, prev_weight_step[:, :num_stocks].unsqueeze(-1)],
                dim=-1,
            )
            stock_state_repr = self.stock_prev_weight_mlp(stock_state_input)
            stock_logit_t = self.stock_cross_attention_score(stock_state_repr).squeeze(-1)
            cash_base_input = attended_stock_t.mean(dim=1)
            cash_state_base = self.cash_state_mlp_base(cash_base_input)
            cash_state_input = torch.cat(
                [cash_state_base, prev_weight_step[:, -1:].contiguous()],
                dim=-1,
            )
            cash_state_repr = self.cash_prev_weight_mlp(cash_state_input)
            cash_logit_t = self.cash_cross_attention_score(cash_state_repr).squeeze(-1)

            allocation_logits_t = torch.cat([stock_logit_t, cash_logit_t.unsqueeze(-1)], dim=-1)
            allocation_distribution_result_t = self.allocation_distribution(
                allocation_logits_t,
                debug_context=(
                    "AttentionPortfolioHead.forward use_prev_weight_feature=True "
                    f"time_index={time_index}"
                ),
                logits_name="allocation_logits_t",
            )
            allocation_distribution_debug_info = allocation_distribution_result_t.debug_info
            raw_t = allocation_distribution_result_t.raw_allocation
            allocation_t = smoothing_alpha * raw_t + (1.0 - smoothing_alpha) * prev_weight_step
            allocation_delta_t = allocation_t - prev_weight_step
            turnover_t = 0.5 * torch.abs(allocation_delta_t).sum(dim=-1)

            stock_logits_by_step.append(stock_logit_t)
            cash_logit_by_step.append(cash_logit_t)
            raw_allocations_by_step.append(raw_t)
            allocations_by_step.append(allocation_t)
            turnovers_by_step.append(turnover_t)
            previous_allocations_by_step.append(prev_weight_step)
            prev_weight = allocation_t

        stock_logits = torch.stack(stock_logits_by_step, dim=1)
        cash_logit = torch.stack(cash_logit_by_step, dim=1)
        raw_allocation = torch.stack(raw_allocations_by_step, dim=1)
        precomputed_smoothing = (
            raw_allocation,
            torch.stack(allocations_by_step, dim=1),
            torch.stack(turnovers_by_step, dim=1),
            torch.stack(previous_allocations_by_step, dim=1),
        )
        return TaskHeadResult(
            stock_logits=stock_logits,
            cash_logit=cash_logit,
            debug_info={"used_prev_weight_feature": True},
            raw_allocation=raw_allocation,
            precomputed_smoothing=precomputed_smoothing,
            allocation_distribution_debug_info=allocation_distribution_debug_info,
        )
