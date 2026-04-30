"""Cross-sectional stock and cash scoring modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math
import torch
from torch import nn

from .task_head import AttentionPortfolioHead, MLPPortfolioHead


@dataclass
class CrossSectionalScoreResult:
    stock_logits: torch.Tensor
    cash_logit: torch.Tensor
    debug_info: dict[str, Any]
    raw_allocation: torch.Tensor | None = None
    precomputed_smoothing: tuple[torch.Tensor, ...] | None = None


class MLPCrossSectionalScorer(nn.Module):
    """Score stocks independently with market and pooled-stock cash context."""

    def __init__(
        self,
        *,
        stock_temporal_dim: int,
        market_temporal_dim: int,
        stock_identity_dim: int,
        cross_sectional_dim: int,
        dropout: float,
        uses_post_temporal_identity: bool,
    ) -> None:
        super().__init__()
        self.uses_post_temporal_identity = uses_post_temporal_identity

        stock_feature_width = (
            stock_temporal_dim * 2
            + market_temporal_dim * 2
            + (stock_identity_dim if uses_post_temporal_identity else 0)
        )
        cash_feature_width = stock_temporal_dim * 2 + market_temporal_dim * 2
        cash_hidden_dim = max(4, market_temporal_dim)

        self.task_head = MLPPortfolioHead(
            stock_feature_width=stock_feature_width,
            cash_feature_width=cash_feature_width,
            cross_sectional_dim=cross_sectional_dim,
            cash_hidden_dim=cash_hidden_dim,
            dropout=dropout,
        )

    @property
    def stock_score(self) -> nn.Module | None:
        return self.task_head.stock_score

    @property
    def cash_score(self) -> nn.Module | None:
        return self.task_head.cash_score

    def forward(
        self,
        *,
        stock_temporal_current: torch.Tensor,
        stock_temporal_summary: torch.Tensor,
        market_current: torch.Tensor,
        market_summary: torch.Tensor,
        stock_identity: torch.Tensor | None,
        initial_allocation: torch.Tensor | None = None,
    ) -> CrossSectionalScoreResult:
        del initial_allocation
        _, _, num_stocks, _ = stock_temporal_current.shape
        market_current_expanded = market_current.unsqueeze(2).expand(-1, -1, num_stocks, -1)
        market_summary_expanded = market_summary.unsqueeze(2).expand(-1, -1, num_stocks, -1)

        stock_feature_parts = [
            stock_temporal_current,
            stock_temporal_summary,
            market_current_expanded,
            market_summary_expanded,
        ]
        if self.uses_post_temporal_identity:
            if stock_identity is None:
                raise RuntimeError("stock_identity must be provided for concat stock embedding.")
            stock_feature_parts.append(stock_identity)
        stock_features = torch.cat(stock_feature_parts, dim=-1)

        pooled_stock_current = stock_temporal_current.mean(dim=2)
        pooled_stock_running = stock_temporal_summary.mean(dim=2)
        cash_features = torch.cat(
            [pooled_stock_current, pooled_stock_running, market_current, market_summary],
            dim=-1,
        )
        head_result = self.task_head(
            stock_features=stock_features,
            cash_features=cash_features,
        )

        return CrossSectionalScoreResult(
            stock_logits=head_result.stock_logits,
            cash_logit=head_result.cash_logit,
            debug_info={
                "stock_feature_shape": tuple(stock_features.shape),
                "stock_content_shape": None,
                "stock_attention_input_shape": None,
                "stock_attention_weight_shape": None,
                **head_result.debug_info,
            },
            raw_allocation=head_result.raw_allocation,
            precomputed_smoothing=head_result.precomputed_smoothing,
        )


class AttentionCrossSectionalScorer(nn.Module):
    """Score stocks with cross-sectional self-attention and optional previous weights."""

    def __init__(
        self,
        *,
        stock_temporal_dim: int,
        market_temporal_dim: int,
        stock_identity_dim: int,
        cross_sectional_dim: int,
        dropout: float,
        uses_post_temporal_identity: bool,
        allocation_smoothing_alpha: float,
        detach_prev_weight: bool,
        use_prev_weight_feature: bool,
    ) -> None:
        super().__init__()
        self.uses_post_temporal_identity = uses_post_temporal_identity
        self.stock_attention_dim = (
            cross_sectional_dim + stock_identity_dim
            if uses_post_temporal_identity
            else cross_sectional_dim
        )
        self.use_prev_weight_feature = use_prev_weight_feature

        stock_content_width = stock_temporal_dim * 2 + market_temporal_dim * 2
        self.stock_content_proj = nn.Linear(stock_content_width, cross_sectional_dim)

        self.stock_cross_attention_query = nn.Linear(
            self.stock_attention_dim, self.stock_attention_dim, bias=False
        )
        self.stock_cross_attention_key = nn.Linear(
            self.stock_attention_dim, self.stock_attention_dim, bias=False
        )
        self.stock_cross_attention_value = nn.Linear(
            self.stock_attention_dim, self.stock_attention_dim, bias=False
        )

        self.stock_cross_attn_out = nn.Linear(self.stock_attention_dim, self.stock_attention_dim)
        self.stock_cross_norm1 = nn.LayerNorm(self.stock_attention_dim)
        self.stock_cross_norm2 = nn.LayerNorm(self.stock_attention_dim)
        self.stock_cross_ffn = nn.Sequential(
            nn.Linear(self.stock_attention_dim, 4 * self.stock_attention_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * self.stock_attention_dim, self.stock_attention_dim),
        )

        self.task_head = AttentionPortfolioHead(
            stock_attention_dim=self.stock_attention_dim,
            dropout=dropout,
            allocation_smoothing_alpha=allocation_smoothing_alpha,
            detach_prev_weight=detach_prev_weight,
            use_prev_weight_feature=use_prev_weight_feature,
        )

    @property
    def stock_cross_attention_score(self) -> nn.Module | None:
        return self.task_head.stock_cross_attention_score

    @property
    def cash_cross_attention_score(self) -> nn.Module | None:
        return self.task_head.cash_cross_attention_score

    @property
    def cash_state_mlp_base(self) -> nn.Module | None:
        return self.task_head.cash_state_mlp_base

    @property
    def stock_prev_weight_mlp(self) -> nn.Module | None:
        return self.task_head.stock_prev_weight_mlp

    @property
    def cash_prev_weight_mlp(self) -> nn.Module | None:
        return self.task_head.cash_prev_weight_mlp

    def forward(
        self,
        *,
        stock_temporal_current: torch.Tensor,
        stock_temporal_summary: torch.Tensor,
        market_current: torch.Tensor,
        market_summary: torch.Tensor,
        stock_identity: torch.Tensor | None,
        initial_allocation: torch.Tensor | None = None,
    ) -> CrossSectionalScoreResult:
        num_scenarios, time_steps, num_stocks, _ = stock_temporal_current.shape
        market_current_expanded = market_current.unsqueeze(2).expand(-1, -1, num_stocks, -1)
        market_summary_expanded = market_summary.unsqueeze(2).expand(-1, -1, num_stocks, -1)

        stock_content_inputs = torch.cat(
            [
                stock_temporal_current,
                stock_temporal_summary,
                market_current_expanded,
                market_summary_expanded,
            ],
            dim=-1,
        )
        stock_content = self.stock_content_proj(stock_content_inputs)
        if self.uses_post_temporal_identity:
            if stock_identity is None:
                raise RuntimeError("stock_identity must be provided for concat stock embedding.")
            stock_attention_inputs = torch.cat([stock_content, stock_identity], dim=-1)
        else:
            stock_attention_inputs = stock_content

        flattened_attention_inputs = stock_attention_inputs.reshape(
            num_scenarios * time_steps,
            num_stocks,
            self.stock_attention_dim,
        )

        x = flattened_attention_inputs
        x_norm = self.stock_cross_norm1(x)

        query = self.stock_cross_attention_query(x_norm)
        key = self.stock_cross_attention_key(x_norm)
        value = self.stock_cross_attention_value(x_norm)

        attention_scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(
            self.stock_attention_dim
        )
        attention_weights = torch.softmax(attention_scores, dim=-1)

        attn_out = torch.matmul(attention_weights, value)
        attn_out = self.stock_cross_attn_out(attn_out)

        x = x + attn_out
        x = x + self.stock_cross_ffn(self.stock_cross_norm2(x))

        attended_stock = x.reshape(
            num_scenarios,
            time_steps,
            num_stocks,
            self.stock_attention_dim,
        )
        debug_info = {
            "stock_feature_shape": tuple(stock_attention_inputs.shape),
            "stock_content_shape": tuple(stock_content.shape),
            "stock_attention_input_shape": tuple(stock_attention_inputs.shape),
            "stock_attention_weight_shape": tuple(attention_weights.shape),
        }

        head_result = self.task_head(
            attended_stock=attended_stock,
            initial_allocation=initial_allocation,
        )
        return CrossSectionalScoreResult(
            stock_logits=head_result.stock_logits,
            cash_logit=head_result.cash_logit,
            debug_info={**debug_info, **head_result.debug_info},
            raw_allocation=head_result.raw_allocation,
            precomputed_smoothing=head_result.precomputed_smoothing,
        )
