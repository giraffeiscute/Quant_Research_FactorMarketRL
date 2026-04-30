"""Temporal encoder modules for portfolio models."""

from __future__ import annotations

from typing import Any

import math
import torch
from torch import nn

from .temporal_utils import (
    build_local_causal_window_mask,
    causal_running_mean,
    fixed_window_causal_mean,
)


class StockTemporalEncoder(nn.Module):
    """Encode per-stock temporal feature sequences without cross-sectional scoring."""

    def __init__(
        self,
        *,
        stock_feature_dim: int,
        stock_hidden_dim: int,
        stock_temporal_encoder_type: str,
        stock_temporal_attention_window: int,
        dropout: float,
        use_legacy_stock_ffn_noop_for_inference: bool = False,
    ) -> None:
        super().__init__()
        if stock_temporal_attention_window <= 0:
            raise ValueError(
                "stock_temporal_attention_window must be positive before model construction."
            )

        self.stock_hidden_dim = int(stock_hidden_dim)
        self.stock_temporal_encoder_type = stock_temporal_encoder_type
        self.stock_temporal_attention_window = int(stock_temporal_attention_window)
        self.stock_input_proj = nn.Linear(stock_feature_dim, stock_hidden_dim)
        self.stock_ffn = nn.Sequential(
            nn.Linear(stock_hidden_dim, stock_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(stock_hidden_dim, stock_hidden_dim),
        )
        self._legacy_stock_ffn_noop_for_inference = use_legacy_stock_ffn_noop_for_inference

        if self.stock_temporal_encoder_type == "causal_self_attention":
            d = stock_hidden_dim
            self.stock_temporal_query = nn.Linear(d, d, bias=False)
            self.stock_temporal_key = nn.Linear(d, d, bias=False)
            self.stock_temporal_value = nn.Linear(d, d, bias=False)
            self.stock_temporal_attn_out = nn.Linear(d, d)
            self.stock_temporal_norm1 = nn.LayerNorm(d)
            self.stock_temporal_norm2 = nn.LayerNorm(d)
            self.stock_temporal_ffn = nn.Sequential(
                nn.Linear(d, 4 * d),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d, d),
            )

    def enable_legacy_stock_ffn_noop_for_inference(self) -> None:
        self._legacy_stock_ffn_noop_for_inference = True

    def forward(
        self,
        x_stock: torch.Tensor,
        *,
        stock_identity: torch.Tensor | None = None,
        time_encoding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        if x_stock.ndim != 4:
            raise ValueError("x_stock must have shape [S, T, N, F_stock].")

        stock_sequence = self.stock_input_proj(x_stock)
        if time_encoding is not None:
            stock_sequence = stock_sequence + time_encoding
        if stock_identity is not None:
            if stock_identity.shape != stock_sequence.shape:
                raise ValueError(
                    "stock_identity must have shape [S, T, N, D_stock]. "
                    f"Received {tuple(stock_identity.shape)} expected {tuple(stock_sequence.shape)}."
                )
            stock_sequence = stock_sequence + stock_identity

        debug_info: dict[str, Any] = {
            "stock_current_shape": tuple(stock_sequence.shape),
        }
        if not self._legacy_stock_ffn_noop_for_inference:
            stock_sequence = self.stock_ffn(stock_sequence)

        if self.stock_temporal_encoder_type == "running_summary":
            stock_summary = causal_running_mean(stock_sequence)
            debug_info.update(
                {
                    "stock_temporal_encoder_type": self.stock_temporal_encoder_type,
                    "stock_temporal_attention_window": self.stock_temporal_attention_window,
                    "stock_temporal_attention_mask_shape": None,
                    "stock_temporal_attention_weight_shape": None,
                }
            )
            return stock_sequence, stock_summary, debug_info

        if self.stock_temporal_encoder_type != "causal_self_attention":
            raise ValueError(
                "Unsupported stock_temporal_encoder_type: "
                f"{self.stock_temporal_encoder_type!r}."
            )

        num_scenarios, time_steps, num_stocks, _ = stock_sequence.shape
        flattened_stock_sequence = stock_sequence.permute(0, 2, 1, 3).reshape(
            num_scenarios * num_stocks,
            time_steps,
            self.stock_hidden_dim,
        )

        x = flattened_stock_sequence
        x_norm = self.stock_temporal_norm1(x)
        query = self.stock_temporal_query(x_norm)
        key = self.stock_temporal_key(x_norm)
        value = self.stock_temporal_value(x_norm)

        attention_mask = build_local_causal_window_mask(
            time_steps=time_steps,
            window_size=self.stock_temporal_attention_window,
            device=stock_sequence.device,
            dtype=stock_sequence.dtype,
        )

        attention_scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(
            self.stock_hidden_dim
        )
        attention_scores = attention_scores + attention_mask.unsqueeze(0)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        attn_out = torch.matmul(attention_weights, value)
        attn_out = self.stock_temporal_attn_out(attn_out)

        x = x + attn_out
        x = x + self.stock_temporal_ffn(self.stock_temporal_norm2(x))

        attended_stock = x.reshape(
            num_scenarios,
            num_stocks,
            time_steps,
            self.stock_hidden_dim,
        ).permute(0, 2, 1, 3)

        stock_summary = fixed_window_causal_mean(
            attended_stock,
            self.stock_temporal_attention_window,
        )

        debug_info.update(
            {
                "stock_temporal_encoder_type": self.stock_temporal_encoder_type,
                "stock_temporal_attention_window": self.stock_temporal_attention_window,
                "stock_temporal_attention_mask_shape": tuple(attention_mask.shape),
                "stock_temporal_attention_weight_shape": tuple(attention_weights.shape),
            }
        )
        return attended_stock, stock_summary, debug_info


class MarketTemporalEncoder(nn.Module):
    """Encode market temporal feature sequences."""

    def __init__(
        self,
        *,
        market_feature_dim: int,
        market_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.market_input_proj = nn.Linear(market_feature_dim, market_hidden_dim)
        self.market_ffn = nn.Sequential(
            nn.Linear(market_hidden_dim, 4 * market_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * market_hidden_dim, market_hidden_dim),
        )

    def forward(
        self,
        x_market: torch.Tensor,
        *,
        time_encoding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        if x_market.ndim != 3:
            raise ValueError("x_market must have shape [S, T, F_market].")

        market_sequence = self.market_input_proj(x_market)
        if time_encoding is not None:
            market_sequence = market_sequence + time_encoding
        market_sequence = self.market_ffn(market_sequence)
        market_summary = causal_running_mean(market_sequence)
        return (
            market_sequence,
            market_summary,
            {
                "market_current_shape": tuple(market_sequence.shape),
                "market_summary_shape": tuple(market_summary.shape),
            },
        )
