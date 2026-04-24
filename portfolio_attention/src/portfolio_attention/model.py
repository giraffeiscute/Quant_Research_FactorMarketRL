"""Scenario-aware portfolio model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import math

from .config import ModelConfig


class PortfolioAttentionModel(nn.Module):
    """Portfolio model that preserves scenario and time structure.

    Expected tensor layout:
    - `x_stock`: [S, T, N, F_stock]
    - `x_market`: [S, T, F_market]
    - `stock_indices`: [S, N]
    - `target_returns`: [S, T, N]

    The forward pass keeps `S` (scenario) and `T` (time) separate and returns:
    - `stock_weights`: [S, T, N]
    - `cash_weight`: [S, T]
    - `portfolio_return`: [S, T]

    To avoid future leakage, the model keeps scenario/time structure intact and
    uses only current-or-earlier information at each time step. The stock
    branch can switch between causal running summaries and fixed-window causal
    self-attention over time, and can optionally apply a cross-sectional stock
    self-attention scorer; the market branch is a fixed linear-projection plus
    causal-running-summary context branch.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        num_stocks: int,
        max_lookback: int,
        stock_temporal_attention_window: int | None = None,
    ) -> None:
        super().__init__()
        if num_stocks <= 0:
            raise ValueError("num_stocks must be positive before model construction.")
        if max_lookback <= 0:
            raise ValueError("max_lookback must be positive before model construction.")

        self.config = config
        self.num_stocks = num_stocks
        self.max_lookback = max_lookback
        self.stock_temporal_encoder_type = config.stock_temporal_encoder_type
        self.stock_cross_sectional_encoder_type = config.stock_cross_sectional_encoder_type
        self.allocation_smoothing_alpha = float(config.allocation_smoothing_alpha)
        self.initial_allocation_mode = str(config.initial_allocation_mode).strip().lower()
        self.initial_random_concentration = float(config.initial_random_concentration)
        if self.initial_allocation_mode not in {"equal_weight", "random_dirichlet"}:
            raise ValueError(
                "initial_allocation_mode must be one of {'equal_weight', 'random_dirichlet'}, "
                f"received {self.initial_allocation_mode!r}."
            )
        if not 0.0 <= self.allocation_smoothing_alpha <= 1.0:
            raise ValueError(
                "allocation_smoothing_alpha must be in [0.0, 1.0], "
                f"received {self.allocation_smoothing_alpha}."
            )
        if self.initial_random_concentration <= 0.0:
            raise ValueError(
                "initial_random_concentration must be > 0.0, "
                f"received {self.initial_random_concentration}."
            )
        self.stock_temporal_attention_window = (
            max_lookback
            if stock_temporal_attention_window is None
            else int(stock_temporal_attention_window)
        )
        if self.stock_temporal_attention_window <= 0:
            raise ValueError(
                "stock_temporal_attention_window must be positive before model construction."
            )
        self.time_position_mode = config.time_positional_encoding_type
        self.stock_embedding_type = config.stock_embedding_type
        self.id_position_mode = self.stock_embedding_type
        self.stock_id_representation_type = config.stock_id_representation_type
        self.stock_identity_dim = config.stock_id_embedding_dim
        self.uses_post_temporal_identity = self.stock_embedding_type == "concat"
        self.stock_input_proj = nn.Linear(
            config.stock_feature_dim, config.stock_temporal_dim
        )
        self.stock_ffn = nn.Sequential(
            nn.Linear(config.stock_temporal_dim, config.stock_temporal_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.stock_temporal_dim, config.stock_temporal_dim),
        )
        self._legacy_stock_ffn_noop_for_inference = False

        self.market_input_proj = nn.Linear(
            config.market_feature_dim, config.market_temporal_dim
        )
        self.market_ffn = nn.Sequential(
            nn.Linear(config.market_temporal_dim, 4 * config.market_temporal_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.market_temporal_dim, config.market_temporal_dim),
        )

        self.stock_id_embedding = (
            nn.Embedding(num_stocks, config.stock_id_embedding_dim)
            if self.stock_id_representation_type == "learning"
            else None
        )
        if self.stock_id_representation_type == "gaussian":
            self.register_buffer(
                "stock_id_gaussian_code",
                self._build_stock_id_gaussian_code(
                    num_stocks=num_stocks,
                    embedding_dim=config.stock_id_embedding_dim,
                ),
                persistent=True,
            )
        else:
            self.register_buffer("stock_id_gaussian_code", None, persistent=False)
        self.stock_attention_dim = (
            config.cross_sectional_dim + self.stock_identity_dim
            if self.uses_post_temporal_identity
            else config.cross_sectional_dim
        )

        d = config.stock_temporal_dim

        if self.stock_temporal_encoder_type == "causal_self_attention":
            self.stock_temporal_query = nn.Linear(d, d, bias=False)
            self.stock_temporal_key = nn.Linear(d, d, bias=False)
            self.stock_temporal_value = nn.Linear(d, d, bias=False)

            self.stock_temporal_attn_out = nn.Linear(d, d)
            self.stock_temporal_norm1 = nn.LayerNorm(d)
            self.stock_temporal_norm2 = nn.LayerNorm(d)
            self.stock_temporal_ffn = nn.Sequential(
                nn.Linear(d, 4 * d),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(4 * d, d),
            )

        self.stock_content_proj = None
        self.stock_cross_attention_query = None
        self.stock_cross_attention_key = None
        self.stock_cross_attention_value = None
        self.stock_cross_attn_out = None
        self.stock_cross_norm1 = None
        self.stock_cross_norm2 = None
        self.stock_cross_ffn = None
        self.stock_cross_attention_score = None
        self.cash_cross_attention_score = None
        self.cash_state_mlp_base = None
        self.stock_prev_weight_mlp = None
        self.cash_prev_weight_mlp = None
        self.stock_score = None
        self.cash_score = None

        if self.stock_cross_sectional_encoder_type == "self_attention":
            stock_content_width = config.stock_temporal_dim * 2 + config.market_temporal_dim * 2
            self.stock_content_proj = nn.Linear(stock_content_width, config.cross_sectional_dim)

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
                nn.Dropout(config.dropout),
                nn.Linear(4 * self.stock_attention_dim, self.stock_attention_dim),
            )

            self.stock_cross_attention_score = nn.Linear(self.stock_attention_dim, 1)
            self.cash_cross_attention_score = nn.Linear(self.stock_attention_dim, 1)
            self.cash_state_mlp_base = nn.Sequential(
                nn.Linear(self.stock_attention_dim, self.stock_attention_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(self.stock_attention_dim, self.stock_attention_dim),
            )
            self.stock_prev_weight_mlp = nn.Sequential(
                nn.Linear(self.stock_attention_dim + 1, self.stock_attention_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(self.stock_attention_dim, self.stock_attention_dim),
            )
            self.cash_prev_weight_mlp = nn.Sequential(
                nn.Linear(self.stock_attention_dim + 1, self.stock_attention_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(self.stock_attention_dim, self.stock_attention_dim),
            )
        elif self.stock_cross_sectional_encoder_type == "mlp":
            stock_feature_width = (
                config.stock_temporal_dim * 2
                + config.market_temporal_dim * 2
                + (self.stock_identity_dim if self.uses_post_temporal_identity else 0)
            )
            cash_feature_width = (
                config.stock_temporal_dim * 2 + config.market_temporal_dim * 2
            )
            cash_hidden_dim = max(4, config.market_temporal_dim)

            self.stock_score = nn.Sequential(
                nn.Linear(stock_feature_width, config.cross_sectional_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.cross_sectional_dim, 1),
            )
            self.cash_score = nn.Sequential(
                nn.Linear(cash_feature_width, cash_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(cash_hidden_dim, 1),
            )
        else:
            raise ValueError(
                "Unsupported stock_cross_sectional_encoder_type: "
                f"{self.stock_cross_sectional_encoder_type!r}."
            )

    @staticmethod
    def _build_stock_id_gaussian_code(
        *,
        num_stocks: int,
        embedding_dim: int,
    ) -> torch.Tensor:
        # Fixed Gaussian ID codes keep stock identities dense without materializing one-hot vectors.
        gaussian_code = torch.randn(num_stocks, embedding_dim)
        gaussian_code = gaussian_code / gaussian_code.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return gaussian_code

    @staticmethod
    def _causal_running_mean(values: torch.Tensor) -> torch.Tensor:
        if values.ndim < 2:
            raise ValueError("Expected at least 2 dimensions for causal running mean.")
        steps = torch.arange(
            1,
            values.shape[1] + 1,
            device=values.device,
            dtype=values.dtype,
        )
        view_shape = [1, values.shape[1]] + [1] * (values.ndim - 2)
        return values.cumsum(dim=1) / steps.view(*view_shape)

    @staticmethod
    def _fixed_window_causal_mean(values: torch.Tensor, window_size: int) -> torch.Tensor:
        if values.ndim < 2:
            raise ValueError("Expected at least 2 dimensions for fixed-window causal mean.")
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, received {window_size}.")

        time_steps = values.shape[1]
        if window_size >= time_steps:
            return PortfolioAttentionModel._causal_running_mean(values)

        cumsum = values.cumsum(dim=1)
        window_sums = cumsum.clone()
        window_sums[:, window_size:] = cumsum[:, window_size:] - cumsum[:, :-window_size]
        counts = torch.arange(
            1,
            time_steps + 1,
            device=values.device,
            dtype=values.dtype,
        ).clamp(max=window_size)
        view_shape = [1, time_steps] + [1] * (values.ndim - 2)
        return window_sums / counts.view(*view_shape)

    @staticmethod
    def _build_local_causal_window_mask(
        *,
        time_steps: int,
        window_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if time_steps <= 0:
            raise ValueError(f"time_steps must be positive, received {time_steps}.")
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, received {window_size}.")

        query_positions = torch.arange(time_steps, device=device).unsqueeze(1)
        key_positions = torch.arange(time_steps, device=device).unsqueeze(0)
        earliest_allowed = query_positions - (window_size - 1)
        disallowed = (key_positions > query_positions) | (key_positions < earliest_allowed)
        mask = torch.zeros((time_steps, time_steps), device=device, dtype=dtype)
        return mask.masked_fill(disallowed, float("-inf"))

    @staticmethod
    def _build_sinusoidal_time_encoding(
        *,
        time_steps: int,
        embedding_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if time_steps <= 0:
            raise ValueError(f"time_steps must be positive, received {time_steps}.")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, received {embedding_dim}.")

        positions = torch.arange(time_steps, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / embedding_dim)
        )
        encoding = torch.zeros((time_steps, embedding_dim), device=device, dtype=dtype)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term[: encoding[:, 1::2].shape[1]])
        return encoding.unsqueeze(0)

    def _encode_market_sequence(
        self,
        market_sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if market_sequence.ndim != 3:
            raise ValueError("market_sequence must have shape [S, T, D_market].")

        market_sequence = self.market_ffn(market_sequence)
        market_summary = self._causal_running_mean(market_sequence)
        return market_sequence, market_summary

    def _build_stock_identity(
        self,
        *,
        stock_indices: torch.Tensor,
        time_steps: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.stock_id_representation_type == "learning":
            if self.stock_id_embedding is None:
                raise RuntimeError("stock_id_embedding must be initialized in learning mode.")
            return self.stock_id_embedding(stock_indices).unsqueeze(1).expand(-1, time_steps, -1, -1)

        if self.stock_id_representation_type != "gaussian":
            raise ValueError(
                "Unsupported stock_id_representation_type: "
                f"{self.stock_id_representation_type!r}."
            )

        if self.stock_id_gaussian_code is None:
            raise RuntimeError("stock_id_gaussian_code must be initialized in gaussian mode.")
        identity = self.stock_id_gaussian_code[stock_indices.to(dtype=torch.long)]
        return identity.to(dtype=dtype).unsqueeze(1).expand(-1, time_steps, -1, -1)

    def _encode_stock_sequence(
        self,
        stock_sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        if stock_sequence.ndim != 4:
            raise ValueError("stock_sequence must have shape [S, T, N, D_stock].")
        if not self._legacy_stock_ffn_noop_for_inference:
            stock_sequence = self.stock_ffn(stock_sequence)
        if self.stock_temporal_encoder_type == "running_summary":
            stock_summary = self._causal_running_mean(stock_sequence)
            return (
                stock_sequence,
                stock_summary,
                {
                    "stock_temporal_encoder_type": self.stock_temporal_encoder_type,
                    "stock_temporal_attention_window": self.stock_temporal_attention_window,
                    "stock_temporal_attention_mask_shape": None,
                    "stock_temporal_attention_weight_shape": None,
                },
            )

        if self.stock_temporal_encoder_type != "causal_self_attention":
            raise ValueError(
                "Unsupported stock_temporal_encoder_type: "
                f"{self.stock_temporal_encoder_type!r}."
            )

        num_scenarios, time_steps, num_stocks, _ = stock_sequence.shape

        flattened_stock_sequence = stock_sequence.permute(0, 2, 1, 3).reshape(
            num_scenarios * num_stocks,
            time_steps,
            self.config.stock_temporal_dim,
        )

        x = flattened_stock_sequence

        x_norm = self.stock_temporal_norm1(x)
        query = self.stock_temporal_query(x_norm)
        key = self.stock_temporal_key(x_norm)
        value = self.stock_temporal_value(x_norm)

        attention_mask = self._build_local_causal_window_mask(
            time_steps=time_steps,
            window_size=self.stock_temporal_attention_window,
            device=stock_sequence.device,
            dtype=stock_sequence.dtype,
        )

        attention_scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(
            self.config.stock_temporal_dim
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
            self.config.stock_temporal_dim,
        ).permute(0, 2, 1, 3)

        stock_summary = self._fixed_window_causal_mean(
            attended_stock,
            self.stock_temporal_attention_window,
        )

        return (
            attended_stock,
            stock_summary,
            {
                "stock_temporal_encoder_type": self.stock_temporal_encoder_type,
                "stock_temporal_attention_window": self.stock_temporal_attention_window,
                "stock_temporal_attention_mask_shape": tuple(attention_mask.shape),
                "stock_temporal_attention_weight_shape": tuple(attention_weights.shape),
            },
        )

    def enable_legacy_stock_ffn_noop_for_inference(self) -> None:
        self._legacy_stock_ffn_noop_for_inference = True

    def _initial_allocation(
        self,
        *,
        num_scenarios: int,
        num_stocks: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.initial_allocation_mode == "equal_weight":
            return torch.full(
                (num_scenarios, num_stocks + 1),
                1.0 / (num_stocks + 1),
                device=device,
                dtype=dtype,
            )

        if self.initial_allocation_mode == "random_dirichlet":
            concentration = torch.full(
                (num_stocks + 1,),
                self.initial_random_concentration,
                device=device,
                dtype=dtype,
            )
            return torch.distributions.Dirichlet(concentration).sample((num_scenarios,))

        raise ValueError(f"Unsupported initial_allocation_mode: {self.initial_allocation_mode!r}")

    def _smooth_allocation_from_logits(
        self,
        *,
        allocation_logits: torch.Tensor,
        num_stocks: int,
        initial_allocation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if allocation_logits.ndim != 3:
            raise ValueError("allocation_logits must have shape [S, T, N+1].")
        num_scenarios, time_steps, total_assets = allocation_logits.shape
        expected_assets = num_stocks + 1
        if total_assets != expected_assets:
            raise ValueError(
                f"allocation_logits last dimension must equal {expected_assets}, received {total_assets}."
            )
        if initial_allocation is None:
            prev_weight = self._initial_allocation(
                num_scenarios=num_scenarios,
                num_stocks=num_stocks,
                device=allocation_logits.device,
                dtype=allocation_logits.dtype,
            )
        else:
            prev_weight = initial_allocation
            if prev_weight.shape != (num_scenarios, expected_assets):
                raise ValueError(
                    "initial_allocation must have shape [S, N+1]. "
                    f"Received {tuple(prev_weight.shape)} expected {(num_scenarios, expected_assets)}."
                )
        alpha = float(self.allocation_smoothing_alpha)
        raw_allocations: list[torch.Tensor] = []
        allocations: list[torch.Tensor] = []
        turnovers: list[torch.Tensor] = []
        for time_index in range(time_steps):
            logits_t = allocation_logits[:, time_index, :]
            raw_t = torch.softmax(logits_t, dim=-1)
            allocation_t = alpha * raw_t + (1.0 - alpha) * prev_weight
            turnover_t = 0.5 * torch.abs(allocation_t - prev_weight).sum(dim=-1)
            raw_allocations.append(raw_t)
            allocations.append(allocation_t)
            turnovers.append(turnover_t)
            prev_weight = allocation_t

        return (
            torch.stack(raw_allocations, dim=1),
            torch.stack(allocations, dim=1),
            torch.stack(turnovers, dim=1),
        )

    def _score_stock_allocation(
        self,
        *,
        stock_temporal_current: torch.Tensor,
        stock_temporal_summary: torch.Tensor,
        market_current: torch.Tensor,
        market_summary: torch.Tensor,
        stock_identity: torch.Tensor | None,
        initial_allocation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        num_scenarios, time_steps, num_stocks, _ = stock_temporal_current.shape
        market_current_expanded = market_current.unsqueeze(2).expand(-1, -1, num_stocks, -1)
        market_summary_expanded = market_summary.unsqueeze(2).expand(-1, -1, num_stocks, -1)

        if self.stock_cross_sectional_encoder_type == "mlp":
            if self.stock_score is None or self.cash_score is None:
                raise RuntimeError("MLP cross-sectional scorer modules must be initialized.")
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
            stock_logits = self.stock_score(stock_features).squeeze(-1)
            pooled_stock_current = stock_temporal_current.mean(dim=2)
            pooled_stock_running = stock_temporal_summary.mean(dim=2)
            cash_features = torch.cat(
                [pooled_stock_current, pooled_stock_running, market_current, market_summary],
                dim=-1,
            )
            cash_logit = self.cash_score(cash_features).squeeze(-1)
            return (
                stock_logits,
                cash_logit,
                {
                    "stock_feature_shape": tuple(stock_features.shape),
                    "stock_content_shape": None,
                    "stock_attention_input_shape": None,
                    "stock_attention_weight_shape": None,
                },
            )

        if self.stock_cross_sectional_encoder_type != "self_attention":
            raise ValueError(
                "Unsupported stock_cross_sectional_encoder_type: "
                f"{self.stock_cross_sectional_encoder_type!r}."
            )
        if (
            self.stock_content_proj is None
            or self.stock_cross_attention_query is None
            or self.stock_cross_attention_key is None
            or self.stock_cross_attention_value is None
            or self.stock_cross_attn_out is None
            or self.stock_cross_norm1 is None
            or self.stock_cross_norm2 is None
            or self.stock_cross_ffn is None
            or self.stock_cross_attention_score is None
            or self.cash_cross_attention_score is None
            or self.cash_state_mlp_base is None
            or self.stock_prev_weight_mlp is None
            or self.cash_prev_weight_mlp is None
        ):
            raise RuntimeError("Self-attention cross-sectional scorer modules must be initialized.")

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
        if initial_allocation is None:
            prev_weight = self._initial_allocation(
                num_scenarios=num_scenarios,
                num_stocks=num_stocks,
                device=attended_stock.device,
                dtype=attended_stock.dtype,
            )
        else:
            prev_weight = initial_allocation
            if prev_weight.shape != (num_scenarios, num_stocks + 1):
                raise ValueError(
                    "initial_allocation must have shape [S, N+1]. "
                    f"Received {tuple(prev_weight.shape)} expected {(num_scenarios, num_stocks + 1)}."
                )
        alpha = float(self.allocation_smoothing_alpha)
        stock_logits_by_step: list[torch.Tensor] = []
        cash_logit_by_step: list[torch.Tensor] = []
        for time_index in range(time_steps):
            attended_stock_t = attended_stock[:, time_index, :, :]
            stock_state_input = torch.cat(
                [attended_stock_t, prev_weight[:, :num_stocks].unsqueeze(-1)],
                dim=-1,
            )
            stock_state_repr = self.stock_prev_weight_mlp(stock_state_input)
            stock_logit_t = self.stock_cross_attention_score(stock_state_repr).squeeze(-1)
            cash_base_input = attended_stock_t.mean(dim=1)
            cash_state_base = self.cash_state_mlp_base(cash_base_input)
            cash_state_input = torch.cat([cash_state_base, prev_weight[:, -1:].contiguous()], dim=-1)
            cash_state_repr = self.cash_prev_weight_mlp(cash_state_input)
            cash_logit_t = self.cash_cross_attention_score(cash_state_repr).squeeze(-1)

            allocation_logits_t = torch.cat([stock_logit_t, cash_logit_t.unsqueeze(-1)], dim=-1)
            raw_t = torch.softmax(allocation_logits_t, dim=-1)
            prev_weight = alpha * raw_t + (1.0 - alpha) * prev_weight

            stock_logits_by_step.append(stock_logit_t)
            cash_logit_by_step.append(cash_logit_t)
        stock_logits = torch.stack(stock_logits_by_step, dim=1)
        cash_logit = torch.stack(cash_logit_by_step, dim=1)
        return (
            stock_logits,
            cash_logit,
            {
                "stock_feature_shape": tuple(stock_attention_inputs.shape),
                "stock_content_shape": tuple(stock_content.shape),
                "stock_attention_input_shape": tuple(stock_attention_inputs.shape),
                "stock_attention_weight_shape": tuple(attention_weights.shape),
            },
        )

    def forward(
        self,
        x_stock: torch.Tensor,
        x_market: torch.Tensor,
        stock_indices: torch.Tensor,
        target_returns: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if x_stock.ndim != 4:
            raise ValueError("x_stock must have shape [S, T, N, F_stock].")
        if x_market.ndim != 3:
            raise ValueError("x_market must have shape [S, T, F_market].")
        if stock_indices.ndim != 2:
            raise ValueError("stock_indices must have shape [S, N].")

        num_scenarios, time_steps, num_stocks, stock_feature_dim = x_stock.shape
        assert stock_feature_dim == self.config.stock_feature_dim
        assert x_market.shape == (num_scenarios, time_steps, self.config.market_feature_dim)
        assert stock_indices.shape == (num_scenarios, num_stocks)
        if time_steps > self.max_lookback:
            raise ValueError(
                f"Received time_steps={time_steps}, but model was constructed for max_lookback={self.max_lookback}."
            )
        if num_stocks > self.num_stocks:
            raise ValueError(
                f"Received num_stocks={num_stocks}, but model was constructed for {self.num_stocks}."
            )
        stock_current = self.stock_input_proj(x_stock)
        market_current = self.market_input_proj(x_market)

        if self.time_position_mode == "sinusoidal":
            stock_time_encoding = self._build_sinusoidal_time_encoding(
                time_steps=time_steps,
                embedding_dim=self.config.stock_temporal_dim,
                device=stock_current.device,
                dtype=stock_current.dtype,
            ).unsqueeze(2)
            market_time_encoding = self._build_sinusoidal_time_encoding(
                time_steps=time_steps,
                embedding_dim=self.config.market_temporal_dim,
                device=market_current.device,
                dtype=market_current.dtype,
            )
            stock_current = stock_current + stock_time_encoding
            market_current = market_current + market_time_encoding

        stock_identity = self._build_stock_identity(
            stock_indices=stock_indices,
            time_steps=time_steps,
            dtype=stock_current.dtype,
        )
        if self.stock_embedding_type == "pre_temporal":
            stock_current = stock_current + stock_identity
            stock_identity_for_scoring = None
        else:
            stock_identity_for_scoring = stock_identity

        stock_temporal_current, stock_temporal_summary, stock_temporal_debug_info = (
            self._encode_stock_sequence(stock_current)
        )
        market_current, market_summary = self._encode_market_sequence(market_current)
        initial_allocation = self._initial_allocation(
            num_scenarios=num_scenarios,
            num_stocks=num_stocks,
            device=stock_temporal_current.device,
            dtype=stock_temporal_current.dtype,
        )

        stock_logits, cash_logit, stock_debug_info = self._score_stock_allocation(
            stock_temporal_current=stock_temporal_current,
            stock_temporal_summary=stock_temporal_summary,
            market_current=market_current,
            market_summary=market_summary,
            stock_identity=stock_identity_for_scoring,
            initial_allocation=initial_allocation,
        )

        allocation_logits = torch.cat([stock_logits, cash_logit.unsqueeze(-1)], dim=-1)
        raw_allocation, allocation, turnover = self._smooth_allocation_from_logits(
            allocation_logits=allocation_logits,
            num_stocks=num_stocks,
            initial_allocation=initial_allocation,
        )
        stock_weights = allocation[..., :-1]
        cash_weight = allocation[..., -1]

        portfolio_return = None
        if target_returns is not None:
            expected_shape = (num_scenarios, time_steps, num_stocks)
            if target_returns.shape != expected_shape:
                raise ValueError(
                    f"target_returns must have shape {expected_shape}, received {tuple(target_returns.shape)}."
                )
            portfolio_return = (stock_weights * target_returns).sum(dim=-1)

        debug_info = {
            "time_position_mode": self.time_position_mode,
            "stock_embedding_type": self.stock_embedding_type,
            "id_position_mode": self.id_position_mode,
            "stock_id_representation_type": self.stock_id_representation_type,
            "stock_temporal_encoder_type": self.stock_temporal_encoder_type,
            "stock_cross_sectional_encoder_type": self.stock_cross_sectional_encoder_type,
            "stock_temporal_attention_window": self.stock_temporal_attention_window,
            "allocation_smoothing_alpha": self.allocation_smoothing_alpha,
            "initial_allocation_mode": self.initial_allocation_mode,
            "initial_random_concentration": self.initial_random_concentration,
            "stock_current_shape": tuple(stock_current.shape),
            "stock_temporal_current_shape": tuple(stock_temporal_current.shape),
            "stock_running_shape": tuple(stock_temporal_summary.shape),
            "stock_temporal_summary_shape": tuple(stock_temporal_summary.shape),
            "market_current_shape": tuple(market_current.shape),
            "market_summary_shape": tuple(market_summary.shape),
            **stock_debug_info,
            **stock_temporal_debug_info,
        }

        return {
            "stock_weights": stock_weights,
            "cash_weight": cash_weight,
            "stock_logits": stock_logits,
            "cash_logit": cash_logit,
            "allocation_logits": allocation_logits,
            "raw_allocation": raw_allocation,
            "allocation": allocation,
            "turnover": turnover,
            "portfolio_return": portfolio_return,
            "debug_info": debug_info,
        }
