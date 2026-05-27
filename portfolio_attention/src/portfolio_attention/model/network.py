"""Scenario-aware portfolio model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..config import ModelConfig
from .allocation_path import (
    AllocationResult,
    AllocationSmoother,
)
from .cross_sectional import (
    AttentionCrossSectionalScorer,
    CrossSectionalScoreResult,
    MLPCrossSectionalScorer,
)
from .critic import PortfolioCritic
from .stock_embedding import StockIdentityEmbedding
from .temp_encoders import MarketTemporalEncoder, StockTemporalEncoder
from .temporal_utils import build_sinusoidal_time_encoding


_LEGACY_STATE_DICT_PREFIX_RENAMES = {
    "stock_input_proj.": "stock_temporal_encoder.stock_input_proj.",
    "stock_ffn.": "stock_temporal_encoder.stock_ffn.",
    "stock_temporal_query.": "stock_temporal_encoder.stock_temporal_query.",
    "stock_temporal_key.": "stock_temporal_encoder.stock_temporal_key.",
    "stock_temporal_value.": "stock_temporal_encoder.stock_temporal_value.",
    "stock_temporal_attn_out.": "stock_temporal_encoder.stock_temporal_attn_out.",
    "stock_temporal_norm1.": "stock_temporal_encoder.stock_temporal_norm1.",
    "stock_temporal_norm2.": "stock_temporal_encoder.stock_temporal_norm2.",
    "stock_temporal_ffn.": "stock_temporal_encoder.stock_temporal_ffn.",
    "market_input_proj.": "market_temporal_encoder.market_input_proj.",
    "market_ffn.": "market_temporal_encoder.market_ffn.",
    "stock_score.": "cross_sectional_scorer.task_head.stock_score.",
    "cash_score.": "cross_sectional_scorer.task_head.cash_score.",
    "stock_content_proj.": "cross_sectional_scorer.stock_content_proj.",
    "stock_cross_attention_query.": "cross_sectional_scorer.stock_cross_attention_query.",
    "stock_cross_attention_key.": "cross_sectional_scorer.stock_cross_attention_key.",
    "stock_cross_attention_value.": "cross_sectional_scorer.stock_cross_attention_value.",
    "stock_cross_attn_out.": "cross_sectional_scorer.stock_cross_attn_out.",
    "stock_cross_norm1.": "cross_sectional_scorer.stock_cross_norm1.",
    "stock_cross_norm2.": "cross_sectional_scorer.stock_cross_norm2.",
    "stock_cross_ffn.": "cross_sectional_scorer.stock_cross_ffn.",
    "stock_cross_attention_score.": (
        "cross_sectional_scorer.task_head.stock_cross_attention_score."
    ),
    "cash_cross_attention_score.": (
        "cross_sectional_scorer.task_head.cash_cross_attention_score."
    ),
    "cash_state_mlp_base.": "cross_sectional_scorer.task_head.cash_state_mlp_base.",
    "stock_prev_weight_mlp.": "cross_sectional_scorer.task_head.stock_prev_weight_mlp.",
    "cash_prev_weight_mlp.": "cross_sectional_scorer.task_head.cash_prev_weight_mlp.",
    "cross_sectional_scorer.stock_score.": "cross_sectional_scorer.task_head.stock_score.",
    "cross_sectional_scorer.cash_score.": "cross_sectional_scorer.task_head.cash_score.",
    "cross_sectional_scorer.stock_cross_attention_score.": (
        "cross_sectional_scorer.task_head.stock_cross_attention_score."
    ),
    "cross_sectional_scorer.cash_cross_attention_score.": (
        "cross_sectional_scorer.task_head.cash_cross_attention_score."
    ),
    "cross_sectional_scorer.cash_state_mlp_base.": (
        "cross_sectional_scorer.task_head.cash_state_mlp_base."
    ),
    "cross_sectional_scorer.stock_prev_weight_mlp.": (
        "cross_sectional_scorer.task_head.stock_prev_weight_mlp."
    ),
    "cross_sectional_scorer.cash_prev_weight_mlp.": (
        "cross_sectional_scorer.task_head.cash_prev_weight_mlp."
    ),
}


def _remap_legacy_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
    *,
    module_prefix: str,
) -> None:
    for legacy_prefix, new_prefix in _LEGACY_STATE_DICT_PREFIX_RENAMES.items():
        full_legacy_prefix = module_prefix + legacy_prefix
        full_new_prefix = module_prefix + new_prefix
        for key in list(state_dict.keys()):
            if not key.startswith(full_legacy_prefix):
                continue
            new_key = full_new_prefix + key[len(full_legacy_prefix) :]
            if new_key not in state_dict:
                state_dict[new_key] = state_dict[key]
            del state_dict[key]


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
    - `value_prediction`: [S, T] when requested, otherwise None
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
        self.inference_allocation_mode = str(config.inference_allocation_mode).strip().lower()
        self.initial_random_concentration = float(config.initial_random_concentration)
        if not isinstance(config.detach_prev_weight, bool):
            raise ValueError(
                "detach_prev_weight must be a bool, "
                f"received {config.detach_prev_weight!r}."
            )
        self.detach_prev_weight = config.detach_prev_weight
        self.use_prev_weight_feature = bool(getattr(config, "use_prev_weight_feature", False))
        if self.initial_allocation_mode not in {"equal_weight", "random_dirichlet"}:
            raise ValueError(
                "initial_allocation_mode must be one of {'equal_weight', 'random_dirichlet'}, "
                f"received {self.initial_allocation_mode!r}."
            )
        if self.inference_allocation_mode not in {"softmax", "dirichlet_mean"}:
            raise ValueError(
                "inference_allocation_mode must be one of {'softmax', 'dirichlet_mean'}, "
                f"received {self.inference_allocation_mode!r}."
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
        self.stock_temporal_encoder = StockTemporalEncoder(
            stock_feature_dim=config.stock_feature_dim,
            stock_hidden_dim=config.stock_temporal_dim,
            stock_temporal_encoder_type=self.stock_temporal_encoder_type,
            stock_temporal_attention_window=self.stock_temporal_attention_window,
            dropout=config.dropout,
        )
        self.market_temporal_encoder = MarketTemporalEncoder(
            market_feature_dim=config.market_feature_dim,
            market_hidden_dim=config.market_temporal_dim,
            dropout=config.dropout,
        )
        self.allocation_smoother = AllocationSmoother(
            initial_allocation_mode=self.initial_allocation_mode,
            initial_random_concentration=self.initial_random_concentration,
            allocation_smoothing_alpha=self.allocation_smoothing_alpha,
            detach_prev_weight=self.detach_prev_weight,
        )

        self.stock_identity_embedding = StockIdentityEmbedding(
            num_stocks=num_stocks,
            representation_type=config.stock_id_representation_type,
            embedding_dim=config.stock_id_embedding_dim,
        )
        self.stock_attention_dim = (
            config.cross_sectional_dim + self.stock_identity_dim
            if self.uses_post_temporal_identity
            else config.cross_sectional_dim
        )

        if self.stock_cross_sectional_encoder_type == "self_attention":
            self.cross_sectional_scorer: nn.Module = AttentionCrossSectionalScorer(
                stock_temporal_dim=config.stock_temporal_dim,
                market_temporal_dim=config.market_temporal_dim,
                stock_identity_dim=self.stock_identity_dim,
                cross_sectional_dim=config.cross_sectional_dim,
                dropout=config.dropout,
                uses_post_temporal_identity=self.uses_post_temporal_identity,
                allocation_smoothing_alpha=self.allocation_smoothing_alpha,
                detach_prev_weight=self.detach_prev_weight,
                use_prev_weight_feature=self.use_prev_weight_feature,
                inference_allocation_mode=self.inference_allocation_mode,
                dirichlet_logit_scale=float(config.dirichlet_logit_scale),
            )
        elif self.stock_cross_sectional_encoder_type == "mlp":
            self.cross_sectional_scorer = MLPCrossSectionalScorer(
                stock_temporal_dim=config.stock_temporal_dim,
                market_temporal_dim=config.market_temporal_dim,
                stock_identity_dim=self.stock_identity_dim,
                cross_sectional_dim=config.cross_sectional_dim,
                dropout=config.dropout,
                uses_post_temporal_identity=self.uses_post_temporal_identity,
                inference_allocation_mode=self.inference_allocation_mode,
                dirichlet_logit_scale=float(config.dirichlet_logit_scale),
            )
        else:
            raise ValueError(
                "Unsupported stock_cross_sectional_encoder_type: "
                f"{self.stock_cross_sectional_encoder_type!r}."
            )

        # Keep adding the critic from advancing the policy model's historical RNG sequence.
        rng_state = torch.random.get_rng_state()
        self.critic = PortfolioCritic(
            stock_temporal_dim=config.stock_temporal_dim,
            market_temporal_dim=config.market_temporal_dim,
            hidden_dim=config.cross_sectional_dim,
            dropout=config.dropout,
        )
        torch.random.set_rng_state(rng_state)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        _remap_legacy_state_dict_keys(state_dict, module_prefix=prefix)
        legacy_value_head_prefix = prefix + "value_head."
        critic_value_head_prefix = prefix + "critic.value_head."
        for key in list(state_dict.keys()):
            if not key.startswith(legacy_value_head_prefix):
                continue
            new_key = critic_value_head_prefix + key[len(legacy_value_head_prefix) :]
            if new_key not in state_dict:
                state_dict[new_key] = state_dict[key]
            del state_dict[key]
        for key, value in self.state_dict().items():
            if not key.startswith("critic."):
                continue
            full_key = prefix + key
            if full_key not in state_dict:
                state_dict[full_key] = value.detach().clone()

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def enable_legacy_stock_ffn_noop_for_inference(self) -> None:
        self.stock_temporal_encoder.enable_legacy_stock_ffn_noop_for_inference()

    def _build_allocation_distribution_debug_info(
        self,
        *,
        include_alpha_debug_stats: bool = False,
    ) -> dict[str, Any]:
        del include_alpha_debug_stats
        return {
            "allocation_distribution_type": self.inference_allocation_mode,
            "allocation_sampling_mode": "deterministic",
        }

    def forward(
        self,
        x_stock: torch.Tensor,
        x_market: torch.Tensor,
        stock_indices: torch.Tensor,
        target_returns: torch.Tensor | None = None,
        compute_value_prediction: bool = False,
        initial_allocation_override: torch.Tensor | None = None,
        return_state_features: bool = False,
        return_last_step_only: bool = False,
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
        stock_time_encoding = None
        market_time_encoding = None
        if self.time_position_mode == "sinusoidal":
            stock_time_encoding = build_sinusoidal_time_encoding(
                time_steps=time_steps,
                embedding_dim=self.config.stock_temporal_dim,
                device=x_stock.device,
                dtype=x_stock.dtype,
            ).unsqueeze(2)
            market_time_encoding = build_sinusoidal_time_encoding(
                time_steps=time_steps,
                embedding_dim=self.config.market_temporal_dim,
                device=x_market.device,
                dtype=x_market.dtype,
            )

        stock_identity = self.stock_identity_embedding(
            stock_indices=stock_indices,
            time_steps=time_steps,
            dtype=x_stock.dtype,
        )
        if self.stock_embedding_type == "pre_temporal":
            stock_identity_for_temporal = stock_identity
            stock_identity_for_scoring = None
        else:
            stock_identity_for_temporal = None
            stock_identity_for_scoring = stock_identity

        stock_temporal_current, stock_temporal_summary, stock_temporal_debug_info = (
            self.stock_temporal_encoder(
                x_stock,
                stock_identity=stock_identity_for_temporal,
                time_encoding=stock_time_encoding,
            )
        )
        market_current, market_summary, market_temporal_debug_info = (
            self.market_temporal_encoder(x_market, time_encoding=market_time_encoding)
        )
        if bool(return_last_step_only):
            stock_temporal_current = stock_temporal_current[:, -1:, :, :]
            stock_temporal_summary = stock_temporal_summary[:, -1:, :, :]
            market_current = market_current[:, -1:, :]
            market_summary = market_summary[:, -1:, :]
            if stock_identity_for_scoring is not None:
                stock_identity_for_scoring = stock_identity_for_scoring[:, -1:, :, :]
        if initial_allocation_override is None:
            initial_allocation = self.allocation_smoother.initial_allocation(
                num_scenarios=num_scenarios,
                total_assets=num_stocks + 1,
                device=stock_temporal_current.device,
                dtype=stock_temporal_current.dtype,
            )
        else:
            expected_initial_shape = (num_scenarios, num_stocks + 1)
            if tuple(initial_allocation_override.shape) != expected_initial_shape:
                raise ValueError(
                    "initial_allocation_override must have shape [S, N+1]. "
                    f"Expected {expected_initial_shape}, received {tuple(initial_allocation_override.shape)}."
                )
            initial_allocation = initial_allocation_override.to(
                device=stock_temporal_current.device,
                dtype=stock_temporal_current.dtype,
            )

        score_outputs = self.cross_sectional_scorer(
            stock_temporal_current=stock_temporal_current,
            stock_temporal_summary=stock_temporal_summary,
            market_current=market_current,
            market_summary=market_summary,
            stock_identity=stock_identity_for_scoring,
            initial_allocation=initial_allocation,
        )
        if not isinstance(score_outputs, CrossSectionalScoreResult):
            raise ValueError(
                "cross_sectional_scorer must return a CrossSectionalScoreResult."
            )
        stock_logits = score_outputs.stock_logits
        cash_logit = score_outputs.cash_logit
        stock_debug_info = score_outputs.debug_info
        precomputed_smoothing = score_outputs.precomputed_smoothing

        allocation_logits = torch.cat([stock_logits, cash_logit.unsqueeze(-1)], dim=-1)
        if precomputed_smoothing is None:
            raw_allocation = score_outputs.raw_allocation
            if raw_allocation is None:
                raise RuntimeError(
                    "CrossSectionalScoreResult.raw_allocation must be set when "
                    "precomputed_smoothing is not provided."
                )
            allocation_result: AllocationResult = self.allocation_smoother(
                raw_allocation=raw_allocation,
                initial_allocation=initial_allocation,
            )
            raw_allocation = allocation_result.raw_allocation
            allocation = allocation_result.allocation
            turnover = allocation_result.turnover
            previous_allocation = allocation_result.previous_allocation
        else:
            if len(precomputed_smoothing) != 4:
                raise ValueError(
                    "precomputed_smoothing must contain "
                    "(raw_allocation, allocation, turnover, previous_allocation)."
                )
            raw_allocation, allocation, turnover, previous_allocation = precomputed_smoothing
        allocation_delta = allocation - previous_allocation
        allocation_change_l2 = allocation_delta.pow(2).sum(dim=-1)
        stock_weights = allocation[..., :-1]
        cash_weight = allocation[..., -1]
        value_prediction = None
        if bool(compute_value_prediction):
            value_prediction = self.critic(
                stock_temporal_current=stock_temporal_current,
                stock_temporal_summary=stock_temporal_summary,
                market_current=market_current,
                market_summary=market_summary,
                previous_allocation=previous_allocation,
            )

        portfolio_return = None
        if target_returns is not None:
            output_time_steps = int(stock_weights.shape[1])
            expected_shape = (num_scenarios, time_steps, num_stocks)
            if target_returns.shape != expected_shape:
                raise ValueError(
                    f"target_returns must have shape {expected_shape}, received {tuple(target_returns.shape)}."
                )
            target_returns_for_output = (
                target_returns[:, -output_time_steps:, :]
                if bool(return_last_step_only)
                else target_returns
            )
            portfolio_return = (stock_weights * target_returns_for_output).sum(dim=-1)

        debug_info = {
            "time_position_mode": self.time_position_mode,
            "stock_embedding_type": self.stock_embedding_type,
            "id_position_mode": self.id_position_mode,
            "stock_id_representation_type": self.stock_id_representation_type,
            "stock_temporal_encoder_type": self.stock_temporal_encoder_type,
            "stock_cross_sectional_encoder_type": self.stock_cross_sectional_encoder_type,
            "stock_temporal_attention_window": self.stock_temporal_attention_window,
            "allocation_smoothing_alpha": self.allocation_smoothing_alpha,
            "detach_prev_weight": self.detach_prev_weight,
            "use_prev_weight_feature": self.use_prev_weight_feature,
            "initial_allocation_mode": self.initial_allocation_mode,
            "initial_random_concentration": self.initial_random_concentration,
            "stock_temporal_current_shape": tuple(stock_temporal_current.shape),
            "stock_running_shape": tuple(stock_temporal_summary.shape),
            "stock_temporal_summary_shape": tuple(stock_temporal_summary.shape),
            **self._build_allocation_distribution_debug_info(),
            **stock_debug_info,
            **stock_temporal_debug_info,
            **market_temporal_debug_info,
        }

        outputs = {
            "stock_weights": stock_weights,
            "cash_weight": cash_weight,
            "stock_logits": stock_logits,
            "cash_logit": cash_logit,
            "allocation_logits": allocation_logits,
            "raw_allocation": raw_allocation,
            "allocation": allocation,
            "initial_allocation": initial_allocation,
            "previous_allocation": previous_allocation,
            "value_prediction": value_prediction,
            "turnover": turnover,
            "allocation_change_l2": allocation_change_l2,
            "portfolio_return": portfolio_return,
            "debug_info": debug_info,
        }
        if bool(return_state_features):
            outputs.update(
                {
                    "stock_temporal_current": stock_temporal_current,
                    "stock_temporal_summary": stock_temporal_summary,
                    "market_current": market_current,
                    "market_summary": market_summary,
                }
            )
        return outputs
