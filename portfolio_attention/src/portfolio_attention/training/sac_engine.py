"""SAC training collection helpers.

This module owns the repo-native SAC rollout-to-replay collection step. The
LightningModule keeps optimizer state, manual backward, and logging; this file
keeps the algorithm data contract out of the frontend runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..common.utils import apply_score_mask
from ..config import DataConfig, ModelConfig, TrainConfig
from ..model import PortfolioAttentionModel
from ..rl.replay import SACTransitionBatch, collect_sac_transitions_from_rollout
from ..rl.rebalance import build_rebalance_schedule, gather_decision_steps
from ..rl.sac_policy import build_dirichlet_sac_policy


@dataclass(frozen=True)
class SACTrainingCollection:
    """Collected SAC replay transitions plus training diagnostics."""

    transitions: SACTransitionBatch
    summary: dict[str, torch.Tensor]
    dummy_loss: torch.Tensor
    batch_size: int
    context_window_steps: int


def collect_sac_training_batch(
    model: PortfolioAttentionModel,
    batch: dict[str, Any],
    *,
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> SACTrainingCollection:
    """Sample SAC rollout actions and convert the rollout into replay transitions."""
    with torch.no_grad():
        outputs = model(
            batch["x_stock"],
            batch["x_market"],
            batch["stock_indices"],
            target_returns=batch["r_stock"],
        )
        score_mask = _require_score_mask(batch)
        allocation_logits = _require_tensor(outputs, "allocation_logits")
        scored_logits = apply_score_mask(allocation_logits, score_mask)
        schedule = build_rebalance_schedule(
            horizon_steps=int(scored_logits.shape[1]),
            rebalance_interval_days=int(data_config.rebalance_interval_days),
        )
        decision_logits = gather_decision_steps(scored_logits, schedule=schedule)
        policy = build_dirichlet_sac_policy(
            decision_logits,
            model_config=model_config,
            train_config=train_config,
        )
        actions = policy.sample_action()
        transitions = collect_sac_transitions_from_rollout(
            batch,
            actions,
            transaction_cost_rate=float(train_config.transaction_cost_rate),
            reward_scale=float(train_config.rl_training.reward_scale),
            rebalance_interval_days=int(data_config.rebalance_interval_days),
            context_window_steps=_resolve_sac_context_window_steps(
                train_config,
                full_time_steps=int(batch["x_stock"].shape[1]),
            ),
        )
        raw_allocation = _require_tensor(outputs, "raw_allocation")
        sampled_net_returns = _reshape_transition_values(
            transitions.reward,
            batch_size=int(actions.shape[0]),
            horizon_steps=schedule.num_decisions,
        ) * float(train_config.rl_training.reward_scale)
        sampled_turnover = _sampled_action_turnover(actions)
        scenario_final_returns = torch.prod(1.0 + sampled_net_returns, dim=1) - 1.0
        summary = {
            "scenario_final_returns": scenario_final_returns.detach(),
            "mean_turnover": sampled_turnover.detach().mean(),
            "sampled_action_reward_mean": transitions.reward.detach().mean(),
            "return_mean_min": sampled_net_returns.mean(dim=1).detach().min(),
            "return_mean_max": sampled_net_returns.mean(dim=1).detach().max(),
            "return_std_min": sampled_net_returns.std(dim=1, unbiased=True).detach().min(),
            "return_std_max": sampled_net_returns.std(dim=1, unbiased=True).detach().max(),
            "allocation_logits_abs_max": allocation_logits.detach().abs().max(),
            "raw_allocation_min": raw_allocation.detach().min(),
            "raw_allocation_max": raw_allocation.detach().max(),
        }

    dummy_loss = next(model.parameters()).sum() * 0.0
    return SACTrainingCollection(
        transitions=transitions,
        summary=summary,
        dummy_loss=dummy_loss,
        batch_size=int(transitions.batch_size),
        context_window_steps=int(transitions.x_stock.shape[1]),
    )


def _require_tensor(outputs: dict[str, Any], key: str) -> torch.Tensor:
    value = outputs.get(key)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"SAC training requires model output {key!r}.")
    return value


def _require_score_mask(batch: dict[str, Any]) -> torch.Tensor:
    score_mask = batch.get("score_mask")
    if not isinstance(score_mask, torch.Tensor):
        raise RuntimeError("SAC training requires score_mask tensor in batch.")
    return score_mask.to(dtype=torch.bool)


def _reshape_transition_values(
    values: torch.Tensor,
    *,
    batch_size: int,
    horizon_steps: int,
) -> torch.Tensor:
    expected_shape = (int(batch_size) * int(horizon_steps), 1)
    if tuple(values.shape) != expected_shape:
        raise ValueError(
            "SAC transition values must flatten [B, T] in batch-major order. "
            f"Expected {expected_shape}, received {tuple(values.shape)}."
        )
    return values.reshape(int(batch_size), int(horizon_steps))


def _sampled_action_turnover(actions: torch.Tensor) -> torch.Tensor:
    previous_allocation = torch.empty_like(actions)
    previous_allocation[:, 0, :] = 0.0
    previous_allocation[:, 0, -1] = 1.0
    if int(actions.shape[1]) > 1:
        previous_allocation[:, 1:, :] = actions[:, :-1, :]
    return 0.5 * torch.abs(actions - previous_allocation).sum(dim=-1)


def _resolve_sac_context_window_steps(
    train_config: TrainConfig,
    *,
    full_time_steps: int,
) -> int:
    configured_steps = train_config.rl_training.sac.context_window_steps
    if configured_steps is None:
        return int(full_time_steps)
    context_steps = int(configured_steps)
    if context_steps > int(full_time_steps):
        raise ValueError(
            "TrainConfig.rl_training.sac_context_window_steps cannot exceed the "
            "training batch time dimension. "
            f"Received context_window_steps={context_steps} full_time_steps={int(full_time_steps)}."
        )
    return context_steps
