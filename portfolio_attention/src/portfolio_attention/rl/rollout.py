"""Sampled rollout path simulation for future rolling PPO."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Dirichlet

from ..common.net_return import apply_transaction_cost_to_returns
from ..config import ModelConfig, TrainConfig
from ..model.allocation_distribution import logits_to_rl_post_train_dirichlet_alpha
from ..model.reward import apply_turnover_reward_penalty


@dataclass(frozen=True)
class RolloutPathResult:
    """Detached rollout-time tensors collected for later PPO-style updates.

    ``sampled_log_probs`` are old log-probabilities from collection time.
    ``old_values`` are old value predictions from collection time, not fresh
    value predictions for a value-loss update.
    """

    sampled_actions: torch.Tensor
    sampled_log_probs: torch.Tensor
    sampled_turnover: torch.Tensor
    sampled_gross_returns: torch.Tensor
    sampled_net_returns: torch.Tensor
    sampled_base_rewards: torch.Tensor
    sampled_rewards: torch.Tensor
    sampled_reward_penalty: torch.Tensor
    old_values: torch.Tensor
    entropy: torch.Tensor


@torch.no_grad()
def sample_rollout_path(
    scored_logits: torch.Tensor,
    scored_r_stock: torch.Tensor,
    initial_previous_allocation: torch.Tensor,
    value_prediction: torch.Tensor,
    *,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> RolloutPathResult:
    """Collect one sampled rolling allocation path per scenario/window.

    This is simulator-only rollout collection for future rolling PPO. It does
    not compute PPO losses, discounted returns, advantages, GAE, or a rollout
    buffer. Rewards are currently per-step portfolio returns scaled by
    ``rl_training.reward_scale`` and clipped by ``rl_training.ppo.reward_clip``.
    """
    _validate_rollout_inputs(
        scored_logits,
        scored_r_stock,
        initial_previous_allocation,
        value_prediction,
    )
    reward_scale, reward_clip = _validate_rollout_reward_config(train_config)

    time_steps = int(scored_logits.shape[1])
    previous_action = initial_previous_allocation

    sampled_actions: list[torch.Tensor] = []
    sampled_log_probs: list[torch.Tensor] = []
    sampled_turnover: list[torch.Tensor] = []
    sampled_gross_returns: list[torch.Tensor] = []
    sampled_net_returns: list[torch.Tensor] = []
    sampled_base_rewards: list[torch.Tensor] = []
    sampled_rewards: list[torch.Tensor] = []
    sampled_reward_penalty: list[torch.Tensor] = []
    entropy: list[torch.Tensor] = []

    for timestep in range(time_steps):
        alpha_t = logits_to_rl_post_train_dirichlet_alpha(
            scored_logits[:, timestep, :],
            alpha_min=float(train_config.rl_training.alpha_min),
            alpha_max=float(train_config.rl_training.alpha_max),
            logit_scale=float(model_config.dirichlet_logit_scale),
            evidence_scale=float(train_config.rl_training.rl_post_train_evidence_scale),
        )

        dist_t = Dirichlet(alpha_t)
        action_t = dist_t.sample()
        log_prob_t = dist_t.log_prob(action_t)
        entropy_t = dist_t.entropy()

        turnover_t = 0.5 * torch.abs(action_t - previous_action).sum(dim=-1)
        stock_weights_t = action_t[:, :-1]
        gross_return_t = (stock_weights_t * scored_r_stock[:, timestep, :]).sum(dim=-1)
        net_return_t = apply_transaction_cost_to_returns(
            gross_return_t,
            turnover_t,
            transaction_cost_rate=float(train_config.transaction_cost_rate),
        )
        base_reward_t = net_return_t / reward_scale
        base_reward_t = base_reward_t.clamp(
            min=-reward_clip,
            max=reward_clip,
        )
        reward_t, reward_penalty_t = apply_turnover_reward_penalty(
            base_reward_t,
            action_t,
            previous_action,
            turnover_penalty=float(train_config.turnover_penalty),
            turnover_penalty_norm=str(train_config.turnover_penalty_norm),
            reward_scale=reward_scale,
        )

        sampled_actions.append(action_t)
        sampled_log_probs.append(log_prob_t)
        sampled_turnover.append(turnover_t)
        sampled_gross_returns.append(gross_return_t)
        sampled_net_returns.append(net_return_t)
        sampled_base_rewards.append(base_reward_t)
        sampled_rewards.append(reward_t)
        sampled_reward_penalty.append(reward_penalty_t)
        entropy.append(entropy_t)

        previous_action = action_t.detach()

    return RolloutPathResult(
        sampled_actions=torch.stack(sampled_actions, dim=1),
        sampled_log_probs=torch.stack(sampled_log_probs, dim=1),
        sampled_turnover=torch.stack(sampled_turnover, dim=1),
        sampled_gross_returns=torch.stack(sampled_gross_returns, dim=1),
        sampled_net_returns=torch.stack(sampled_net_returns, dim=1),
        sampled_base_rewards=torch.stack(sampled_base_rewards, dim=1),
        sampled_rewards=torch.stack(sampled_rewards, dim=1),
        sampled_reward_penalty=torch.stack(sampled_reward_penalty, dim=1),
        old_values=value_prediction.detach(),
        entropy=torch.stack(entropy, dim=1),
    )


def _validate_rollout_reward_config(train_config: TrainConfig) -> tuple[float, float]:
    reward_scale = float(train_config.rl_training.reward_scale)
    if reward_scale <= 0.0:
        raise ValueError(f"reward_scale must be > 0, received {reward_scale}.")
    reward_clip = float(train_config.rl_training.ppo.reward_clip)
    if reward_clip <= 0.0:
        raise ValueError(f"reward_clip must be > 0, received {reward_clip}.")
    return reward_scale, reward_clip


def _validate_rollout_inputs(
    scored_logits: torch.Tensor,
    scored_r_stock: torch.Tensor,
    initial_previous_allocation: torch.Tensor,
    value_prediction: torch.Tensor,
) -> None:
    if scored_logits.ndim != 3:
        raise ValueError(f"scored_logits must have shape [B, T, A], received {tuple(scored_logits.shape)}.")
    if scored_r_stock.ndim != 3:
        raise ValueError(f"scored_r_stock must have shape [B, T, N], received {tuple(scored_r_stock.shape)}.")
    if initial_previous_allocation.ndim != 2:
        raise ValueError(
            "initial_previous_allocation must have shape [B, A], "
            f"received {tuple(initial_previous_allocation.shape)}."
        )
    if value_prediction.ndim != 2:
        raise ValueError(
            f"value_prediction must have shape [B, T], received {tuple(value_prediction.shape)}."
        )

    batch_size, time_steps, action_dim = scored_logits.shape
    stock_batch_size, stock_time_steps, num_stocks = scored_r_stock.shape
    if (stock_batch_size, stock_time_steps) != (batch_size, time_steps):
        raise ValueError(
            "scored_logits and scored_r_stock must share [B, T] dimensions. "
            f"Received scored_logits={tuple(scored_logits.shape)} "
            f"scored_r_stock={tuple(scored_r_stock.shape)}."
        )
    if initial_previous_allocation.shape != (batch_size, action_dim):
        raise ValueError(
            "initial_previous_allocation must match scored_logits [B, A] dimensions. "
            f"Received scored_logits={tuple(scored_logits.shape)} "
            f"initial_previous_allocation={tuple(initial_previous_allocation.shape)}."
        )
    if value_prediction.shape != (batch_size, time_steps):
        raise ValueError(
            "value_prediction must match scored_logits [B, T] dimensions. "
            f"Received scored_logits={tuple(scored_logits.shape)} "
            f"value_prediction={tuple(value_prediction.shape)}."
        )
    if num_stocks != action_dim - 1:
        raise ValueError(
            "scored_r_stock stock dimension must equal scored_logits action dimension minus cash. "
            f"Received scored_logits={tuple(scored_logits.shape)} "
            f"scored_r_stock={tuple(scored_r_stock.shape)}."
        )
