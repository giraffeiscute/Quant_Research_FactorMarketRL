"""PPO rollout batch and update helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..config import ModelConfig, TrainConfig
from . import algorithms as rl_algorithms
from .distributions import RolloutPPOPolicyScoring, compute_dirichlet_log_probs_from_logits
from .rollout import sample_rollout_path
from .rollout_targets import (
    compute_discounted_reward_targets,
    compute_rollout_advantages_from_targets,
)


@dataclass(frozen=True)
class RolloutPPOBatch:
    """Detached rollout tensors plus fixed PPO targets for one update batch."""

    sampled_actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    targets: torch.Tensor
    advantages: torch.Tensor
    sampled_turnover: torch.Tensor
    sampled_gross_returns: torch.Tensor
    sampled_net_returns: torch.Tensor
    sampled_base_rewards: torch.Tensor
    sampled_rewards: torch.Tensor
    sampled_reward_penalty: torch.Tensor
    old_entropy: torch.Tensor


@dataclass(frozen=True)
class RolloutPPOUpdateResult:
    """Loss components from scoring a fixed rollout against a fresh policy."""

    policy_loss: torch.Tensor
    ppo_policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy_loss: torch.Tensor
    ppo_ratio: torch.Tensor
    ppo_clip_fraction: torch.Tensor
    approx_kl: torch.Tensor
    policy_scoring: RolloutPPOPolicyScoring
    scored_value_prediction: torch.Tensor


@dataclass(frozen=True)
class RolloutPPOTrainingBatch:
    """Collected rollout batch plus diagnostics shared by PPO update loops."""

    ppo_batch: RolloutPPOBatch
    summary: dict[str, torch.Tensor]
    batch_size: int


def collect_rollout_ppo_batch(
    *,
    scored_logits: torch.Tensor,
    scored_r_stock: torch.Tensor,
    initial_previous_allocation: torch.Tensor,
    value_prediction: torch.Tensor,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> RolloutPPOBatch:
    ppo_config = train_config.rl_training.ppo
    rollout = sample_rollout_path(
        scored_logits=scored_logits,
        scored_r_stock=scored_r_stock,
        initial_previous_allocation=initial_previous_allocation,
        value_prediction=value_prediction,
        model_config=model_config,
        train_config=train_config,
    )
    targets = compute_discounted_reward_targets(
        rollout.sampled_rewards,
        gamma=float(ppo_config.gamma),
    )
    advantages = compute_rollout_advantages_from_targets(
        targets,
        rollout.old_values,
        normalize=bool(ppo_config.normalize_advantages),
    )
    return RolloutPPOBatch(
        sampled_actions=rollout.sampled_actions.detach(),
        old_log_probs=rollout.sampled_log_probs.detach(),
        old_values=rollout.old_values.detach(),
        targets=targets.detach(),
        advantages=advantages.detach(),
        sampled_turnover=rollout.sampled_turnover.detach(),
        sampled_gross_returns=rollout.sampled_gross_returns.detach(),
        sampled_net_returns=rollout.sampled_net_returns.detach(),
        sampled_base_rewards=rollout.sampled_base_rewards.detach(),
        sampled_rewards=rollout.sampled_rewards.detach(),
        sampled_reward_penalty=rollout.sampled_reward_penalty.detach(),
        old_entropy=rollout.entropy.detach(),
    )


def compute_rollout_ppo_update_loss(
    *,
    scored_logits: torch.Tensor,
    scored_value_prediction: torch.Tensor,
    ppo_batch: RolloutPPOBatch,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> RolloutPPOUpdateResult:
    ppo_config = train_config.rl_training.ppo
    policy_scoring = compute_dirichlet_log_probs_from_logits(
        scored_logits,
        ppo_batch.sampled_actions,
        model_config=model_config,
        train_config=train_config,
    )
    entropy_per_dim = policy_scoring.entropy / float(scored_logits.shape[-1])
    entropy_loss = -float(ppo_config.entropy_coef) * entropy_per_dim.mean()

    # With a fresh model forward, the ratio can drift away from 1.0 while the
    # old log-probabilities, actions, and advantages remain fixed.
    ppo_policy_loss, ppo_ratio, ppo_clip_fraction = rl_algorithms.compute_ppo_clipped_policy_loss(
        policy_scoring.new_log_probs,
        ppo_batch.old_log_probs,
        ppo_batch.advantages,
        clip_range=float(ppo_config.clip_range),
    )
    log_ratio = policy_scoring.new_log_probs - ppo_batch.old_log_probs
    approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean().detach()
    value_loss = rl_algorithms.compute_value_loss(
        scored_value_prediction,
        ppo_batch.targets.detach(),
    )
    policy_loss = (
        ppo_policy_loss
        + float(ppo_config.value_loss_coef) * value_loss
        + entropy_loss
    )
    return RolloutPPOUpdateResult(
        policy_loss=policy_loss,
        ppo_policy_loss=ppo_policy_loss,
        value_loss=value_loss,
        entropy_loss=entropy_loss,
        ppo_ratio=ppo_ratio,
        ppo_clip_fraction=ppo_clip_fraction,
        approx_kl=approx_kl,
        policy_scoring=policy_scoring,
        scored_value_prediction=scored_value_prediction,
    )


def build_rollout_ppo_update_metrics(
    ppo_batch: RolloutPPOBatch,
    ppo_update: RolloutPPOUpdateResult,
    *,
    ppo_epoch: int = 1,
) -> dict[str, torch.Tensor | float]:
    policy_scoring = ppo_update.policy_scoring
    entropy_per_dim = policy_scoring.entropy / float(policy_scoring.alpha.shape[-1])
    rollout_final_returns = torch.prod(1.0 + ppo_batch.sampled_net_returns, dim=1) - 1.0
    return {
        "train_total_loss": ppo_update.policy_loss.detach(),
        "train_policy_loss": ppo_update.ppo_policy_loss,
        "train_entropy_loss": ppo_update.entropy_loss.detach(),
        "train_entropy_per_dim": entropy_per_dim.detach().mean(),
        "train_alpha_min": policy_scoring.alpha.detach().min(),
        "train_alpha_max": policy_scoring.alpha.detach().max(),
        "train_alpha_mean": policy_scoring.alpha.detach().mean(),
        "train_reward_base": ppo_batch.sampled_base_rewards.detach().mean(),
        "train_reward_final": ppo_batch.sampled_rewards.detach().mean(),
        "train_reward_TO_penalty": ppo_batch.sampled_reward_penalty.detach().mean(),
        "train_advantage_mean": ppo_batch.advantages.detach().mean(),
        "train_advantage_std": ppo_batch.advantages.detach().std(unbiased=False),
        "train_log_prob_mean": policy_scoring.new_log_probs.detach().mean(),
        "train_log_prob_std": policy_scoring.new_log_probs.detach().std(unbiased=False),
        "train_TO": ppo_batch.sampled_turnover.detach().mean(),
        "train_return": ppo_batch.sampled_net_returns.detach().mean(),
        "train_rollout_value_loss": ppo_update.value_loss.detach(),
        "train_rollout_target_mean": ppo_batch.targets.detach().mean(),
        "train_rollout_target_std": ppo_batch.targets.detach().std(unbiased=False),
        "train_rollout_advantage_mean": ppo_batch.advantages.detach().mean(),
        "train_rollout_advantage_std": ppo_batch.advantages.detach().std(unbiased=False),
        "train_rollout_ppo_epoch": float(ppo_epoch),
        "train_rollout_ppo_ratio_mean": ppo_update.ppo_ratio.detach().mean(),
        "train_rollout_ppo_clip_fraction": ppo_update.ppo_clip_fraction.detach(),
        "train_rollout_ppo_approx_kl": ppo_update.approx_kl.detach(),
        "train_rollout_entropy_per_dim": entropy_per_dim.detach().mean(),
        "train_rollout_total_loss": ppo_update.policy_loss.detach(),
        "train_rollout_reward_base": ppo_batch.sampled_base_rewards.detach().mean(),
        "train_rollout_reward_final": ppo_batch.sampled_rewards.detach().mean(),
        "train_rollout_reward_TO_penalty": ppo_batch.sampled_reward_penalty.detach().mean(),
        "train_rollout_return": ppo_batch.sampled_net_returns.detach().mean(),
        "train_rollout_TO": ppo_batch.sampled_turnover.detach().mean(),
        "train_rollout_final_returns": rollout_final_returns.detach().mean(),
    }
