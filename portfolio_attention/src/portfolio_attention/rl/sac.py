"""SAC loss helpers for portfolio RL training.

SAC replay stores raw observations; losses recompute current actor/encoder
features from sampled replay batches so actor gradients update the live policy
instead of stale cached features.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.nn.functional as F

from ..config import ModelConfig, TrainConfig
from ..model import PortfolioAttentionModel, TwinPortfolioQCritic
from .replay import SACTransitionBatch
from .sac_policy import build_dirichlet_sac_policy


@dataclass(frozen=True)
class SACQLossResult:
    q_loss: torch.Tensor
    q1_loss: torch.Tensor
    q2_loss: torch.Tensor


@dataclass(frozen=True)
class SACLossBatchResult:
    q_loss: torch.Tensor
    q1_loss: torch.Tensor
    q2_loss: torch.Tensor
    actor_loss: torch.Tensor
    alpha_loss: torch.Tensor | None
    alpha: torch.Tensor
    target_q: torch.Tensor
    current_q1: torch.Tensor
    current_q2: torch.Tensor
    policy_log_prob: torch.Tensor
    policy_action: torch.Tensor
    target_entropy: torch.Tensor | None


@dataclass(frozen=True)
class SACQReplayLossResult:
    q_loss: torch.Tensor
    q1_loss: torch.Tensor
    q2_loss: torch.Tensor
    alpha: torch.Tensor
    target_q: torch.Tensor
    current_q1: torch.Tensor
    current_q2: torch.Tensor


@dataclass(frozen=True)
class SACActorAlphaReplayLossResult:
    actor_loss: torch.Tensor
    alpha_loss: torch.Tensor | None
    alpha: torch.Tensor
    policy_log_prob: torch.Tensor
    policy_action: torch.Tensor
    target_entropy: torch.Tensor | None


def compute_sac_q_loss(
    q1_current: torch.Tensor,
    q2_current: torch.Tensor,
    target_q: torch.Tensor,
) -> SACQLossResult:
    """Compute twin-Q MSE loss against a detached Bellman target."""
    if tuple(q1_current.shape) != tuple(q2_current.shape):
        raise ValueError(
            "q1_current and q2_current must share shape. "
            f"Received q1={tuple(q1_current.shape)} q2={tuple(q2_current.shape)}."
        )
    if tuple(q1_current.shape) != tuple(target_q.shape):
        raise ValueError(
            "target_q must match q current shape. "
            f"Received q={tuple(q1_current.shape)} target_q={tuple(target_q.shape)}."
        )
    detached_target = target_q.detach()
    q1_loss = F.mse_loss(q1_current, detached_target)
    q2_loss = F.mse_loss(q2_current, detached_target)
    return SACQLossResult(q_loss=q1_loss + q2_loss, q1_loss=q1_loss, q2_loss=q2_loss)


def compute_sac_actor_loss(
    log_prob: torch.Tensor,
    q1_policy: torch.Tensor,
    q2_policy: torch.Tensor,
    alpha: torch.Tensor | float,
) -> torch.Tensor:
    """Compute SAC actor objective: E[alpha * log pi(a|s) - min(Q1, Q2)]."""
    if tuple(log_prob.shape) != tuple(q1_policy.shape) or tuple(log_prob.shape) != tuple(
        q2_policy.shape
    ):
        raise ValueError(
            "log_prob, q1_policy, and q2_policy must share shape. "
            f"Received log_prob={tuple(log_prob.shape)} "
            f"q1={tuple(q1_policy.shape)} q2={tuple(q2_policy.shape)}."
        )
    alpha_tensor = _coerce_alpha(alpha, reference=log_prob)
    return (alpha_tensor.detach() * log_prob - torch.minimum(q1_policy, q2_policy)).mean()


def compute_sac_alpha_loss(
    log_alpha: torch.Tensor,
    log_prob: torch.Tensor,
    target_entropy: float | torch.Tensor,
) -> torch.Tensor:
    """Compute auto-entropy temperature loss; gradients flow only to log_alpha."""
    target_entropy_tensor = torch.as_tensor(
        target_entropy,
        dtype=log_prob.dtype,
        device=log_prob.device,
    )
    return -(log_alpha * (log_prob.detach() + target_entropy_tensor)).mean()


@torch.no_grad()
def soft_update_targets(
    online: torch.nn.Module,
    target: torch.nn.Module,
    *,
    tau: float,
) -> None:
    """Polyak-average target parameters toward online parameters."""
    tau = float(tau)
    if tau <= 0.0 or tau > 1.0:
        raise ValueError(f"tau must be in (0, 1], received {tau}.")
    online_params = list(online.parameters())
    target_params = list(target.parameters())
    if len(online_params) != len(target_params):
        raise ValueError(
            "online and target modules must have the same number of parameters."
        )
    for online_param, target_param in zip(online_params, target_params):
        if tuple(online_param.shape) != tuple(target_param.shape):
            raise ValueError(
                "online and target parameter shapes must match. "
                f"Received online={tuple(online_param.shape)} target={tuple(target_param.shape)}."
            )
        target_param.mul_(1.0 - tau).add_(online_param, alpha=tau)


def build_sac_metrics(result: SACLossBatchResult) -> dict[str, torch.Tensor]:
    """Build detached training metrics for SAC losses."""
    metrics = {
        "train_sac_q_loss": result.q_loss.detach(),
        "train_sac_q1_loss": result.q1_loss.detach(),
        "train_sac_q2_loss": result.q2_loss.detach(),
        "train_sac_actor_loss": result.actor_loss.detach(),
        "train_sac_temp": result.alpha.detach(),
        "train_sac_target_q_mean": result.target_q.detach().mean(),
        "train_sac_current_q1_mean": result.current_q1.detach().mean(),
        "train_sac_current_q2_mean": result.current_q2.detach().mean(),
        "train_sac_log_prob_mean": result.policy_log_prob.detach().mean(),
    }
    if result.alpha_loss is not None:
        metrics["train_sac_temp_loss"] = result.alpha_loss.detach()
    if result.target_entropy is not None:
        metrics["train_sac_target_entropy"] = result.target_entropy.detach()
    return metrics


def compute_sac_losses_from_replay_batch(
    *,
    actor_model: PortfolioAttentionModel,
    q_critic: TwinPortfolioQCritic,
    target_q_critic: TwinPortfolioQCritic,
    replay_batch: SACTransitionBatch,
    model_config: ModelConfig,
    train_config: TrainConfig,
    alpha: torch.Tensor | float | None,
    log_alpha: torch.Tensor | None = None,
    target_entropy: float | None = None,
) -> SACLossBatchResult:
    """Compute all SAC losses in one forward pass for smoke tests.

    PR6 training should prefer ``compute_sac_q_loss_from_replay_batch`` and
    ``compute_sac_actor_alpha_loss_from_replay_batch`` so critic parameters are
    stepped before rebuilding the actor loss graph.
    """
    q_result = compute_sac_q_loss_from_replay_batch(
        actor_model=actor_model,
        q_critic=q_critic,
        target_q_critic=target_q_critic,
        replay_batch=replay_batch,
        model_config=model_config,
        train_config=train_config,
        alpha=alpha,
        log_alpha=log_alpha,
    )
    actor_result = compute_sac_actor_alpha_loss_from_replay_batch(
        actor_model=actor_model,
        q_critic=q_critic,
        replay_batch=replay_batch,
        model_config=model_config,
        train_config=train_config,
        alpha=alpha,
        log_alpha=log_alpha,
        target_entropy=target_entropy,
    )
    return SACLossBatchResult(
        q_loss=q_result.q_loss,
        q1_loss=q_result.q1_loss,
        q2_loss=q_result.q2_loss,
        actor_loss=actor_result.actor_loss,
        alpha_loss=actor_result.alpha_loss,
        alpha=actor_result.alpha,
        target_q=q_result.target_q,
        current_q1=q_result.current_q1,
        current_q2=q_result.current_q2,
        policy_log_prob=actor_result.policy_log_prob,
        policy_action=actor_result.policy_action,
        target_entropy=actor_result.target_entropy,
    )


def compute_sac_q_loss_from_replay_batch(
    *,
    actor_model: PortfolioAttentionModel,
    q_critic: TwinPortfolioQCritic,
    target_q_critic: TwinPortfolioQCritic,
    replay_batch: SACTransitionBatch,
    model_config: ModelConfig,
    train_config: TrainConfig,
    alpha: torch.Tensor | float | None,
    log_alpha: torch.Tensor | None = None,
) -> SACQReplayLossResult:
    """Compute critic loss from replay, recomputing live actor context features."""
    actor_outputs = _forward_actor_for_sac(
        actor_model,
        x_stock=replay_batch.x_stock,
        x_market=replay_batch.x_market,
        stock_indices=replay_batch.stock_indices,
        previous_allocation=replay_batch.previous_allocation,
    )
    current_q1, current_q2 = _evaluate_q(
        q_critic,
        _detach_actor_feature_outputs(actor_outputs),
        previous_allocation=replay_batch.previous_allocation,
        action=replay_batch.action,
    )
    with torch.no_grad():
        next_outputs = _forward_actor_for_sac(
            actor_model,
            x_stock=replay_batch.next_x_stock,
            x_market=replay_batch.next_x_market,
            stock_indices=replay_batch.next_stock_indices,
            previous_allocation=replay_batch.next_previous_allocation,
        )
        next_policy = build_dirichlet_sac_policy(
            _require_tensor(next_outputs, "allocation_logits"),
            model_config=model_config,
            train_config=train_config,
        )
        next_action = next_policy.rsample_action()
        next_log_prob = next_policy.log_prob(next_action)
        resolved_alpha = _resolve_alpha(
            alpha,
            log_alpha=log_alpha,
            reference=next_log_prob,
        )
        target_q1, target_q2 = _evaluate_q(
            target_q_critic,
            next_outputs,
            previous_allocation=replay_batch.next_previous_allocation,
            action=next_action,
        )
        target_value = torch.minimum(target_q1, target_q2) - (
            resolved_alpha * next_log_prob
        )
        target_q = replay_batch.reward + float(train_config.rl_training.sac.gamma) * (
            1.0 - replay_batch.done.to(dtype=replay_batch.reward.dtype)
        ) * target_value
    q_loss_result = compute_sac_q_loss(current_q1, current_q2, target_q)
    return SACQReplayLossResult(
        q_loss=q_loss_result.q_loss,
        q1_loss=q_loss_result.q1_loss,
        q2_loss=q_loss_result.q2_loss,
        alpha=resolved_alpha,
        target_q=target_q.detach(),
        current_q1=current_q1,
        current_q2=current_q2,
    )


def compute_sac_actor_alpha_loss_from_replay_batch(
    *,
    actor_model: PortfolioAttentionModel,
    q_critic: TwinPortfolioQCritic,
    replay_batch: SACTransitionBatch,
    model_config: ModelConfig,
    train_config: TrainConfig,
    alpha: torch.Tensor | float | None,
    log_alpha: torch.Tensor | None = None,
    target_entropy: float | None = None,
) -> SACActorAlphaReplayLossResult:
    """Compute actor and optional auto-entropy losses from a fresh actor forward."""
    actor_outputs = _forward_actor_for_sac(
        actor_model,
        x_stock=replay_batch.x_stock,
        x_market=replay_batch.x_market,
        stock_indices=replay_batch.stock_indices,
        previous_allocation=replay_batch.previous_allocation,
    )

    policy = build_dirichlet_sac_policy(
        _require_tensor(actor_outputs, "allocation_logits"),
        model_config=model_config,
        train_config=train_config,
    )
    policy_action = policy.rsample_action()
    policy_log_prob = policy.log_prob(policy_action)
    with _temporarily_freeze_parameters(q_critic):
        q1_policy, q2_policy = _evaluate_q(
            q_critic,
            actor_outputs,
            previous_allocation=replay_batch.previous_allocation,
            action=policy_action,
        )
    resolved_alpha = _resolve_alpha(
        alpha,
        log_alpha=log_alpha,
        reference=policy_log_prob,
    )
    actor_loss = compute_sac_actor_loss(
        policy_log_prob,
        q1_policy,
        q2_policy,
        resolved_alpha,
    )

    alpha_loss = None
    target_entropy_tensor = None
    if log_alpha is not None:
        resolved_target_entropy = (
            float(target_entropy)
            if target_entropy is not None
            else _default_target_entropy(action_dim=int(policy_action.shape[-1]))
        )
        target_entropy_tensor = torch.as_tensor(
            resolved_target_entropy,
            dtype=policy_log_prob.dtype,
            device=policy_log_prob.device,
        )
        alpha_loss = compute_sac_alpha_loss(
            log_alpha,
            policy_log_prob,
            target_entropy_tensor,
        )

    return SACActorAlphaReplayLossResult(
        actor_loss=actor_loss,
        alpha_loss=alpha_loss,
        alpha=resolved_alpha,
        policy_log_prob=policy_log_prob,
        policy_action=policy_action,
        target_entropy=target_entropy_tensor,
    )


def _forward_actor_for_sac(
    actor_model: PortfolioAttentionModel,
    *,
    x_stock: torch.Tensor,
    x_market: torch.Tensor,
    stock_indices: torch.Tensor,
    previous_allocation: torch.Tensor,
) -> dict[str, Any]:
    if int(previous_allocation.shape[1]) != 1:
        raise ValueError(
            "SAC replay batches must use a single-step previous_allocation time dimension."
        )
    return actor_model(
        x_stock,
        x_market,
        stock_indices,
        initial_allocation_override=previous_allocation[:, 0, :],
        return_state_features=True,
        return_last_step_only=True,
    )


def _evaluate_q(
    q_critic: TwinPortfolioQCritic,
    actor_outputs: dict[str, Any],
    *,
    previous_allocation: torch.Tensor,
    action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return q_critic(
        stock_temporal_current=_require_tensor(actor_outputs, "stock_temporal_current"),
        stock_temporal_summary=_require_tensor(actor_outputs, "stock_temporal_summary"),
        market_current=_require_tensor(actor_outputs, "market_current"),
        market_summary=_require_tensor(actor_outputs, "market_summary"),
        previous_allocation=previous_allocation,
        action=action,
    )


def _require_tensor(outputs: dict[str, Any], key: str) -> torch.Tensor:
    value = outputs.get(key)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"SAC loss computation requires actor output {key!r}.")
    return value


def _detach_actor_feature_outputs(outputs: dict[str, Any]) -> dict[str, Any]:
    detached = dict(outputs)
    for key in (
        "stock_temporal_current",
        "stock_temporal_summary",
        "market_current",
        "market_summary",
    ):
        value = detached.get(key)
        if isinstance(value, torch.Tensor):
            detached[key] = value.detach()
    return detached


@contextmanager
def _temporarily_freeze_parameters(module: torch.nn.Module) -> Iterator[None]:
    parameters = list(module.parameters())
    original_requires_grad = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(parameters, original_requires_grad):
            parameter.requires_grad_(requires_grad)


def _coerce_alpha(alpha: torch.Tensor | float, *, reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(alpha, dtype=reference.dtype, device=reference.device)


def _resolve_alpha(
    alpha: torch.Tensor | float | None,
    *,
    log_alpha: torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor:
    if log_alpha is not None:
        return log_alpha.exp().to(dtype=reference.dtype, device=reference.device)
    if alpha is None:
        raise ValueError("alpha must be provided when log_alpha is None.")
    return _coerce_alpha(alpha, reference=reference)


def _default_target_entropy(*, action_dim: int) -> float:
    # Heuristic for Dirichlet-simplex SAC; real experiments should prefer an explicit config.
    return -float(action_dim)
