"""Reward utilities for reinforcement-learning style portfolio training."""

from __future__ import annotations

from typing import Literal

import torch

from ..common.win_rate import compute_binary_win_rate_reward


def _coerce_portfolio_returns(portfolio_returns: torch.Tensor) -> torch.Tensor:
    if portfolio_returns.numel() == 0:
        raise ValueError("portfolio_returns must not be empty.")
    if portfolio_returns.ndim == 1:
        return portfolio_returns.unsqueeze(0)
    if portfolio_returns.ndim != 2:
        raise ValueError(
            "portfolio_returns must have shape [num_scenarios_in_batch, time_steps]. "
            f"Received {tuple(portfolio_returns.shape)}."
        )
    return portfolio_returns


def differential_sharpe_loss(
    portfolio_returns: torch.Tensor,
    eta: float = 0.2,
    A0: float = 0.0,
    B0: float = 1e-4,
    eps: float = 1e-8,
    reduction: Literal["mean", "sum", "last"] = "mean",
) -> torch.Tensor:
    """Compute negative Differential Sharpe Ratio loss over each scenario path."""
    portfolio_returns = _coerce_portfolio_returns(portfolio_returns)

    batch_size, time_steps = portfolio_returns.shape
    device = portfolio_returns.device

    A = torch.full((batch_size,), A0, device=device)
    B = torch.full((batch_size,), B0, device=device)
    scores = []

    for t in range(time_steps):
        Rt = portfolio_returns[:, t]
        delta_A = Rt - A
        delta_B = Rt**2 - B

        numerator = B * delta_A - 0.5 * A * delta_B
        denominator = (B - A**2 + eps) ** 1.5
        score_t = numerator / (denominator + eps)
        scores.append(score_t)

        A = A + eta * delta_A
        B = B + eta * delta_B

    all_scores = torch.stack(scores, dim=1)

    if reduction == "last":
        score = all_scores[:, -1]
    elif reduction == "sum":
        score = all_scores.sum(dim=1)
    else:
        score = all_scores.mean(dim=1)

    return -score.mean()


def compute_differential_sharpe_scores(
    portfolio_returns: torch.Tensor,
    *,
    eta: float = 0.2,
    A0: float | torch.Tensor = 0.0,
    B0: float | torch.Tensor = 1e-4,
    dsr_var_eps: float = 1e-8,
    reward_clip: float | None = None,
) -> torch.Tensor:
    """Return per-step Differential Sharpe scores with positive reward sign."""
    scored_returns = _coerce_portfolio_returns(portfolio_returns)
    batch_size, time_steps = scored_returns.shape

    if float(dsr_var_eps) <= 0.0:
        raise ValueError(f"dsr_var_eps must be > 0, received {dsr_var_eps}.")

    def _expand_stat(value: float | torch.Tensor, name: str) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor_value = value.to(device=scored_returns.device, dtype=scored_returns.dtype)
            if tensor_value.ndim == 0:
                return tensor_value.expand(batch_size)
            if tensor_value.ndim == 1 and int(tensor_value.shape[0]) == batch_size:
                return tensor_value
            raise ValueError(
                f"{name} tensor must be scalar or shape [batch_size={batch_size}], "
                f"received {tuple(tensor_value.shape)}."
            )
        return torch.full(
            (batch_size,),
            float(value),
            device=scored_returns.device,
            dtype=scored_returns.dtype,
        )

    A = _expand_stat(A0, "A0")
    B = _expand_stat(B0, "B0")
    eta_value = float(eta)
    scores: list[torch.Tensor] = []

    for time_index in range(time_steps):
        Rt = scored_returns[:, time_index]
        delta_A = Rt - A
        delta_B = Rt.pow(2) - B
        var_prev = torch.clamp(B - A.pow(2), min=float(dsr_var_eps))
        numerator = B * delta_A - 0.5 * A * delta_B
        denominator = var_prev.pow(1.5) + float(dsr_var_eps)
        score_t = numerator / denominator
        if reward_clip is not None:
            clip_value = float(reward_clip)
            if clip_value <= 0.0:
                raise ValueError(f"reward_clip must be > 0 when set, received {reward_clip}.")
            score_t = torch.clamp(score_t, min=-clip_value, max=clip_value)
        scores.append(score_t)
        A = A + eta_value * delta_A
        B = B + eta_value * delta_B

    return torch.stack(scores, dim=1)


def compute_dsr_warmup_stats(
    prediction_returns: torch.Tensor,
    *,
    rolling_horizon_days: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use first H-1 horizon returns to initialize A0/B0, where H=rolling_horizon_days."""
    scored_returns = _coerce_portfolio_returns(prediction_returns)
    horizon_days = int(rolling_horizon_days)
    if horizon_days < 2:
        raise ValueError(
            "rolling_horizon_days must be >= 2 to compute DSR warmup stats, "
            f"received {horizon_days}."
        )
    if int(scored_returns.shape[1]) != horizon_days:
        raise ValueError(
            "prediction_returns must have shape [batch, rolling_horizon_days]. "
            f"Received {tuple(scored_returns.shape)} with rolling_horizon_days={horizon_days}."
        )
    warmup_returns = scored_returns[:, : horizon_days - 1]
    return warmup_returns.mean(dim=1), warmup_returns.pow(2).mean(dim=1)


def compute_dsr_day_reward(
    prediction_returns: torch.Tensor,
    *,
    rolling_horizon_days: int,
    A0: torch.Tensor | None = None,
    B0: torch.Tensor | None = None,
    dsr_var_eps: float = 1e-8,
    reward_clip: float | None = 5.0,
) -> torch.Tensor:
    """Compute DSR reward for final horizon day (index H-1), warmup from 0..H-2."""
    scored_returns = _coerce_portfolio_returns(prediction_returns)
    horizon_days = int(rolling_horizon_days)
    if horizon_days < 2:
        raise ValueError(
            "rolling_horizon_days must be >= 2 to compute a final-day DSR reward, "
            f"received {horizon_days}."
        )
    if int(scored_returns.shape[1]) != horizon_days:
        raise ValueError(
            "prediction_returns must have shape [batch, rolling_horizon_days]. "
            f"Received {tuple(scored_returns.shape)} with rolling_horizon_days={horizon_days}."
        )

    warmup_A0, warmup_B0 = compute_dsr_warmup_stats(
        scored_returns,
        rolling_horizon_days=horizon_days,
    )
    A_prev = warmup_A0 if A0 is None else A0.to(device=scored_returns.device, dtype=scored_returns.dtype)
    B_prev = warmup_B0 if B0 is None else B0.to(device=scored_returns.device, dtype=scored_returns.dtype)
    if A_prev.ndim == 0:
        A_prev = A_prev.expand(scored_returns.shape[0])
    if B_prev.ndim == 0:
        B_prev = B_prev.expand(scored_returns.shape[0])
    if tuple(A_prev.shape) != (scored_returns.shape[0],):
        raise ValueError(
            "A0 must be scalar or shape [batch_size]. "
            f"Received {tuple(A_prev.shape)} for batch_size={scored_returns.shape[0]}."
        )
    if tuple(B_prev.shape) != (scored_returns.shape[0],):
        raise ValueError(
            "B0 must be scalar or shape [batch_size]. "
            f"Received {tuple(B_prev.shape)} for batch_size={scored_returns.shape[0]}."
        )

    reward_return = scored_returns[:, horizon_days - 1]
    var_prev = torch.clamp(B_prev - A_prev.pow(2), min=float(dsr_var_eps))
    numerator = B_prev * (reward_return - A_prev) - 0.5 * A_prev * (reward_return.pow(2) - B_prev)
    reward = numerator / (var_prev.pow(1.5) + float(dsr_var_eps))
    if reward_clip is not None:
        clip_value = float(reward_clip)
        if clip_value <= 0.0:
            raise ValueError(f"reward_clip must be > 0 when set, received {reward_clip}.")
        reward = torch.clamp(reward, min=-clip_value, max=clip_value)
    return reward


def compute_rolling_sharpe_reward(
    portfolio_returns: torch.Tensor,
    *,
    eps: float = 1e-6,
    reward_clip: float | None = 5.0,
) -> torch.Tensor:
    """Compute per-path rolling-window Sharpe reward with positive reward sign."""
    scored_returns = _coerce_portfolio_returns(portfolio_returns)
    if int(scored_returns.shape[1]) < 2:
        raise ValueError(
            "rolling Sharpe reward requires at least two time steps, "
            f"received {int(scored_returns.shape[1])}."
        )
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be > 0, received {eps}.")

    mean_ret = scored_returns.mean(dim=1)
    std_ret = scored_returns.std(dim=1, unbiased=True)
    reward = mean_ret / (std_ret + float(eps))
    if reward_clip is not None:
        clip_value = float(reward_clip)
        if clip_value <= 0.0:
            raise ValueError(f"reward_clip must be > 0 when set, received {reward_clip}.")
        reward = torch.clamp(reward, min=-clip_value, max=clip_value)
    return reward


def compute_return_reward(
    portfolio_returns: torch.Tensor,
    *,
    reward_scale: float = 1.0,
    reward_clip: float | None = 5.0,
) -> torch.Tensor:
    """Compute final-day simple return reward with positive reward sign."""
    scored_returns = _coerce_portfolio_returns(portfolio_returns)
    scale = float(reward_scale)
    if scale <= 0.0:
        raise ValueError(f"reward_scale must be > 0, received {reward_scale}.")
    reward = scored_returns[:, -1] / scale
    if reward_clip is not None:
        clip_value = float(reward_clip)
        if clip_value <= 0.0:
            raise ValueError(f"reward_clip must be > 0 when set, received {reward_clip}.")
        reward = torch.clamp(reward, min=-clip_value, max=clip_value)
    return reward


def compute_turnover_reward_regularizer(
    action: torch.Tensor,
    previous_allocation: torch.Tensor,
    *,
    norm: str = "l1",
) -> torch.Tensor:
    """Compute per-action turnover regularizer for RL reward shaping."""
    if action.numel() == 0:
        raise ValueError("action must not be empty.")
    if previous_allocation.numel() == 0:
        raise ValueError("previous_allocation must not be empty.")
    if action.ndim < 1 or previous_allocation.ndim < 1:
        raise ValueError(
            "action and previous_allocation must include an asset dimension. "
            f"Received action={tuple(action.shape)} previous_allocation={tuple(previous_allocation.shape)}."
        )
    if int(action.shape[-1]) != int(previous_allocation.shape[-1]):
        raise ValueError(
            "action and previous_allocation must share the asset dimension. "
            f"Received action={tuple(action.shape)} previous_allocation={tuple(previous_allocation.shape)}."
        )
    action_tensor, previous_tensor = torch.broadcast_tensors(action, previous_allocation)
    allocation_delta = action_tensor - previous_tensor
    resolved_norm = str(norm).strip().lower()
    if resolved_norm == "l1":
        return 0.5 * allocation_delta.abs().sum(dim=-1)
    if resolved_norm == "l2":
        return allocation_delta.pow(2).sum(dim=-1)
    raise ValueError(
        "turnover_penalty_norm must be one of {'l1', 'l2'}, "
        f"received {norm!r}."
    )


def apply_turnover_reward_penalty(
    base_reward: torch.Tensor,
    action: torch.Tensor,
    previous_allocation: torch.Tensor,
    *,
    turnover_penalty: float = 0.0,
    turnover_penalty_norm: str = "l1",
    reward_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subtract turnover reward penalty after the base reward has been clipped."""
    scale = float(reward_scale)
    if scale <= 0.0:
        raise ValueError(f"reward_scale must be > 0, received {reward_scale}.")
    regularizer = compute_turnover_reward_regularizer(
        action,
        previous_allocation,
        norm=turnover_penalty_norm,
    )
    if tuple(regularizer.shape) != tuple(base_reward.shape):
        raise ValueError(
            "turnover reward regularizer must match base_reward shape. "
            f"Received regularizer={tuple(regularizer.shape)} base_reward={tuple(base_reward.shape)}."
        )
    penalty = (float(turnover_penalty) / scale) * regularizer
    return base_reward - penalty, penalty


def compute_win_rate_reward(
    portfolio_returns: torch.Tensor,
    baseline_returns: torch.Tensor,
) -> torch.Tensor:
    """Return +1 when portfolio return strictly beats baseline return, otherwise -1."""
    return compute_binary_win_rate_reward(portfolio_returns, baseline_returns)
