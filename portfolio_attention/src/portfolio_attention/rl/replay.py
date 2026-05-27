"""Replay buffer and transition collection for SAC portfolio training."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch

from ..common.net_return import apply_transaction_cost_to_returns
from ..common.utils import apply_score_mask


@dataclass(frozen=True)
class SACTransitionBatch:
    """Single-step SAC transitions backed by causal raw observation windows.

    Reward alignment assumption: ``r_stock[:, t]`` is the realized next-period
    stock return earned by the action sampled from ``x_stock[:, t]``. Terminal
    transitions keep ``next_*`` tensors as placeholders and must be guarded by
    ``done`` during Bellman target computation.

    ``x_stock`` and ``x_market`` retain a causal context window ending at the
    decision timestep. ``action``, ``reward``, ``done``, and allocation tensors
    keep a single-step time dimension of 1.
    """

    x_stock: torch.Tensor
    x_market: torch.Tensor
    stock_indices: torch.Tensor
    previous_allocation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_x_stock: torch.Tensor
    next_x_market: torch.Tensor
    next_stock_indices: torch.Tensor
    next_previous_allocation: torch.Tensor
    done: torch.Tensor
    stock_temporal_current: torch.Tensor | None = None
    stock_temporal_summary: torch.Tensor | None = None
    market_current: torch.Tensor | None = None
    market_summary: torch.Tensor | None = None
    next_stock_temporal_current: torch.Tensor | None = None
    next_stock_temporal_summary: torch.Tensor | None = None
    next_market_current: torch.Tensor | None = None
    next_market_summary: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.action.shape[0])

    def detached_cpu(self) -> "SACTransitionBatch":
        return _map_transition_tensors(
            self,
            lambda tensor: tensor.detach().cpu(),
        )

    def to(
        self,
        device: torch.device | str | None = None,
        *,
        dtype: torch.dtype | None = None,
    ) -> "SACTransitionBatch":
        return _map_transition_tensors(
            self,
            lambda tensor: _move_transition_tensor(tensor, device=device, dtype=dtype),
        )

    def index_select(self, indices: torch.Tensor) -> "SACTransitionBatch":
        return _map_transition_tensors(
            self,
            lambda tensor: tensor.index_select(0, indices.to(device=tensor.device)),
        )


class SACReplayBuffer:
    """In-memory ring buffer for detached SAC transitions."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, received {self.capacity}.")
        self._storage: list[SACTransitionBatch] = []
        self._position = 0

    def __len__(self) -> int:
        return len(self._storage)

    def push(self, transitions: SACTransitionBatch) -> None:
        detached = transitions.detached_cpu()
        _validate_transition_batch(detached)
        for row_index in range(detached.batch_size):
            row = detached.index_select(torch.tensor([row_index], dtype=torch.long))
            if len(self._storage) < self.capacity:
                self._storage.append(row)
            else:
                self._storage[self._position] = row
            self._position = (self._position + 1) % self.capacity

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
    ) -> SACTransitionBatch:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, received {batch_size}.")
        if not self._storage:
            raise ValueError("Cannot sample from an empty SACReplayBuffer.")
        indices = torch.randint(
            low=0,
            high=len(self._storage),
            size=(batch_size,),
            generator=generator,
        )
        sampled = _concat_transition_batches([self._storage[int(index)] for index in indices])
        if device is not None or dtype is not None:
            sampled = sampled.to(device=device, dtype=dtype)
        return sampled


def collect_sac_transitions_from_rollout(
    batch: dict[str, Any],
    actions: torch.Tensor,
    *,
    transaction_cost_rate: float = 0.0,
    reward_scale: float = 1.0,
    context_window_steps: int | None = None,
    stock_temporal_current: torch.Tensor | None = None,
    stock_temporal_summary: torch.Tensor | None = None,
    market_current: torch.Tensor | None = None,
    market_summary: torch.Tensor | None = None,
) -> SACTransitionBatch:
    """Split a scored rolling-window rollout into per-step SAC transitions.

    Assumption: ``scored_r_stock[:, t]`` is the realized next-period stock
    return earned by ``actions[:, t]`` sampled from ``scored_x_stock[:, t]``.
    Raw observations are stored as fixed-length causal windows ending at the
    decision timestep so SAC losses can recompute live temporal features from
    replay without losing rolling context. For terminal transitions,
    ``next_x_*`` is a placeholder copied from the terminal observation window
    and must be ignored via ``done=True``.
    """
    score_mask = _require_score_mask(batch)
    x_stock = _require_tensor(batch, "x_stock")
    x_market = _require_tensor(batch, "x_market")
    scored_x_stock = apply_score_mask(x_stock, score_mask)
    scored_x_market = apply_score_mask(x_market, score_mask)
    scored_r_stock = apply_score_mask(_require_tensor(batch, "r_stock"), score_mask)
    stock_indices = _require_tensor(batch, "stock_indices")
    _validate_collector_inputs(
        scored_x_stock=scored_x_stock,
        scored_x_market=scored_x_market,
        scored_r_stock=scored_r_stock,
        stock_indices=stock_indices,
        actions=actions,
    )
    reward_scale = float(reward_scale)
    if reward_scale <= 0.0:
        raise ValueError(f"reward_scale must be > 0, received {reward_scale}.")

    batch_size, horizon_steps, num_stocks, _ = scored_x_stock.shape
    full_time_steps = int(x_stock.shape[1])
    context_steps = (
        full_time_steps
        if context_window_steps is None
        else int(context_window_steps)
    )
    if context_steps <= 0:
        raise ValueError(f"context_window_steps must be positive, received {context_steps}.")
    action_dim = num_stocks + 1
    previous_allocation = _previous_allocations_from_actions(actions)
    next_previous_allocation = actions.detach()
    stock_weights = actions[..., :-1]
    gross_returns = (stock_weights * scored_r_stock).sum(dim=-1)
    turnover = 0.5 * torch.abs(actions - previous_allocation).sum(dim=-1)
    net_returns = apply_transaction_cost_to_returns(
        gross_returns,
        turnover,
        transaction_cost_rate=float(transaction_cost_rate),
    )
    rewards = net_returns / reward_scale
    done = torch.zeros(batch_size, horizon_steps, dtype=torch.bool, device=actions.device)
    done[:, -1] = True

    flat_indices = _flatten_transition_indices(batch_size, horizon_steps, device=actions.device)
    score_positions = _score_positions(score_mask).to(device=actions.device)
    next_time_indices = torch.clamp(flat_indices["time"] + 1, max=horizon_steps - 1)
    flat_stock_indices = stock_indices.index_select(0, flat_indices["batch"])
    x_stock_windows = _gather_causal_context_windows(
        x_stock,
        score_positions=score_positions,
        flat_batch_indices=flat_indices["batch"],
        flat_time_indices=flat_indices["time"],
        context_window_steps=context_steps,
    )
    x_market_windows = _gather_causal_context_windows(
        x_market,
        score_positions=score_positions,
        flat_batch_indices=flat_indices["batch"],
        flat_time_indices=flat_indices["time"],
        context_window_steps=context_steps,
    )
    next_x_stock_windows = _gather_causal_context_windows(
        x_stock,
        score_positions=score_positions,
        flat_batch_indices=flat_indices["batch"],
        flat_time_indices=next_time_indices,
        context_window_steps=context_steps,
    )
    next_x_market_windows = _gather_causal_context_windows(
        x_market,
        score_positions=score_positions,
        flat_batch_indices=flat_indices["batch"],
        flat_time_indices=next_time_indices,
        context_window_steps=context_steps,
    )

    feature_kwargs = _collect_feature_transition_kwargs(
        score_mask=score_mask,
        flat_batch_indices=flat_indices["batch"],
        flat_time_indices=flat_indices["time"],
        next_time_indices=next_time_indices,
        stock_temporal_current=stock_temporal_current,
        stock_temporal_summary=stock_temporal_summary,
        market_current=market_current,
        market_summary=market_summary,
    )
    transitions = SACTransitionBatch(
        x_stock=x_stock_windows,
        x_market=x_market_windows,
        stock_indices=flat_stock_indices,
        previous_allocation=previous_allocation[
            flat_indices["batch"], flat_indices["time"]
        ].unsqueeze(1),
        action=actions[flat_indices["batch"], flat_indices["time"]].unsqueeze(1),
        reward=rewards[flat_indices["batch"], flat_indices["time"]].unsqueeze(1),
        next_x_stock=next_x_stock_windows,
        next_x_market=next_x_market_windows,
        next_stock_indices=flat_stock_indices,
        next_previous_allocation=next_previous_allocation[
            flat_indices["batch"], flat_indices["time"]
        ].unsqueeze(1),
        done=done[flat_indices["batch"], flat_indices["time"]].unsqueeze(1),
        **feature_kwargs,
    )
    _validate_transition_batch(transitions)
    return transitions


def _previous_allocations_from_actions(actions: torch.Tensor) -> torch.Tensor:
    if actions.ndim != 3:
        raise ValueError(
            "actions must have shape [B, T, N+1] including cash. "
            f"Received {tuple(actions.shape)}."
        )
    previous = torch.empty_like(actions)
    previous[:, 0, :] = 0.0
    previous[:, 0, -1] = 1.0
    if int(actions.shape[1]) > 1:
        previous[:, 1:, :] = actions[:, :-1, :]
    return previous


def _flatten_transition_indices(
    batch_size: int,
    horizon_steps: int,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch_indices = torch.arange(batch_size, device=device).repeat_interleave(horizon_steps)
    time_indices = torch.arange(horizon_steps, device=device).repeat(batch_size)
    return {"batch": batch_indices, "time": time_indices}


def _score_positions(score_mask: torch.Tensor) -> torch.Tensor:
    if score_mask.ndim != 2:
        raise ValueError("score_mask must have shape [B, T].")
    positions_by_batch = [
        torch.nonzero(row.to(dtype=torch.bool), as_tuple=False).flatten()
        for row in score_mask
    ]
    if not positions_by_batch:
        raise ValueError("score_mask must contain at least one scenario.")
    scored_steps = int(positions_by_batch[0].numel())
    if scored_steps <= 0:
        raise ValueError("score_mask must select at least one time step.")
    if any(int(positions.numel()) != scored_steps for positions in positions_by_batch):
        raise ValueError("All scenarios must share the same number of scored time steps.")
    return torch.stack(positions_by_batch, dim=0)


def _gather_causal_context_windows(
    values: torch.Tensor,
    *,
    score_positions: torch.Tensor,
    flat_batch_indices: torch.Tensor,
    flat_time_indices: torch.Tensor,
    context_window_steps: int,
) -> torch.Tensor:
    decision_positions = score_positions[flat_batch_indices, flat_time_indices]
    offsets = torch.arange(
        int(context_window_steps),
        device=values.device,
        dtype=decision_positions.dtype,
    ) - (int(context_window_steps) - 1)
    window_indices = decision_positions.unsqueeze(1) + offsets.unsqueeze(0)
    window_indices = torch.clamp(window_indices, min=0)
    window_indices = torch.minimum(window_indices, decision_positions.unsqueeze(1))
    batch_indices = flat_batch_indices.unsqueeze(1).expand_as(window_indices)
    return values[batch_indices, window_indices]


def _collect_feature_transition_kwargs(
    *,
    score_mask: torch.Tensor,
    flat_batch_indices: torch.Tensor,
    flat_time_indices: torch.Tensor,
    next_time_indices: torch.Tensor,
    stock_temporal_current: torch.Tensor | None,
    stock_temporal_summary: torch.Tensor | None,
    market_current: torch.Tensor | None,
    market_summary: torch.Tensor | None,
) -> dict[str, torch.Tensor | None]:
    feature_inputs = {
        "stock_temporal_current": stock_temporal_current,
        "stock_temporal_summary": stock_temporal_summary,
        "market_current": market_current,
        "market_summary": market_summary,
    }
    if all(value is None for value in feature_inputs.values()):
        return {}
    if any(value is None for value in feature_inputs.values()):
        raise ValueError("SAC feature transition collection requires all feature tensors or none.")
    scored_features = {
        name: apply_score_mask(value, score_mask)  # type: ignore[arg-type]
        for name, value in feature_inputs.items()
    }
    return {
        "stock_temporal_current": scored_features["stock_temporal_current"][
            flat_batch_indices, flat_time_indices
        ].unsqueeze(1),
        "stock_temporal_summary": scored_features["stock_temporal_summary"][
            flat_batch_indices, flat_time_indices
        ].unsqueeze(1),
        "market_current": scored_features["market_current"][
            flat_batch_indices, flat_time_indices
        ].unsqueeze(1),
        "market_summary": scored_features["market_summary"][
            flat_batch_indices, flat_time_indices
        ].unsqueeze(1),
        "next_stock_temporal_current": scored_features["stock_temporal_current"][
            flat_batch_indices, next_time_indices
        ].unsqueeze(1),
        "next_stock_temporal_summary": scored_features["stock_temporal_summary"][
            flat_batch_indices, next_time_indices
        ].unsqueeze(1),
        "next_market_current": scored_features["market_current"][
            flat_batch_indices, next_time_indices
        ].unsqueeze(1),
        "next_market_summary": scored_features["market_summary"][
            flat_batch_indices, next_time_indices
        ].unsqueeze(1),
    }


def _map_transition_tensors(
    transitions: SACTransitionBatch,
    fn: Any,
) -> SACTransitionBatch:
    values = {}
    for field_info in fields(SACTransitionBatch):
        value = getattr(transitions, field_info.name)
        values[field_info.name] = None if value is None else fn(value)
    return SACTransitionBatch(**values)


def _move_transition_tensor(
    tensor: torch.Tensor,
    *,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    if dtype is not None and torch.is_floating_point(tensor):
        return tensor.to(device=device, dtype=dtype)
    return tensor.to(device=device)


def _concat_transition_batches(batches: list[SACTransitionBatch]) -> SACTransitionBatch:
    values = {}
    for field_info in fields(SACTransitionBatch):
        field_values = [getattr(batch, field_info.name) for batch in batches]
        if all(value is None for value in field_values):
            values[field_info.name] = None
            continue
        if any(value is None for value in field_values):
            raise ValueError(f"Cannot concatenate mixed optional replay field {field_info.name!r}.")
        values[field_info.name] = torch.cat(field_values, dim=0)
    return SACTransitionBatch(**values)


def _validate_collector_inputs(
    *,
    scored_x_stock: torch.Tensor,
    scored_x_market: torch.Tensor,
    scored_r_stock: torch.Tensor,
    stock_indices: torch.Tensor,
    actions: torch.Tensor,
) -> None:
    if scored_x_stock.ndim != 4:
        raise ValueError("scored x_stock must have shape [B, T, N, F_stock].")
    batch_size, horizon_steps, num_stocks, _ = scored_x_stock.shape
    if horizon_steps <= 0:
        raise ValueError("SAC transition collection requires at least one scored step.")
    if tuple(scored_x_market.shape[:2]) != (batch_size, horizon_steps):
        raise ValueError("x_market must share scored [B, T] dimensions with x_stock.")
    if tuple(scored_r_stock.shape) != (batch_size, horizon_steps, num_stocks):
        raise ValueError("r_stock must have scored shape [B, T, N].")
    if tuple(stock_indices.shape) != (batch_size, num_stocks):
        raise ValueError("stock_indices must have shape [B, N].")
    if tuple(actions.shape) != (batch_size, horizon_steps, num_stocks + 1):
        raise ValueError(
            "actions must have shape [B, T, N+1] matching scored rollout and cash. "
            f"Received actions={tuple(actions.shape)} scored_x_stock={tuple(scored_x_stock.shape)}."
        )
    _validate_simplex(actions, name="actions")


def _validate_transition_batch(transitions: SACTransitionBatch) -> None:
    batch_size = transitions.batch_size
    if batch_size <= 0:
        raise ValueError("SACTransitionBatch must contain at least one transition.")
    if transitions.x_stock.ndim != 4 or int(transitions.x_stock.shape[1]) <= 0:
        raise ValueError("x_stock must have shape [B, C, N, F_stock] with C >= 1.")
    if transitions.next_x_stock.shape != transitions.x_stock.shape:
        raise ValueError("next_x_stock must match x_stock shape.")
    if transitions.x_market.ndim != 3 or int(transitions.x_market.shape[1]) != int(
        transitions.x_stock.shape[1]
    ):
        raise ValueError(
            "x_market must have shape [B, C, F_market] matching x_stock context length."
        )
    if transitions.next_x_market.shape != transitions.x_market.shape:
        raise ValueError("next_x_market must match x_market shape.")
    if transitions.stock_indices.ndim != 2:
        raise ValueError("stock_indices must have shape [B, N].")
    if transitions.stock_indices.shape != transitions.next_stock_indices.shape:
        raise ValueError("stock_indices and next_stock_indices must share shape.")
    if int(transitions.stock_indices.shape[0]) != batch_size:
        raise ValueError("stock_indices batch dimension must match action batch size.")
    if int(transitions.stock_indices.shape[1]) + 1 != int(transitions.action.shape[-1]):
        raise ValueError("action dimension must equal stock count + cash.")
    if not torch.is_floating_point(transitions.reward):
        raise TypeError("reward must be a floating point tensor.")
    if transitions.done.dtype != torch.bool:
        raise TypeError("done must be bool.")
    for field_name in (
        "x_stock",
        "x_market",
        "previous_allocation",
        "action",
        "reward",
        "next_x_stock",
        "next_x_market",
        "next_previous_allocation",
        "done",
    ):
        value = getattr(transitions, field_name)
        if int(value.shape[0]) != batch_size:
            raise ValueError(f"{field_name} batch dimension must match action batch size.")
    if tuple(transitions.previous_allocation.shape) != tuple(transitions.action.shape):
        raise ValueError("previous_allocation and action must share shape.")
    if tuple(transitions.next_previous_allocation.shape) != tuple(transitions.action.shape):
        raise ValueError("next_previous_allocation and action must share shape.")
    if tuple(transitions.reward.shape) != tuple(transitions.done.shape):
        raise ValueError("reward and done must share shape [B, 1].")
    if int(transitions.action.shape[1]) != 1:
        raise ValueError("SAC transitions must retain a single-step time dimension of 1.")
    if tuple(transitions.reward.shape) != (batch_size, 1):
        raise ValueError("reward must have shape [B, 1].")
    if tuple(transitions.done.shape) != (batch_size, 1):
        raise ValueError("done must have shape [B, 1].")
    _validate_simplex(transitions.previous_allocation, name="previous_allocation")
    _validate_simplex(transitions.action, name="action")
    _validate_simplex(transitions.next_previous_allocation, name="next_previous_allocation")
    if not torch.allclose(
        transitions.next_previous_allocation,
        transitions.action,
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError(
            "next_previous_allocation must equal action for portfolio SAC transitions."
        )
    _validate_finite(transitions.reward, name="reward")
    _validate_optional_feature_shapes(transitions)


def _validate_optional_feature_shapes(transitions: SACTransitionBatch) -> None:
    feature_pairs = (
        ("stock_temporal_current", "next_stock_temporal_current", 4, "stock feature"),
        ("stock_temporal_summary", "next_stock_temporal_summary", 4, "stock feature"),
        ("market_current", "next_market_current", 3, "market feature"),
        ("market_summary", "next_market_summary", 3, "market feature"),
    )
    batch_size = transitions.batch_size
    expected_stock_prefix = (
        batch_size,
        1,
        int(transitions.stock_indices.shape[1]),
    )
    expected_market_prefix = (batch_size, 1)
    for current_name, next_name, ndim, label in feature_pairs:
        current = getattr(transitions, current_name)
        next_value = getattr(transitions, next_name)
        if current is None and next_value is None:
            continue
        if current is None or next_value is None:
            raise ValueError(
                f"{current_name} and {next_name} must either both be set or both be None."
            )
        if current.ndim != ndim:
            raise ValueError(f"{current_name} must have a valid {label} transition shape.")
        if next_value.shape != current.shape:
            raise ValueError(f"{next_name} must match {current_name} shape.")
        expected_prefix = expected_stock_prefix if ndim == 4 else expected_market_prefix
        if tuple(current.shape[: len(expected_prefix)]) != expected_prefix:
            raise ValueError(
                f"{current_name} leading dimensions must match replay transition shape."
            )
        _validate_finite(current, name=current_name)
        _validate_finite(next_value, name=next_name)


def _validate_simplex(tensor: torch.Tensor, *, name: str) -> None:
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must be a floating point tensor.")
    _validate_finite(tensor, name=name)
    if (tensor < 0).any():
        raise ValueError(f"{name} must be non-negative.")
    sums = tensor.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-4, rtol=1e-4):
        raise ValueError(f"{name} must sum to 1 over the last dimension.")


def _validate_finite(tensor: torch.Tensor, *, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite.")


def _require_tensor(batch: dict[str, Any], key: str) -> torch.Tensor:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"SAC transition collection requires {key!r} tensor in batch.")
    return value


def _require_score_mask(batch: dict[str, Any]) -> torch.Tensor:
    value = batch.get("score_mask")
    if not isinstance(value, torch.Tensor):
        raise RuntimeError("SAC transition collection requires 'score_mask' tensor in batch.")
    return value.to(dtype=torch.bool)
