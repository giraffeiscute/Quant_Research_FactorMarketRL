"""Distributed coordination helpers for Lightning orchestration."""

from __future__ import annotations

import pytorch_lightning as pl
import torch


def state_transition_barrier(*, trainer: pl.Trainer, barrier_name: str) -> None:
    resolved_world_size = int(getattr(trainer, "world_size", 1) or 1)
    if resolved_world_size <= 1:
        return

    strategy = getattr(trainer, "strategy", None)
    strategy_barrier = getattr(strategy, "barrier", None)
    if callable(strategy_barrier):
        try:
            strategy_barrier(barrier_name)
        except TypeError:
            strategy_barrier()
        return

    if not torch.distributed.is_available():
        return
    if not torch.distributed.is_initialized():
        return
    torch.distributed.barrier()


def sync_bool_flag_across_ranks(*, trainer: pl.Trainer, flag: bool) -> bool:
    resolved_world_size = int(getattr(trainer, "world_size", 1) or 1)
    if resolved_world_size <= 1:
        return bool(flag)

    if not torch.distributed.is_available():
        return bool(flag)
    if not torch.distributed.is_initialized():
        return bool(flag)

    tensor_device = torch.device("cpu")
    if torch.cuda.is_available():
        tensor_device = torch.device("cuda", torch.cuda.current_device())
    sync_tensor = torch.tensor(1 if bool(flag) else 0, device=tensor_device, dtype=torch.int32)
    torch.distributed.all_reduce(sync_tensor, op=torch.distributed.ReduceOp.MAX)
    return bool(int(sync_tensor.item()))


def sync_bool_flag_across_initialized_ranks(flag: bool) -> bool:
    if not torch.distributed.is_available():
        return bool(flag)
    if not torch.distributed.is_initialized():
        return bool(flag)
    tensor_device = torch.device("cpu")
    if torch.cuda.is_available():
        tensor_device = torch.device("cuda", torch.cuda.current_device())
    sync_tensor = torch.tensor(1 if bool(flag) else 0, device=tensor_device, dtype=torch.int32)
    torch.distributed.all_reduce(sync_tensor, op=torch.distributed.ReduceOp.MAX)
    return bool(int(sync_tensor.item()))


_state_transition_barrier = state_transition_barrier
_sync_bool_flag_across_ranks = sync_bool_flag_across_ranks
_sync_bool_flag_across_initialized_ranks = sync_bool_flag_across_initialized_ranks
