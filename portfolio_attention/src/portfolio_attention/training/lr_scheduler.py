"""Learning-rate schedule helpers shared by training runtimes."""

from __future__ import annotations

import math

import torch

from ..config import TrainConfig


def resolve_total_optimizer_steps(*, num_epochs: int, train_batches_per_epoch: int) -> int:
    return max(0, int(num_epochs)) * max(0, int(train_batches_per_epoch))


def resolve_lr_warmup_steps(*, total_steps: int, warmup_fraction: float) -> int:
    resolved_total_steps = max(0, int(total_steps))
    if resolved_total_steps <= 1 or float(warmup_fraction) <= 0.0:
        return 0
    warmup_steps = int(round(resolved_total_steps * float(warmup_fraction)))
    warmup_steps = max(1, warmup_steps)
    return min(warmup_steps, resolved_total_steps - 1)


def build_lr_warmup_decay_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    train_config: TrainConfig,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    """Build a per-step linear warmup + cosine decay scheduler when enabled."""

    if not bool(train_config.enable_lr_warmup_decay):
        return None

    resolved_total_steps = max(1, int(total_steps))
    warmup_steps = resolve_lr_warmup_steps(
        total_steps=resolved_total_steps,
        warmup_fraction=float(train_config.lr_warmup_fraction),
    )
    min_factor = float(train_config.lr_min_factor)

    def lr_lambda(step_index: int) -> float:
        step = min(max(0, int(step_index)), resolved_total_steps - 1)
        if warmup_steps > 0 and step < warmup_steps:
            warmup_progress = float(step + 1) / float(warmup_steps)
            return min_factor + (1.0 - min_factor) * warmup_progress

        decay_steps = max(1, resolved_total_steps - warmup_steps - 1)
        decay_progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return min_factor + (1.0 - min_factor) * cosine_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
