"""Independent CSV logging for gradient diagnostics."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


GRADIENT_DIAGNOSTIC_FIELDS = [
    "event",
    "epoch",
    "global_step",
    "batch_idx",
    "grad_norm",
    "grad_abs_max",
    "grad_nonfinite_count",
    "first_nonfinite_param",
    "train_loss",
    "return_mean_min",
    "return_mean_max",
    "return_std_min",
    "return_std_max",
    "allocation_logits_abs_max",
    "raw_allocation_min",
    "raw_allocation_max",
    "dirichlet_alpha_min",
    "dirichlet_alpha_max",
    "dirichlet_alpha_mean",
    "dirichlet_alpha_sum_mean",
]


@dataclass(frozen=True)
class GradientDiagnostics:
    grad_norm: float
    grad_abs_max: float
    grad_nonfinite_count: int
    first_nonfinite_param: str


class GradientDiagnosticsCSVWriter:
    """Append-only CSV writer with a stable diagnostics schema."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def write_row(self, row: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        should_write_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=GRADIENT_DIAGNOSTIC_FIELDS)
            if should_write_header:
                writer.writeheader()
            writer.writerow(
                {
                    field: _format_diagnostic_value(row.get(field, ""))
                    for field in GRADIENT_DIAGNOSTIC_FIELDS
                }
            )


def gradient_diagnostics_path(
    *,
    outputs_dir: Path,
    state: str,
    loss_name: str,
) -> Path:
    return (
        Path(outputs_dir)
        / "lightning_logs"
        / "gradient_diagnostics"
        / f"{state}_{loss_name}.csv"
    )


def compute_gradient_diagnostics(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> GradientDiagnostics:
    total_sq = 0.0
    finite_abs_max = 0.0
    saw_finite = False
    nonfinite_count = 0
    first_nonfinite_param = ""

    for name, parameter in named_parameters:
        grad = parameter.grad
        if grad is None:
            continue
        detached = grad.detach()
        finite_mask = torch.isfinite(detached)
        if not bool(finite_mask.all().item()):
            current_nonfinite = int((~finite_mask).sum().detach().cpu().item())
            nonfinite_count += current_nonfinite
            if not first_nonfinite_param:
                first_nonfinite_param = name
        finite_values = detached[finite_mask]
        if finite_values.numel() == 0:
            continue
        finite_float = finite_values.float()
        total_sq += float(finite_float.pow(2).sum().detach().cpu().item())
        finite_abs_max = max(
            finite_abs_max,
            float(finite_float.abs().max().detach().cpu().item()),
        )
        saw_finite = True

    grad_norm = total_sq**0.5 if saw_finite else 0.0
    grad_abs_max = finite_abs_max if saw_finite else 0.0
    if nonfinite_count > 0:
        grad_norm = float("nan")

    return GradientDiagnostics(
        grad_norm=grad_norm,
        grad_abs_max=grad_abs_max,
        grad_nonfinite_count=nonfinite_count,
        first_nonfinite_param=first_nonfinite_param,
    )


def _format_diagnostic_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    if isinstance(value, float):
        return f"{value:.8g}"
    return value
