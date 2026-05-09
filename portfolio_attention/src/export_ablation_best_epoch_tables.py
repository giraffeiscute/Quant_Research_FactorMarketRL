"""Export ablation best-epoch tables from prediction experiment directories."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
import sys
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from export_best_epoch_comparison import (  # noqa: E402
    LOSS_NAMES,
    STATE_NAMES,
    _average_loss_rows,
    _select_best_epoch,
)

DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "outputs"
DETAIL_CSV_NAME = "ablation_best_epoch_by_state.csv"
SUMMARY_CSV_NAME = "ablation_best_epoch_summary.csv"
MARKDOWN_NAME = "ablation_best_epoch_tables.md"
MISSING_TABLE_VALUE = "—"
REGIME_TABLE_FIELDS = [
    "Variant",
    "Feedback",
    "λ",
    "Detach",
    "Cost rate",
    "Bear SR ↑",
    "Neutral SR ↑",
    "Bull SR ↑",
    "Bear return",
    "Neutral return",
    "Bull return",
    "Bear TO ↓",
    "Neutral TO ↓",
    "Bull TO ↓",
    "Bear stocks",
    "Neutral stocks",
    "Bull stocks",
    "Bear epoch",
    "Neutral epoch",
    "Bull epoch",
]


@dataclass(frozen=True)
class ExperimentConfig:
    sample_num_stocks: int | None
    train_batch_size: int | None
    detach_prev_weight: bool | None
    use_prev_weight_feature: bool | None
    turnover_penalty: float | None
    transaction_cost_rate: float | None
    rolling_stride_days: int | None
    learning_rate: float | None

    @property
    def detach_label(self) -> str:
        if self.detach_prev_weight is True:
            return "D"
        if self.detach_prev_weight is False:
            return "noD"
        return "unknownD"

    @property
    def prev_weight_label(self) -> str:
        if self.use_prev_weight_feature is True:
            return "W"
        if self.use_prev_weight_feature is False:
            return "noW"
        return "unknownW"

    @property
    def label(self) -> str:
        sample = f"s{self.sample_num_stocks}" if self.sample_num_stocks is not None else "s?"
        penalty = _format_token_number(self.turnover_penalty)
        cost = _format_token_number(self.transaction_cost_rate)
        stride = self.rolling_stride_days if self.rolling_stride_days is not None else "?"
        return (
            f"{sample}_{self.detach_label}_{self.prev_weight_label}_"
            f"p{penalty}_cost{cost}_stride{stride}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan outputs/predictions_s* ablation directories, select the best "
            "holdout epoch per state by mean SR, and export comparison tables."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Root directory containing predictions_s* experiment directories, "
            "or one predictions_s* experiment directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where CSV and Markdown tables should be written.",
    )
    return parser


def _format_token_number(value: float | None) -> str:
    if value is None:
        return "?"
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):g}"


def _format_cell(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Integral):
        return str(value)
    if isinstance(value, Real):
        return f"{float(value):.{digits}f}"
    return str(value)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, Real) and not isinstance(value, Integral):
        return round(float(value), 6)
    return value


def _load_runtime_config(experiment_dir: Path) -> dict[str, Any] | None:
    runtime_paths = sorted(experiment_dir.glob("*/*runtime_config*.json"))
    if not runtime_paths:
        return None
    return json.loads(runtime_paths[0].read_text(encoding="utf-8"))


def _experiment_config(experiment_dir: Path) -> ExperimentConfig:
    payload = _load_runtime_config(experiment_dir)
    if payload is None:
        return ExperimentConfig(None, None, None, None, None, None, None, None)
    data = payload.get("data_config", {})
    model = payload.get("model_config", {})
    train = payload.get("train_config", {})
    return ExperimentConfig(
        sample_num_stocks=_to_optional_int(data.get("sample_num_stocks")),
        train_batch_size=_to_optional_int(data.get("train_batch_size")),
        detach_prev_weight=_to_optional_bool(model.get("detach_prev_weight")),
        use_prev_weight_feature=_to_optional_bool(model.get("use_prev_weight_feature")),
        turnover_penalty=_to_optional_float(train.get("turnover_penalty")),
        transaction_cost_rate=_to_optional_float(train.get("transaction_cost_rate")),
        rolling_stride_days=_to_optional_int(data.get("rolling_stride_days")),
        learning_rate=_to_optional_float(train.get("learning_rate")),
    )


def _to_optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _to_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _epoch_from_dir(epoch_dir: Path) -> int | None:
    match = re.match(r"(\d+)_holdout_backtest$", epoch_dir.name)
    return int(match.group(1)) if match else None


def _complete_epoch_count(state_dir: Path) -> int:
    if not state_dir.is_dir():
        return 0
    count = 0
    for epoch_dir in state_dir.glob("*_holdout_backtest"):
        if _epoch_from_dir(epoch_dir) is None:
            continue
        if all((epoch_dir / f"{loss}_monitoring_holdout_backtest.json").is_file() for loss in LOSS_NAMES):
            count += 1
    return count


def _is_prediction_experiment_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and path.name.startswith("predictions_s")
        and any((path / state).is_dir() for state in STATE_NAMES)
    )


def _prediction_experiment_dirs(output_root: Path) -> list[Path]:
    if _is_prediction_experiment_dir(output_root):
        return [output_root]
    return [
        path
        for path in output_root.glob("predictions_s*")
        if _is_prediction_experiment_dir(path)
    ]


def _state_metrics(best_payload: dict[str, Any], *, context: str) -> dict[str, Any]:
    if best_payload.get("epoch") is None:
        return {
            "best_epoch": None,
            "mean_return": None,
            "mean_sr": None,
            "mean_stocks": None,
            "mean_turnover": None,
        }
    loss_name = LOSS_NAMES[0]
    avg_return, avg_sr, avg_stocks, avg_turnover = _average_loss_rows(
        best_payload["per_loss_scenarios"][loss_name],
        context=context,
    )
    return {
        "best_epoch": best_payload["epoch"],
        "mean_return": avg_return,
        "mean_sr": avg_sr,
        "mean_stocks": avg_stocks,
        "mean_turnover": avg_turnover,
    }


def _sort_key(item: tuple[Path, ExperimentConfig]) -> tuple[Any, ...]:
    path, config = item
    return (
        config.sample_num_stocks if config.sample_num_stocks is not None else 10**9,
        0 if config.use_prev_weight_feature is True else 1,
        0 if config.detach_prev_weight is True else 1,
        config.turnover_penalty if config.turnover_penalty is not None else 10**9,
        config.transaction_cost_rate if config.transaction_cost_rate is not None else 10**9,
        path.name,
    )


def _collect_rows(output_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    experiments = [
        (path, _experiment_config(path))
        for path in _prediction_experiment_dirs(output_root)
    ]
    experiments.sort(key=_sort_key)

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for experiment_dir, config in experiments:
        covered_metrics: list[dict[str, Any]] = []
        missing_states: list[str] = []
        for state in STATE_NAMES:
            state_dir = experiment_dir / state
            best_payload = _select_best_epoch(state_dir)
            metrics = _state_metrics(best_payload, context=f"{experiment_dir.name}/{state}")
            epoch_count = _complete_epoch_count(state_dir)
            if metrics["best_epoch"] is None:
                missing_states.append(state)
            else:
                covered_metrics.append(metrics)
            detail_rows.append(
                {
                    "experiment": experiment_dir.name,
                    "label": config.label,
                    "state": state,
                    "sample_num_stocks": config.sample_num_stocks,
                    "detach_prev_weight": config.detach_prev_weight,
                    "use_prev_weight_feature": config.use_prev_weight_feature,
                    "turnover_penalty": config.turnover_penalty,
                    "transaction_cost_rate": config.transaction_cost_rate,
                    "rolling_stride_days": config.rolling_stride_days,
                    "complete_epoch_count": epoch_count,
                    **metrics,
                }
            )

        summary_rows.append(_build_regime_table_row(experiment_dir.name, config, detail_rows))

    summary_rows.sort(key=_regime_table_sort_key)
    return detail_rows, summary_rows


def _build_regime_table_row(
    experiment_name: str,
    config: ExperimentConfig,
    detail_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics_by_state = {
        str(row["state"]): row
        for row in detail_rows
        if row.get("experiment") == experiment_name
    }
    variant, feedback = _variant_and_feedback(config)
    row: dict[str, Any] = {
        "Variant": variant,
        "Feedback": feedback,
        "λ": _format_lambda(config.turnover_penalty),
        "Detach": _format_bool_flag(config.detach_prev_weight),
        "Cost rate": _format_cost_rate(config.transaction_cost_rate),
    }
    for state, title in (("bear", "Bear"), ("neutral", "Neutral"), ("bull", "Bull")):
        metrics = metrics_by_state.get(state, {})
        row[f"{title} SR ↑"] = _format_summary_metric(metrics.get("mean_sr"))
        row[f"{title} TO ↓"] = _format_summary_metric(metrics.get("mean_turnover"))
        row[f"{title} epoch"] = _format_epoch(metrics.get("best_epoch"))
        row[f"{title} return"] = _format_summary_metric(metrics.get("mean_return"))
        row[f"{title} stocks"] = _format_summary_metric(metrics.get("mean_stocks"))
    return row


def _variant_and_feedback(config: ExperimentConfig) -> tuple[str, str]:
    penalty = config.turnover_penalty
    cost = config.transaction_cost_rate or 0.0
    sample = config.sample_num_stocks
    use_prev_weight = config.use_prev_weight_feature
    detach = config.detach_prev_weight

    if sample == 1000 and use_prev_weight is True and detach is False and penalty == 5000.0:
        return "Main", "End-to-end"
    if sample == 1000 and use_prev_weight is True and detach is True:
        if penalty == 5000.0 and cost == 0.0:
            return "Stop-gradient feedback", "Stop-gradient"
        if penalty == 0.0 and cost == 0.0:
            return "No turnover penalty", "Stop-gradient"
        if penalty == 0.0 and cost == 0.001:
            return "With transaction cost", "Stop-gradient"
    if sample == 1000 and use_prev_weight is False:
        if penalty == 0.0:
            return "No prev-weight feedback", "None"
        if penalty == 5000.0:
            return "No prev-weight feedback + penalty", "None"
    if sample == 1500 and use_prev_weight is True and detach is False and penalty == 5000.0:
        return "Larger universe", "End-to-end"
    if sample == 200 and use_prev_weight is True and detach is False and penalty == 5000.0:
        if config.train_batch_size == 30:
            return "Small sample + larger batch", "End-to-end"
        return "Small sample", "End-to-end"
    return config.label, config.detach_label


def _format_lambda(value: float | None) -> str:
    if value is None:
        return MISSING_TABLE_VALUE
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):g}"


def _format_bool_flag(value: bool | None) -> str:
    if value is None:
        return MISSING_TABLE_VALUE
    return str(value)


def _format_cost_rate(value: float | None) -> str:
    if value is None:
        return MISSING_TABLE_VALUE
    return f"{float(value):g}"


def _format_epoch(value: Any) -> str:
    if value is None:
        return MISSING_TABLE_VALUE
    return str(int(value))


def _format_summary_metric(value: Any) -> str:
    if value is None:
        return MISSING_TABLE_VALUE
    return f"{float(value):.4f}"


def _regime_table_sort_key(row: dict[str, Any]) -> int:
    order = {
        "Main": 0,
        "Stop-gradient feedback": 1,
        "No turnover penalty": 2,
        "With transaction cost": 3,
        "No prev-weight feedback": 4,
        "No prev-weight feedback + penalty": 5,
        "Larger universe": 6,
        "Small sample": 7,
        "Small sample + larger batch": 8,
    }
    return order.get(str(row.get("Variant")), 10**9)


def _mean_present(values: Any) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _write_csv(output_path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _markdown_table(rows: list[dict[str, Any]], fieldnames: list[str], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(field)) for field in fieldnames) + " |")
    return "\n".join(lines)


def _write_markdown(output_path: Path, detail_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    state_fields = [
        "experiment",
        "state",
        "best_epoch",
        "mean_sr",
        "mean_return",
        "mean_turnover",
        "mean_stocks",
        "complete_epoch_count",
    ]
    lines = [
        "# Ablation Best-Epoch Tables",
        "",
        "Best epoch is selected per experiment/state by the highest mean holdout portfolio SR.",
        "",
        "## Table 2. Regime-wise Ablation With Turnover",
        "",
        _markdown_table(
            summary_rows,
            REGIME_TABLE_FIELDS,
            REGIME_TABLE_FIELDS,
        ),
        "",
        "## By State",
        "",
        _markdown_table(
            detail_rows,
            state_fields,
            ["experiment", "state", "best_epoch", "mean_sr", "mean_return", "mean_turnover", "mean_stocks", "epochs"],
        ),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_arg_parser().parse_args()
    output_root = args.output_root.resolve()
    output_dir = args.output_dir.resolve()
    detail_rows, summary_rows = _collect_rows(output_root)

    detail_fields = [
        "experiment",
        "label",
        "state",
        "sample_num_stocks",
        "detach_prev_weight",
        "use_prev_weight_feature",
        "turnover_penalty",
        "transaction_cost_rate",
        "rolling_stride_days",
        "complete_epoch_count",
        "best_epoch",
        "mean_return",
        "mean_sr",
        "mean_stocks",
        "mean_turnover",
    ]
    summary_fields = REGIME_TABLE_FIELDS

    detail_path = output_dir / DETAIL_CSV_NAME
    summary_path = output_dir / SUMMARY_CSV_NAME
    markdown_path = output_dir / MARKDOWN_NAME
    _write_csv(detail_path, detail_rows, detail_fields)
    _write_csv(summary_path, summary_rows, summary_fields)
    _write_markdown(markdown_path, detail_rows, summary_rows)

    print(
        json.dumps(
            {
                "detail_csv": str(detail_path),
                "summary_csv": str(summary_path),
                "markdown": str(markdown_path),
                "num_experiments": len(summary_rows),
                "num_state_rows": len(detail_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
