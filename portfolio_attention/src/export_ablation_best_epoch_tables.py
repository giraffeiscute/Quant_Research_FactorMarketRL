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
RL_SUMMARY_CSV_NAME = "ablation_best_epoch_summary_rl.csv"
MARKDOWN_NAME = "ablation_best_epoch_tables.md"
MISSING_TABLE_VALUE = "—"
HIGH_EXPLORATION_EVIDENCE_SCALE = 0.3
LOW_EXPLORATION_EVIDENCE_SCALE = 1.0
REGIME_TABLE_FIELDS = [
    "Variant",
    "Feedback",
    "Group size",
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
    "Bear cash",
    "Neutral cash",
    "Bull cash",
    "Bear stocks",
    "Neutral stocks",
    "Bull stocks",
    "Bear epoch",
    "Neutral epoch",
    "Bull epoch",
]
RL_SUMMARY_PREFIX_FIELDS = [
    "Experiment",
    "Training",
    "Task head",
    "lr",
    "Reward style",
    "Exploration",
]
RL_SUMMARY_EXCLUDED_FIELDS = frozenset({"λ", "Detach", "Cost rate"})
RL_SUMMARY_TABLE_FIELDS = RL_SUMMARY_PREFIX_FIELDS + [
    field for field in REGIME_TABLE_FIELDS if field not in RL_SUMMARY_EXCLUDED_FIELDS
]
RL_SUMMARY_BASELINE_EXPERIMENTS = frozenset(
    {
        "predictions_s400_p0_noW_stride1",
        "predictions_dirch__s400_p0_noW_stride1",
    }
)


@dataclass(frozen=True)
class ExperimentConfig:
    sample_num_stocks: int | None
    train_batch_size: int | None
    detach_prev_weight: bool | None
    use_prev_weight_feature: bool | None
    turnover_penalty: float | None
    transaction_cost_rate: float | None
    rolling_stride_days: int | None
    learning_rate: str | None
    rl_group_size: int | None
    reward_style: str | None
    exploration: str | None
    rl_post_train_evidence_scale: str | None
    training: str
    task_head: str | None

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
        rl = f"RL_g{self.rl_group_size}_" if self.rl_group_size is not None else ""
        return (
            f"{rl}{sample}_{self.detach_label}_{self.prev_weight_label}_"
            f"p{penalty}_cost{cost}_stride{stride}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan outputs/predictions_* ablation directories, select the best "
            "holdout epoch per state by mean SR, and export comparison tables."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Root directory containing predictions_* experiment directories, "
            "or one supported prediction experiment directory."
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


def _load_runtime_configs_by_state(experiment_dir: Path) -> list[tuple[str | None, dict[str, Any]]]:
    runtime_items: list[tuple[str | None, Path]] = []
    for state in STATE_NAMES:
        runtime_items.extend((state, path) for path in sorted((experiment_dir / state).glob("*runtime_config*.json")))
    if not runtime_items:
        runtime_items = [
            (path.parent.name if path.parent.name in STATE_NAMES else None, path)
            for path in sorted(experiment_dir.glob("*/*runtime_config*.json"))
        ]
    return [
        (state, json.loads(path.read_text(encoding="utf-8")))
        for state, path in runtime_items
    ]


def _group_size_from_experiment_name(experiment_dir: Path) -> int | None:
    match = re.search(r"(?:^|_)g(\d+)(?:_|$)", experiment_dir.name)
    return int(match.group(1)) if match else None


def _to_optional_epoch(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_optional_csv_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _best_epoch_val_cash(experiment_dir: Path, state: str, best_epoch: Any) -> float | None:
    epoch = _to_optional_epoch(best_epoch)
    if epoch is None:
        return None
    if not experiment_dir.name.startswith("predictions_"):
        return None
    logs_dir_name = experiment_dir.name.replace("predictions_", "lightning_logs_", 1)
    metrics_dir = experiment_dir.parent / logs_dir_name / f"{state}_{LOSS_NAMES[0]}"
    metrics_path = metrics_dir / "metrics.csv"
    if not metrics_path.is_file():
        metrics_path = metrics_dir / "RL_metrics.csv"
    if not metrics_path.is_file():
        return None

    val_cash_by_epoch: dict[int, float] = {}
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_epoch = _to_optional_epoch(row.get("epoch"))
            row_val_cash = _to_optional_csv_float(row.get("val_cash"))
            if row_epoch is None or row_val_cash is None:
                continue
            val_cash_by_epoch[row_epoch] = row_val_cash
    return val_cash_by_epoch.get(epoch)


def _format_config_float(value: Any) -> str:
    return f"{float(value):g}"


def _train_config(payload: dict[str, Any]) -> dict[str, Any]:
    train = payload.get("train_config", {})
    return train if isinstance(train, dict) else {}


def _model_config(payload: dict[str, Any]) -> dict[str, Any]:
    model = payload.get("model_config", {})
    return model if isinstance(model, dict) else {}


def _rl_training_config(payload: dict[str, Any]) -> dict[str, Any]:
    rl_training = _train_config(payload).get("rl_training", {})
    return rl_training if isinstance(rl_training, dict) else {}


def _coalesced_runtime_label(
    runtime_configs: list[tuple[str | None, dict[str, Any]]],
    getter: Any,
    formatter: Any = str,
) -> str | None:
    state_labels: list[tuple[str | None, str]] = []
    unique_labels: list[str] = []
    for state, payload in runtime_configs:
        value = getter(payload)
        if value in (None, ""):
            continue
        label = str(formatter(value))
        state_labels.append((state, label))
        if label not in unique_labels:
            unique_labels.append(label)
    if not unique_labels:
        return None
    if len(unique_labels) == 1:
        return unique_labels[0]
    return "; ".join(
        f"{state}={label}" if state is not None else label
        for state, label in state_labels
    )


def _reward_style_from_type(value: Any) -> str:
    reward_type = str(value).strip().lower()
    if reward_type == "rolling_sharpe":
        return "rollingSR"
    if reward_type == "dsr_day_last":
        return "DSR"
    return str(value)


def _task_head_from_model(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    mode = str(_model_config(payload).get("inference_allocation_mode", "")).strip().lower()
    if mode == "dirichlet_mean":
        return "dirichlet"
    if mode:
        return mode
    return "softmax"


def _training_label(experiment_name: str) -> str:
    if experiment_name.startswith("predictions_RL"):
        return "RL_from_scratch"
    if experiment_name.startswith("predictions_dirch__"):
        return "pretrain_alpha"
    if experiment_name.startswith("predictions_postRL"):
        return "postRL_from_pretrain_alpha"
    return "raw"


def _find_evidence_scale(value: Any) -> Any:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = str(key).strip().lower()
            if normalized_key in {
                "evidence_scale",
                "post_train_evidence_scale",
                "rl_post_train_evidence_scale",
                "default_rl_post_train_evidence_scale",
            }:
                return item
            found = _find_evidence_scale(item)
            if found is not None:
                return found
    if isinstance(value, list):
        for item in value:
            found = _find_evidence_scale(item)
            if found is not None:
                return found
    return None


def _rl_enabled_from_payloads(runtime_configs: list[tuple[str | None, dict[str, Any]]]) -> bool:
    return any(_rl_training_config(payload).get("enabled") is True for _, payload in runtime_configs)


def _exploration_from_experiment_name(experiment_name: str, *, rl_enabled: bool) -> str | None:
    lower_name = experiment_name.lower()
    if "lowexpro" in lower_name or "lowexploration" in lower_name:
        return "low"
    if "highexpro" in lower_name or "highexploration" in lower_name:
        return "high"
    if rl_enabled:
        return "medium"
    return None


def _fallback_evidence_scale_label(experiment_name: str, *, rl_enabled: bool) -> str | None:
    exploration = _exploration_from_experiment_name(experiment_name, rl_enabled=rl_enabled)
    if exploration == "high":
        return _format_config_float(HIGH_EXPLORATION_EVIDENCE_SCALE)
    if exploration == "low":
        return _format_config_float(LOW_EXPLORATION_EVIDENCE_SCALE)
    return None


def _experiment_config(experiment_dir: Path) -> ExperimentConfig:
    runtime_configs = _load_runtime_configs_by_state(experiment_dir)
    payload = runtime_configs[0][1] if runtime_configs else None
    training = _training_label(experiment_dir.name)
    is_rl_experiment = experiment_dir.name.startswith(("predictions_RL", "predictions_postRL"))
    if payload is None:
        return ExperimentConfig(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            _group_size_from_experiment_name(experiment_dir),
            None,
            None,
            None,
            training,
            None,
        )
    data = payload.get("data_config", {})
    model = _model_config(payload)
    train = _train_config(payload)
    rl_training = _rl_training_config(payload)
    dirname_group_size = _group_size_from_experiment_name(experiment_dir)
    rl_group_size = (
        _to_optional_int(rl_training.get("group_size"))
        if isinstance(rl_training, dict) and rl_training.get("enabled") is True
        else None
    )
    rl_enabled = (
        is_rl_experiment
        and (_rl_enabled_from_payloads(runtime_configs) or rl_group_size is not None or dirname_group_size is not None)
    )
    evidence_scale = (
        _coalesced_runtime_label(runtime_configs, _find_evidence_scale, _format_config_float)
        if is_rl_experiment
        else None
    )
    return ExperimentConfig(
        sample_num_stocks=_to_optional_int(data.get("sample_num_stocks")),
        train_batch_size=_to_optional_int(data.get("train_batch_size")),
        detach_prev_weight=_to_optional_bool(model.get("detach_prev_weight")),
        use_prev_weight_feature=_to_optional_bool(model.get("use_prev_weight_feature")),
        turnover_penalty=_to_optional_float(train.get("turnover_penalty")),
        transaction_cost_rate=_to_optional_float(train.get("transaction_cost_rate")),
        rolling_stride_days=_to_optional_int(data.get("rolling_stride_days")),
        learning_rate=_coalesced_runtime_label(
            runtime_configs,
            lambda runtime_payload: _train_config(runtime_payload).get("learning_rate"),
            _format_config_float,
        ),
        rl_group_size=rl_group_size or dirname_group_size,
        reward_style=(
            _coalesced_runtime_label(
                runtime_configs,
                lambda runtime_payload: _rl_training_config(runtime_payload).get("reward_type")
                if _rl_training_config(runtime_payload).get("enabled") is True
                else None,
                _reward_style_from_type,
            )
            if is_rl_experiment
            else None
        ),
        exploration=(
            _exploration_from_experiment_name(experiment_dir.name, rl_enabled=rl_enabled)
            if is_rl_experiment
            else None
        ),
        rl_post_train_evidence_scale=evidence_scale
        or _fallback_evidence_scale_label(experiment_dir.name, rl_enabled=rl_enabled),
        training=training,
        task_head=_task_head_from_model(payload),
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
        and path.name.startswith("predictions_")
        and any((path / state).is_dir() for state in STATE_NAMES)
    )


def _prediction_experiment_dirs(output_root: Path) -> list[Path]:
    if _is_prediction_experiment_dir(output_root):
        return [output_root]
    return [
        path
        for path in output_root.glob("predictions_*")
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
        config.rl_group_size if config.rl_group_size is not None else 10**9,
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
        for state in STATE_NAMES:
            state_dir = experiment_dir / state
            best_payload = _select_best_epoch(state_dir)
            metrics = _state_metrics(best_payload, context=f"{experiment_dir.name}/{state}")
            metrics["mean_cash"] = _best_epoch_val_cash(experiment_dir, state, metrics.get("best_epoch"))
            epoch_count = _complete_epoch_count(state_dir)
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
                    "learning_rate": config.learning_rate,
                    "rl_group_size": config.rl_group_size,
                    "reward_style": config.reward_style,
                    "exploration": config.exploration,
                    "rl_post_train_evidence_scale": config.rl_post_train_evidence_scale,
                    "training": config.training,
                    "task_head": config.task_head,
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
        "Experiment": experiment_name,
        "Training": config.training,
        "Task head": config.task_head or MISSING_TABLE_VALUE,
        "lr": config.learning_rate or MISSING_TABLE_VALUE,
        "Reward style": config.reward_style or MISSING_TABLE_VALUE,
        "Exploration": config.exploration or MISSING_TABLE_VALUE,
        "Evidence scale": config.rl_post_train_evidence_scale or MISSING_TABLE_VALUE,
        "Variant": variant,
        "Feedback": feedback,
        "Group size": _format_group_size(config.rl_group_size),
        "λ": _format_lambda(config.turnover_penalty),
        "Detach": _format_bool_flag(config.detach_prev_weight),
        "Cost rate": _format_cost_rate(config.transaction_cost_rate),
    }
    for state, title in (("bear", "Bear"), ("neutral", "Neutral"), ("bull", "Bull")):
        metrics = metrics_by_state.get(state, {})
        row[f"{title} SR ↑"] = _format_summary_metric(metrics.get("mean_sr"))
        row[f"{title} TO ↓"] = _format_summary_metric(metrics.get("mean_turnover"))
        row[f"{title} cash"] = _format_summary_metric(metrics.get("mean_cash"))
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
    rl_group_size = config.rl_group_size

    if rl_group_size is not None:
        return f"RL group size {rl_group_size}", "RL"

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
            return "Small sample (200) + larger batch", "End-to-end"
        return "Small sample (200)", "End-to-end"
    if sample == 400 and use_prev_weight is True and detach is False and penalty == 5000.0:
        return "Small sample (400)", "End-to-end"
    if sample == 400 and use_prev_weight is False:
        if penalty == 0.0:
            return "Small sample (400) no prev-weight feedback", "None"
        if penalty == 5000.0:
            return "Small sample (400) no prev-weight feedback + penalty", "None"
    if sample == 600 and use_prev_weight is True and detach is False and penalty == 5000.0:
        return "Small sample (600)", "End-to-end"
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


def _format_group_size(value: int | None) -> str:
    if value is None:
        return MISSING_TABLE_VALUE
    return str(value)


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
        "Small sample (200)": 7,
        "Small sample (200) + larger batch": 8,
        "Small sample (400)": 9,
        "Small sample (400) no prev-weight feedback": 10,
        "RL group size 128": 11,
        "Small sample (400) no prev-weight feedback + penalty": 12,
        "Small sample (600)": 13,
    }
    return order.get(str(row.get("Variant")), 10**9)


def _is_rl_summary_experiment(experiment_name: str) -> bool:
    return (
        experiment_name in RL_SUMMARY_BASELINE_EXPERIMENTS
        or experiment_name.startswith("predictions_RL")
        or experiment_name.startswith("predictions_postRL")
    )


def _is_standard_summary_experiment(experiment_name: str) -> bool:
    return experiment_name.startswith("predictions_s") or experiment_name.startswith("predictions_RL")


def _rl_summary_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    experiment_name = str(row.get("Experiment") or "")
    order = {
        "predictions_s400_p0_noW_stride1": 0,
        "predictions_RL_s400_g128_lr4": 1,
        "predictions_RL_s1500_g64_lr2^4": 2,
        "predictions_dirch__s400_p0_noW_stride1": 3,
        "predictions_postRL_s400_p0_noW_stride1": 4,
        "predictions_postRL_s400_p0_noW_stride1_lowexpro": 5,
        "predictions_postRL_s400_p0_lowexpro_rollingSR": 6,
        "predictions_postRL_s400_p0_highexpro_lowclip_rollingSR": 7,
        "predictions_postRL_s400_lr5^6_highexpro_lowclip_DSR": 8,
        "predictions_postRL_s400_lr1^5_highexpro_lowclip_DSR": 9,
    }
    if experiment_name in order:
        return (order[experiment_name], experiment_name)
    if experiment_name.startswith("predictions_postRL"):
        return (50, experiment_name)
    if experiment_name.startswith("predictions_RL"):
        return (40, experiment_name)
    return (60, experiment_name)


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
        "rl_group_size",
        "learning_rate",
        "reward_style",
        "exploration",
        "rl_post_train_evidence_scale",
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
            [
                "experiment",
                "state",
                "best_epoch",
                "mean_sr",
                "mean_return",
                "mean_turnover",
                "mean_stocks",
                "group_size",
                "lr",
                "reward_style",
                "exploration",
                "evidence_scale",
                "epochs",
            ],
        ),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_arg_parser().parse_args()
    output_root = args.output_root.resolve()
    output_dir = args.output_dir.resolve()
    detail_rows, summary_rows = _collect_rows(output_root)
    standard_summary_rows = [
        row
        for row in summary_rows
        if _is_standard_summary_experiment(str(row.get("Experiment") or ""))
    ]
    rl_summary_rows = [
        row
        for row in summary_rows
        if _is_rl_summary_experiment(str(row.get("Experiment") or ""))
    ]
    rl_summary_rows.sort(key=_rl_summary_sort_key)

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
        "learning_rate",
        "rl_group_size",
        "reward_style",
        "exploration",
        "rl_post_train_evidence_scale",
        "training",
        "task_head",
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
    rl_summary_path = output_dir / RL_SUMMARY_CSV_NAME
    markdown_path = output_dir / MARKDOWN_NAME
    _write_csv(detail_path, detail_rows, detail_fields)
    _write_csv(summary_path, standard_summary_rows, summary_fields)
    _write_csv(rl_summary_path, rl_summary_rows, RL_SUMMARY_TABLE_FIELDS)
    _write_markdown(markdown_path, detail_rows, standard_summary_rows)

    print(
        json.dumps(
            {
                "detail_csv": str(detail_path),
                "summary_csv": str(summary_path),
                "rl_summary_csv": str(rl_summary_path),
                "markdown": str(markdown_path),
                "num_experiments": len(summary_rows),
                "num_standard_summary_experiments": len(standard_summary_rows),
                "num_rl_summary_experiments": len(rl_summary_rows),
                "num_state_rows": len(detail_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
