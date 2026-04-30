"""Export best-epoch comparison tables for multiple result directories."""

from __future__ import annotations

import argparse
import csv
import json
import re
from numbers import Integral, Real
from pathlib import Path
import sys
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

STATE_NAMES = ("bear", "neutral", "bull")
LOSS_NAMES = ("sharpe",)
# Edit these when switching the comparison targets.
RESULT_DIR_1 = PROJECT_DIR / "outputs" / "result v18 add dropout 0.1"
RESULT_DIR_2 = PROJECT_DIR / "outputs" / "result v19 lamda0.2 l1 a=0.7"
RESULT_DIR_3 = PROJECT_DIR / "outputs" / "result v20 lamda 0.05 l2 a=1"
RESULT_DIR_4 = PROJECT_DIR / "outputs" / "result v22 lamda 2500 sample1000 l2"
OUTPUT_DIR = PROJECT_DIR / "outputs"
# Leave blank to auto-generate a readable label from the directory name.
LABEL_1 = "no weight"
LABEL_2 = "l1"
LABEL_3 = "l2"
LABEL_4 = "l2 sample1000"
COMPARISON_CSV_NAME = "best_epoch_state_loss_comparison.csv"
SUMMARY_CSV_NAME = "best_epoch_state_summary.csv"
NET_FEE_RATE = 0.001
NET_VERSION_THRESHOLD = 22


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export best-epoch comparison CSV files for four result directories."
    )
    parser.add_argument(
        "--dir-1",
        "--mlp-dir",
        dest="dir_1",
        type=Path,
        default=RESULT_DIR_1,
        help="Directory containing the first set of state folders.",
    )
    parser.add_argument(
        "--dir-2",
        "--attention-dir",
        dest="dir_2",
        type=Path,
        default=RESULT_DIR_2,
        help="Directory containing the second set of state folders.",
    )
    parser.add_argument(
        "--dir-3",
        dest="dir_3",
        type=Path,
        default=RESULT_DIR_3,
        help="Directory containing the third set of state folders.",
    )
    parser.add_argument(
        "--dir-4",
        dest="dir_4",
        type=Path,
        default=RESULT_DIR_4,
        help="Directory containing the fourth set of state folders.",
    )
    parser.add_argument(
        "--label-1",
        default=LABEL_1,
        help="Display label for the first result set. Defaults to a prettified directory name.",
    )
    parser.add_argument(
        "--label-2",
        default=LABEL_2,
        help="Display label for the second result set. Defaults to a prettified directory name.",
    )
    parser.add_argument(
        "--label-3",
        default=LABEL_3,
        help="Display label for the third result set. Defaults to a prettified directory name.",
    )
    parser.add_argument(
        "--label-4",
        default=LABEL_4,
        help="Display label for the fourth result set. Defaults to a prettified directory name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where the CSV files should be written.",
    )
    return parser


def _prettify_label(raw_label: str) -> str:
    tokens = raw_label.replace("_", " ").replace("-", " ").split()
    pretty_tokens: list[str] = []
    for token in tokens:
        if token.isupper():
            pretty_tokens.append(token)
            continue
        if len(token) > 1 and token[0].lower() == "v" and token[1:].isdigit():
            pretty_tokens.append(token.upper())
            continue
        if token.isdigit():
            pretty_tokens.append(token)
            continue
        pretty_tokens.append(token.capitalize())
    return " ".join(pretty_tokens)


def _resolve_display_label(raw_label: str, result_dir: Path, fallback_label: str) -> str:
    label = str(raw_label).strip()
    if label:
        return label
    prettified = _prettify_label(result_dir.name)
    return prettified or fallback_label


def _require_distinct_labels(labels: list[str]) -> None:
    if len(set(labels)) != len(labels):
        raise ValueError(
            "All labels must be distinct so the exported columns remain unambiguous."
        )


def _parse_result_version(result_dir: Path) -> int | None:
    match = re.search(r"\bv(\d+)\b", result_dir.name, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _build_net_metric_flags(
    labels: list[str],
    result_dirs: list[Path],
) -> tuple[dict[str, bool], list[dict[str, str | int | None]]]:
    use_net_metrics: dict[str, bool] = {}
    warnings: list[dict[str, str | int | None]] = []
    for label, result_dir in zip(labels, result_dirs):
        version = _parse_result_version(result_dir)
        if version is None:
            use_net_metrics[label] = False
            warnings.append(
                {
                    "label": label,
                    "result_dir": str(result_dir),
                    "reason": "unparseable_version_token",
                    "expected_pattern": "v<number>",
                    "use_net_metrics": False,
                }
            )
            continue
        use_net_metrics[label] = version <= NET_VERSION_THRESHOLD
    return use_net_metrics, warnings


def _to_net_return_and_sr(
    gross_return: float | None,
    gross_sr: float | None,
    turnover: float | None,
) -> tuple[float | None, float | None]:
    if gross_return is None or gross_sr is None:
        return gross_return, gross_sr
    if turnover is None:
        return gross_return, gross_sr
    cost = NET_FEE_RATE * turnover * 100.0
    net_return = gross_return - cost
    if gross_return == 0:
        return net_return, None
    net_sr = gross_sr * (net_return / gross_return)
    return net_return, net_sr


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing manifest: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON manifest: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must contain a JSON object: {manifest_path}")
    return payload


def _coerce_float(value: Any, *, field_name: str, manifest_path: Path, scenario_id: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Scenario {scenario_id!r} in {manifest_path} has invalid {field_name}: {value!r}"
        ) from exc


def _coerce_optional_float(
    value: Any,
    *,
    field_name: str,
    manifest_path: Path,
    scenario_id: str,
) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Scenario {scenario_id!r} in {manifest_path} has invalid {field_name}: {value!r}"
        ) from exc


def _pick_optional_float(
    source: dict[str, Any],
    *,
    keys: tuple[str, ...],
    manifest_path: Path,
    scenario_id: str,
) -> float | None:
    for key in keys:
        if key in source:
            return _coerce_optional_float(
                source.get(key),
                field_name=key,
                manifest_path=manifest_path,
                scenario_id=scenario_id,
            )
    return None


def _extract_epoch_from_dir(epoch_dir: Path) -> int:
    token = epoch_dir.name.split("_", 1)[0].strip()
    if not token:
        raise ValueError(f"Epoch directory name is invalid: {epoch_dir}")
    try:
        epoch = int(token)
    except ValueError as exc:
        raise ValueError(f"Epoch directory does not start with an integer: {epoch_dir}") from exc
    if epoch <= 0:
        raise ValueError(f"Epoch must be positive in directory name: {epoch_dir}")
    return epoch


def _mean(values: list[float], *, context: str) -> float:
    if not values:
        raise ValueError(f"Cannot compute mean for empty value list: {context}")
    return sum(values) / len(values)


def _mean_optional(values: list[float], *, context: str) -> float | None:
    if not values:
        return None
    return _mean(values, context=context)


def _format_scenario_id(raw_scenario_id: str) -> str:
    token = str(raw_scenario_id).strip()
    if not token:
        raise ValueError("Scenario id cannot be empty.")
    parts = token.split("_")
    if parts and parts[-1].isdigit():
        return parts[-1]
    raise ValueError(f"Scenario id does not end with a numeric suffix: {raw_scenario_id!r}")


def _scenario_metrics_from_manifest(
    manifest_path: Path,
) -> list[dict[str, float | str | None]]:
    payload = _load_manifest(manifest_path)
    scenario_artifacts = payload.get("scenario_artifacts")
    if not isinstance(scenario_artifacts, list) or not scenario_artifacts:
        raise ValueError(f"Manifest must provide a non-empty scenario_artifacts list: {manifest_path}")

    manifest_mean_turnover = _pick_optional_float(
        payload,
        keys=("mean_average_turnover", "mean_turnover", "turnover_rate"),
        manifest_path=manifest_path,
        scenario_id="__manifest__",
    )
    rows: list[dict[str, float | str | None]] = []
    for item in scenario_artifacts:
        if not isinstance(item, dict):
            raise ValueError(f"Manifest scenario_artifacts entries must be objects: {manifest_path}")
        scenario_id = str(item.get("scenario_id", "")).strip()
        if not scenario_id:
            raise ValueError(f"Manifest scenario_artifacts entries must include scenario_id: {manifest_path}")
        scenario_turnover = _pick_optional_float(
            item,
            keys=("average_turnover", "turnover_rate", "turnover", "mean_average_turnover"),
            manifest_path=manifest_path,
            scenario_id=scenario_id,
        )
        rows.append(
            {
                "scenario_id": _format_scenario_id(scenario_id),
                "portfolio_return": _coerce_float(
                    item.get("final_return"),
                    field_name="final_return",
                    manifest_path=manifest_path,
                    scenario_id=scenario_id,
                ),
                "portfolio_sr": _coerce_float(
                    item.get("backtest_portfolio_sr"),
                    field_name="backtest_portfolio_sr",
                    manifest_path=manifest_path,
                    scenario_id=scenario_id,
                ),
                "selected_stock_count": _coerce_float(
                    item.get("total_selected_stock_count"),
                    field_name="total_selected_stock_count",
                    manifest_path=manifest_path,
                    scenario_id=scenario_id,
                ),
                "turnover_rate": scenario_turnover if scenario_turnover is not None else manifest_mean_turnover,
            }
        )
    return rows


def _collect_state_epoch_manifests(state_dir: Path) -> tuple[dict[int, dict[str, Path]], list[dict[str, Any]]]:
    if not state_dir.is_dir():
        return {}, [{"state_dir": str(state_dir), "reason": "missing_state_directory"}]

    epoch_to_manifests: dict[int, dict[str, Path]] = {}
    skipped_epochs: list[dict[str, Any]] = []
    for epoch_dir in sorted(path for path in state_dir.iterdir() if path.is_dir()):
        epoch = _extract_epoch_from_dir(epoch_dir)
        manifest_paths = {
            loss_name: epoch_dir / f"{loss_name}_monitoring_holdout_backtest.json"
            for loss_name in LOSS_NAMES
        }
        missing = [str(path) for path in manifest_paths.values() if not path.is_file()]
        if missing:
            skipped_epochs.append(
                {
                    "epoch": epoch,
                    "epoch_dir": str(epoch_dir),
                    "missing_manifests": missing,
                }
            )
            continue
        epoch_to_manifests[epoch] = manifest_paths

    if not epoch_to_manifests:
        skipped_epochs.append({"state_dir": str(state_dir), "reason": "no_complete_epoch_directories"})
    return epoch_to_manifests, skipped_epochs


def _epoch_aggregate_from_manifests(epoch_manifests: dict[str, Path]) -> dict[str, Any]:
    per_loss_scenarios: dict[str, dict[str, dict[str, float | str]]] = {}
    sr_values: list[float] = []
    return_values: list[float] = []

    for loss_name in LOSS_NAMES:
        manifest_path = epoch_manifests[loss_name]
        rows = _scenario_metrics_from_manifest(manifest_path)
        scenario_map: dict[str, dict[str, float | str]] = {}
        for row in rows:
            scenario_id = str(row["scenario_id"])
            scenario_map[scenario_id] = row
            sr_values.append(float(row["portfolio_sr"]))
            return_values.append(float(row["portfolio_return"]))
        per_loss_scenarios[loss_name] = scenario_map

    return {
        "epoch_mean_sr": _mean(sr_values, context="epoch_mean_sr"),
        "epoch_mean_return": _mean(return_values, context="epoch_mean_return"),
        "per_loss_scenarios": per_loss_scenarios,
    }


def _select_best_epoch(state_dir: Path) -> dict[str, Any]:
    epoch_manifests, skipped_epochs = _collect_state_epoch_manifests(state_dir)
    best_payload: dict[str, Any] | None = None

    for epoch in sorted(epoch_manifests):
        aggregate = _epoch_aggregate_from_manifests(epoch_manifests[epoch])
        candidate = {
            "epoch": epoch,
            "epoch_mean_sr": aggregate["epoch_mean_sr"],
            "epoch_mean_return": aggregate["epoch_mean_return"],
            "per_loss_scenarios": aggregate["per_loss_scenarios"],
        }
        if best_payload is None:
            best_payload = candidate
            continue

        if candidate["epoch_mean_sr"] > best_payload["epoch_mean_sr"]:
            best_payload = candidate
        elif (
            candidate["epoch_mean_sr"] == best_payload["epoch_mean_sr"]
            and candidate["epoch"] < best_payload["epoch"]
        ):
            # Keep the earlier epoch on exact ties so the choice is deterministic.
            best_payload = candidate

    if best_payload is None:
        return {
            "epoch": None,
            "epoch_mean_sr": None,
            "epoch_mean_return": None,
            "per_loss_scenarios": {loss_name: {} for loss_name in LOSS_NAMES},
            "skipped_epochs": skipped_epochs,
        }
    best_payload["skipped_epochs"] = skipped_epochs
    return best_payload


def _average_loss_rows(
    loss_scenarios: dict[str, dict[str, float | str]],
    *,
    context: str,
) -> tuple[float | None, float | None, float | None, float | None]:
    if not loss_scenarios:
        return None, None, None, None
    return_values = [float(item["portfolio_return"]) for item in loss_scenarios.values()]
    sr_values = [float(item["portfolio_sr"]) for item in loss_scenarios.values()]
    selected_stock_counts = [float(item["selected_stock_count"]) for item in loss_scenarios.values()]
    turnover_values = [
        float(item["turnover_rate"])
        for item in loss_scenarios.values()
        if item.get("turnover_rate") is not None
    ]
    return (
        _mean(return_values, context=f"{context}: portfolio_return"),
        _mean(sr_values, context=f"{context}: portfolio_sr"),
        _mean(selected_stock_counts, context=f"{context}: selected_stock_count"),
        _mean_optional(turnover_values, context=f"{context}: turnover_rate"),
    )


def _build_comparison_rows(
    best_by_label: list[tuple[str, dict[str, dict[str, Any]]]],
    use_net_metrics: dict[str, bool],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for state in STATE_NAMES:
        state_payloads = [(label, best_by_state[state]) for label, best_by_state in best_by_label]

        for loss_name in LOSS_NAMES:
            loss_rows_by_label = {
                label: best_payload["per_loss_scenarios"][loss_name]
                for label, best_payload in state_payloads
            }

            average_row: dict[str, Any] = {
                "state": state,
                "loss": loss_name,
                "row_type": "average",
                "scenario_id": "",
            }
            for label, best_payload in state_payloads:
                average_row[f"{label}_best_epoch"] = best_payload["epoch"]
                avg_return, avg_sr, _, avg_turnover = _average_loss_rows(
                    loss_rows_by_label[label],
                    context=f"{state}/{loss_name}/{label}",
                )
                if use_net_metrics.get(label, False):
                    net_return, net_sr = _to_net_return_and_sr(avg_return, avg_sr, avg_turnover)
                    average_row[f"{label}_return"] = net_return
                    average_row[f"{label}_sr"] = net_sr
                else:
                    average_row[f"{label}_return"] = avg_return
                    average_row[f"{label}_sr"] = avg_sr
                average_row[f"{label}_turnover"] = avg_turnover
            rows.append(average_row)

            all_scenario_ids = sorted(
                {
                    scenario_id
                    for loss_rows in loss_rows_by_label.values()
                    for scenario_id in loss_rows
                }
            )
            for scenario_id in all_scenario_ids:
                scenario_row: dict[str, Any] = {
                    "state": state,
                    "loss": loss_name,
                    "row_type": "scenario",
                    "scenario_id": scenario_id,
                }
                for label, best_payload in state_payloads:
                    row = loss_rows_by_label[label].get(scenario_id)
                    scenario_row[f"{label}_best_epoch"] = best_payload["epoch"]
                    gross_return = (
                        float(row["portfolio_return"]) if row is not None else None
                    )
                    gross_sr = (
                        float(row["portfolio_sr"]) if row is not None else None
                    )
                    scenario_turnover = (
                        float(row["turnover_rate"])
                        if row is not None and row.get("turnover_rate") is not None
                        else None
                    )
                    if use_net_metrics.get(label, False):
                        net_return, net_sr = _to_net_return_and_sr(
                            gross_return,
                            gross_sr,
                            scenario_turnover,
                        )
                        scenario_row[f"{label}_return"] = net_return
                        scenario_row[f"{label}_sr"] = net_sr
                    else:
                        scenario_row[f"{label}_return"] = gross_return
                        scenario_row[f"{label}_sr"] = gross_sr
                    scenario_row[f"{label}_turnover"] = scenario_turnover
                rows.append(scenario_row)

    return rows


def _build_summary_rows(
    best_by_label: list[tuple[str, dict[str, dict[str, Any]]]],
    use_net_metrics: dict[str, bool],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in STATE_NAMES:
        state_payloads = [(label, best_by_state[state]) for label, best_by_state in best_by_label]
        row: dict[str, Any] = {"state": state}
        for label, best_payload in state_payloads:
            row[f"{label}_best_epoch"] = best_payload["epoch"]
        for loss_name in LOSS_NAMES:
            mean_return_by_label: dict[str, float | None] = {}
            mean_sr_by_label: dict[str, float | None] = {}
            mean_stocks_by_label: dict[str, float | None] = {}
            mean_turnover_by_label: dict[str, float | None] = {}
            for label, best_payload in state_payloads:
                avg_return, avg_sr, avg_stocks_bought, avg_turnover = _average_loss_rows(
                    best_payload["per_loss_scenarios"][loss_name],
                    context=f"{state}/{loss_name}/{label}/summary",
                )
                mean_return_by_label[label] = avg_return
                mean_sr_by_label[label] = avg_sr
                mean_stocks_by_label[label] = avg_stocks_bought
                mean_turnover_by_label[label] = avg_turnover
            for label, _ in state_payloads:
                if use_net_metrics.get(label, False):
                    net_return, _ = _to_net_return_and_sr(
                        mean_return_by_label[label],
                        mean_sr_by_label[label],
                        mean_turnover_by_label[label],
                    )
                    row[f"{label}_{loss_name}_mean_return"] = net_return
                else:
                    row[f"{label}_{loss_name}_mean_return"] = mean_return_by_label[label]
            for label, _ in state_payloads:
                if use_net_metrics.get(label, False):
                    _, net_sr = _to_net_return_and_sr(
                        mean_return_by_label[label],
                        mean_sr_by_label[label],
                        mean_turnover_by_label[label],
                    )
                    row[f"{label}_{loss_name}_mean_sr"] = net_sr
                else:
                    row[f"{label}_{loss_name}_mean_sr"] = mean_sr_by_label[label]
            for label, _ in state_payloads:
                row[f"{label}_{loss_name}_mean_stocks"] = mean_stocks_by_label[label]
            for label, _ in state_payloads:
                row[f"{label}_{loss_name}_mean_turnover"] = mean_turnover_by_label[label]
        rows.append(row)
    return rows


def _write_csv(output_path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized_row = {
                key: (
                    "None"
                    if value is None
                    else
                    round(float(value))
                    if key.endswith("_mean_stocks") and isinstance(value, Real)
                    else round(float(value), 2)
                    if isinstance(value, Real) and not isinstance(value, Integral)
                    else value
                )
                for key, value in row.items()
            }
            writer.writerow(normalized_row)


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir.resolve()
    result_dirs = [
        args.dir_1.resolve(),
        args.dir_2.resolve(),
        args.dir_3.resolve(),
        args.dir_4.resolve(),
    ]
    labels = [
        _resolve_display_label(args.label_1, result_dirs[0], "Label 1"),
        _resolve_display_label(args.label_2, result_dirs[1], "Label 2"),
        _resolve_display_label(args.label_3, result_dirs[2], "Label 3"),
        _resolve_display_label(args.label_4, result_dirs[3], "Label 4"),
    ]
    _require_distinct_labels(labels)
    use_net_metrics, version_warnings = _build_net_metric_flags(labels, result_dirs)

    best_by_label = [
        (
            label,
            {state: _select_best_epoch(result_dir / state) for state in STATE_NAMES},
        )
        for label, result_dir in zip(labels, result_dirs)
    ]

    comparison_rows = _build_comparison_rows(
        best_by_label,
        use_net_metrics,
    )
    summary_rows = _build_summary_rows(
        best_by_label,
        use_net_metrics,
    )

    comparison_path = output_dir / COMPARISON_CSV_NAME
    summary_path = output_dir / SUMMARY_CSV_NAME
    _write_csv(
        comparison_path,
        comparison_rows,
        fieldnames=[
            "state",
            "loss",
            "row_type",
            "scenario_id",
            *[f"{label}_best_epoch" for label in labels],
            *[f"{label}_return" for label in labels],
            *[f"{label}_sr" for label in labels],
            *[f"{label}_turnover" for label in labels],
        ],
    )
    _write_csv(
        summary_path,
        summary_rows,
        fieldnames=[
            "state",
            *[f"{label}_best_epoch" for label in labels],
            *[
                field_name
                for loss_name in LOSS_NAMES
                for field_name in (
                    *[f"{label}_{loss_name}_mean_return" for label in labels],
                    *[f"{label}_{loss_name}_mean_sr" for label in labels],
                    *[f"{label}_{loss_name}_mean_stocks" for label in labels],
                    *[f"{label}_{loss_name}_mean_turnover" for label in labels],
                )
            ],
        ],
    )

    status_payload = {
        "comparison_csv": str(comparison_path),
        "summary_csv": str(summary_path),
        "use_net_metrics_by_label": use_net_metrics,
        "version_warnings": version_warnings,
        "states": {
            state: {
                **{
                    f"{label}_best_epoch": best_by_state[state]["epoch"]
                    for label, best_by_state in best_by_label
                },
                **{
                    f"{label}_skipped_epochs": best_by_state[state]["skipped_epochs"]
                    for label, best_by_state in best_by_label
                },
            }
            for state in STATE_NAMES
        },
    }
    print(json.dumps(status_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
