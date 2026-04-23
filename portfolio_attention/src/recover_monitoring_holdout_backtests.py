#!/usr/bin/env python3
"""Rebuild monitoring holdout backtest artifacts from saved checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from torch.utils.data import Dataset

from portfolio_attention.config import (
    DataConfig,
    EvaluationConfig,
    ModelConfig,
    PathsConfig,
    normalize_model_config_dict,
)
from portfolio_attention.dataset import PortfolioPanelDataset
from portfolio_attention.evaluate import (
    _normalize_overview_loss_order,
    rebuild_monitoring_holdout_backtest_overviews,
    run_monitoring_holdout_backtest,
)
from portfolio_attention.model import PortfolioAttentionModel
from portfolio_attention.utils import ensure_output_dirs

VALID_LOSSES = ("return", "sharpe", "dsr", "sortino", "mdd", "cvar")


def _parse_csv_epochs(raw_value: str) -> list[int]:
    epochs: list[int] = []
    seen: set[int] = set()
    for chunk in raw_value.split(","):
        token = chunk.strip()
        if not token:
            continue
        epoch = int(token)
        if epoch <= 0:
            raise ValueError(f"Epoch values must be positive, received {epoch}.")
        if epoch not in seen:
            seen.add(epoch)
            epochs.append(epoch)
    if not epochs:
        raise ValueError("At least one epoch must be provided.")
    return epochs


def _parse_csv_losses(raw_value: str) -> list[str]:
    losses: list[str] = []
    seen: set[str] = set()
    for chunk in raw_value.split(","):
        loss_name = chunk.strip().lower()
        if not loss_name:
            continue
        if loss_name not in VALID_LOSSES:
            raise ValueError(f"Unsupported loss {loss_name!r}. Valid options: {sorted(VALID_LOSSES)}.")
        if loss_name not in seen:
            seen.add(loss_name)
            losses.append(loss_name)
    if not losses:
        raise ValueError("At least one loss must be provided.")
    if len(losses) != 4:
        raise ValueError(
            "Monitoring overview recovery requires exactly 4 unique losses to build the four-panel chart."
        )
    return list(_normalize_overview_loss_order(losses))


def _checkpoint_path(checkpoint_dir: Path, state: str, loss_name: str, epoch: int) -> Path:
    return checkpoint_dir / f"{state}_train_monitoring_{loss_name}_epoch_{epoch}.pt"


def _serialize_for_comparison(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def _load_checkpoint(checkpoint_path: Path, *, map_location: str | torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a dict payload: {checkpoint_path}")
    return checkpoint


def _preflight_cuda(device_name: str) -> torch.device:
    if not str(device_name).startswith("cuda"):
        raise ValueError(
            f"This recovery script requires a CUDA device string such as 'cuda:0', received {device_name!r}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA preflight failed: torch.cuda.is_available() returned False in the requested runtime."
        )

    device = torch.device(device_name)
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(
            f"Requested CUDA device index {device.index} but only {torch.cuda.device_count()} devices are visible."
        )

    # Touch the device once so failures surface before the long-running recovery starts.
    torch.empty((1,), device=device)
    return device


def _build_dataset_from_checkpoint(
    data_config_payload: dict[str, Any],
) -> tuple[DataConfig, PortfolioPanelDataset, Dataset]:
    legacy_ratio_keys = (
        "scenario_train_split_ratio",
        "scenario_validation_split_ratio",
        "scenario_test_split_ratio",
    )
    legacy_ratio_payload = {
        key: data_config_payload[key] for key in legacy_ratio_keys if key in data_config_payload
    }
    if legacy_ratio_payload:
        raise ValueError(
            "Monitoring holdout recovery does not support legacy scenario-internal "
            f"time split checkpoints: {sorted(legacy_ratio_payload)}."
        )
    data_config = DataConfig(**data_config_payload)
    dataset = PortfolioPanelDataset(data_config)
    _, _, test_dataset = dataset.build_train_validation_test_datasets()
    if len(test_dataset) == 0:
        raise RuntimeError("Holdout test dataset is empty; cannot rebuild monitoring backtests.")
    return data_config, dataset, test_dataset


def _build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    *,
    dataset: PortfolioPanelDataset,
    device: torch.device,
) -> PortfolioAttentionModel:
    checkpoint_model_config = checkpoint.get("model_config", {})
    if not isinstance(checkpoint_model_config, dict):
        raise ValueError("Checkpoint is missing a valid model_config payload.")
    if "stock_temporal_encoder_type" not in checkpoint_model_config:
        raise ValueError(
            "Checkpoint model_config is missing 'stock_temporal_encoder_type'. "
            "This checkpoint was saved with an older architecture and is not compatible with the current model."
        )

    normalized_model_config = normalize_model_config_dict(checkpoint_model_config)
    filtered_model_config = {
        key: value
        for key, value in normalized_model_config.items()
        if key in ModelConfig.__dataclass_fields__
    }
    model_config = ModelConfig(**filtered_model_config)
    max_lookback = checkpoint.get("max_lookback")
    if max_lookback is None:
        max_lookback = checkpoint.get("metadata", {}).get("max_context_time_steps")
    if max_lookback is None:
        max_lookback = dataset.max_time_steps

    model = PortfolioAttentionModel(
        model_config,
        num_stocks=dataset.num_stocks,
        max_lookback=int(max_lookback),
        stock_temporal_attention_window=int(
            checkpoint.get("data_config", {}).get("lookback_days", dataset.metadata.lookback_days)
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _discover_checkpoints(
    *,
    checkpoint_dir: Path,
    state: str,
    epochs: list[int],
    losses: list[str],
    map_location: str,
) -> tuple[list[tuple[int, str, Path]], dict[str, Any]]:
    discovered: list[tuple[int, str, Path]] = []
    missing_paths: list[Path] = []
    baseline_data_config: dict[str, Any] | None = None
    baseline_data_config_signature: str | None = None

    for epoch in epochs:
        for loss_name in losses:
            checkpoint_path = _checkpoint_path(checkpoint_dir, state, loss_name, epoch)
            if not checkpoint_path.exists():
                missing_paths.append(checkpoint_path)
                continue

            checkpoint = _load_checkpoint(checkpoint_path, map_location=map_location)
            checkpoint_epoch = int(checkpoint.get("epoch") or -1)
            if checkpoint_epoch != epoch:
                raise ValueError(
                    f"Checkpoint epoch mismatch for {checkpoint_path}: expected {epoch}, got {checkpoint_epoch}."
                )

            checkpoint_train_config = checkpoint.get("train_config", {})
            checkpoint_loss_name = str(checkpoint_train_config.get("loss_name", "")).strip().lower()
            if checkpoint_loss_name != loss_name:
                raise ValueError(
                    f"Checkpoint loss mismatch for {checkpoint_path}: expected {loss_name}, got {checkpoint_loss_name}."
                )

            checkpoint_data_config = checkpoint.get("data_config", {})
            if not isinstance(checkpoint_data_config, dict):
                raise ValueError(f"Checkpoint is missing a valid data_config payload: {checkpoint_path}")

            checkpoint_state = str(checkpoint_data_config.get("state", "")).strip().lower()
            if checkpoint_state != state:
                raise ValueError(
                    f"Checkpoint state mismatch for {checkpoint_path}: expected {state}, got {checkpoint_state}."
                )

            current_signature = _serialize_for_comparison(checkpoint_data_config)
            if baseline_data_config_signature is None:
                baseline_data_config = dict(checkpoint_data_config)
                baseline_data_config_signature = current_signature
            elif current_signature != baseline_data_config_signature:
                raise ValueError(
                    "All monitoring checkpoints must share the same data_config to reuse one holdout split. "
                    f"Mismatch found in {checkpoint_path}."
                )

            discovered.append((epoch, loss_name, checkpoint_path))

    if missing_paths:
        formatted = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing monitoring checkpoints:\n{formatted}")
    if baseline_data_config is None:
        raise RuntimeError("No monitoring checkpoints were discovered.")

    return discovered, baseline_data_config


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must contain an object payload: {path}")
    return payload


def _verify_epoch_outputs(
    *,
    paths: PathsConfig,
    state: str,
    epoch: int,
    losses: list[str],
    expected_scenarios: int,
    overview_paths: list[str],
) -> dict[str, Any]:
    output_dir = paths.get_state_predictions_dir(state) / f"{epoch}_holdout_backtest"
    if not output_dir.exists():
        raise FileNotFoundError(f"Recovered output directory was not created: {output_dir}")

    verified_manifests: list[str] = []
    scenario_count: int | None = None
    for loss_name in losses:
        manifest_path = output_dir / f"{loss_name}_monitoring_holdout_backtest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing recovered manifest: {manifest_path}")

        manifest = _load_manifest(manifest_path)
        if manifest.get("artifact_type") != "monitoring_holdout_backtest_manifest":
            raise ValueError(f"Unexpected artifact_type in {manifest_path}")
        if str(manifest.get("state", "")).lower() != state:
            raise ValueError(f"Manifest state mismatch in {manifest_path}")
        if int(manifest.get("epoch", -1)) != epoch:
            raise ValueError(f"Manifest epoch mismatch in {manifest_path}")
        if str(manifest.get("loss_name", "")).lower() != loss_name:
            raise ValueError(f"Manifest loss_name mismatch in {manifest_path}")
        if manifest.get("overview_loss_order") != losses:
            raise ValueError(f"Manifest overview_loss_order mismatch in {manifest_path}")

        scenario_ids = manifest.get("scenario_ids")
        scenario_artifacts = manifest.get("scenario_artifacts")
        if not isinstance(scenario_ids, list) or not isinstance(scenario_artifacts, list):
            raise ValueError(f"Manifest scenario payload is malformed: {manifest_path}")
        if len(scenario_ids) != len(scenario_artifacts):
            raise ValueError(f"Manifest scenario count mismatch in {manifest_path}")
        if len(scenario_ids) != expected_scenarios:
            raise ValueError(
                f"Manifest scenario count mismatch in {manifest_path}: "
                f"expected {expected_scenarios}, got {len(scenario_ids)}."
            )

        if scenario_count is None:
            scenario_count = len(scenario_ids)

        for item in scenario_artifacts:
            chart_path = item.get("weight_trajectory_overview_chart")
            if not isinstance(chart_path, str) or not Path(chart_path).exists():
                raise FileNotFoundError(
                    f"Scenario overview chart is missing or invalid in {manifest_path}: {chart_path!r}"
                )
            benchmark_market_index_csv = item.get("benchmark_market_index_csv")
            if benchmark_market_index_csv not in {None, ""} and not Path(
                str(benchmark_market_index_csv)
            ).exists():
                raise FileNotFoundError(
                    "Scenario benchmark market index CSV is missing or invalid in "
                    f"{manifest_path}: {benchmark_market_index_csv!r}"
                )
            for metric_name in (
                "benchmark_excess_return",
                "benchmark_information_ratio",
                "benchmark_excess_max_drawdown",
            ):
                metric_value = item.get(metric_name)
                if metric_value in {None, ""}:
                    continue
                float(metric_value)

        verified_manifests.append(str(manifest_path))

    for path in overview_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Overview path does not exist after rebuild: {path}")

    return {
        "epoch": epoch,
        "output_dir": str(output_dir),
        "manifest_count": len(verified_manifests),
        "overview_count": len(overview_paths),
        "scenario_count": scenario_count or 0,
        "manifests": verified_manifests,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild monitoring holdout backtest JSON and overview PNG artifacts from saved checkpoints."
    )
    parser.add_argument("--state", required=True)
    parser.add_argument("--epochs", required=True, help="Comma-separated epoch list, e.g. 30,60,90,120")
    parser.add_argument(
        "--losses",
        required=True,
        help="Comma-separated loss list used for the four-panel overview, e.g. mdd,return,sharpe,sortino",
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    state = str(args.state).strip().lower()
    epochs = _parse_csv_epochs(args.epochs)
    losses = _parse_csv_losses(args.losses)
    checkpoint_dir = args.checkpoint_dir.resolve()
    device_name = str(args.device).strip()

    device = _preflight_cuda(device_name)
    paths = PathsConfig(project_dir=PROJECT_DIR)
    ensure_output_dirs(paths)

    discovered, baseline_data_config = _discover_checkpoints(
        checkpoint_dir=checkpoint_dir,
        state=state,
        epochs=epochs,
        losses=losses,
        map_location="cpu",
    )
    data_config, dataset, test_dataset = _build_dataset_from_checkpoint(baseline_data_config)
    resolved_state = str(data_config.state).lower()
    if resolved_state != state:
        raise ValueError(
            f"Recovered dataset state mismatch: requested {state}, checkpoint data_config resolved to {resolved_state}."
        )

    if len(test_dataset) <= 0:
        raise RuntimeError("Holdout dataset is empty; cannot rebuild monitoring outputs.")

    evaluation_config = EvaluationConfig()
    checkpoint_index = {(epoch, loss_name): checkpoint_path for epoch, loss_name, checkpoint_path in discovered}
    epoch_summaries: list[dict[str, Any]] = []

    print(
        json.dumps(
            {
                "status": "starting",
                "python": sys.executable,
                "device": str(device),
                "state": state,
                "epochs": epochs,
                "losses": losses,
                "holdout_scenarios": len(test_dataset),
                "checkpoint_dir": str(checkpoint_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for epoch in epochs:
        for loss_name in losses:
            checkpoint_path = checkpoint_index[(epoch, loss_name)]
            checkpoint = _load_checkpoint(checkpoint_path, map_location=device)
            model = _build_model_from_checkpoint(checkpoint, dataset=dataset, device=device)
            try:
                payload = run_monitoring_holdout_backtest(
                    model=model,
                    dataset=dataset,
                    holdout_dataset=test_dataset,
                    loss_name=loss_name,
                    epoch=epoch,
                    paths=paths,
                    device=device,
                    evaluation_config=evaluation_config,
                )
            finally:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            print(
                json.dumps(
                    {
                        "status": "recovered_loss",
                        "epoch": epoch,
                        "loss_name": loss_name,
                        "output_dir": payload["holdout_backtest_output_dir"],
                        "scenario_count": len(payload["scenario_artifacts"]),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        overview_paths = rebuild_monitoring_holdout_backtest_overviews(
            paths,
            state=state,
            epoch=epoch,
            loss_order=losses,
        )
        summary = _verify_epoch_outputs(
            paths=paths,
            state=state,
            epoch=epoch,
            losses=losses,
            expected_scenarios=len(test_dataset),
            overview_paths=overview_paths,
        )
        epoch_summaries.append(summary)
        print(json.dumps({"status": "verified_epoch", **summary}, ensure_ascii=False), flush=True)

    print(
        json.dumps(
            {
                "status": "completed",
                "state": state,
                "device": str(device),
                "epochs": epoch_summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
