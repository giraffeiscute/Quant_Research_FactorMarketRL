"""Evaluation entrypoint."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from portfolio_attention import (
        artifact_paths,
        evaluate_rebuild,
        evaluation_artifacts,
        evaluation_monitoring,
        evaluation_pipeline,
        evaluation_presentation,
        evaluation_runtime,
        evaluation_shared,
    )
    from portfolio_attention.config import (
        DataConfig,
        ModelConfig,
        PathsConfig,
    )
    from portfolio_attention.config_validation import (
        normalize_model_config_dict,
        raise_if_checkpoint_uses_legacy_stock_id_representation_type,
        validated_data_config,
        validated_model_config,
    )
else:
    from . import (
        artifact_paths,
        evaluate_rebuild,
        evaluation_artifacts,
        evaluation_monitoring,
        evaluation_pipeline,
        evaluation_presentation,
        evaluation_runtime,
        evaluation_shared,
    )
    from .config import (
        DataConfig,
        ModelConfig,
        PathsConfig,
    )
    from .config_validation import (
        normalize_model_config_dict,
        raise_if_checkpoint_uses_legacy_stock_id_representation_type,
        validated_data_config,
        validated_model_config,
    )

TERMINAL_OUTPUT_KEYS = [
    "state",
    "loss_name",
    "num_holdout_scenarios",
    "mean_final_return",
    "mean_average_turnover",
    "std_final_return",
    "median_final_return",
    "worst_scenario_final_return",
    "best_scenario_final_return",
    "best_scenario_id",
]

# Legacy facade compatibility for root scripts.
ROLLING_ONE_STEP_EVALUATION_MODE = evaluation_runtime.ROLLING_ONE_STEP_EVALUATION_MODE
_compute_backtest_portfolio_sr = evaluation_artifacts._compute_backtest_portfolio_sr
_extract_exported_train_config = evaluation_artifacts.extract_exported_train_config
_get_aux_lookup = evaluation_presentation.get_aux_lookup
_is_weight_above_threshold = evaluation_shared.is_weight_above_threshold
_load_aux_frame = evaluation_presentation.load_aux_frame
_validate_checkpoint_metadata = evaluation_pipeline._validate_checkpoint_metadata
format_allocation_group_label = evaluation_presentation.format_allocation_group_label
_normalize_overview_loss_order = evaluation_shared.normalize_overview_loss_order
rebuild_monitoring_holdout_backtest_overviews = (
    evaluate_rebuild.rebuild_monitoring_holdout_backtest_overviews
)
run_monitoring_holdout_backtest = evaluation_monitoring.run_monitoring_holdout_backtest


def _resolve_checkpoint_state(data_config: DataConfig) -> str | None:
    return data_config.state


def _resolve_checkpoint_path(
    *,
    paths: PathsConfig,
    data_config: DataConfig,
    checkpoint_path: Path | None,
    loss_name: str | None,
) -> Path:
    return checkpoint_path or artifact_paths.train_best_checkpoint_path(
        paths,
        loss_name or "dsr",
        state=_resolve_checkpoint_state(data_config),
    )


def _resolve_checkpoint_metadata_dict(
    checkpoint: dict[str, Any],
    key: str,
) -> Any:
    checkpoint_payload = checkpoint.get(key)
    if checkpoint_payload:
        return checkpoint_payload
    checkpoint_metadata = checkpoint.get("portfolio_attention_metadata", {})
    if not isinstance(checkpoint_metadata, dict):
        return checkpoint_payload
    return checkpoint_metadata.get(key, checkpoint_payload)


def _build_model_config_from_checkpoint(checkpoint: dict[str, Any]) -> ModelConfig:
    checkpoint_model_config = _resolve_checkpoint_metadata_dict(checkpoint, "model_config")
    if checkpoint_model_config is None:
        checkpoint_model_config = {}
    if not isinstance(checkpoint_model_config, dict):
        raise ValueError("Checkpoint model_config payload must be a dictionary.")
    raise_if_checkpoint_uses_legacy_stock_id_representation_type(
        checkpoint_model_config,
        context="Checkpoint model_config",
    )
    if "stock_temporal_encoder_type" not in checkpoint_model_config:
        raise ValueError(
            "Checkpoint model_config is missing 'stock_temporal_encoder_type'. "
            "This checkpoint was saved with an older architecture and is not compatible with the current model."
        )
    normalized_model_config = normalize_model_config_dict(checkpoint_model_config)
    filtered_config_dict = {
        key: value
        for key, value in normalized_model_config.items()
        if key in ModelConfig.__dataclass_fields__
    }
    return validated_model_config(ModelConfig(**filtered_config_dict))


def _build_data_config_from_checkpoint(
    checkpoint: dict[str, Any],
    *,
    fallback_data_config: DataConfig,
) -> DataConfig:
    checkpoint_data_config = _resolve_checkpoint_metadata_dict(checkpoint, "data_config")
    if not isinstance(checkpoint_data_config, dict):
        return fallback_data_config
    filtered_config_dict = {
        key: value
        for key, value in checkpoint_data_config.items()
        if key in DataConfig.__dataclass_fields__
    }
    if not filtered_config_dict:
        return validated_data_config(fallback_data_config)

    fallback_dict = fallback_data_config.__dict__.copy()
    fallback_dict.update(filtered_config_dict)
    return validated_data_config(DataConfig(**fallback_dict))


def _validate_requested_runtime_configs_against_checkpoint(
    *,
    requested_data_config: DataConfig,
    requested_model_config: ModelConfig,
    checkpoint: dict[str, Any],
    args_dict: dict[str, Any],
) -> None:
    checkpoint_data_config = _build_data_config_from_checkpoint(
        checkpoint,
        fallback_data_config=requested_data_config,
    )
    checkpoint_model_config = _build_model_config_from_checkpoint(checkpoint)

    if "num_stocks" in args_dict and checkpoint_data_config.num_stocks != requested_data_config.num_stocks:
        raise ValueError(
            "Requested num_stocks does not match the checkpoint data configuration. "
            f"checkpoint={checkpoint_data_config.num_stocks} requested={requested_data_config.num_stocks}"
        )
    if (
        "stock_id_representation_type" in args_dict
        and checkpoint_model_config.stock_id_representation_type
        != requested_model_config.stock_id_representation_type
    ):
        raise ValueError(
            "Requested stock_id_representation_type does not match the checkpoint model configuration. "
            f"checkpoint={checkpoint_model_config.stock_id_representation_type!r} "
            f"requested={requested_model_config.stock_id_representation_type!r}"
        )
    if (
        "stock_embedding_type" in args_dict
        and checkpoint_model_config.stock_embedding_type
        != requested_model_config.stock_embedding_type
    ):
        raise ValueError(
            "Requested stock_embedding_type does not match the checkpoint model configuration. "
            f"checkpoint={checkpoint_model_config.stock_embedding_type!r} "
            f"requested={requested_model_config.stock_embedding_type!r}"
        )
    if (
        "stock_temporal_encoder_type" in args_dict
        and checkpoint_model_config.stock_temporal_encoder_type
        != requested_model_config.stock_temporal_encoder_type
    ):
        raise ValueError(
            "Requested stock_temporal_encoder_type does not match the checkpoint model configuration. "
            f"checkpoint={checkpoint_model_config.stock_temporal_encoder_type!r} "
            f"requested={requested_model_config.stock_temporal_encoder_type!r}"
        )
    if (
        "stock_cross_sectional_encoder_type" in args_dict
        and checkpoint_model_config.stock_cross_sectional_encoder_type
        != requested_model_config.stock_cross_sectional_encoder_type
    ):
        raise ValueError(
            "Requested stock_cross_sectional_encoder_type does not match the checkpoint model configuration. "
            f"checkpoint={checkpoint_model_config.stock_cross_sectional_encoder_type!r} "
            f"requested={requested_model_config.stock_cross_sectional_encoder_type!r}"
        )


def _format_terminal_summary(payload: dict[str, Any]) -> str:
    lines = [f"{key}: {payload[key]}" for key in TERMINAL_OUTPUT_KEYS if key in payload]
    scenario_artifacts = payload.get("scenario_artifacts", [])
    if isinstance(scenario_artifacts, list) and scenario_artifacts:
        lines.append("scenario_results:")
        for item in sorted(
            scenario_artifacts,
            key=lambda artifact: float(artifact["final_return"]),
            reverse=True,
        ):
            lines.append(
                "  "
                f"{item['scenario_id']} | final_return={float(item['final_return']):.8f} | "
                f"backtest_portfolio_sr={float(item['backtest_portfolio_sr']):.8f}"
            )
    return "\n".join(lines)


def _format_refresh_terminal_summary(payloads: list[dict[str, Any]]) -> str:
    lines = [f"refreshed_artifact_count: {len(payloads)}"]
    lines.extend(
        f"refreshed: {payload['state']} | {payload['loss_name']} | scenarios={len(payload.get('scenario_artifacts', []))}"
        for payload in payloads
    )
    return "\n".join(lines)


def _format_monitoring_overview_backfill_terminal_summary(generated_paths: list[str]) -> str:
    lines = [f"rebuild_overview_count: {len(generated_paths)}"]
    lines.extend(f"rebuild: {path}" for path in generated_paths)
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run evaluation for portfolio_attention.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--num-stocks", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--stock-id-representation-type",
        choices=["learning", "gaussian"],
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--stock-embedding-type",
        choices=["concat", "pre_temporal"],
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--stock-temporal-encoder-type",
        choices=["running_summary", "causal_self_attention"],
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--stock-cross-sectional-encoder-type",
        choices=["mlp", "self_attention"],
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--loss",
        default=None,
        choices=["return", "sharpe", "dsr", "sortino", "mdd", "cvar"],
    )
    parser.add_argument(
        "--refresh-existing-scenario-artifacts",
        action="store_true",
        help="Rebuild holdout scenario PNG/JSON/CSV artifacts under outputs/predictions using current checkpoints.",
    )
    parser.add_argument(
        "--backfill-monitoring-holdout-overviews",
        action="store_true",
        help="Rebuild monitoring multi-loss overview PNGs from existing *_holdout_backtest JSON manifests.",
    )
    parser.add_argument(
        "--monitoring-output-dir",
        type=Path,
        action="append",
        default=None,
        help="Specific *_holdout_backtest directory to rebuild. May be provided multiple times.",
    )
    return parser


def resolve_paths_config_from_args(
    args: argparse.Namespace,
    *,
    paths: PathsConfig | None = None,
) -> PathsConfig:
    resolved_paths = paths or PathsConfig()
    args_dict = vars(args)
    if "output_root" in args_dict:
        return PathsConfig(
            project_dir=resolved_paths.project_dir,
            output_root=args_dict["output_root"],
        )
    return resolved_paths


def resolve_data_config_from_args(
    args: argparse.Namespace,
    *,
    data_config: DataConfig | None = None,
) -> DataConfig:
    resolved_data_config = validated_data_config(data_config or DataConfig())
    args_dict = vars(args)
    data_overrides: dict[str, Any] = {}
    if "num_stocks" in args_dict:
        data_overrides["num_stocks"] = args_dict["num_stocks"]
    if data_overrides:
        resolved_data_config = replace(resolved_data_config, **data_overrides)
    return validated_data_config(resolved_data_config)


def resolve_model_config_from_args(
    args: argparse.Namespace,
    *,
    model_config: ModelConfig | None = None,
) -> ModelConfig:
    resolved_model_config = validated_model_config(model_config or ModelConfig())
    args_dict = vars(args)
    model_overrides: dict[str, Any] = {}
    if "stock_id_representation_type" in args_dict:
        model_overrides["stock_id_representation_type"] = args_dict[
            "stock_id_representation_type"
        ]
    if "stock_embedding_type" in args_dict:
        model_overrides["stock_embedding_type"] = args_dict["stock_embedding_type"]
    if "stock_temporal_encoder_type" in args_dict:
        model_overrides["stock_temporal_encoder_type"] = args_dict[
            "stock_temporal_encoder_type"
        ]
    if "stock_cross_sectional_encoder_type" in args_dict:
        model_overrides["stock_cross_sectional_encoder_type"] = args_dict[
            "stock_cross_sectional_encoder_type"
        ]
    if model_overrides:
        resolved_model_config = replace(resolved_model_config, **model_overrides)
    return validated_model_config(resolved_model_config)


def main() -> None:
    args = build_arg_parser().parse_args()
    args_dict = vars(args)
    paths = resolve_paths_config_from_args(args)
    if args.backfill_monitoring_holdout_overviews:
        if args.checkpoint is not None:
            raise ValueError("--checkpoint cannot be used with --backfill-monitoring-holdout-overviews.")
        if args.refresh_existing_scenario_artifacts:
            raise ValueError(
                "--refresh-existing-scenario-artifacts cannot be used with "
                "--backfill-monitoring-holdout-overviews."
            )
        if args.loss is not None:
            raise ValueError("--loss cannot be used with --backfill-monitoring-holdout-overviews.")
        generated_paths = evaluate_rebuild.backfill_monitoring_holdout_backtest_overviews(
            paths=paths,
            output_dirs=args.monitoring_output_dir,
        )
        print(_format_monitoring_overview_backfill_terminal_summary(generated_paths))
        return
    if args.refresh_existing_scenario_artifacts:
        if args.checkpoint is not None:
            raise ValueError("--checkpoint cannot be used with --refresh-existing-scenario-artifacts.")
        if args.monitoring_output_dir:
            raise ValueError(
                "--monitoring-output-dir cannot be used with --refresh-existing-scenario-artifacts."
            )
        payloads = evaluate_rebuild.refresh_existing_scenario_artifacts(
            paths=paths,
            run_evaluation_fn=evaluation_pipeline.run_evaluation,
            device_name=args.device,
            loss_name=args.loss,
        )
        print(_format_refresh_terminal_summary(payloads))
        return
    if args.monitoring_output_dir:
        raise ValueError("--monitoring-output-dir requires --backfill-monitoring-holdout-overviews.")
    requested_data_config = resolve_data_config_from_args(args)
    requested_model_config = resolve_model_config_from_args(args)
    resolved_checkpoint = _resolve_checkpoint_path(
        paths=paths,
        data_config=requested_data_config,
        checkpoint_path=args.checkpoint,
        loss_name=args.loss,
    )
    checkpoint = torch.load(resolved_checkpoint, map_location="cpu", weights_only=False)
    _validate_requested_runtime_configs_against_checkpoint(
        requested_data_config=requested_data_config,
        requested_model_config=requested_model_config,
        checkpoint=checkpoint,
        args_dict=args_dict,
    )
    payload = evaluation_pipeline.run_evaluation(
        data_config=requested_data_config,
        paths=paths,
        checkpoint_path=resolved_checkpoint,
        device_name=args.device,
        loss_name=args.loss,
    )
    print(_format_terminal_summary(payload))


if __name__ == "__main__":
    main()
