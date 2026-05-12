"""Canonical evaluation CLI entrypoint."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from portfolio_attention.config import (
    DataConfig,
    EvaluationConfig,
    ModelConfig,
    PathsConfig,
    load_experiment_config,
)
from portfolio_attention.config.validation import (
    validated_data_config,
    validated_evaluation_config,
    validated_model_config,
)
from portfolio_attention.evaluation import (
    artifacts as evaluation_artifacts,
    checkpoints as evaluation_checkpoints,
    monitoring as evaluation_monitoring,
    pipeline as evaluation_pipeline,
    presentation as evaluation_presentation,
    rebuild as evaluation_rebuild,
    runtime as evaluation_runtime,
    shared as evaluation_shared,
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
_resolve_checkpoint_state = evaluation_checkpoints._resolve_checkpoint_state
_resolve_checkpoint_path = evaluation_checkpoints._resolve_checkpoint_path
_resolve_checkpoint_metadata_dict = evaluation_checkpoints._resolve_checkpoint_metadata_dict
_build_model_config_from_checkpoint = evaluation_checkpoints._build_model_config_from_checkpoint
_build_data_config_from_checkpoint = evaluation_checkpoints._build_data_config_from_checkpoint
_validate_requested_runtime_configs_against_checkpoint = (
    evaluation_checkpoints._validate_requested_runtime_configs_against_checkpoint
)
rebuild_monitoring_holdout_backtest_overviews = (
    evaluation_rebuild.rebuild_monitoring_holdout_backtest_overviews
)
run_monitoring_holdout_backtest = evaluation_monitoring.run_monitoring_holdout_backtest


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
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--sample-num-stocks", type=int, default=argparse.SUPPRESS)
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
        "--inference-allocation-mode",
        choices=["softmax", "dirichlet_mean"],
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--loss",
        default=None,
        choices=["return", "sharpe", "sortino", "mdd", "cvar"],
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


def _experiment_config_from_args(args: argparse.Namespace):
    return load_experiment_config(vars(args).get("config"))


def resolve_paths_config_from_args(
    args: argparse.Namespace,
    *,
    paths: PathsConfig | None = None,
) -> PathsConfig:
    resolved_paths = paths or _experiment_config_from_args(args).paths
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
    resolved_data_config = validated_data_config(
        data_config or _experiment_config_from_args(args).data
    )
    args_dict = vars(args)
    data_overrides: dict[str, Any] = {}
    if "sample_num_stocks" in args_dict:
        data_overrides["sample_num_stocks"] = args_dict["sample_num_stocks"]
    if data_overrides:
        resolved_data_config = replace(resolved_data_config, **data_overrides)
    return validated_data_config(resolved_data_config)


def resolve_model_config_from_args(
    args: argparse.Namespace,
    *,
    model_config: ModelConfig | None = None,
) -> ModelConfig:
    resolved_model_config = validated_model_config(
        model_config or _experiment_config_from_args(args).model
    )
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
    if "inference_allocation_mode" in args_dict:
        model_overrides["inference_allocation_mode"] = args_dict["inference_allocation_mode"]
    if model_overrides:
        resolved_model_config = replace(resolved_model_config, **model_overrides)
    return validated_model_config(resolved_model_config)


def resolve_evaluation_config_from_args(
    args: argparse.Namespace,
    *,
    evaluation_config: EvaluationConfig | None = None,
) -> EvaluationConfig:
    if evaluation_config is not None:
        return validated_evaluation_config(evaluation_config)
    return _experiment_config_from_args(args).evaluation


def main() -> None:
    args = build_arg_parser().parse_args()
    args_dict = vars(args)
    paths = resolve_paths_config_from_args(args)
    evaluation_config = resolve_evaluation_config_from_args(args)
    loss_name = args.loss or _experiment_config_from_args(args).train.loss_name or None
    if args.backfill_monitoring_holdout_overviews:
        if args.checkpoint is not None:
            raise ValueError("--checkpoint cannot be used with --backfill-monitoring-holdout-overviews.")
        if args.refresh_existing_scenario_artifacts:
            raise ValueError(
                "--refresh-existing-scenario-artifacts cannot be used with "
                "--backfill-monitoring-holdout-overviews."
            )
        if loss_name is not None:
            raise ValueError("--loss cannot be used with --backfill-monitoring-holdout-overviews.")
        generated_paths = evaluation_rebuild.backfill_monitoring_holdout_backtest_overviews(
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
        payloads = evaluation_rebuild.refresh_existing_scenario_artifacts(
            paths=paths,
            run_evaluation_fn=evaluation_pipeline.run_evaluation,
            device_name=args.device,
            loss_name=loss_name,
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
        loss_name=loss_name,
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
        evaluation_config=evaluation_config,
        loss_name=loss_name,
    )
    print(_format_terminal_summary(payload))


if __name__ == "__main__":
    main()
