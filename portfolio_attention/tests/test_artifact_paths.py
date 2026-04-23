from __future__ import annotations

from pathlib import Path

from portfolio_attention import artifact_paths
from portfolio_attention.config import PathsConfig


def test_metrics_paths_match_existing_rules() -> None:
    paths = PathsConfig(output_root=Path("synthetic_outputs"))

    assert artifact_paths.train_metrics_path(paths, "sharpe") == paths.metrics_dir / "train_metrics_sharpe.json"
    assert artifact_paths.train_metrics_path(paths, "sharpe", state="bear") == (
        paths.get_state_metrics_dir("bear") / "train_metrics_sharpe.json"
    )
    assert artifact_paths.evaluation_metrics_path(paths, "mdd", state="neutral") == (
        paths.get_state_metrics_dir("neutral") / "evaluation_metrics_mdd.json"
    )
    assert artifact_paths.evaluation_per_scenario_metrics_path(paths, "return", state="bull") == (
        paths.get_state_metrics_dir("bull") / "evaluation_metrics_return_per_scenario.csv"
    )


def test_candidate_train_metrics_paths_keep_fallback_order() -> None:
    paths = PathsConfig(output_root=Path("synthetic_outputs"))
    checkpoint_path = paths.checkpoints_dir / "bear_train_last_sharpe.pt"

    candidates = artifact_paths.candidate_train_metrics_paths(
        paths,
        "sharpe",
        state="bear",
        checkpoint_path=checkpoint_path,
    )

    sibling_metrics_dir = checkpoint_path.parent.parent / "metrics"
    assert candidates == [
        paths.get_state_metrics_dir("bear") / "train_metrics_sharpe.json",
        paths.metrics_dir / "train_metrics_sharpe.json",
        sibling_metrics_dir / "bear" / "train_metrics_sharpe.json",
        sibling_metrics_dir / "train_metrics_sharpe.json",
    ]


def test_checkpoint_manifest_and_overview_paths_match_existing_rules() -> None:
    paths = PathsConfig(output_root=Path("synthetic_outputs"))
    output_dir = paths.get_state_predictions_dir("bear") / "50_holdout_backtest"

    assert artifact_paths.train_best_checkpoint_name("sharpe", state="bear") == "bear_train_best_sharpe.pt"
    assert artifact_paths.train_last_checkpoint_name("sharpe", state="bear") == "bear_train_last_sharpe.pt"
    assert artifact_paths.train_best_checkpoint_name("", state="bear") == "bear_train_best.pt"
    assert artifact_paths.train_last_checkpoint_name("", state="bear") == "bear_train_last.pt"
    assert artifact_paths.train_best_checkpoint_path(paths, "sharpe", state="bear") == (
        paths.checkpoints_dir / "bear_train_best_sharpe.pt"
    )
    assert artifact_paths.train_last_checkpoint_path(paths, "sharpe", state="bear") == (
        paths.checkpoints_dir / "bear_train_last_sharpe.pt"
    )
    assert artifact_paths.epoch_candidate_checkpoint_path(paths, "sharpe", 7) == (
        paths.checkpoints_dir / "train_candidate_sharpe_epoch_7.pt"
    )
    assert artifact_paths.monitoring_epoch_checkpoint_path(paths, "sharpe", 7, state="bear") == (
        paths.checkpoints_dir / "bear_train_monitoring_sharpe_epoch_7.pt"
    )
    assert artifact_paths.monitoring_epoch_checkpoint_path(paths, "sharpe", 7) == (
        paths.checkpoints_dir / "train_monitoring_sharpe_epoch_7.pt"
    )
    assert artifact_paths.monitoring_manifest_path(output_dir, "sharpe") == (
        output_dir / "sharpe_monitoring_holdout_backtest.json"
    )
    assert artifact_paths.monitoring_overview_path(output_dir, "bear_4860_200_PL_1") == (
        output_dir / "bear_4860_200_PL_1_weight_trajectory_overview.png"
    )
