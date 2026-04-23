from __future__ import annotations

import pytest

from portfolio_attention import run_metadata


def test_update_monitoring_manifest_overview_paths_matches_existing_behavior() -> None:
    manifest_payload = {
        "scenario_artifacts": [
            {"scenario_id": "s1", "weight_trajectory_overview_chart": None},
            {"scenario_id": "s2", "weight_trajectory_overview_chart": "old"},
        ],
        "overview_loss_order": ["sharpe"],
        "holdout_backtest_overview_paths": [],
    }

    changed = run_metadata.update_monitoring_manifest_overview_paths(
        manifest_payload,
        overview_path_by_scenario={"s1": "/tmp/s1.png"},
        loss_order=("mdd", "return", "sharpe", "sortino"),
    )

    assert changed is True
    assert manifest_payload["overview_loss_order"] == ["mdd", "return", "sharpe", "sortino"]
    assert manifest_payload["scenario_artifacts"][0]["weight_trajectory_overview_chart"] == "/tmp/s1.png"
    assert manifest_payload["scenario_artifacts"][1]["weight_trajectory_overview_chart"] is None
    assert manifest_payload["holdout_backtest_overview_paths"] == ["/tmp/s1.png"]

    changed_again = run_metadata.update_monitoring_manifest_overview_paths(
        manifest_payload,
        overview_path_by_scenario={"s1": "/tmp/s1.png"},
        loss_order=("mdd", "return", "sharpe", "sortino"),
    )
    assert changed_again is False


def test_validate_manifest_scenario_artifacts_strict_errors() -> None:
    with pytest.raises(ValueError, match="scenario_artifacts list"):
        run_metadata.validate_manifest_scenario_artifacts({})
    with pytest.raises(ValueError, match="entries must be objects"):
        run_metadata.validate_manifest_scenario_artifacts({"scenario_artifacts": ["not-dict"]})


def test_update_train_metrics_history_overview_paths_and_state_resolution() -> None:
    payload = {
        "final_backtest": {"state": "bear"},
        "history": [
            {
                "epoch": 5,
                "holdout_backtest_output_dir": "outputs/predictions/bear/5_holdout_backtest",
                "holdout_backtest_overview_paths": [],
            }
        ],
    }

    changed = run_metadata.update_train_metrics_history_overview_paths(
        payload,
        state="bear",
        epoch=5,
        holdout_backtest_output_dir="outputs/predictions/bear/5_holdout_backtest",
        overview_paths=["/tmp/overview_1.png"],
    )
    assert changed is True
    assert payload["history"][0]["holdout_backtest_overview_paths"] == ["/tmp/overview_1.png"]

    unchanged = run_metadata.update_train_metrics_history_overview_paths(
        payload,
        state="bull",
        epoch=5,
        holdout_backtest_output_dir="outputs/predictions/bear/5_holdout_backtest",
        overview_paths=["/tmp/overview_2.png"],
    )
    assert unchanged is False
    assert payload["history"][0]["holdout_backtest_overview_paths"] == ["/tmp/overview_1.png"]


def test_update_payload_overview_paths_updates_nested_final_backtest() -> None:
    payload = {
        "state": "neutral",
        "scenario_artifacts": [
            {"scenario_id": "s1", "weight_trajectory_overview_chart": None},
            {"scenario_id": "s2", "weight_trajectory_overview_chart": None},
        ],
        "final_backtest": {
            "scenario_artifacts": [
                {"scenario_id": "s1", "weight_trajectory_overview_chart": None},
                {"scenario_id": "s2", "weight_trajectory_overview_chart": None},
            ]
        },
    }

    changed = run_metadata.update_payload_overview_paths(
        payload,
        state="neutral",
        scenario_id="s2",
        overview_path="/tmp/s2.png",
    )
    assert changed is True
    assert payload["scenario_artifacts"][0]["weight_trajectory_overview_chart"] is None
    assert payload["scenario_artifacts"][1]["weight_trajectory_overview_chart"] == "/tmp/s2.png"
    assert payload["final_backtest"]["scenario_artifacts"][0]["weight_trajectory_overview_chart"] is None
    assert payload["final_backtest"]["scenario_artifacts"][1]["weight_trajectory_overview_chart"] == "/tmp/s2.png"


def test_resume_history_item_match_uses_shared_float_and_exact_keys() -> None:
    history_item = {
        "train_loss": 1.0,
        "train_mean_final_return": 0.2,
        "val_loss": 0.5,
        "val_mean_final_return": 0.3,
        "holdout_backtest_ran": True,
        "holdout_backtest_epoch": 7,
        "holdout_backtest_output_dir": "some_dir",
        "monitoring_checkpoint_path": "ckpt.pt",
    }
    checkpoint_metrics = {
        "train_loss": 1.0 + 1e-12,
        "train_mean_final_return": 0.2 + 1e-12,
        "val_loss": 0.5 + 1e-12,
        "val_mean_final_return": 0.3 + 1e-12,
        "holdout_backtest_ran": True,
        "holdout_backtest_epoch": 7,
        "holdout_backtest_output_dir": "some_dir",
        "monitoring_checkpoint_path": "ckpt.pt",
    }

    assert run_metadata.resume_history_item_matches_checkpoint(
        history_item,
        checkpoint_metrics,
        history_epoch=7,
    )

    checkpoint_metrics["monitoring_checkpoint_path"] = "other.pt"
    assert not run_metadata.resume_history_item_matches_checkpoint(
        history_item,
        checkpoint_metrics,
        history_epoch=7,
    )

    assert run_metadata.resume_history_item_matches_checkpoint({}, {"epoch": 3}, history_epoch=3)
    assert not run_metadata.resume_history_item_matches_checkpoint({}, {"epoch": 3}, history_epoch=4)


def test_epoch_metrics_mutation_helpers() -> None:
    metrics = run_metadata.create_epoch_metrics(
        epoch=9,
        train_loss=0.1,
        train_mean_final_return=0.2,
        val_loss=0.3,
        val_mean_final_return=0.4,
        validation_epoch_metadata={"validation_num_rolling_windows_total": 42},
    )
    assert metrics["epoch"] == 9
    assert metrics["holdout_backtest_ran"] is False
    assert metrics["monitoring_checkpoint_path"] is None

    run_metadata.apply_monitoring_backtest_to_epoch_metrics(
        metrics,
        epoch=9,
        monitoring_backtest={
            "holdout_backtest_loss": 0.9,
            "mean_final_return": 0.11,
            "std_final_return": 0.12,
            "median_final_return": 0.13,
            "worst_scenario_final_return": -0.2,
            "best_scenario_final_return": 0.8,
            "best_scenario_id": "s9",
            "holdout_backtest_output_dir": "dir_9",
            "holdout_backtest_overview_paths": ["o1.png"],
        },
    )
    run_metadata.inject_best_state_fields(
        metrics,
        current_window_best_epoch=9,
        current_window_best_val_loss=0.3,
        global_best_val_loss=0.25,
        global_best_checkpoint_updated=True,
        epochs_without_improvement=0,
    )
    run_metadata.set_monitoring_checkpoint_path(metrics, "monitoring_9.pt")

    assert metrics["holdout_backtest_ran"] is True
    assert metrics["holdout_backtest_overview_paths"] == ["o1.png"]
    assert metrics["current_window_best_epoch"] == 9
    assert metrics["global_best_checkpoint_updated"] is True
    assert metrics["monitoring_checkpoint_path"] == "monitoring_9.pt"
