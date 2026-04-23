from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from export_best_epoch_comparison import (
    LOSS_NAMES,
    _build_summary_rows,
    _scenario_metrics_from_manifest,
)


def test_scenario_metrics_from_manifest_includes_selected_stock_count() -> None:
    temp_dir = Path.cwd() / ".pytest_fixture_tmp" / f"manifest-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        manifest_path = temp_dir / "sharpe_monitoring_holdout_backtest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "scenario_artifacts": [
                        {
                            "scenario_id": "neutral_4860_200_PL_3",
                            "final_return": 0.25,
                            "backtest_portfolio_sr": 0.5,
                            "total_selected_stock_count": 860,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        rows = _scenario_metrics_from_manifest(manifest_path)

        assert rows == [
            {
                "scenario_id": "3",
                "portfolio_return": 0.25,
                "portfolio_sr": 0.5,
                "selected_stock_count": 860.0,
            }
        ]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_build_summary_rows_adds_mean_stocks_after_mean_sr() -> None:
    def loss_rows(start: int) -> dict[str, dict[str, float | str]]:
        return {
            "3": {
                "scenario_id": "3",
                "portfolio_return": 0.1,
                "portfolio_sr": 0.2,
                "selected_stock_count": float(start),
            },
            "4": {
                "scenario_id": "4",
                "portfolio_return": 0.3,
                "portfolio_sr": 0.4,
                "selected_stock_count": float(start + 20),
            },
        }

    best_by_state_1 = {
        state: {
            "epoch": 10,
            "per_loss_scenarios": {
                loss_name: loss_rows((index + 1) * 100)
                for index, loss_name in enumerate(LOSS_NAMES)
            },
        }
        for state in ("bear", "neutral", "bull")
    }
    best_by_state_2 = {
        state: {
            "epoch": 20,
            "per_loss_scenarios": {
                loss_name: loss_rows((index + 1) * 200)
                for index, loss_name in enumerate(LOSS_NAMES)
            },
        }
        for state in ("bear", "neutral", "bull")
    }

    rows = _build_summary_rows(
        best_by_state_1,
        best_by_state_2,
        label_1="mlp v8",
        label_2="mlp v9",
    )

    first_row = rows[0]
    keys = list(first_row.keys())
    sharpe_sr_index = keys.index("mlp v8_sharpe_mean_sr")
    sharpe_stocks_index = keys.index("mlp v8_sharpe_mean_stocks")

    assert sharpe_stocks_index == sharpe_sr_index + 2
    assert first_row["mlp v8_sharpe_mean_stocks"] == 310.0
    assert first_row["mlp v9_sharpe_mean_stocks"] == 610.0
