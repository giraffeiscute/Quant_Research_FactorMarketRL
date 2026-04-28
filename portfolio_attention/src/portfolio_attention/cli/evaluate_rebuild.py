"""Compatibility facade for evaluation artifact rebuild helpers."""

from __future__ import annotations

import argparse

from portfolio_attention.evaluation.rebuild import (
    _build_refresh_data_config,
    backfill_monitoring_holdout_backtest_overviews,
    cleanup_monitoring_holdout_backtest_artifacts,
    cleanup_multi_loss_weight_trajectory_overviews,
    rebuild_monitoring_holdout_backtest_overviews,
    rebuild_multi_loss_weight_trajectory_overviews,
    refresh_existing_scenario_artifacts,
)

__all__ = [
    "_build_refresh_data_config",
    "backfill_monitoring_holdout_backtest_overviews",
    "cleanup_monitoring_holdout_backtest_artifacts",
    "cleanup_multi_loss_weight_trajectory_overviews",
    "rebuild_monitoring_holdout_backtest_overviews",
    "rebuild_multi_loss_weight_trajectory_overviews",
    "refresh_existing_scenario_artifacts",
    "build_arg_parser",
    "main",
]


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Evaluation rebuild helper entrypoint for portfolio_attention."
    )


def main() -> None:
    build_arg_parser().parse_args()


if __name__ == "__main__":
    main()
