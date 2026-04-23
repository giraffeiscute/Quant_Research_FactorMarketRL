from __future__ import annotations

import torch

from portfolio_attention.evaluation_types import RuntimePayloadAdapter


def test_runtime_payload_adapter_round_trip_preserves_runtime_tensors() -> None:
    payload = {
        "scenario_id": "PL_1",
        "source_path": "dummy.parquet",
        "loss_name": "sharpe",
        "state": "bull",
        "evaluation_split": "holdout_test",
        "train_config": {"num_epochs": 3},
        "final_return": 0.12,
        "backtest_portfolio_sr": 1.23,
        "mean_step_return": 0.01,
        "std_step_return": 0.03,
        "final_cash_weight": 0.1,
        "mean_cash_weight": 0.12,
        "num_time_steps": 2,
        "scored_num_time_steps": 2,
        "context_num_time_steps": 3,
        "warmup_time_steps": 1,
        "analysis_time_index": 42,
        "feature_time_start_index": 10,
        "feature_time_end_index": 11,
        "target_time_start_index": 11,
        "target_time_end_index": 12,
        "scored_feature_time_start_index": 10,
        "scored_feature_time_end_index": 11,
        "scored_target_time_start_index": 11,
        "scored_target_time_end_index": 12,
        "context_feature_time_start_index": 9,
        "context_feature_time_end_index": 11,
        "context_target_time_start_index": 10,
        "context_target_time_end_index": 12,
        "top_k_stock_weights": [{"rank": 1, "stock_id": "A", "weight": 0.5}],
        "allocation_group_top_n": 5,
        "stock_count_weight_threshold": 0.01,
        "stock_count_min_active_days": 1,
        "effective_stock_count_min_active_days": 1,
        "stock_count_lookback_days": 2,
        "total_selected_stock_count": 3,
        "benchmark_market_index_csv": None,
        "benchmark_excess_return": 0.02,
        "benchmark_information_ratio": 0.6,
        "benchmark_excess_max_drawdown": -0.1,
        "_final_stock_weights_tensor": torch.tensor([0.3, 0.7]),
        "_stock_weights_tensor": torch.tensor([[0.3, 0.7], [0.4, 0.6]]),
        "_cash_weights_tensor": torch.tensor([0.1, 0.2]),
        "_portfolio_returns_tensor": torch.tensor([0.01, 0.02]),
        "_target_time_indices_tensor": torch.tensor([11, 12]),
    }

    parsed = RuntimePayloadAdapter.from_legacy_payload(payload, require_runtime_tensors=True)
    serialized = RuntimePayloadAdapter.to_legacy_payload(parsed, include_runtime_tensors=True)

    assert serialized["scenario_id"] == payload["scenario_id"]
    assert serialized["loss_name"] == payload["loss_name"]
    assert torch.equal(serialized["_stock_weights_tensor"], payload["_stock_weights_tensor"])
    assert torch.equal(serialized["_cash_weights_tensor"], payload["_cash_weights_tensor"])
    assert torch.equal(
        serialized["_target_time_indices_tensor"],
        payload["_target_time_indices_tensor"],
    )


def test_runtime_payload_adapter_strip_runtime_fields() -> None:
    payloads = [
        {
            "scenario_id": "PL_1",
            "_stock_weights_tensor": torch.tensor([[0.1]]),
            "_cash_weights_tensor": torch.tensor([0.2]),
        }
    ]
    RuntimePayloadAdapter.strip_runtime_fields(payloads)
    assert "_stock_weights_tensor" not in payloads[0]
    assert "_cash_weights_tensor" not in payloads[0]

