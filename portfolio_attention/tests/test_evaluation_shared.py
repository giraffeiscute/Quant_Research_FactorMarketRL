from __future__ import annotations

from pathlib import Path

import torch

from portfolio_attention import evaluation_shared


def test_weight_trajectory_export_data_round_trip() -> None:
    grouped = [
        {"label": "GroupA", "weights": torch.tensor([0.1, 0.2, 0.3])},
        {"label": "Cash", "weights": torch.tensor([0.9, 0.8, 0.7])},
    ]
    target_time_indices = torch.tensor([100, 101, 102], dtype=torch.int64)
    payload = evaluation_shared.build_weight_trajectory_export_data(
        grouped_weight_trajectories=grouped,
        target_time_indices=target_time_indices,
    )
    loaded_grouped, loaded_indices = evaluation_shared.load_weight_trajectory_export_data(payload)

    assert [item["label"] for item in loaded_grouped] == ["GroupA", "Cash"]
    assert torch.equal(loaded_indices, target_time_indices)


def test_persisted_artifact_loader_day_weight_payload() -> None:
    path = Path("portfolio_attention/tests/.tmp_day_weights.pt")
    expected = {
        "artifact_type": "holdout_scenario_day_weights",
        "target_time_indices": torch.tensor([1, 2]),
    }
    try:
        torch.save(expected, path)
        loaded = evaluation_shared.PersistedArtifactLoader.load_day_weight_artifact(path)
        assert loaded["artifact_type"] == "holdout_scenario_day_weights"
        assert torch.equal(loaded["target_time_indices"], torch.tensor([1, 2]))
    finally:
        path.unlink(missing_ok=True)
