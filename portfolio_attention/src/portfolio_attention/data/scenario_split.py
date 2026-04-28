"""Scenario file discovery, sorting, and train/validation/test splitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import DataConfig
from .parsing import parse_scenario_file_info
from .records import ScenarioFileRecord


def scenario_sort_key(path: Path) -> tuple[str, int]:
    info = parse_scenario_file_info(path.name)
    return str(info["state"]), int(info["scenario_index"])


def discover_scenario_records(
    *,
    scenario_dir: Path,
    scenario_glob: str,
    state: str,
) -> list[ScenarioFileRecord]:
    matched_paths = sorted(
        scenario_dir.glob(scenario_glob),
        key=scenario_sort_key,
    )
    records: list[ScenarioFileRecord] = []
    for path in matched_paths:
        info = parse_scenario_file_info(path.name)
        if str(info["state"]) != state:
            continue
        records.append(
            ScenarioFileRecord(
                scenario_id=path.stem,
                source_path=str(path),
                state=str(info["state"]),
                scenario_index=int(info["scenario_index"]),
                parsed_n=int(info["parsed_n"]),
                parsed_t=int(info["parsed_t"]),
            )
        )
    return records


def split_scenario_records(
    records: list[ScenarioFileRecord],
    config: DataConfig,
) -> tuple[
    list[ScenarioFileRecord],
    list[ScenarioFileRecord],
    list[ScenarioFileRecord],
    list[ScenarioFileRecord],
]:
    split_records = list(records)
    if config.shuffle_scenario_splits:
        generator = np.random.default_rng(int(config.scenario_split_seed))
        permutation = generator.permutation(len(split_records))
        split_records = [split_records[int(index)] for index in permutation.tolist()]

    train_records = split_records[: config.num_train_scenarios]
    val_start = config.num_train_scenarios
    val_end = val_start + config.num_validation_scenarios
    validation_records = split_records[val_start:val_end]
    test_records = split_records[val_end:]
    return split_records, train_records, validation_records, test_records
