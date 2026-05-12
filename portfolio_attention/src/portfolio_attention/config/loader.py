"""Sparse YAML experiment config loading."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

from .schema import DataConfig, EvaluationConfig, ModelConfig, PathsConfig, TrainConfig
from .validation import (
    validate_train_config_against_data_config,
    validated_data_config,
    validated_evaluation_config,
    validated_model_config,
    validated_train_config,
)


@dataclass
class ExecutionConfig:
    states: list[str] | None = None
    losses: list[str] | None = None
    accelerator: str | None = None
    lightning_devices: int | None = None
    num_workers: int | None = None


@dataclass
class ExperimentConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)


_SECTION_TYPES = {
    "paths": PathsConfig,
    "data": DataConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "evaluation": EvaluationConfig,
    "execution": ExecutionConfig,
}
_PATH_FIELDS = {
    "paths": {"project_dir", "output_root"},
    "data": {"scenario_dir"},
    "train": {"resume_from"},
}


def _legal_keys(section: str) -> list[str]:
    return sorted(field_info.name for field_info in fields(_SECTION_TYPES[section]))


def _raise_unknown_section(section: str) -> None:
    raise ValueError(
        f"Unknown config section {section!r}. "
        f"Legal sections: {sorted(_SECTION_TYPES)}."
    )


def _raise_unknown_key(section: str, key: str) -> None:
    raise ValueError(
        f"Unknown field {key!r} in section {section!r}. "
        f"Legal keys for section {section!r}: {_legal_keys(section)}."
    )


def _load_yaml_mapping(yaml_path: str | Path | None) -> dict[str, Any]:
    if yaml_path is None:
        return {}

    path = Path(yaml_path)
    raw_payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw_payload is None:
        return {}
    if not isinstance(raw_payload, dict):
        raise ValueError(
            "Experiment config YAML must contain a mapping at the document root. "
            f"Legal sections: {sorted(_SECTION_TYPES)}."
        )
    return raw_payload


def _normalize_section_overrides(section: str, payload: object) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(
            f"Config section {section!r} must be a mapping. "
            f"Legal keys for section {section!r}: {_legal_keys(section)}."
        )

    legal_keys = set(_legal_keys(section))
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in legal_keys:
            _raise_unknown_key(section, str(key))
        if key in _PATH_FIELDS.get(section, set()) and value is not None:
            value = Path(value)
        normalized[key] = value
    return normalized


def load_experiment_config(yaml_path: str | Path | None = None) -> ExperimentConfig:
    """Load schema defaults plus sparse YAML overrides, then validate sections."""

    cfg = ExperimentConfig()
    payload = _load_yaml_mapping(yaml_path)

    for section, section_payload in payload.items():
        if section not in _SECTION_TYPES:
            _raise_unknown_section(str(section))
        section_config = getattr(cfg, section)
        overrides = _normalize_section_overrides(str(section), section_payload)
        if overrides:
            cfg = replace(cfg, **{section: replace(section_config, **overrides)})

    if cfg.paths.project_dir is not None:
        cfg.paths.project_dir = Path(cfg.paths.project_dir)
    if cfg.paths.output_root is not None:
        cfg.paths.output_root = Path(cfg.paths.output_root)

    validated_data = validated_data_config(cfg.data)
    validated_train = validated_train_config(cfg.train)
    validate_train_config_against_data_config(validated_train, validated_data)

    return ExperimentConfig(
        paths=cfg.paths,
        data=validated_data,
        model=validated_model_config(cfg.model),
        train=validated_train,
        evaluation=validated_evaluation_config(cfg.evaluation),
        execution=cfg.execution,
    )
