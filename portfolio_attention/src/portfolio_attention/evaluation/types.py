"""Typed evaluation payload models and adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, TypedDict
from typing_extensions import NotRequired

import torch

TRANSIENT_SCENARIO_TENSOR_FIELDS = (
    "_final_stock_weights_tensor",
    "_stock_weights_tensor",
    "_cash_weights_tensor",
    "_portfolio_returns_tensor",
    "_target_time_indices_tensor",
)


class WeightTrajectorySeriesEntry(TypedDict):
    label: str
    weights: list[float]


class WeightTrajectoryExportData(TypedDict):
    reference_day: int
    target_time_indices: list[int]
    series: list[WeightTrajectorySeriesEntry]


class ScenarioExportArtifact(TypedDict):
    artifact_type: str
    scenario_id: str
    source_path: str
    loss_name: str
    state: str
    evaluation_split: str
    final_return: float
    backtest_portfolio_sr: float
    weight_trajectory_data: WeightTrajectoryExportData
    weight_trajectory_overview_chart: str | None
    day_weight_artifact: str
    train_config: dict[str, object]
    checkpoint_path: str
    all_stock_weights: list[dict[str, object]]
    grouped_allocations: list[dict[str, object]]
    grouped_allocations_top_n: list[dict[str, object]]
    allocation_group_top_n: int
    top_k_stock_weights: NotRequired[list[dict[str, object]]]
    benchmark_market_index_csv: NotRequired[str | None]
    benchmark_excess_return: NotRequired[float | None]
    benchmark_information_ratio: NotRequired[float | None]
    average_turnover: NotRequired[float]
    mean_cash_weight: NotRequired[float]


class MonitoringScenarioArtifact(TypedDict):
    scenario_index: int
    scenario_id: str
    final_return: float
    backtest_portfolio_sr: float
    metrics_text: str
    weight_trajectory_data: WeightTrajectoryExportData
    weight_trajectory_chart: str | None
    weight_trajectory_overview_chart: str | None
    benchmark_market_index_csv: NotRequired[str | None]
    benchmark_excess_return: NotRequired[float | None]
    benchmark_information_ratio: NotRequired[float | None]
    average_turnover: NotRequired[float]
    mean_cash_weight: NotRequired[float]
    evaluation_mode: NotRequired[str | None]
    rolling_window_lookback_days: NotRequired[int | None]
    rolling_window_horizon_days: NotRequired[int | None]
    rolling_window_stride_days: NotRequired[int | None]
    num_rolling_windows: NotRequired[int | None]
    total_selected_stock_count: NotRequired[int]
    stock_count_weight_threshold: NotRequired[float]


class HoldoutSummary(TypedDict):
    state: str
    loss_name: str
    evaluation_split: str
    num_holdout_scenarios: int
    mean_final_return: float
    mean_average_turnover: float
    mean_cash_weight: float
    std_final_return: float
    median_final_return: float
    worst_scenario_final_return: float
    best_scenario_final_return: float
    best_scenario_id: str
    worst_scenario_id: str
    best_scenario_source_path: str
    worst_scenario_source_path: str
    evaluation_mode: NotRequired[str]
    rolling_window_lookback_days: NotRequired[int]
    rolling_window_horizon_days: NotRequired[int]
    rolling_window_stride_days: NotRequired[int]
    num_rolling_windows: NotRequired[int]
    evaluation_price_anchor_mode: NotRequired[str]


@dataclass
class ScenarioIdentity:
    scenario_id: str
    source_path: str
    loss_name: str
    state: str
    evaluation_split: str


@dataclass
class ScenarioWindowMeta:
    num_time_steps: int
    scored_num_time_steps: int
    context_num_time_steps: int
    warmup_time_steps: int
    analysis_time_index: int
    feature_time_start_index: int
    feature_time_end_index: int
    target_time_start_index: int
    target_time_end_index: int
    scored_feature_time_start_index: int
    scored_feature_time_end_index: int
    scored_target_time_start_index: int
    scored_target_time_end_index: int
    context_feature_time_start_index: int
    context_feature_time_end_index: int
    context_target_time_start_index: int
    context_target_time_end_index: int
    evaluation_mode: str | None = None
    rolling_window_lookback_days: int | None = None
    rolling_window_horizon_days: int | None = None
    rolling_window_stride_days: int | None = None
    num_rolling_windows: int | None = None
    evaluation_price_anchor_mode: str | None = None


@dataclass
class ScenarioReturnStats:
    final_return: float
    backtest_portfolio_sr: float
    mean_step_return: float
    std_step_return: float
    average_turnover: float
    sharpe_like: float | None = None


@dataclass
class ScenarioCashStats:
    final_cash_weight: float
    mean_cash_weight: float


@dataclass
class ScenarioSelectionStats:
    allocation_group_top_n: int
    stock_count_weight_threshold: float
    stock_count_min_active_days: int
    effective_stock_count_min_active_days: int
    stock_count_lookback_days: int
    total_selected_stock_count: int


@dataclass
class BenchmarkMetrics:
    benchmark_market_index_csv: str | None
    benchmark_excess_return: float | None
    benchmark_information_ratio: float | None


@dataclass
class ScenarioRuntimeTensors:
    final_stock_weights: torch.Tensor | None
    stock_weights: torch.Tensor | None
    cash_weights: torch.Tensor | None
    portfolio_returns: torch.Tensor | None
    target_time_indices: torch.Tensor | None


@dataclass
class RollingScenarioOutputs:
    scenario_id: str
    source_path: str
    portfolio_returns: torch.Tensor
    turnover: torch.Tensor
    scored_target_time_indices: torch.Tensor
    context_target_time_indices: torch.Tensor
    lookback_days: int
    context_time_steps: int
    num_rolling_windows: int
    evaluation_price_anchor_mode: str
    stock_weights: torch.Tensor | None = None
    cash_weights: torch.Tensor | None = None
    previous_allocation: torch.Tensor | None = None

    def to_legacy_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "scenario_id": self.scenario_id,
            "source_path": Path(self.source_path),
            "portfolio_returns": self.portfolio_returns,
            "turnover": self.turnover,
            "scored_target_time_indices": self.scored_target_time_indices,
            "context_target_time_indices": self.context_target_time_indices,
            "lookback_days": int(self.lookback_days),
            "context_time_steps": int(self.context_time_steps),
            "num_rolling_windows": int(self.num_rolling_windows),
            "evaluation_price_anchor_mode": self.evaluation_price_anchor_mode,
        }
        if self.stock_weights is not None:
            payload["stock_weights"] = self.stock_weights
        if self.cash_weights is not None:
            payload["cash_weights"] = self.cash_weights
        if self.previous_allocation is not None:
            payload["previous_allocation"] = self.previous_allocation
        return payload


@dataclass
class ScenarioEvalResult:
    identity: ScenarioIdentity
    window_meta: ScenarioWindowMeta
    return_stats: ScenarioReturnStats
    cash_stats: ScenarioCashStats
    selection_stats: ScenarioSelectionStats
    benchmark_metrics: BenchmarkMetrics
    train_config: dict[str, object]
    top_k_stock_weights: list[dict[str, object]]
    runtime_tensors: ScenarioRuntimeTensors | None = None
    extra_payload: dict[str, object] = field(default_factory=dict)


class RuntimePayloadAdapter:
    """Converts between legacy per-scenario dict payloads and typed runtime objects."""

    _KNOWN_KEYS = {
        "scenario_id",
        "source_path",
        "loss_name",
        "state",
        "evaluation_split",
        "train_config",
        "final_return",
        "backtest_portfolio_sr",
        "mean_step_return",
        "std_step_return",
        "average_turnover",
        "final_cash_weight",
        "mean_cash_weight",
        "num_time_steps",
        "scored_num_time_steps",
        "context_num_time_steps",
        "warmup_time_steps",
        "analysis_time_index",
        "feature_time_start_index",
        "feature_time_end_index",
        "target_time_start_index",
        "target_time_end_index",
        "scored_feature_time_start_index",
        "scored_feature_time_end_index",
        "scored_target_time_start_index",
        "scored_target_time_end_index",
        "context_feature_time_start_index",
        "context_feature_time_end_index",
        "context_target_time_start_index",
        "context_target_time_end_index",
        "top_k_stock_weights",
        "allocation_group_top_n",
        "stock_count_weight_threshold",
        "stock_count_min_active_days",
        "effective_stock_count_min_active_days",
        "stock_count_lookback_days",
        "total_selected_stock_count",
        "benchmark_market_index_csv",
        "benchmark_excess_return",
        "benchmark_information_ratio",
        "evaluation_mode",
        "rolling_window_lookback_days",
        "rolling_window_horizon_days",
        "rolling_window_stride_days",
        "num_rolling_windows",
        "evaluation_price_anchor_mode",
        "sharpe_like",
        # Legacy artifact field that is intentionally ignored in new outputs.
        "benchmark_excess_max_drawdown",
        *TRANSIENT_SCENARIO_TENSOR_FIELDS,
    }

    @staticmethod
    def _to_int(value: object, *, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _to_float(value: object, *, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _to_optional_float(value: object) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def from_legacy_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        require_runtime_tensors: bool = False,
    ) -> ScenarioEvalResult:
        runtime_tensors = ScenarioRuntimeTensors(
            final_stock_weights=payload.get("_final_stock_weights_tensor")
            if isinstance(payload.get("_final_stock_weights_tensor"), torch.Tensor)
            else None,
            stock_weights=payload.get("_stock_weights_tensor")
            if isinstance(payload.get("_stock_weights_tensor"), torch.Tensor)
            else None,
            cash_weights=payload.get("_cash_weights_tensor")
            if isinstance(payload.get("_cash_weights_tensor"), torch.Tensor)
            else None,
            portfolio_returns=payload.get("_portfolio_returns_tensor")
            if isinstance(payload.get("_portfolio_returns_tensor"), torch.Tensor)
            else None,
            target_time_indices=payload.get("_target_time_indices_tensor")
            if isinstance(payload.get("_target_time_indices_tensor"), torch.Tensor)
            else None,
        )
        if require_runtime_tensors:
            missing = [
                field_name
                for field_name, value in (
                    ("_final_stock_weights_tensor", runtime_tensors.final_stock_weights),
                    ("_stock_weights_tensor", runtime_tensors.stock_weights),
                    ("_cash_weights_tensor", runtime_tensors.cash_weights),
                    ("_portfolio_returns_tensor", runtime_tensors.portfolio_returns),
                    ("_target_time_indices_tensor", runtime_tensors.target_time_indices),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    "Legacy payload is missing runtime tensor fields required for this operation: "
                    f"{missing}"
                )

        identity = ScenarioIdentity(
            scenario_id=str(payload.get("scenario_id", "")),
            source_path=str(payload.get("source_path", "")),
            loss_name=str(payload.get("loss_name", "")),
            state=str(payload.get("state", "")),
            evaluation_split=str(payload.get("evaluation_split", "holdout_test")),
        )
        window_meta = ScenarioWindowMeta(
            num_time_steps=cls._to_int(payload.get("num_time_steps")),
            scored_num_time_steps=cls._to_int(
                payload.get("scored_num_time_steps", payload.get("num_time_steps"))
            ),
            context_num_time_steps=cls._to_int(payload.get("context_num_time_steps")),
            warmup_time_steps=cls._to_int(payload.get("warmup_time_steps")),
            analysis_time_index=cls._to_int(payload.get("analysis_time_index")),
            feature_time_start_index=cls._to_int(payload.get("feature_time_start_index")),
            feature_time_end_index=cls._to_int(payload.get("feature_time_end_index")),
            target_time_start_index=cls._to_int(payload.get("target_time_start_index")),
            target_time_end_index=cls._to_int(payload.get("target_time_end_index")),
            scored_feature_time_start_index=cls._to_int(payload.get("scored_feature_time_start_index")),
            scored_feature_time_end_index=cls._to_int(payload.get("scored_feature_time_end_index")),
            scored_target_time_start_index=cls._to_int(payload.get("scored_target_time_start_index")),
            scored_target_time_end_index=cls._to_int(payload.get("scored_target_time_end_index")),
            context_feature_time_start_index=cls._to_int(payload.get("context_feature_time_start_index")),
            context_feature_time_end_index=cls._to_int(payload.get("context_feature_time_end_index")),
            context_target_time_start_index=cls._to_int(payload.get("context_target_time_start_index")),
            context_target_time_end_index=cls._to_int(payload.get("context_target_time_end_index")),
            evaluation_mode=str(payload["evaluation_mode"]) if payload.get("evaluation_mode") is not None else None,
            rolling_window_lookback_days=cls._to_int(payload["rolling_window_lookback_days"])
            if payload.get("rolling_window_lookback_days") is not None
            else None,
            rolling_window_horizon_days=cls._to_int(payload["rolling_window_horizon_days"])
            if payload.get("rolling_window_horizon_days") is not None
            else None,
            rolling_window_stride_days=cls._to_int(payload["rolling_window_stride_days"])
            if payload.get("rolling_window_stride_days") is not None
            else None,
            num_rolling_windows=cls._to_int(payload["num_rolling_windows"])
            if payload.get("num_rolling_windows") is not None
            else None,
            evaluation_price_anchor_mode=str(payload["evaluation_price_anchor_mode"])
            if payload.get("evaluation_price_anchor_mode") is not None
            else None,
        )
        return_stats = ScenarioReturnStats(
            final_return=cls._to_float(payload.get("final_return")),
            backtest_portfolio_sr=cls._to_float(payload.get("backtest_portfolio_sr")),
            mean_step_return=cls._to_float(payload.get("mean_step_return")),
            std_step_return=cls._to_float(payload.get("std_step_return")),
            average_turnover=cls._to_float(payload.get("average_turnover")),
            sharpe_like=cls._to_optional_float(payload.get("sharpe_like")),
        )
        cash_stats = ScenarioCashStats(
            final_cash_weight=cls._to_float(payload.get("final_cash_weight")),
            mean_cash_weight=cls._to_float(payload.get("mean_cash_weight")),
        )
        selection_stats = ScenarioSelectionStats(
            allocation_group_top_n=cls._to_int(payload.get("allocation_group_top_n")),
            stock_count_weight_threshold=cls._to_float(payload.get("stock_count_weight_threshold")),
            stock_count_min_active_days=cls._to_int(payload.get("stock_count_min_active_days")),
            effective_stock_count_min_active_days=cls._to_int(
                payload.get("effective_stock_count_min_active_days")
            ),
            stock_count_lookback_days=cls._to_int(payload.get("stock_count_lookback_days")),
            total_selected_stock_count=cls._to_int(payload.get("total_selected_stock_count")),
        )
        benchmark_metrics = BenchmarkMetrics(
            benchmark_market_index_csv=str(payload["benchmark_market_index_csv"])
            if payload.get("benchmark_market_index_csv") not in {None, ""}
            else None,
            benchmark_excess_return=cls._to_optional_float(payload.get("benchmark_excess_return")),
            benchmark_information_ratio=cls._to_optional_float(payload.get("benchmark_information_ratio")),
        )
        raw_train_config = payload.get("train_config")
        train_config = dict(raw_train_config) if isinstance(raw_train_config, dict) else {}
        raw_top_k = payload.get("top_k_stock_weights")
        top_k_stock_weights = list(raw_top_k) if isinstance(raw_top_k, list) else []
        extra_payload = {
            key: value for key, value in payload.items() if key not in cls._KNOWN_KEYS
        }
        return ScenarioEvalResult(
            identity=identity,
            window_meta=window_meta,
            return_stats=return_stats,
            cash_stats=cash_stats,
            selection_stats=selection_stats,
            benchmark_metrics=benchmark_metrics,
            train_config=train_config,
            top_k_stock_weights=top_k_stock_weights,
            runtime_tensors=runtime_tensors,
            extra_payload=extra_payload,
        )

    @classmethod
    def to_legacy_payload(
        cls,
        result: ScenarioEvalResult,
        *,
        include_runtime_tensors: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = dict(result.extra_payload)
        payload.update(
            {
                "scenario_id": result.identity.scenario_id,
                "source_path": result.identity.source_path,
                "loss_name": result.identity.loss_name,
                "state": result.identity.state,
                "evaluation_split": result.identity.evaluation_split,
                "train_config": dict(result.train_config),
                "final_return": float(result.return_stats.final_return),
                "backtest_portfolio_sr": float(result.return_stats.backtest_portfolio_sr),
                "mean_step_return": float(result.return_stats.mean_step_return),
                "std_step_return": float(result.return_stats.std_step_return),
                "average_turnover": float(result.return_stats.average_turnover),
                "final_cash_weight": float(result.cash_stats.final_cash_weight),
                "mean_cash_weight": float(result.cash_stats.mean_cash_weight),
                "num_time_steps": int(result.window_meta.num_time_steps),
                "scored_num_time_steps": int(result.window_meta.scored_num_time_steps),
                "context_num_time_steps": int(result.window_meta.context_num_time_steps),
                "warmup_time_steps": int(result.window_meta.warmup_time_steps),
                "analysis_time_index": int(result.window_meta.analysis_time_index),
                "feature_time_start_index": int(result.window_meta.feature_time_start_index),
                "feature_time_end_index": int(result.window_meta.feature_time_end_index),
                "target_time_start_index": int(result.window_meta.target_time_start_index),
                "target_time_end_index": int(result.window_meta.target_time_end_index),
                "scored_feature_time_start_index": int(result.window_meta.scored_feature_time_start_index),
                "scored_feature_time_end_index": int(result.window_meta.scored_feature_time_end_index),
                "scored_target_time_start_index": int(result.window_meta.scored_target_time_start_index),
                "scored_target_time_end_index": int(result.window_meta.scored_target_time_end_index),
                "context_feature_time_start_index": int(result.window_meta.context_feature_time_start_index),
                "context_feature_time_end_index": int(result.window_meta.context_feature_time_end_index),
                "context_target_time_start_index": int(result.window_meta.context_target_time_start_index),
                "context_target_time_end_index": int(result.window_meta.context_target_time_end_index),
                "top_k_stock_weights": list(result.top_k_stock_weights),
                "allocation_group_top_n": int(result.selection_stats.allocation_group_top_n),
                "stock_count_weight_threshold": float(
                    result.selection_stats.stock_count_weight_threshold
                ),
                "stock_count_min_active_days": int(result.selection_stats.stock_count_min_active_days),
                "effective_stock_count_min_active_days": int(
                    result.selection_stats.effective_stock_count_min_active_days
                ),
                "stock_count_lookback_days": int(result.selection_stats.stock_count_lookback_days),
                "total_selected_stock_count": int(result.selection_stats.total_selected_stock_count),
                "benchmark_market_index_csv": result.benchmark_metrics.benchmark_market_index_csv,
                "benchmark_excess_return": result.benchmark_metrics.benchmark_excess_return,
                "benchmark_information_ratio": result.benchmark_metrics.benchmark_information_ratio,
            }
        )
        if result.window_meta.evaluation_mode is not None:
            payload["evaluation_mode"] = result.window_meta.evaluation_mode
        if result.window_meta.rolling_window_lookback_days is not None:
            payload["rolling_window_lookback_days"] = int(result.window_meta.rolling_window_lookback_days)
        if result.window_meta.rolling_window_horizon_days is not None:
            payload["rolling_window_horizon_days"] = int(result.window_meta.rolling_window_horizon_days)
        if result.window_meta.rolling_window_stride_days is not None:
            payload["rolling_window_stride_days"] = int(result.window_meta.rolling_window_stride_days)
        if result.window_meta.num_rolling_windows is not None:
            payload["num_rolling_windows"] = int(result.window_meta.num_rolling_windows)
        if result.window_meta.evaluation_price_anchor_mode is not None:
            payload["evaluation_price_anchor_mode"] = result.window_meta.evaluation_price_anchor_mode
        if result.return_stats.sharpe_like is not None:
            payload["sharpe_like"] = float(result.return_stats.sharpe_like)
        if include_runtime_tensors and result.runtime_tensors is not None:
            payload["_final_stock_weights_tensor"] = result.runtime_tensors.final_stock_weights
            payload["_stock_weights_tensor"] = result.runtime_tensors.stock_weights
            payload["_cash_weights_tensor"] = result.runtime_tensors.cash_weights
            payload["_portfolio_returns_tensor"] = result.runtime_tensors.portfolio_returns
            payload["_target_time_indices_tensor"] = result.runtime_tensors.target_time_indices
        return payload

    @staticmethod
    def strip_runtime_fields(per_scenario_payloads: list[dict[str, Any]]) -> None:
        for item in per_scenario_payloads:
            for field_name in TRANSIENT_SCENARIO_TENSOR_FIELDS:
                item.pop(field_name, None)
