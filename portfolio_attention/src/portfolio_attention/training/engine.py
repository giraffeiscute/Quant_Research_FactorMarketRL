"""Training engine primitives: dataset/runtime/train/validation execution."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import threading
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from ..config import DataConfig, ModelConfig, PathsConfig, TrainConfig
from ..data.dataset import PortfolioPanelDataset
from ..evaluation.metrics import apply_transaction_cost_to_returns
from ..evaluation.runtime import (
    EVALUATION_PRICE_ANCHOR_MODE_PER_WINDOW,
    ROLLING_ONE_STEP_EVALUATION_MODE,
    ROLLING_ONE_STEP_HORIZON_DAYS,
    ROLLING_ONE_STEP_STRIDE_DAYS,
    _collect_single_scenario_rolling_one_step_outputs,
)
from ..model.losses import build_loss, build_portfolio_objective_loss, compute_turnover_penalty
from ..model import PortfolioAttentionModel
from .resume import advance_train_loader_generator, load_resume_training_state
from .status import TrainingStatusReporter, build_dataset_progress_callback
from ..common.utils import (
    apply_score_mask,
    append_log,
    format_determinism_status,
    get_determinism_status,
    set_seed,
)


_TRAINING_INITIALIZATION_LOCK = threading.Lock()


@dataclass
class DatasetBundle:
    dataset: PortfolioPanelDataset
    train_dataset: Dataset
    validation_dataset: Dataset
    test_dataset: Dataset


@dataclass
class TrainingRuntimeBundle:
    model: PortfolioAttentionModel
    optimizer: torch.optim.Optimizer
    train_loader: DataLoader
    validation_loader: DataLoader
    resolved_shuffle_seed: int
    train_batch_size: int
    resume_state: dict[str, Any] | None


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _build_validation_rolling_metadata(
    *,
    lookback_days: int,
    num_rolling_windows_total: int | None = None,
) -> dict[str, Any]:
    resolved_lookback_days = int(lookback_days)
    return {
        "validation_evaluation_mode": ROLLING_ONE_STEP_EVALUATION_MODE,
        "validation_price_anchor_mode": EVALUATION_PRICE_ANCHOR_MODE_PER_WINDOW,
        "validation_rolling_window_lookback_days": resolved_lookback_days,
        "validation_rolling_window_horizon_days": ROLLING_ONE_STEP_HORIZON_DAYS,
        "validation_rolling_window_stride_days": ROLLING_ONE_STEP_STRIDE_DAYS,
        "validation_context_num_time_steps": (
            resolved_lookback_days + ROLLING_ONE_STEP_HORIZON_DAYS
        ),
        "validation_warmup_time_steps": resolved_lookback_days,
        "validation_num_rolling_windows_total": (
            None
            if num_rolling_windows_total is None
            else int(num_rolling_windows_total)
        ),
    }


def _log_reproducibility_status(log_path: Path, train_config: TrainConfig, device: torch.device) -> None:
    status = get_determinism_status(device=device, seed=train_config.seed)
    message = format_determinism_status(status)
    append_log(log_path, message)


def _append_dataset_split_summary(log_path: Path, dataset: PortfolioPanelDataset) -> None:
    metadata = dataset.metadata
    train_samples = int(metadata.train_window_count)
    train_batch_size = int(metadata.train_batch_size)
    steps_per_epoch = math.ceil(train_samples / train_batch_size)
    append_log(
        log_path,
        (
            f"Found {metadata.total_scenarios_found} scenarios in {metadata.scenario_dir} "
            f"using glob='{metadata.scenario_glob}'."
        ),
    )
    append_log(log_path, f"Train scenarios ({metadata.num_train_scenarios}): {metadata.train_scenarios}")
    append_log(
        log_path,
        f"Validation scenarios ({metadata.num_validation_scenarios}): {metadata.validation_scenarios}",
    )
    append_log(log_path, f"Holdout test scenarios ({metadata.num_test_scenarios}): {metadata.test_scenarios}")
    append_log(
        log_path,
        (
            "Scenario time coverage: "
            f"train_raw={metadata.train_segment_raw_length} "
            f"train_context={metadata.train_context_time_steps} "
            f"train_scored={metadata.train_score_time_steps} | "
            f"validation_raw={metadata.validation_segment_raw_length} "
            f"validation_context={metadata.validation_context_time_steps} "
            f"validation_scored={metadata.validation_score_time_steps} | "
            f"test_raw={metadata.test_segment_raw_length} "
            f"test_context={metadata.test_context_time_steps} "
            f"test_scored={metadata.test_score_time_steps}"
        ),
    )
    append_log(
        log_path,
        (
            f"train_batch_size={metadata.train_batch_size} "
            f"shuffle_scenario_splits={metadata.shuffle_scenario_splits} "
            f"scenario_split_seed={metadata.scenario_split_seed} "
            f"shuffle_train_scenarios={metadata.shuffle_train_scenarios}"
        ),
    )
    append_log(
        log_path,
        (
            f"train_samples={train_samples} "
            f"steps_per_epoch={steps_per_epoch} "
            f"windows_per_scenario={int(metadata.train_windows_per_scenario)}"
        ),
    )
    append_log(
        log_path,
        (
            "Rolling train windows: "
            f"lookback_days={metadata.lookback_days} "
            f"rolling_horizon_days={metadata.rolling_horizon_days} "
            f"rolling_stride_days={metadata.rolling_stride_days} "
            f"train_windows_per_scenario={metadata.train_windows_per_scenario} "
            f"train_window_count={metadata.train_window_count} "
            f"rolling_train_dataset_mode={metadata.rolling_train_dataset_mode}"
        ),
    )


def _run_loss_step(
    model: PortfolioAttentionModel,
    batch: dict[str, Any],
    loss_name: str,
    *,
    turnover_penalty: float = 0.0,
    transaction_cost_rate: float = 0.0,
    turnover_penalty_norm: str = "l1",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    outputs = model(
        batch["x_stock"],
        batch["x_market"],
        batch["stock_indices"],
        target_returns=batch["r_stock"],
    )
    portfolio_returns = outputs["portfolio_return"]
    if portfolio_returns is None:
        raise RuntimeError("Training batch must provide target returns.")
    if portfolio_returns.ndim != 2:
        raise ValueError(
            "portfolio_returns must have shape [num_scenarios_in_batch, time_steps]. "
            f"Received {tuple(portfolio_returns.shape)}."
        )
    turnover = outputs.get("turnover")
    if not isinstance(turnover, torch.Tensor):
        raise RuntimeError("Training batch requires model outputs to include turnover tensor.")
    if turnover.shape != portfolio_returns.shape:
        raise ValueError(
            "turnover must match portfolio_returns shape. "
            f"Received turnover={tuple(turnover.shape)} portfolio_returns={tuple(portfolio_returns.shape)}."
        )
    allocation = outputs.get("allocation")
    if not isinstance(allocation, torch.Tensor):
        raise RuntimeError("Training batch requires model outputs to include allocation tensor.")
    if allocation.ndim != 3 or allocation.shape[:2] != portfolio_returns.shape:
        raise ValueError(
            "allocation must have shape [num_scenarios_in_batch, time_steps, num_assets]. "
            f"Received allocation={tuple(allocation.shape)} portfolio_returns={tuple(portfolio_returns.shape)}."
        )
    previous_allocation = outputs.get("previous_allocation")
    if not isinstance(previous_allocation, torch.Tensor):
        raise RuntimeError("Training batch requires model outputs to include previous_allocation tensor.")
    if previous_allocation.shape != allocation.shape:
        raise ValueError(
            "previous_allocation must match allocation shape. "
            f"Received previous_allocation={tuple(previous_allocation.shape)} "
            f"allocation={tuple(allocation.shape)}."
        )
    score_mask = batch.get("score_mask")
    if score_mask is None:
        scored_returns = portfolio_returns
        scored_turnover = turnover
        scored_allocation = allocation
        scored_previous_allocation = previous_allocation
    else:
        if not isinstance(score_mask, torch.Tensor):
            raise ValueError("score_mask must be a tensor when provided in the batch.")
        score_mask_bool = score_mask.to(dtype=torch.bool)
        scored_returns = apply_score_mask(portfolio_returns, score_mask_bool)
        scored_turnover = apply_score_mask(turnover, score_mask_bool)
        scored_allocation = apply_score_mask(allocation, score_mask_bool)
        scored_previous_allocation = apply_score_mask(previous_allocation, score_mask_bool)

    loss = build_portfolio_objective_loss(
        loss_name,
        scored_returns,
        turnover=scored_turnover,
        allocation=scored_allocation,
        previous_allocation=scored_previous_allocation,
        turnover_penalty=turnover_penalty,
        transaction_cost_rate=transaction_cost_rate,
        turnover_penalty_norm=turnover_penalty_norm,
    )
    net_scored_returns = scored_returns - float(transaction_cost_rate) * scored_turnover
    if float(turnover_penalty) > 0.0:
        weight_loss = float(turnover_penalty) * compute_turnover_penalty(
            scored_turnover,
            norm=turnover_penalty_norm,
            portfolio_returns=net_scored_returns,
            allocation=scored_allocation,
            previous_allocation=scored_previous_allocation,
        )
    else:
        weight_loss = scored_returns.new_zeros(())
    scenario_final_returns = torch.prod(1.0 + net_scored_returns, dim=1) - 1.0
    summary = {
        "scenario_final_returns": scenario_final_returns,
        "scenario_mean_step_returns": net_scored_returns.mean(dim=1),
        "scenario_gross_final_returns": torch.prod(1.0 + scored_returns, dim=1) - 1.0,
        "weight_loss": weight_loss,
        "mean_turnover": scored_turnover.mean(),
    }
    return loss, net_scored_returns, summary


def _resolve_shuffle_train_scenarios_seed(data_config: DataConfig, train_config: TrainConfig) -> int:
    if data_config.shuffle_train_scenarios_seed is not None:
        return int(data_config.shuffle_train_scenarios_seed)
    return int(train_config.seed)


@torch.no_grad()
def _evaluate_epoch(
    model: PortfolioAttentionModel,
    dataset: PortfolioPanelDataset,
    loader: DataLoader,
    device: torch.device,
    loss_name: str,
    lookback_days: int,
    evaluation_transaction_cost_rate: float = 0.0,
    heartbeat_callback: Any | None = None,
) -> tuple[float, float, dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    total_final_return = 0.0
    total_scenarios = 0
    total_rolling_windows = 0

    num_batches = len(loader)
    for batch_index, raw_batch in enumerate(loader, start=1):
        rolling_outputs = _collect_single_scenario_rolling_one_step_outputs(
            model=model,
            dataset=dataset,
            raw_batch=raw_batch,
            device=device,
            lookback_days=int(lookback_days),
            evaluation_label="Validation rolling evaluation",
            collect_weights=False,
        )
        scored_returns = apply_transaction_cost_to_returns(
            rolling_outputs["portfolio_returns"],
            rolling_outputs["turnover"],
            transaction_cost_rate=evaluation_transaction_cost_rate,
        ).unsqueeze(0)
        loss = build_loss(loss_name, scored_returns)
        summary = {
            "scenario_final_returns": torch.prod(1.0 + scored_returns, dim=1) - 1.0,
        }
        scenario_count = 1
        total_loss += float(loss.detach().cpu().item()) * scenario_count
        total_final_return += float(summary["scenario_final_returns"].mean().detach().cpu().item()) * scenario_count
        total_scenarios += scenario_count
        total_rolling_windows += int(rolling_outputs["num_rolling_windows"])
        if heartbeat_callback is not None:
            heartbeat_callback(batch_index, num_batches)

    if total_scenarios == 0:
        raise RuntimeError("Evaluation loader produced no scenarios.")

    return (
        total_loss / total_scenarios,
        total_final_return / total_scenarios,
        _build_validation_rolling_metadata(
            lookback_days=int(lookback_days),
            num_rolling_windows_total=total_rolling_windows,
        ),
    )


def build_training_model(
    *,
    model_config: ModelConfig,
    dataset: PortfolioPanelDataset,
    data_config: DataConfig,
    device: torch.device,
) -> PortfolioAttentionModel:
    return PortfolioAttentionModel(
        model_config,
        num_stocks=dataset.num_stocks,
        max_lookback=dataset.max_time_steps,
        stock_temporal_attention_window=int(data_config.lookback_days),
    ).to(device)


def build_training_optimizer(
    *,
    model: PortfolioAttentionModel,
    train_config: TrainConfig,
) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )


def build_training_dataloaders(
    *,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    data_config: DataConfig,
    train_batch_size: int,
    generator: torch.Generator,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=bool(data_config.shuffle_train_scenarios),
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
    )
    return train_loader, validation_loader


def build_or_load_resume_state(
    *,
    paths: PathsConfig,
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    dataset: PortfolioPanelDataset,
    model: PortfolioAttentionModel,
    optimizer: torch.optim.Optimizer,
    train_dataset: Dataset,
    generator: torch.Generator,
) -> dict[str, Any] | None:
    resume_state = load_resume_training_state(
        paths=paths,
        data_config=data_config,
        model_config=model_config,
        train_config=train_config,
        dataset=dataset,
        model=model,
        optimizer=optimizer,
    )
    if resume_state is not None:
        advance_train_loader_generator(
            generator=generator,
            train_dataset=train_dataset,
            completed_epochs=int(resume_state["checkpoint_epoch"]),
            shuffle_enabled=bool(data_config.shuffle_train_scenarios),
        )
    return resume_state


def _initialize_training_runtime(
    *,
    paths: PathsConfig,
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    dataset: PortfolioPanelDataset,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    device: torch.device,
) -> tuple[
    PortfolioAttentionModel,
    torch.optim.Optimizer,
    DataLoader,
    DataLoader,
    int,
    int,
    dict[str, Any] | None,
]:
    set_seed(train_config.seed)
    model = build_training_model(
        model_config=model_config,
        dataset=dataset,
        data_config=data_config,
        device=device,
    )
    optimizer = build_training_optimizer(model=model, train_config=train_config)
    resolved_shuffle_seed = _resolve_shuffle_train_scenarios_seed(data_config, train_config)
    train_batch_size = int(data_config.train_batch_size)
    generator = torch.Generator()
    generator.manual_seed(resolved_shuffle_seed)
    resume_state = build_or_load_resume_state(
        paths=paths,
        data_config=data_config,
        model_config=model_config,
        train_config=train_config,
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        generator=generator,
    )
    train_loader, validation_loader = build_training_dataloaders(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        data_config=data_config,
        train_batch_size=train_batch_size,
        generator=generator,
    )
    return (
        model,
        optimizer,
        train_loader,
        validation_loader,
        resolved_shuffle_seed,
        train_batch_size,
        resume_state,
    )


def _initialize_training_runtime_bundle(
    *,
    paths: PathsConfig,
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    dataset: PortfolioPanelDataset,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    device: torch.device,
) -> TrainingRuntimeBundle:
    (
        model,
        optimizer,
        train_loader,
        validation_loader,
        resolved_shuffle_seed,
        train_batch_size,
        resume_state,
    ) = _initialize_training_runtime(
        paths=paths,
        data_config=data_config,
        model_config=model_config,
        train_config=train_config,
        dataset=dataset,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        device=device,
    )
    return TrainingRuntimeBundle(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=validation_loader,
        resolved_shuffle_seed=resolved_shuffle_seed,
        train_batch_size=train_batch_size,
        resume_state=resume_state,
    )


def _prepare_training_runtime(
    *,
    status_reporter: TrainingStatusReporter,
    initialization_lock: threading.Lock | None,
    paths: PathsConfig,
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    dataset: PortfolioPanelDataset,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    device: torch.device,
) -> TrainingRuntimeBundle:
    def _write_initialization_status(phase: str, message: str) -> None:
        status_reporter.update(
            "PREPARING_DATA",
            phase=phase,
            message=message,
        )

    if initialization_lock is None:
        _write_initialization_status(
            "initializing_runtime",
            "Preparing model, optimizer, DataLoaders, and resume state.",
        )
        return _initialize_training_runtime_bundle(
            paths=paths,
            data_config=data_config,
            model_config=model_config,
            train_config=train_config,
            dataset=dataset,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            device=device,
        )

    _write_initialization_status(
        "waiting_for_initialization_lock",
        "Shared dataset attached; waiting for serialized runtime initialization.",
    )
    with initialization_lock:
        _write_initialization_status(
            "initializing_runtime",
            "Preparing model, optimizer, DataLoaders, and resume state.",
        )
        return _initialize_training_runtime_bundle(
            paths=paths,
            data_config=data_config,
            model_config=model_config,
            train_config=train_config,
            dataset=dataset,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            device=device,
        )


def _emit_training_heartbeat(
    *,
    status_reporter: TrainingStatusReporter,
    epoch_started_at: float,
    epoch: int,
    num_epochs: int,
    batch_index: int,
    num_train_batches: int,
    force: bool = False,
) -> None:
    status_reporter.heartbeat(
        force=force,
        epoch_started_at=epoch_started_at,
        epoch=epoch,
        progress_ratio=((epoch - 1) + (batch_index / max(1, num_train_batches))) / num_epochs,
        phase="training",
        message=f"Training batch {batch_index}/{num_train_batches}",
        epoch_batch_index=batch_index,
        epoch_num_batches=num_train_batches,
        epoch_batch_progress_ratio=batch_index / max(1, num_train_batches),
        validation_batch_index=None,
        validation_num_batches=None,
        validation_batch_progress_ratio=None,
    )


def _log_first_training_batch_shapes(
    *,
    log_path: Path,
    batch: dict[str, Any],
    portfolio_returns: torch.Tensor,
    shape_logged: bool,
) -> bool:
    if shape_logged:
        return shape_logged
    append_log(
        log_path,
        (
            "Training tensor shapes: "
            f"x_stock={tuple(batch['x_stock'].shape)} "
            f"x_market={tuple(batch['x_market'].shape)} "
            f"r_stock={tuple(batch['r_stock'].shape)} "
            f"portfolio_returns={tuple(portfolio_returns.shape)}"
        ),
    )
    return True


def _run_single_train_batch(
    *,
    model: PortfolioAttentionModel,
    optimizer: torch.optim.Optimizer,
    raw_batch: dict[str, Any],
    device: torch.device,
    loss_name: str,
    turnover_penalty: float = 0.0,
    transaction_cost_rate: float = 0.0,
    turnover_penalty_norm: str = "l1",
    grad_clip_norm: float,
    log_path: Path,
    shape_logged: bool,
) -> tuple[float, float, int, bool]:
    batch = _move_batch_to_device(raw_batch, device)
    optimizer.zero_grad(set_to_none=True)
    loss, portfolio_returns, summary = _run_loss_step(
        model,
        batch,
        loss_name,
        turnover_penalty=turnover_penalty,
        transaction_cost_rate=transaction_cost_rate,
        turnover_penalty_norm=turnover_penalty_norm,
    )
    shape_logged = _log_first_training_batch_shapes(
        log_path=log_path,
        batch=batch,
        portfolio_returns=portfolio_returns,
        shape_logged=shape_logged,
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    sample_count = int(batch["x_stock"].shape[0])
    return (
        float(loss.detach().cpu().item()),
        float(summary["scenario_final_returns"].mean().detach().cpu().item()),
        sample_count,
        shape_logged,
    )


def _run_training_epoch(
    *,
    model: PortfolioAttentionModel,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device,
    loss_name: str,
    turnover_penalty: float = 0.0,
    transaction_cost_rate: float = 0.0,
    turnover_penalty_norm: str = "l1",
    grad_clip_norm: float,
    epoch: int,
    num_epochs: int,
    epoch_started_at: float,
    status_reporter: TrainingStatusReporter,
    log_path: Path,
    shape_logged: bool,
) -> tuple[float, float, int, bool]:
    model.train()
    total_train_loss = 0.0
    total_train_final_return = 0.0
    total_train_scenarios = 0
    status_reporter.last_heartbeat_at = None
    num_train_batches = len(train_loader)
    _emit_training_heartbeat(
        status_reporter=status_reporter,
        epoch_started_at=epoch_started_at,
        epoch=epoch,
        num_epochs=num_epochs,
        batch_index=0,
        num_train_batches=num_train_batches,
        force=True,
    )

    for batch_index, raw_batch in enumerate(train_loader, start=1):
        loss_value, mean_final_return, sample_count, shape_logged = _run_single_train_batch(
            model=model,
            optimizer=optimizer,
            raw_batch=raw_batch,
            device=device,
            loss_name=loss_name,
            turnover_penalty=turnover_penalty,
            transaction_cost_rate=transaction_cost_rate,
            turnover_penalty_norm=turnover_penalty_norm,
            grad_clip_norm=grad_clip_norm,
            log_path=log_path,
            shape_logged=shape_logged,
        )
        total_train_loss += loss_value * sample_count
        total_train_final_return += mean_final_return * sample_count
        total_train_scenarios += sample_count
        _emit_training_heartbeat(
            status_reporter=status_reporter,
            epoch_started_at=epoch_started_at,
            epoch=epoch,
            num_epochs=num_epochs,
            batch_index=batch_index,
            num_train_batches=num_train_batches,
        )

    if total_train_scenarios == 0:
        raise RuntimeError("Train loader produced no scenarios.")

    return (
        total_train_loss / total_train_scenarios,
        total_train_final_return / total_train_scenarios,
        num_train_batches,
        shape_logged,
    )


def _run_validation_epoch(
    *,
    model: PortfolioAttentionModel,
    dataset: PortfolioPanelDataset,
    validation_loader: DataLoader,
    device: torch.device,
    loss_name: str,
    lookback_days: int,
    epoch: int,
    num_epochs: int,
    num_train_batches: int,
    epoch_started_at: float,
    status_reporter: TrainingStatusReporter,
    evaluation_transaction_cost_rate: float = 0.0,
) -> tuple[float, float, dict[str, Any]]:
    num_validation_batches = len(validation_loader)
    status_reporter.heartbeat(
        force=True,
        epoch_started_at=epoch_started_at,
        epoch=epoch,
        progress_ratio=(epoch / num_epochs),
        phase="validation",
        message=f"Validating batch 0/{num_validation_batches}",
        epoch_batch_index=num_train_batches,
        epoch_num_batches=num_train_batches,
        epoch_batch_progress_ratio=1.0 if num_train_batches > 0 else 0.0,
        validation_batch_index=0,
        validation_num_batches=num_validation_batches,
        validation_batch_progress_ratio=0.0,
    )

    def _validation_heartbeat(batch_index: int, num_batches: int) -> None:
        status_reporter.heartbeat(
            epoch_started_at=epoch_started_at,
            epoch=epoch,
            progress_ratio=(epoch / num_epochs),
            phase="validation",
            message=f"Validating batch {batch_index}/{num_batches}",
            epoch_batch_index=num_train_batches,
            epoch_num_batches=num_train_batches,
            epoch_batch_progress_ratio=1.0 if num_train_batches > 0 else 0.0,
            validation_batch_index=batch_index,
            validation_num_batches=num_batches,
            validation_batch_progress_ratio=batch_index / max(1, num_batches),
        )

    return _evaluate_epoch(
        model,
        dataset,
        validation_loader,
        device,
        loss_name,
        lookback_days=lookback_days,
        evaluation_transaction_cost_rate=evaluation_transaction_cost_rate,
        heartbeat_callback=_validation_heartbeat,
    )


def build_dataset_bundle(
    *,
    data_config: DataConfig,
    paths: PathsConfig,
    loss_name: str,
    device: torch.device,
    num_epochs: int,
    log_path: Path,
) -> DatasetBundle:
    dataset = PortfolioPanelDataset(
        data_config,
        progress_callback=build_dataset_progress_callback(
            paths=paths,
            loss_names=[loss_name],
            device=str(device),
            num_epochs=num_epochs,
            log_path=log_path,
        ),
    )
    train_dataset, validation_dataset, test_dataset = dataset.build_train_validation_test_datasets()
    return DatasetBundle(
        dataset=dataset,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
    )
