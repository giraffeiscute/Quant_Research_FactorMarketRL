"""Training status, heartbeat, and dashboard helpers."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable

from rich.console import Console
from rich.live import Live
from rich.table import Table

if __package__ is None or __package__ == "":
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from portfolio_attention.config import PathsConfig
    from portfolio_attention.utils import append_log
else:
    from .config import PathsConfig
    from .utils import append_log


NON_TERMINAL_STATUSES = {"QUEUED", "STARTING", "PREPARING_DATA", "RUNNING"}
STATUS_PERSISTED_KEYS = (
    "train_loss",
    "train_mean_final_return",
    "val_loss",
    "val_mean_final_return",
    "holdout_backtest_loss",
    "holdout_backtest_mean_final_return",
    "validation_evaluation_mode",
    "validation_price_anchor_mode",
    "validation_rolling_window_lookback_days",
    "validation_rolling_window_horizon_days",
    "validation_rolling_window_stride_days",
    "validation_context_num_time_steps",
    "validation_warmup_time_steps",
    "validation_num_rolling_windows_total",
    "best_epoch",
    "best_val_loss",
    "global_best_val_loss",
    "epochs_without_improvement",
    "select_best_from_last_x_epochs",
    "epoch_batch_index",
    "epoch_num_batches",
    "epoch_batch_progress_ratio",
    "validation_batch_index",
    "validation_num_batches",
    "validation_batch_progress_ratio",
    "epoch_elapsed_seconds",
)

HEARTBEAT_INTERVAL_SECONDS = 10.0
SHARED_DASHBOARD_REFRESH_INTERVAL_SECONDS = 0.25

_STATUS_FILE_LOCK = threading.Lock()
_LATEST_TRAINING_STATUS_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def _status_path_for_loss(paths: PathsConfig, loss_name: str) -> Path:
    return paths.status_dir / f"train_status_{loss_name}.json"


def _log_path_for_loss(
    paths: PathsConfig,
    loss_name: str,
    *,
    state: str | None = None,
) -> Path:
    logs_dir = paths.logs_dir if state is None else paths.get_state_logs_dir(state)
    return logs_dir / f"train_{loss_name}.log"


def _console_log_path_for_loss(
    paths: PathsConfig,
    loss_name: str,
    *,
    state: str | None = None,
) -> Path:
    logs_dir = paths.logs_dir if state is None else paths.get_state_logs_dir(state)
    return logs_dir / f"train_{loss_name}.console.log"


def _dataset_progress_message(message: str) -> str:
    return f"[dataset] {message}"


def _status_cache_key(paths: PathsConfig, loss_name: str) -> tuple[str, str]:
    return (str(paths.outputs_dir.resolve()), str(loss_name))


def _read_status_payload_from_disk(status_path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _write_status_payload_to_disk(status_path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2)
    temp_path = status_path.with_name(
        f".{status_path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
    )
    try:
        temp_path.write_text(serialized, encoding="utf-8")
        os.replace(temp_path, status_path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _clear_cached_training_status(paths: PathsConfig, loss_name: str) -> None:
    with _STATUS_FILE_LOCK:
        _LATEST_TRAINING_STATUS_CACHE.pop(_status_cache_key(paths, loss_name), None)


def _write_training_status(
    paths: PathsConfig,
    loss_name: str,
    status: str,
    **kwargs: Any,
) -> None:
    status_path = _status_path_for_loss(paths, loss_name)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    cache_key = _status_cache_key(paths, loss_name)
    with _STATUS_FILE_LOCK:
        previous_payload = dict(_LATEST_TRAINING_STATUS_CACHE.get(cache_key, {}))
        if not previous_payload:
            previous_payload = _read_status_payload_from_disk(status_path) or {}
        current_time = time.time()

        payload = {
            "loss_name": loss_name,
            "status": status,
            "pid": os.getpid(),
            "updated_at": current_time,
            "phase": kwargs.get("phase", previous_payload.get("phase", "queued")),
            "message": kwargs.get("message", previous_payload.get("message", "")),
            **kwargs,
        }
        if "started_at" not in payload:
            if status in NON_TERMINAL_STATUSES:
                payload["started_at"] = previous_payload.get("started_at", current_time)
            elif status in {"DONE", "FAILED"} and "started_at" in previous_payload:
                payload["started_at"] = previous_payload["started_at"]

        started_at = payload.get("started_at")
        if "elapsed_seconds" not in payload:
            if started_at is not None:
                try:
                    payload["elapsed_seconds"] = max(0.0, current_time - float(started_at))
                except (TypeError, ValueError):
                    payload["elapsed_seconds"] = None
            else:
                payload["elapsed_seconds"] = None
        if "avg_epoch_seconds" not in payload:
            payload["avg_epoch_seconds"] = previous_payload.get("avg_epoch_seconds")
        if "eta_seconds" not in payload:
            payload["eta_seconds"] = (
                None if payload.get("phase") != "training" else previous_payload.get("eta_seconds")
            )
        for key in STATUS_PERSISTED_KEYS:
            if key not in payload and key in previous_payload:
                payload[key] = previous_payload[key]

        _write_status_payload_to_disk(status_path, payload)
        _LATEST_TRAINING_STATUS_CACHE[cache_key] = dict(payload)


def _load_training_status(
    paths: PathsConfig,
    loss_name: str,
    *,
    prefer_cache: bool = False,
) -> dict[str, Any]:
    status_path = _status_path_for_loss(paths, loss_name)
    cache_key = _status_cache_key(paths, loss_name)
    with _STATUS_FILE_LOCK:
        if prefer_cache:
            cached_payload = _LATEST_TRAINING_STATUS_CACHE.get(cache_key)
            if cached_payload is not None:
                payload = dict(cached_payload)
                payload.setdefault("loss_name", loss_name)
                payload.setdefault("status", "UNKNOWN")
                payload.setdefault("phase", "unknown")
                payload.setdefault("message", "")
                return payload

        payload = _read_status_payload_from_disk(status_path)
        if payload is None:
            if not status_path.exists():
                return {
                    "loss_name": loss_name,
                    "status": "QUEUED",
                    "phase": "queued",
                    "message": "Waiting to start.",
                }
            return {
                "loss_name": loss_name,
                "status": "UNKNOWN",
                "phase": "status_error",
                "message": "Could not parse status file.",
            }
        _LATEST_TRAINING_STATUS_CACHE[cache_key] = dict(payload)
    payload.setdefault("loss_name", loss_name)
    payload.setdefault("status", "UNKNOWN")
    payload.setdefault("phase", "unknown")
    payload.setdefault("message", "")
    return payload


def _status_snapshot(
    paths: PathsConfig,
    losses: list[str],
    *,
    prefer_cache: bool = False,
) -> list[dict[str, Any]]:
    return [_load_training_status(paths, loss, prefer_cache=prefer_cache) for loss in losses]


def _dashboard_signature(status_rows: list[dict[str, Any]]) -> str:
    projected_rows = [
        {
            "loss_name": row.get("loss_name"),
            "status": row.get("status"),
            "phase": row.get("phase"),
            "device": row.get("device"),
            "epoch": row.get("epoch"),
            "num_epochs": row.get("num_epochs"),
            "train_loss": row.get("train_loss"),
            "val_loss": row.get("val_loss"),
            "best_val_loss": row.get("best_val_loss"),
            "elapsed_seconds": row.get("elapsed_seconds"),
            "avg_epoch_seconds": row.get("avg_epoch_seconds"),
            "eta_seconds": row.get("eta_seconds"),
            "epoch_batch_index": row.get("epoch_batch_index"),
            "epoch_num_batches": row.get("epoch_num_batches"),
            "epoch_batch_progress_ratio": row.get("epoch_batch_progress_ratio"),
            "validation_batch_index": row.get("validation_batch_index"),
            "validation_num_batches": row.get("validation_num_batches"),
            "validation_batch_progress_ratio": row.get("validation_batch_progress_ratio"),
            "epoch_elapsed_seconds": row.get("epoch_elapsed_seconds"),
            "message": row.get("message"),
        }
        for row in status_rows
    ]
    return json.dumps(projected_rows, sort_keys=True, default=str)


def _should_use_live_dashboard() -> bool:
    is_tty = getattr(sys.stdout, "isatty", lambda: False)()
    term = os.environ.get("TERM", "")
    return bool(is_tty and term and term.lower() != "dumb")


def _format_duration(seconds: object | None) -> str:
    if seconds is None:
        return "N/A"
    try:
        numeric_seconds = float(seconds)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(numeric_seconds):
        return "N/A"
    total_seconds = max(0, int(round(numeric_seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_loss_metric(value: object | None) -> str:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        numeric_value = 0.0
    if not math.isfinite(numeric_value):
        numeric_value = 0.0
    return f"{numeric_value:.6f}"


def _format_batch_progress(status_data: dict[str, Any]) -> str:
    phase = str(status_data.get("phase", "")).lower()
    if phase == "validation":
        batch_index = status_data.get("validation_batch_index")
        num_batches = status_data.get("validation_num_batches")
        ratio = status_data.get("validation_batch_progress_ratio")
        prefix = "Val"
    else:
        batch_index = status_data.get("epoch_batch_index")
        num_batches = status_data.get("epoch_num_batches")
        ratio = status_data.get("epoch_batch_progress_ratio")
        prefix = "Train"
    if batch_index in (None, "") or num_batches in (None, "", 0):
        return "-"
    try:
        index_value = int(batch_index)
        total_value = int(num_batches)
    except (TypeError, ValueError):
        return "-"
    progress_text = f"{prefix} {index_value}/{total_value}"
    try:
        if ratio is None:
            raise TypeError
        progress_text += f" ({float(ratio) * 100.0:.1f}%)"
    except (TypeError, ValueError):
        pass
    return progress_text


def _render_multi_loss_dashboard(status_rows: list[dict[str, Any]]) -> Any:
    if not _should_use_live_dashboard():
        lines = ["Multi-loss training status"]
        for status_data in status_rows:
            loss = str(status_data.get("loss_name", "unknown"))
            epoch = int(status_data.get("epoch", 0))
            raw_num_epochs = status_data.get("num_epochs", "?")
            num_epochs = raw_num_epochs if raw_num_epochs not in (None, 0) else "?"
            lines.append(
                f"{loss:<10} | {str(status_data.get('status', 'QUEUED')):<14} | "
                f"{str(status_data.get('phase', 'queued')):<16} | "
                f"Epoch {epoch}/{num_epochs} | "
                f"Train {_format_loss_metric(status_data.get('train_loss'))} | "
                f"Val {_format_loss_metric(status_data.get('val_loss'))} | "
                f"Best {_format_loss_metric(status_data.get('best_val_loss'))} | "
                f"Progress {_format_batch_progress(status_data):<20} | "
                f"Elapsed {_format_duration(status_data.get('elapsed_seconds'))} | "
                f"Avg/Epoch {_format_duration(status_data.get('avg_epoch_seconds'))} | "
                f"ETA {_format_duration(status_data.get('eta_seconds'))} | "
                f"{str(status_data.get('message', ''))}"
            )
        return "\n".join(lines)

    table = Table(
        title="Multi-Loss Portfolio Training Dashboard",
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    table.add_column("Loss", style="cyan", no_wrap=True)
    table.add_column("Status", style="bold")
    table.add_column("Phase", style="white")
    table.add_column("Device", style="green")
    table.add_column("Epoch", justify="right")
    table.add_column("Train Loss", justify="right")
    table.add_column("Val Loss", justify="right")
    table.add_column("Best Val", justify="right")
    table.add_column("Progress", style="white")
    table.add_column("Elapsed", justify="right")
    table.add_column("Avg/Epoch", justify="right")
    table.add_column("ETA", justify="right")
    table.add_column("Message", style="white")

    for data in status_rows:
        loss = str(data.get("loss_name", "unknown"))
        status = data.get("status", "QUEUED")
        status_display = status
        if status == "RUNNING":
            status_display = "[bold blue]RUNNING[/bold blue]"
        elif status == "DONE":
            status_display = "[bold green]DONE[/bold green]"
        elif status == "FAILED":
            status_display = "[bold red]FAILED[/bold red]"
        elif status == "STARTING":
            status_display = "[yellow]STARTING[/yellow]"
        elif status == "PREPARING_DATA":
            status_display = "[yellow]PREPARING[/yellow]"
        elif status == "QUEUED":
            status_display = "[dim]QUEUED[/dim]"
        raw_num_epochs = data.get("num_epochs", 0)
        num_epochs = raw_num_epochs if raw_num_epochs not in (None, 0) else "?"

        table.add_row(
            loss,
            status_display,
            str(data.get("phase", "-")),
            str(data.get("device", "-")),
            f"{data.get('epoch', 0)}/{num_epochs}",
            _format_loss_metric(data.get("train_loss")),
            _format_loss_metric(data.get("val_loss")),
            _format_loss_metric(data.get("best_val_loss")),
            _format_batch_progress(data),
            _format_duration(data.get("elapsed_seconds")),
            _format_duration(data.get("avg_epoch_seconds")),
            _format_duration(data.get("eta_seconds")),
            str(data.get("message", "")),
        )
    return table


def _print_dashboard_update(rendered: str) -> None:
    print(rendered, flush=True)


def _emit_dashboard_update(
    status_rows: list[dict[str, Any]],
    *,
    live: Live | None,
    printer: Callable[[Any], None],
    last_signature: str | None,
    force: bool = False,
) -> str:
    dashboard_signature = _dashboard_signature(status_rows)
    if not force and dashboard_signature == last_signature:
        return dashboard_signature
    rendered = _render_multi_loss_dashboard(status_rows)
    if live is not None:
        live.update(rendered, refresh=True)
    else:
        printer(rendered)
    return dashboard_signature


def _monitor_multi_loss_dashboard(
    *,
    paths: PathsConfig,
    losses: list[str],
    stop_event: threading.Event,
    prefer_cache: bool,
    refresh_interval_seconds: float = SHARED_DASHBOARD_REFRESH_INTERVAL_SECONDS,
    printer: Callable[[Any], None] = _print_dashboard_update,
) -> None:
    use_live_dashboard = _should_use_live_dashboard()
    live: Live | None = None
    last_signature: str | None = None
    if use_live_dashboard:
        console = Console(file=sys.stdout)
        initial_rows = _status_snapshot(paths, losses, prefer_cache=prefer_cache)
        live = Live(
            _render_multi_loss_dashboard(initial_rows),
            console=console,
            auto_refresh=False,
            transient=False,
        )
        live.start()
    try:
        while True:
            status_rows = _status_snapshot(paths, losses, prefer_cache=prefer_cache)
            last_signature = _emit_dashboard_update(
                status_rows,
                live=live,
                printer=printer,
                last_signature=last_signature,
                force=last_signature is None,
            )
            if stop_event.wait(refresh_interval_seconds):
                break
        final_rows = _status_snapshot(paths, losses, prefer_cache=prefer_cache)
        _emit_dashboard_update(
            final_rows,
            live=live,
            printer=printer,
            last_signature=last_signature,
            force=True,
        )
    finally:
        if live is not None:
            live.stop()


def _tail_console_log(
    paths: PathsConfig,
    loss_name: str,
    *,
    state: str | None = None,
    num_lines: int = 8,
) -> str:
    console_log_path = _console_log_path_for_loss(paths, loss_name, state=state)
    try:
        lines = console_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return ""
    tail = lines[-num_lines:]
    return "\n".join(tail)


def _build_failure_summary(
    paths: PathsConfig,
    loss_name: str,
    returncode: int | None,
    *,
    state: str | None = None,
) -> str:
    status_data = _load_training_status(paths, loss_name)
    exit_description = (
        f"exit code {returncode}" if returncode is not None else "an in-process worker exception"
    )
    summary_lines = [
        f"Loss '{loss_name}' failed with {exit_description}.",
    ]
    error_message = str(status_data.get("error_message", "")).strip()
    if error_message:
        summary_lines.append(f"Status error: {error_message}")
    log_tail = _tail_console_log(paths, loss_name, state=state)
    if log_tail:
        summary_lines.append("Console log tail:")
        summary_lines.append(log_tail)
    return "\n".join(summary_lines)


def _should_emit_heartbeat(
    *,
    last_emitted_at: float | None,
    now_seconds: float,
    interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS,
) -> bool:
    if last_emitted_at is None:
        return True
    return (float(now_seconds) - float(last_emitted_at)) >= float(interval_seconds)


class TrainingStatusReporter:
    """Reusable status reporter with a base payload and heartbeat support."""

    def __init__(
        self,
        *,
        paths: PathsConfig,
        loss_name: str,
        base_status: dict[str, Any] | None = None,
        heartbeat_interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        self.paths = paths
        self.loss_name = loss_name
        self.base_status: dict[str, Any] = dict(base_status or {})
        self.heartbeat_interval_seconds = float(heartbeat_interval_seconds)
        self.last_heartbeat_at: float | None = None

    def set_base(self, **updates: Any) -> None:
        self.base_status.update(updates)

    def update(self, status: str, **overrides: Any) -> None:
        payload = dict(self.base_status)
        payload.update(overrides)
        _write_training_status(self.paths, self.loss_name, status, **payload)

    def heartbeat(
        self,
        *,
        epoch_started_at: float,
        force: bool = False,
        now_seconds: float | None = None,
        **overrides: Any,
    ) -> float | None:
        resolved_now = float(now_seconds if now_seconds is not None else time.time())
        if not force and not _should_emit_heartbeat(
            last_emitted_at=self.last_heartbeat_at,
            now_seconds=resolved_now,
            interval_seconds=self.heartbeat_interval_seconds,
        ):
            return self.last_heartbeat_at
        payload = dict(overrides)
        payload["epoch_elapsed_seconds"] = max(0.0, resolved_now - float(epoch_started_at))
        self.update("RUNNING", **payload)
        self.last_heartbeat_at = resolved_now
        return resolved_now


def build_dataset_progress_callback(
    *,
    paths: PathsConfig,
    loss_names: list[str],
    device: str,
    num_epochs: int,
    phase: str = "building_dataset",
    log_path: Path | None = None,
    print_to_stdout: bool = False,
) -> Callable[[str], None]:
    def _callback(message: str) -> None:
        formatted_message = _dataset_progress_message(message)
        if log_path is not None:
            append_log(log_path, formatted_message)
        for loss_name in loss_names:
            _write_training_status(
                paths,
                loss_name,
                "PREPARING_DATA",
                device=device,
                epoch=0,
                num_epochs=num_epochs,
                progress_ratio=0.0,
                phase=phase,
                message=formatted_message,
            )
        if print_to_stdout:
            print(formatted_message, flush=True)

    return _callback


def status_path_for_loss(paths: PathsConfig, loss_name: str) -> Path:
    return _status_path_for_loss(paths, loss_name)


def log_path_for_loss(paths: PathsConfig, loss_name: str, *, state: str | None = None) -> Path:
    return _log_path_for_loss(paths, loss_name, state=state)


def console_log_path_for_loss(paths: PathsConfig, loss_name: str, *, state: str | None = None) -> Path:
    return _console_log_path_for_loss(paths, loss_name, state=state)


def write_training_status(paths: PathsConfig, loss_name: str, status: str, **kwargs: Any) -> None:
    _write_training_status(paths, loss_name, status, **kwargs)


def load_training_status(
    paths: PathsConfig,
    loss_name: str,
    *,
    prefer_cache: bool = False,
) -> dict[str, Any]:
    return _load_training_status(paths, loss_name, prefer_cache=prefer_cache)


def clear_cached_training_status(paths: PathsConfig, loss_name: str) -> None:
    _clear_cached_training_status(paths, loss_name)


def dataset_progress_message(message: str) -> str:
    return _dataset_progress_message(message)


def monitor_multi_loss_dashboard(
    *,
    paths: PathsConfig,
    losses: list[str],
    stop_event: threading.Event,
    prefer_cache: bool,
    refresh_interval_seconds: float = SHARED_DASHBOARD_REFRESH_INTERVAL_SECONDS,
) -> None:
    _monitor_multi_loss_dashboard(
        paths=paths,
        losses=losses,
        stop_event=stop_event,
        prefer_cache=prefer_cache,
        refresh_interval_seconds=refresh_interval_seconds,
    )


def build_failure_summary(
    paths: PathsConfig,
    loss_name: str,
    returncode: int | None,
    *,
    state: str | None = None,
) -> str:
    return _build_failure_summary(paths, loss_name, returncode, state=state)
