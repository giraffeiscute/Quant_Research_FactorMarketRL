"""Safety helpers for Lightning run orchestration."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import signal
from typing import TYPE_CHECKING, Any
import warnings

if TYPE_CHECKING:
    from .config import PathsConfig


_EARLY_WARNING_BUFFER: list[tuple[Warning | str, type[Warning], str, int, str | None]] = []


def _buffer_warning_before_main(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: Any | None = None,
    line: str | None = None,
) -> None:
    del file
    try:
        resolved_lineno = int(lineno)
    except (TypeError, ValueError):
        resolved_lineno = -1
    _EARLY_WARNING_BUFFER.append((message, category, filename, resolved_lineno, line))


# Capture warnings as early as possible so import-time warnings do not flush the terminal.
warnings.showwarning = _buffer_warning_before_main
warnings.simplefilter("always")


def _is_global_rank_zero() -> bool:
    for rank_key in ("RANK", "GLOBAL_RANK"):
        raw_rank = os.environ.get(rank_key)
        if raw_rank is not None:
            try:
                return int(raw_rank) == 0
            except ValueError:
                return raw_rank == "0"
    raw_local_rank = os.environ.get("LOCAL_RANK")
    if raw_local_rank is not None:
        try:
            return int(raw_local_rank) == 0
        except ValueError:
            return raw_local_rank == "0"
    return True


def _emit_lightning_console_message(message: str) -> None:
    if not _is_global_rank_zero():
        return
    print(f"[lightning_train] {message}", flush=True)


class GracefulInterruptController:
    """Capture SIGINT/SIGTERM and expose an explicit interruption check."""

    def __init__(self) -> None:
        self._interrupted = False
        self._installed = False
        self._previous_handlers: dict[int, Any] = {}

    def install(self) -> None:
        if self._installed:
            return
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle_signal)
        self._installed = True

    def restore(self) -> None:
        if not self._installed:
            return
        for signum, previous_handler in self._previous_handlers.items():
            signal.signal(signum, previous_handler)
        self._previous_handlers.clear()
        self._installed = False

    def raise_if_interrupted(self) -> None:
        if self._interrupted:
            raise KeyboardInterrupt("Interrupt requested.")

    @property
    def interrupted(self) -> bool:
        return self._interrupted

    def _handle_signal(self, signum: int, frame: Any | None) -> None:
        del frame
        self._interrupted = True
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = str(signum)
        raise KeyboardInterrupt(f"Received {signal_name}.")


_INTERRUPT_CONTROLLER = GracefulInterruptController()


def _destroy_distributed_process_group_if_initialized() -> None:
    import torch

    if not torch.distributed.is_available():
        return
    if not torch.distributed.is_initialized():
        return
    try:
        torch.distributed.destroy_process_group()
    except Exception:
        # Best-effort teardown during interrupt paths.
        return


def _configure_warning_routing(state: str, paths: PathsConfig) -> None:
    warning_log_path = paths.get_state_logs_dir(state) / "warning.log"
    warning_log_path.parent.mkdir(parents=True, exist_ok=True)

    is_rank_zero = _is_global_rank_zero()
    warning_logger = logging.getLogger("portfolio_attention.warning_routing")
    warning_logger.setLevel(logging.WARNING)
    warning_logger.propagate = False

    for handler in list(warning_logger.handlers):
        warning_logger.removeHandler(handler)
        handler.close()

    if is_rank_zero:
        file_handler = logging.FileHandler(warning_log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.WARNING)
        file_handler.setFormatter(
            logging.Formatter(
                (
                    "%(asctime)s | category=%(warning_category)s | message=%(warning_message)s | "
                    "source=%(source_file)s:%(source_line)s | module=%(source_module)s | rank=%(rank)s"
                ),
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        warning_logger.addHandler(file_handler)
    else:
        warning_logger.addHandler(logging.NullHandler())

    def _route_warning(
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: Any | None = None,
        line: str | None = None,
    ) -> None:
        del file, line
        if not is_rank_zero:
            return
        try:
            source_line = int(lineno)
        except (TypeError, ValueError):
            source_line = -1
        warning_logger.warning(
            "python_warning",
            extra={
                "warning_category": getattr(category, "__name__", str(category)),
                "warning_message": " ".join(str(message).splitlines()),
                "source_file": filename,
                "source_line": source_line,
                "source_module": Path(filename).stem if filename else "<unknown>",
                "rank": os.environ.get("RANK", "0"),
            },
        )

    warnings.showwarning = _route_warning
    warnings.simplefilter("default")
    # Avoid warning-log storms from DataLoader pin-memory deprecations under DDP.
    warnings.filterwarnings(
        "ignore",
        message=r"The argument 'device' of Tensor\.pin_memory\(\) is deprecated\..*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The argument 'device' of Tensor\.is_pinned\(\) is deprecated\..*",
        category=DeprecationWarning,
    )
    buffered_warnings = list(_EARLY_WARNING_BUFFER)
    _EARLY_WARNING_BUFFER.clear()
    for message, category, filename, lineno, line in buffered_warnings:
        _route_warning(
            message,
            category,
            filename,
            lineno,
            line=line,
        )
