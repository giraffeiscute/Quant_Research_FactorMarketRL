"""Backward-compatible module alias for Lightning run safety helpers."""

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from portfolio_attention.lightning import run_safety as _impl
else:
    from .lightning import run_safety as _impl

if __name__ != "__main__":
    sys.modules[__name__] = _impl
