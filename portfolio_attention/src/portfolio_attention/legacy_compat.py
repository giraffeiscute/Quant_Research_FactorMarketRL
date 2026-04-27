"""Backward-compatible module alias for legacy compatibility helpers."""

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from portfolio_attention.legacy import compat as _impl
else:
    from .legacy import compat as _impl

sys.modules[__name__] = _impl
