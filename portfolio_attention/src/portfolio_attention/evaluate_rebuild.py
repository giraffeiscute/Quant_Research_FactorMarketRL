"""Backward-compatible module alias for evaluation rebuild helpers."""

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from portfolio_attention.cli import evaluate_rebuild as _impl
else:
    from .cli import evaluate_rebuild as _impl

if __name__ == "__main__":
    if hasattr(_impl, "main"):
        _impl.main()
else:
    sys.modules[__name__] = _impl
