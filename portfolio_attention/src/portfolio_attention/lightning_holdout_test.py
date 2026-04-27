"""Backward-compatible module alias for Lightning holdout testing."""

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from portfolio_attention.lightning import holdout_test as _impl
else:
    from .lightning import holdout_test as _impl

if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
