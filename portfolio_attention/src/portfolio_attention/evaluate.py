"""Backward-compatible module alias for the evaluation CLI entrypoint."""

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from portfolio_attention.cli import evaluate as _impl
else:
    from .cli import evaluate as _impl

if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
