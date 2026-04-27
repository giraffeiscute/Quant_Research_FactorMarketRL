"""Backward-compatible module alias for the training CLI entrypoint."""

import sys

from .cli import train as _impl

sys.modules[__name__] = _impl
