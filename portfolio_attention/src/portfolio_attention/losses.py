"""Backward-compatible module alias for portfolio loss helpers."""

import sys

from .model import losses as _impl

sys.modules[__name__] = _impl
