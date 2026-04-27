"""Backward-compatible module alias for training engine primitives."""

import sys

from .training import engine as _impl

sys.modules[__name__] = _impl
