"""Backward-compatible module alias for training finalization helpers."""

import sys

from .training import finalization as _impl

sys.modules[__name__] = _impl
