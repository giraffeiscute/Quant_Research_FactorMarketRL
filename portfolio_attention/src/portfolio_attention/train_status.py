"""Backward-compatible module alias for training status helpers."""

import sys

from .training import status as _impl

sys.modules[__name__] = _impl
