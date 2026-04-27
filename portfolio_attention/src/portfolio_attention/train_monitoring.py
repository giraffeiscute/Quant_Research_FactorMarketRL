"""Backward-compatible module alias for training monitoring helpers."""

import sys

from .training import monitoring as _impl

sys.modules[__name__] = _impl
