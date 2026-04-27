"""Backward-compatible module alias for training resume helpers."""

import sys

from .training import resume as _impl

sys.modules[__name__] = _impl
