"""Backward-compatible module alias for config validation helpers."""

import sys

from .config import validation as _impl

sys.modules[__name__] = _impl
