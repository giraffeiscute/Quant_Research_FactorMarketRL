"""Backward-compatible module alias for config path helpers."""

import sys

from .config import paths as _impl

sys.modules[__name__] = _impl
