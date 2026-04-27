"""Backward-compatible wrapper for utility helpers."""

from .common import utils as _impl

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)

__all__ = getattr(_impl, "__all__", [name for name in globals() if not name.startswith("__")])

del _impl, _name
