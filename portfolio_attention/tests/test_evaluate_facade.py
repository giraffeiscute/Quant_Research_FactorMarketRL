from __future__ import annotations

import importlib
from pathlib import Path

from portfolio_attention import evaluate


def test_known_moved_symbol_facade_exports_are_importable() -> None:
    module = importlib.import_module("portfolio_attention.evaluate")
    for symbol_name in evaluate.MOVED_SYMBOL_FACADE_EXPORTS:
        assert hasattr(module, symbol_name), f"Missing moved symbol facade export: {symbol_name}"


def test_split_modules_do_not_import_evaluate_module() -> None:
    package_dir = Path(__file__).resolve().parents[1] / "src" / "portfolio_attention"
    for module_name in ("evaluate_outputs.py", "evaluate_rebuild.py"):
        source = (package_dir / module_name).read_text(encoding="utf-8")
        assert "from .evaluate import" not in source
        assert "from portfolio_attention.evaluate import" not in source
