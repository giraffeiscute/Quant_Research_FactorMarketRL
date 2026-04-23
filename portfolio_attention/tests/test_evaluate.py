from __future__ import annotations

from inspect import signature

import pytest

from portfolio_attention.evaluate import run_evaluation


def test_run_evaluation_signature_rejects_legacy_top_k_argument() -> None:
    run_evaluation_signature = signature(run_evaluation)
    assert "top_k" not in run_evaluation_signature.parameters

    with pytest.raises(TypeError):
        run_evaluation_signature.bind_partial(
            data_config=object(),
            paths=object(),
            top_k=5,
        )
