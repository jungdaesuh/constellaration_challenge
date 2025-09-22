from __future__ import annotations

from typing import Any, Dict

from constelx.eval import forward
from constelx.physics.constel_api import example_boundary


def test_forward_enforcement_guard(monkeypatch) -> None:
    # Enforce real-only; synthetic placeholder path should raise when dev not enabled
    monkeypatch.setenv("CONSTELX_ENFORCE_REAL", "1")
    monkeypatch.delenv("CONSTELX_DEV", raising=False)

    b: Dict[str, Any] = example_boundary()

    raised = False
    try:
        _ = forward(b, use_real=False)
    except RuntimeError:
        raised = True
    assert raised, "Expected enforcement guard to raise on placeholder path"
