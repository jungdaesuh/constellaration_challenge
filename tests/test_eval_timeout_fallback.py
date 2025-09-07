from __future__ import annotations

import os
import pytest

# Physics-gated to avoid importing heavy deps by default
RUN_PHYS = os.getenv("CONSTELX_RUN_PHYSICS_TESTS", "0").lower() in {"1", "true", "yes"}
pytestmark = pytest.mark.skipif(
    not RUN_PHYS, reason="Set CONSTELX_RUN_PHYSICS_TESTS=1 to run physics timeout tests"
)

from constelx.eval import forward  # noqa: E402
from constelx.physics.constel_api import example_boundary  # noqa: E402


def test_timeout_records_failure_and_fallback(monkeypatch) -> None:
    # Force an unrealistically tiny timeout to trigger timeout path in real eval
    monkeypatch.setenv("CONSTELX_REAL_TIMEOUT_MS", "1")
    b = example_boundary()
    m = forward(b, use_real=True, problem="p1")
    # Either a real quick return or a timeout fallback; accept both but check fields
    assert isinstance(m, dict)
    assert "feasible" in m
    assert "fail_reason" in m
    assert "source" in m
