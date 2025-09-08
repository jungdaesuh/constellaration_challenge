from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from constelx.eval import forward, forward_many
from constelx.eval import score as eval_score
from constelx.physics.constel_api import example_boundary


def test_score_infinite_on_failure_flags() -> None:
    # If metrics indicate failure, aggregator should return +inf deterministically.
    metrics_fail_1: Dict[str, Any] = {"feasible": False, "fail_reason": "timeout"}
    metrics_fail_2: Dict[str, Any] = {"feasible": True, "fail_reason": "worker_error"}
    assert eval_score(metrics_fail_1) == float("inf")
    assert eval_score(metrics_fail_2) == float("inf")


def test_forward_sets_placeholder_provenance(tmp_path: Path) -> None:
    b = example_boundary()
    out = forward(b, cache_dir=tmp_path, use_real=False)
    assert out.get("source") == "placeholder"
    # Ensure cached result round-trips without elapsed_ms and preserves source
    out2 = forward(b, cache_dir=tmp_path, use_real=False)
    assert "elapsed_ms" not in out2
    assert out2.get("source") == "placeholder"


def test_forward_many_sets_placeholder_provenance(tmp_path: Path) -> None:
    bs = [example_boundary(), example_boundary()]
    res = forward_many(bs, cache_dir=tmp_path, use_real=False, max_workers=1)
    assert len(res) == 2
    assert all(r.get("source") == "placeholder" for r in res)


def test_cache_ttl_env_is_accepted(tmp_path: Path) -> None:
    # TTL acceptance is best-effort; we only check that the env variable path works
    os.environ["CONSTELX_CACHE_TTL_SECONDS"] = "60"
    try:
        b = example_boundary()
        _ = forward(b, cache_dir=tmp_path, use_real=False)
        _ = forward_many([b], cache_dir=tmp_path, use_real=False, max_workers=1)
    finally:
        os.environ.pop("CONSTELX_CACHE_TTL_SECONDS", None)
