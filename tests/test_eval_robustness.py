from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from constelx.eval import forward, forward_many
from constelx.eval import score as eval_score
from constelx.physics import proxima_eval
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


def test_forward_vmec_metadata_defaults(tmp_path: Path) -> None:
    b = example_boundary()
    low = forward(
        b,
        cache_dir=tmp_path,
        use_real=False,
        vmec_level="low",
        vmec_hot_restart=True,
        vmec_restart_key="run_low",
    )
    assert low["vmec_level"] == "low"
    assert low["vmec_hot_restart"] is True
    assert low.get("vmec_restart_key") == "run_low"

    high = forward(b, cache_dir=tmp_path, use_real=False, vmec_level="high")
    assert high["vmec_level"] == "high"
    assert high["vmec_hot_restart"] is False
    # Ensure we did not collide with the low-level cache entry
    assert low["vmec_level"] != high["vmec_level"]


def test_forward_respects_vmec_env_defaults(tmp_path: Path) -> None:
    b = example_boundary()
    os.environ["CONSTELX_VMEC_LEVEL"] = "medium"
    os.environ["CONSTELX_VMEC_HOT_RESTART"] = "1"
    try:
        metrics = forward(b, cache_dir=tmp_path, use_real=False)
        assert metrics["vmec_level"] == "medium"
        assert metrics["vmec_hot_restart"] is True
    finally:
        os.environ.pop("CONSTELX_VMEC_LEVEL", None)
        os.environ.pop("CONSTELX_VMEC_HOT_RESTART", None)


def test_proxima_forward_metrics_records_vmec_info() -> None:
    b = example_boundary()
    metrics, info = proxima_eval.forward_metrics(
        b,
        problem="p1",
        vmec_opts={"level": "medium", "hot_restart": True, "restart_key": "abc"},
    )
    assert info["vmec_level"] == "medium"
    assert info["vmec_hot_restart"] is True
    assert info.get("vmec_restart_key") == "abc"
    assert metrics  # basic sanity


def test_cache_ttl_env_is_accepted(tmp_path: Path) -> None:
    # TTL acceptance is best-effort; we only check that the env variable path works
    os.environ["CONSTELX_CACHE_TTL_SECONDS"] = "60"
    try:
        b = example_boundary()
        _ = forward(b, cache_dir=tmp_path, use_real=False)
        _ = forward_many([b], cache_dir=tmp_path, use_real=False, max_workers=1)
    finally:
        os.environ.pop("CONSTELX_CACHE_TTL_SECONDS", None)


def test_forward_logging_when_env_enabled(tmp_path: Path, monkeypatch) -> None:
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("CONSTELX_EVAL_LOG_DIR", str(log_dir))
    b = example_boundary()
    result = forward(b, use_real=False, problem="p1")

    logs = list(log_dir.glob("*.json"))
    assert logs, "expected evaluator log output"
    payload = json.loads(logs[0].read_text())
    assert payload["problem"] == "p1"
    assert payload["cache_hit"] is False
    assert payload["metrics"]["source"] == result.get("source")
