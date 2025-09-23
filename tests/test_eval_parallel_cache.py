from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from constelx.eval import forward, forward_many
from constelx.physics.constel_api import example_boundary


def test_forward_writes_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    b = example_boundary()
    cdir = tmp_path / "cache"

    calls: list[None] = []

    def _fake_real_eval(boundary: dict, problem: str, vmec_opts: dict[str, Any]) -> dict[str, Any]:
        calls.append(None)
        # Pretend this is a real, feasible evaluation to exercise caching behavior.
        return {"agg_score": 1.23, "feasible": True, "source": "real"}

    monkeypatch.setattr("constelx.eval._real_eval_with_timeout", _fake_real_eval)
    # First run: compute and cache
    m1 = forward(b, cache_dir=cdir, use_real=True)
    # Second run: should return same result using cache
    m2 = forward(b, cache_dir=cdir, use_real=True)
    assert {k: v for k, v in m1.items() if k != "elapsed_ms"} == {
        k: v for k, v in m2.items() if k != "elapsed_ms"
    }
    assert len(calls) == 1


def test_forward_many_proxy_cache_hits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bs = [example_boundary() for _ in range(3)]
    cdir = tmp_path / "cache"

    calls: list[None] = []

    def _fake_compute(
        boundary: dict[str, Any],
        *,
        use_real: bool | None = None,
        problem: str | None = None,
        attach_proxies: bool = True,
    ) -> dict[str, Any]:
        calls.append(None)
        return {
            "placeholder_metric": 0.5,
            "source": "proxy",
            "qs_residual": 0.01,
            "qi_residual": 0.02,
            "helical_energy": 0.03,
            "mirror_ratio": 0.04,
        }

    monkeypatch.setattr("constelx.physics.metrics.compute", _fake_compute)

    ms = forward_many(
        bs,
        max_workers=1,
        cache_dir=cdir,
        use_real=False,
        mf_proxy=True,
        mf_threshold=0.5,
        mf_metric="qs_residual",
    )
    assert len(ms) == 3
    calls_after_first = len(calls)
    assert calls_after_first >= 1
    ms2 = forward_many(
        bs,
        max_workers=1,
        cache_dir=cdir,
        use_real=False,
        mf_proxy=True,
        mf_threshold=0.5,
        mf_metric="qs_residual",
    )
    assert [{k: v for k, v in r.items() if k != "elapsed_ms"} for r in ms] == [
        {k: v for k, v in r.items() if k != "elapsed_ms"} for r in ms2
    ]
    assert len(calls) == calls_after_first
    assert all(rec.get("source") == "proxy" for rec in ms2)


def test_forward_does_not_cache_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[None] = []

    def _fake_real_eval(boundary: dict, problem: str, vmec_opts: dict) -> dict[str, Any]:
        calls.append(None)
        return {
            "feasible": False,
            "fail_reason": "timeout_after_1ms",
            "source": "placeholder",
            "placeholder_reason": "injected_failure",
        }

    monkeypatch.setattr("constelx.eval._real_eval_with_timeout", _fake_real_eval)
    b = example_boundary()
    cache_dir = tmp_path / "cache"

    res1 = forward(b, cache_dir=cache_dir, use_real=True, problem="p1")
    res2 = forward(b, cache_dir=cache_dir, use_real=True, problem="p1")

    assert len(calls) == 2, "failure results should not be cached"
    assert not any(cache_dir.glob("*.json")), "cache should not store failure entries"
    assert res1.get("source") == "placeholder"
    assert res1.get("feasible") is False
    assert res1.get("fail_reason")
    assert res1.get("placeholder_reason")
    assert res2.get("source") == "placeholder"
    assert res2.get("feasible") is False


def test_forward_many_does_not_cache_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bs = [example_boundary(), example_boundary()]
    calls: list[None] = []

    def _fake_eval(boundary: dict, use_real: bool = False) -> dict[str, Any]:
        calls.append(None)
        return {
            "feasible": False,
            "fail_reason": "timeout_after_1ms",
            "source": "placeholder",
            "placeholder_reason": "injected_failure",
        }

    monkeypatch.setattr("constelx.eval.evaluate_boundary", _fake_eval)
    cache_dir = tmp_path / "cache_many"

    res_batch1 = forward_many(bs, cache_dir=cache_dir, use_real=False)
    res_batch2 = forward_many(bs, cache_dir=cache_dir, use_real=False)

    assert len(calls) == len(bs) * 2, "each call should recompute after failure"
    assert not any(cache_dir.glob("*.json")), "failure rows should be skipped in cache"
    for rec in res_batch1 + res_batch2:
        assert rec.get("source") == "placeholder"
        assert rec.get("feasible") is False
        assert rec.get("fail_reason")
        assert rec.get("placeholder_reason")


def test_real_eval_task_failure_marks_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    from constelx.eval import _real_eval_task

    boundary = example_boundary()

    def _boom(*_args: Any, **_kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        raise RuntimeError("worker boom")

    monkeypatch.setattr("constelx.physics.proxima_eval.forward_metrics", _boom)

    metrics = _real_eval_task((dict(boundary), "p1", {}))

    assert metrics["source"] == "placeholder"
    assert metrics["feasible"] is False
    assert metrics.get("fail_reason", "").startswith("worker_placeholder")
    assert metrics.get("placeholder_reason") == "worker_placeholder"
