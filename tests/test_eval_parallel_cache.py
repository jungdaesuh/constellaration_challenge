from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from constelx.eval import forward, forward_many
from constelx.physics.constel_api import example_boundary


def test_forward_writes_cache(tmp_path: Path) -> None:
    b = example_boundary()
    cdir = tmp_path / "cache"
    # First run: compute and cache
    m1 = forward(b, cache_dir=cdir)
    assert any(p.suffix == ".json" for p in cdir.glob("*.json"))
    # Second run: should return same result using cache
    m2 = forward(b, cache_dir=cdir)
    assert m1 == m2


def test_forward_many_parallel_and_cache(tmp_path: Path) -> None:
    bs = [example_boundary() for _ in range(4)]
    cdir = tmp_path / "cache"
    # Parallel call
    ms = forward_many(bs, max_workers=2, cache_dir=cdir)
    assert len(ms) == 4
    # Boundaries are identical, so at least 1 cache file should exist
    assert len(list(cdir.glob("*.json"))) >= 1
    # Running again should yield identical results
    ms2 = forward_many(bs, max_workers=2, cache_dir=cdir)
    assert ms == ms2


def test_forward_does_not_cache_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[None] = []

    def _fake_real_eval(boundary: dict, problem: str, vmec_opts: dict) -> dict[str, Any]:
        calls.append(None)
        return {"feasible": False, "fail_reason": "timeout_after_1ms", "source": "real"}

    monkeypatch.setattr("constelx.eval._real_eval_with_timeout", _fake_real_eval)
    b = example_boundary()
    cache_dir = tmp_path / "cache"

    forward(b, cache_dir=cache_dir, use_real=True, problem="p1")
    forward(b, cache_dir=cache_dir, use_real=True, problem="p1")

    assert len(calls) == 2, "failure results should not be cached"
    assert not any(cache_dir.glob("*.json")), "cache should not store failure entries"


def test_forward_many_does_not_cache_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bs = [example_boundary(), example_boundary()]
    calls: list[None] = []

    def _fake_eval(boundary: dict, use_real: bool = False) -> dict[str, Any]:
        calls.append(None)
        return {"feasible": False, "fail_reason": "timeout_after_1ms", "source": "placeholder"}

    monkeypatch.setattr("constelx.eval.evaluate_boundary", _fake_eval)
    cache_dir = tmp_path / "cache_many"

    forward_many(bs, cache_dir=cache_dir, use_real=False)
    forward_many(bs, cache_dir=cache_dir, use_real=False)

    assert len(calls) == len(bs) * 2, "each call should recompute after failure"
    assert not any(cache_dir.glob("*.json")), "failure rows should be skipped in cache"
