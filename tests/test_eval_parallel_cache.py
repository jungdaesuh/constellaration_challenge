from __future__ import annotations

from pathlib import Path

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
