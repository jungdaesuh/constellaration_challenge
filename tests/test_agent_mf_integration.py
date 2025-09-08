from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from constelx.agents.simple_agent import AgentConfig, run as run_agent


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def test_agent_writes_phase_with_mf_proxy(tmp_path: Path) -> None:
    out_dir = tmp_path / "runs"
    cfg = AgentConfig(
        nfp=3,
        seed=0,
        out_dir=out_dir,
        algo="random",
        budget=5,
        max_workers=1,
        cache_dir=None,
        use_physics=False,  # placeholder path
        problem=None,
        mf_proxy=True,
        mf_threshold=0.5,
    )
    run_path = run_agent(cfg)
    rows = _read_csv(run_path / "metrics.csv")
    assert rows, "metrics.csv should not be empty"
    assert "phase" in rows[0], "phase column must be present when mf_proxy is enabled"
    assert all(row.get("phase") == "proxy" for row in rows)


def test_agent_real_phase_gated(tmp_path: Path) -> None:
    # Only run when physics tests are enabled and constellaration is available
    run_phys = os.getenv("CONSTELX_RUN_PHYSICS_TESTS", "0").lower() in {"1", "true", "yes"}
    if not run_phys:
        pytest.skip("physics-gated test; set CONSTELX_RUN_PHYSICS_TESTS=1 to enable")
    pytest.importorskip("constellaration")

    out_dir = tmp_path / "runs"
    cfg = AgentConfig(
        nfp=3,
        seed=0,
        out_dir=out_dir,
        algo="random",
        budget=5,
        max_workers=1,
        cache_dir=None,
        use_physics=True,
        problem="p1",
        mf_proxy=True,
        mf_quantile=0.6,
    )
    run_path = run_agent(cfg)
    rows = _read_csv(run_path / "metrics.csv")
    assert rows, "metrics.csv should not be empty"
    assert "phase" in rows[0]
    # At least one real-phase row expected when physics is available
    assert any(row.get("phase") == "real" for row in rows)

