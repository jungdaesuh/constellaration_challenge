from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from constelx.cli import app
from constelx.physics.constel_api import example_boundary


def _to_plain(obj):
    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover
        np = None  # type: ignore
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_to_plain(x) for x in obj)
    if "np" in locals() and np is not None:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    return obj


def test_agent_consumes_init_seeds_first(tmp_path: Path) -> None:
    # Prepare seeds JSONL with two example boundaries
    seeds_path = tmp_path / "seeds.jsonl"
    b = _to_plain(example_boundary())
    with seeds_path.open("w") as f:
        f.write(json.dumps({"boundary": b}) + "\n")
        f.write(json.dumps({"boundary": b}) + "\n")

    runs_dir = tmp_path / "runs"
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "agent",
            "run",
            "--nfp",
            "3",
            "--budget",
            "2",
            "--seed",
            "0",
            "--runs-dir",
            str(runs_dir),
            "--init-seeds",
            str(seeds_path),
        ],
    )
    assert res.exit_code == 0
    out = next(p for p in runs_dir.iterdir() if p.is_dir())
    props = (out / "proposals.jsonl").read_text().splitlines()
    assert len(props) >= 2
    first = json.loads(props[0])
    second = json.loads(props[1])
    assert first.get("boundary") == b
    assert second.get("boundary") == b
