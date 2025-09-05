from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from constelx.cli import app


def test_agent_run_small_creates_artifacts(tmp_path: Path) -> None:
    runner = CliRunner()
    runs_dir = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "agent",
            "run",
            "--nfp",
            "3",
            "--budget",
            "6",
            "--seed",
            "0",
            "--runs-dir",
            str(runs_dir),
        ],
    )
    assert result.exit_code == 0

    # There should be one timestamped subdir under runs_dir
    subdirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    assert len(subdirs) == 1
    out = subdirs[0]

    # Check required files
    for name in ["config.yaml", "proposals.jsonl", "metrics.csv", "best.json", "README.md"]:
        assert (out / name).exists(), f"missing {name}"

    best = json.loads((out / "best.json").read_text())
    assert "score" in best and isinstance(best["score"], (int, float))
