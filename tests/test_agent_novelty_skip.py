from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from constelx.cli import app


def test_agent_novelty_skip_with_duplicate_seeds(tmp_path: Path) -> None:
    # Create two identical seed boundaries
    from constelx.physics.constel_api import example_boundary

    b = example_boundary()
    seeds_path = tmp_path / "seeds.jsonl"
    with seeds_path.open("w") as fh:
        fh.write(json.dumps({"boundary": b}) + "\n")
        fh.write(json.dumps({"boundary": b}) + "\n")

    runner = CliRunner()
    out_dir = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "agent",
            "run",
            "--nfp",
            "3",
            "--budget",
            "4",
            "--runs-dir",
            str(out_dir),
            "--init-seeds",
            str(seeds_path),
            "--novelty-skip",
            "--novelty-eps",
            "1e-12",
        ],
    )
    assert result.exit_code == 0
    # Find the latest run directory and inspect metrics.csv
    runs = sorted(out_dir.iterdir())
    assert runs, "no run folder created"
    metrics = runs[-1] / "metrics.csv"
    assert metrics.exists(), "metrics.csv missing"
    text = metrics.read_text()
    # Ensure at least one novelty skip was recorded
    assert "duplicate_novelty" in text
