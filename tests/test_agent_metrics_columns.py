from __future__ import annotations

import csv
from pathlib import Path

from typer.testing import CliRunner

from constelx.cli import app


def test_agent_metrics_has_required_columns(tmp_path: Path) -> None:
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
            "2",
            "--seed",
            "0",
            "--runs-dir",
            str(runs_dir),
        ],
    )
    assert result.exit_code == 0
    subdirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    assert subdirs, "no run directory created"
    out = subdirs[0]
    csv_path = out / "metrics.csv"
    assert csv_path.exists(), "missing metrics.csv"
    header = next(csv.DictReader(csv_path.open()))
    cols = set(header.keys())
    for c in {"evaluator_score", "agg_score", "elapsed_ms", "feasible", "fail_reason", "source"}:
        assert c in cols, f"missing column: {c}"
