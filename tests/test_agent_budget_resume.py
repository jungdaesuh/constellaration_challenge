from __future__ import annotations

import csv
from pathlib import Path

from typer.testing import CliRunner

from constelx.cli import app


def count_rows_csv(path: Path) -> int:
    with path.open() as f:
        r = csv.DictReader(f)
        return sum(1 for _ in r)


def test_agent_budget_and_resume(tmp_path: Path) -> None:
    runner = CliRunner()
    runs_dir = tmp_path / "runs"

    # First run with budget 3
    r1 = runner.invoke(
        app,
        [
            "agent",
            "run",
            "--nfp",
            "3",
            "--budget",
            "3",
            "--seed",
            "0",
            "--runs-dir",
            str(runs_dir),
        ],
    )
    assert r1.exit_code == 0
    out = next(p for p in runs_dir.iterdir() if p.is_dir())
    metrics_csv = out / "metrics.csv"
    assert metrics_csv.exists()
    first_rows = count_rows_csv(metrics_csv)
    assert first_rows <= 3

    # Resume to budget 5 on the same directory
    r2 = runner.invoke(
        app,
        [
            "agent",
            "run",
            "--nfp",
            "3",
            "--budget",
            "5",
            "--seed",
            "0",
            "--runs-dir",
            str(runs_dir),
            "--resume",
            str(out),
        ],
    )
    assert r2.exit_code == 0
    second_rows = count_rows_csv(metrics_csv)
    assert second_rows <= 5
    assert second_rows >= first_rows
