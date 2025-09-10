from __future__ import annotations

import csv
from pathlib import Path

from typer.testing import CliRunner

from constelx.cli import app


def test_metrics_csv_contains_nfp_and_value_matches(tmp_path: Path) -> None:
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
            "3",
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

    reader = csv.DictReader(csv_path.open())
    rows = list(reader)
    assert rows, "metrics.csv is empty"
    header_cols = reader.fieldnames or []
    assert "nfp" in header_cols, "missing nfp column"
    # Verify values are present and equal to 3 for all rows
    for row in rows:
        assert row.get("nfp") not in (None, ""), "nfp value missing in a row"
        assert int(float(row["nfp"])) == 3

