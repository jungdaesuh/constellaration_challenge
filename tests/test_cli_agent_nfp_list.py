from __future__ import annotations

import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from constelx.cli import app


def test_agent_nfp_list_round_robin_and_provenance(tmp_path: Path) -> None:
    runner = CliRunner()
    runs_dir = tmp_path / "runs"
    res = runner.invoke(
        app,
        [
            "agent",
            "run",
            "--nfp-list",
            "3,4",
            "--budget",
            "4",
            "--seed",
            "0",
            "--runs-dir",
            str(runs_dir),
        ],
    )
    assert res.exit_code == 0

    # Locate the run folder and read artifacts
    subdirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    assert subdirs, "no run directory created"
    out = subdirs[0]

    # Check proposals.jsonl contains an nfp field in round-robin order
    props_lines = (out / "proposals.jsonl").read_text().splitlines()
    assert len(props_lines) == 4
    props_nfp = [json.loads(line).get("nfp") for line in props_lines]
    assert props_nfp == [3, 4, 3, 4]

    # Check metrics.csv has an nfp column with the same order
    rows = list(csv.DictReader((out / "metrics.csv").open()))
    assert rows, "metrics.csv should have rows"
    assert "nfp" in rows[0], "missing nfp column in metrics.csv"
    nfp_vals = [int(r["nfp"]) for r in rows]
    assert nfp_vals == [3, 4, 3, 4]
