from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from constelx.cli import app


def test_agent_with_eci_linear_runs_and_applies_constraint(tmp_path: Path) -> None:
    # Define a simple linear constraint: r_cos[1][5] + z_sin[1][5] = 0
    constraints = [
        {
            "rhs": 0.0,
            "coeffs": [
                {"field": "r_cos", "i": 1, "j": 5, "c": 1.0},
                {"field": "z_sin", "i": 1, "j": 5, "c": 1.0},
            ],
        }
    ]
    constraints_path = tmp_path / "constraints.json"
    constraints_path.write_text(json.dumps(constraints))

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
            "--correction",
            "eci_linear",
            "--constraints-file",
            str(constraints_path),
        ],
    )
    assert result.exit_code == 0

    # Load the first proposal and verify the constraint holds
    subdirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    assert len(subdirs) == 1
    out = subdirs[0]
    proposals = (out / "proposals.jsonl").read_text().splitlines()
    assert len(proposals) >= 1
    first = json.loads(proposals[0])
    b = first["boundary"]
    lhs = float(b["r_cos"][1][5]) + float(b["z_sin"][1][5])
    assert abs(lhs - 0.0) < 1e-9
