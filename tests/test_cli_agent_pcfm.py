from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from constelx.cli import app


def test_agent_with_pcfm_norm_constraint_runs_and_applies(tmp_path: Path) -> None:
    # Constrain (r_cos[1][5], z_sin[1][5]) to lie on circle of radius 0.06
    constraints = [
        {
            "type": "norm_eq",
            "radius": 0.06,
            "terms": [
                {"field": "r_cos", "i": 1, "j": 5, "w": 1.0},
                {"field": "z_sin", "i": 1, "j": 5, "w": 1.0},
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
            "4",
            "--seed",
            "0",
            "--runs-dir",
            str(runs_dir),
            "--correction",
            "pcfm",
            "--constraints-file",
            str(constraints_path),
        ],
    )
    assert result.exit_code == 0

    # Verify proposals exist and first boundary satisfies the norm constraint approximately
    subdirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    assert len(subdirs) == 1
    out = subdirs[0]
    proposals = (out / "proposals.jsonl").read_text().splitlines()
    assert len(proposals) >= 1
    first = json.loads(proposals[0])
    b = first["boundary"]
    x = float(b["r_cos"][1][5])
    y = float(b["z_sin"][1][5])
    assert abs((x * x + y * y) - 0.06 * 0.06) < 1e-4

