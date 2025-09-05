from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from constelx.cli import app


def test_agent_with_pcfm_ratio_constraint(tmp_path: Path) -> None:
    # Enforce z_sin[1][5]/r_cos[1][5] = -1.25
    constraints = [
        {
            "type": "ratio_eq",
            "target": -1.25,
            "num": {"field": "z_sin", "i": 1, "j": 5},
            "den": {"field": "r_cos", "i": 1, "j": 5},
            "eps": 1e-6,
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
            "1",
            "--runs-dir",
            str(runs_dir),
            "--correction",
            "pcfm",
            "--pcfm-gn-iters",
            "3",
            "--constraints-file",
            str(constraints_path),
        ],
    )
    assert result.exit_code == 0

    out = next(p for p in runs_dir.iterdir() if p.is_dir())
    first = json.loads((out / "proposals.jsonl").read_text().splitlines()[0])
    b = first["boundary"]
    x = float(b["r_cos"][1][5])
    y = float(b["z_sin"][1][5])
    ratio = y / (x + 1e-6)
    assert abs(ratio - (-1.25)) < 1e-3
