from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from constelx.cli import app


def test_cli_ablate_spec_plan(tmp_path: Path) -> None:
    # Build a tiny spec with two variants and two seeds
    spec = {
        "base": {"nfp": 3, "budget": 3},
        "seeds": [0, 1],
        "variants": [
            {"name": "baseline", "overrides": {}},
            {
                "name": "eci_linear",
                "overrides": {
                    "correction": "eci_linear",
                    "constraints": [
                        {
                            "rhs": 0.0,
                            "coeffs": [
                                {"field": "r_cos", "i": 1, "j": 5, "c": 1.0},
                                {"field": "z_sin", "i": 1, "j": 5, "c": 1.0},
                            ],
                        }
                    ],
                },
            },
        ],
    }
    spec_path = tmp_path / "plan.json"
    spec_path.write_text(json.dumps(spec))

    runs_dir = tmp_path / "ablations"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "ablate",
            "run",
            "--spec",
            str(spec_path),
            "--runs-dir",
            str(runs_dir),
        ],
    )
    assert result.exit_code == 0

    roots = [p for p in runs_dir.iterdir() if p.is_dir()]
    assert len(roots) == 1
    root = roots[0]
    details = root / "details.csv"
    summary = root / "summary.csv"
    assert details.exists() and summary.exists()
    content = summary.read_text()
    assert "baseline" in content and "eci_linear" in content
