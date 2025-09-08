from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from constelx.cli import app


def test_agent_guard_geom_validate_skips_invalid(tmp_path: Path) -> None:
    # Create an invalid boundary seed with excessive helical amplitude
    b = {
        "r_cos": [[0.0 for _ in range(9)] for _ in range(5)],
        "r_sin": None,
        "z_cos": None,
        "z_sin": [[0.0 for _ in range(9)] for _ in range(5)],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }
    b["r_cos"][0][4] = 1.0  # base radius
    b["r_cos"][1][5] = 2.0  # too large
    b["z_sin"][1][5] = 2.0  # too large

    seeds = tmp_path / "seeds.jsonl"
    seeds.write_text(json.dumps({"boundary": b}) + "\n")

    runs_dir = tmp_path / "runs"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "agent",
            "run",
            "--nfp",
            "3",
            "--budget",
            "1",
            "--seed",
            "0",
            "--runs-dir",
            str(runs_dir),
            "--init-seeds",
            str(seeds),
            "--guard-geom-validate",
        ],
    )
    assert result.exit_code == 0
    subdirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    assert subdirs
    out = subdirs[0]
    metrics_csv = (out / "metrics.csv").read_text()
    assert "invalid_geometry" in metrics_csv
