from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from constelx.cli import app


def test_opt_pareto_smoke(tmp_path: Path) -> None:
    out_path = tmp_path / "pareto.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "opt",
            "pareto",
            "--budget",
            "4",
            "--sweeps",
            "3",
            "--seed",
            "0",
            "--json-out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0
    assert "Pareto front size" in result.stdout
    data = json.loads(out_path.read_text())
    assert "front" in data and data["front"]
