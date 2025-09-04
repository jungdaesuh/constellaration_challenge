from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from constelx.cli import app

runner = CliRunner()


def test_eval_forward_example_runs_and_prints_metrics():
    result = runner.invoke(app, ["eval", "forward", "--example"])
    assert result.exit_code == 0
    # Expect one of the placeholder metrics in output
    assert "placeholder_metric" in result.stdout


def test_eval_score_reads_metrics_json(tmp_path: Path):
    metrics = {"a": 1.25, "b": 2.25}
    p = tmp_path / "metrics.json"
    p.write_text(json.dumps(metrics))
    result = runner.invoke(app, ["eval", "score", "--metrics-json", str(p)])
    assert result.exit_code == 0
    assert "score = 3.5" in result.stdout
