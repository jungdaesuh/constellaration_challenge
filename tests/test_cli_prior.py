from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from constelx.cli import app


def test_cli_prior_train_and_sample(tmp_path: Path, seeds_prior_dataset_jsonl: Path) -> None:
    runner = CliRunner()
    model_path = tmp_path / "prior.joblib"
    result = runner.invoke(
        app,
        [
            "data",
            "prior-train",
            str(seeds_prior_dataset_jsonl),
            "--out",
            str(model_path),
            "--limit",
            "60",
        ],
    )
    assert result.exit_code == 0, result.output
    assert model_path.exists()

    seeds_path = tmp_path / "seeds.jsonl"
    result = runner.invoke(
        app,
        [
            "data",
            "prior-sample",
            str(model_path),
            "--out",
            str(seeds_path),
            "--count",
            "5",
            "--nfp",
            "3",
            "--min-feasibility",
            "0.0",
            "--batch-size",
            "16",
            "--max-draw-batches",
            "8",
        ],
    )
    assert result.exit_code == 0, result.output
    lines = seeds_path.read_text().strip().splitlines()
    assert len(lines) == 5
    record = json.loads(lines[0])
    assert "boundary" in record and "feasibility_score" in record
