from __future__ import annotations

import pandas as pd
from typer.testing import CliRunner

from constelx.cli import app
from constelx.data.seeds_prior import FeasibilitySpec, SeedsPriorConfig, train_prior


def test_agent_seed_mode_prior(tmp_path, seeds_prior_records) -> None:
    config = SeedsPriorConfig(
        pca_components=5,
        generator="gmm",
        gmm_components=5,
        random_state=2,
    )
    model = train_prior(
        seeds_prior_records,
        config,
        FeasibilitySpec(field="metrics.feasible"),
        nfp=3,
    )
    model_path = tmp_path / "prior.joblib"
    model.save(model_path)

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
            "3",
            "--seed",
            "4",
            "--runs-dir",
            str(runs_dir),
            "--seed-mode",
            "prior",
            "--seed-prior",
            str(model_path),
            "--seed-prior-min-prob",
            "0.0",
            "--seed-prior-batch",
            "12",
            "--seed-prior-draw-batches",
            "6",
        ],
    )
    assert result.exit_code == 0, result.output
    run_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    assert run_dirs, "agent run directory not created"
    metrics_csv = run_dirs[0] / "metrics.csv"
    df = pd.read_csv(metrics_csv)
    assert "seed_source" in df.columns
    assert (df["seed_source"] == "prior").any()
