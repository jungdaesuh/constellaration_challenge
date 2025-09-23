from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from constelx.cli import app


def test_agent_run_prior_uses_default_model(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("CONSTELX_DEV", "1")
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
            "--seed-mode",
            "prior",
            "--no-use-physics",
        ],
    )
    assert result.exit_code == 0, result.output
    subdirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    assert subdirs, "expected a run directory"
    metrics = subdirs[0] / "metrics.csv"
    assert metrics.exists()
    # ensure default model was resolved (logged in config)
    cfg = subdirs[0] / "config.yaml"
    content = cfg.read_text()
    assert "seeds_prior_hf_gmm.joblib" in content
