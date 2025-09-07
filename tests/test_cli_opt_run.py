from __future__ import annotations

from typer.testing import CliRunner

from constelx.cli import app


def test_opt_run_trust_constr_smoke() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "opt",
            "run",
            "--baseline",
            "trust-constr",
            "--nfp",
            "3",
            "--budget",
            "5",
        ],
    )
    assert result.exit_code == 0
    assert "Best x:" in result.stdout


def test_opt_run_alm_smoke() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "opt",
            "run",
            "--baseline",
            "alm",
            "--nfp",
            "3",
            "--budget",
            "5",
        ],
    )
    assert result.exit_code == 0
    assert "Best x:" in result.stdout


def test_opt_run_unknown_baseline_errors() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "opt",
            "run",
            "--baseline",
            "unknown",
        ],
    )
    assert result.exit_code != 0
    assert "Unknown baseline" in result.stdout or "Unknown baseline" in result.stderr
