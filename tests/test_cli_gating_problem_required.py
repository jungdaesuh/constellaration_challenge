from __future__ import annotations

from typer.testing import CliRunner

from constelx.cli import app


def test_agent_requires_problem_with_use_physics() -> None:
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
            "--use-physics",
        ],
    )
    assert result.exit_code != 0
    assert "--problem is required" in result.stdout or "--problem is required" in result.stderr


def test_opt_run_requires_problem_with_use_physics() -> None:
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
            "1",
            "--use-physics",
        ],
    )
    assert result.exit_code != 0
    assert "--problem is required" in result.stdout or "--problem is required" in result.stderr
