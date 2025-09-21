from __future__ import annotations

import pytest
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


def test_opt_run_ngopt_smoke() -> None:
    pytest.importorskip("nevergrad")
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "opt",
            "run",
            "--baseline",
            "ngopt",
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


def test_opt_run_desc_trust_missing_desc(monkeypatch: pytest.MonkeyPatch) -> None:
    from constelx.optim import desc_trust_region as mod

    def _raise(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("DESC is not installed; install 'constelx[desc]'")

    monkeypatch.setattr(mod, "run_desc_trust_region", _raise)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "opt",
            "run",
            "--baseline",
            "desc-trust",
        ],
    )
    assert result.exit_code == 1
    assert "DESC is not installed" in result.stderr


def test_opt_run_ngopt_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_args, **_kwargs):
        raise RuntimeError(
            "Nevergrad is required for the NGOpt baseline; install 'constelx[evolution]'"
        )

    monkeypatch.setattr("constelx.optim.baselines.run_ngopt", _raise)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "opt",
            "run",
            "--baseline",
            "ngopt",
        ],
    )
    assert result.exit_code != 0
    assert "Nevergrad is required" in result.stdout or "Nevergrad is required" in result.stderr
