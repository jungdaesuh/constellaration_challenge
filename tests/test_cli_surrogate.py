from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from typer.testing import CliRunner

from constelx.cli import app


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not torch_available(), reason="PyTorch not installed")
def test_surrogate_train_cli_runs(tmp_path: Path) -> None:
    runner = CliRunner()
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "out"
    # Prepare a tiny subset.parquet via data fetch
    result = runner.invoke(
        app,
        [
            "data",
            "fetch",
            "--limit",
            "6",
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert result.exit_code == 0

    # Train surrogate
    result2 = runner.invoke(
        app,
        [
            "surrogate",
            "train",
            "--cache-dir",
            str(cache_dir),
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result2.exit_code == 0
    assert (out_dir / "mlp.pt").exists()


@pytest.mark.skipif(not torch_available(), reason="PyTorch not installed")
def test_surrogate_train_cli_runs_with_pbfm(tmp_path: Path) -> None:
    runner = CliRunner()
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "out"
    runner.invoke(app, ["data", "fetch", "--limit", "6", "--cache-dir", str(cache_dir)])
    res = runner.invoke(
        app,
        [
            "surrogate",
            "train",
            "--cache-dir",
            str(cache_dir),
            "--out-dir",
            str(out_dir),
            "--use-pbfm",
        ],
    )
    assert res.exit_code == 0
    assert (out_dir / "mlp.pt").exists()
