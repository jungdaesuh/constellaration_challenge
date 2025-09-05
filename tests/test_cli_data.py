from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from constelx.cli import app


def test_data_fetch_writes_parquet(tmp_path: Path) -> None:
    runner = CliRunner()
    cache_dir = tmp_path / "cache"
    result = runner.invoke(
        app,
        [
            "data",
            "fetch",
            "--nfp",
            "3",
            "--limit",
            "8",
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert result.exit_code == 0
    out = cache_dir / "subset.parquet"
    assert out.exists()
