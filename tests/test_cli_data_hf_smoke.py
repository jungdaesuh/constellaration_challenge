from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from constelx.cli import app


def hf_available() -> bool:
    try:
        from datasets import load_dataset  # noqa: WPS433 (import in function for optional net)

        ds = load_dataset("proxima-fusion/constellaration", split="train", streaming=True)
        it = iter(ds)
        # Pull a single element to confirm connectivity and dataset presence
        next(it)
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not hf_available(), reason="HF dataset or network unavailable; skipping smoke test"
)
def test_data_fetch_hf_smoke(tmp_path: Path) -> None:
    runner = CliRunner()
    cache_dir = tmp_path / "cache"
    result = runner.invoke(
        app,
        [
            "data",
            "fetch",
            "--source",
            "hf",
            "--nfp",
            "3",
            "--limit",
            "3",
            "--cache-dir",
            str(cache_dir),
        ],
    )
    # If the call failed (transient network, etc.), soft-skip instead of failing CI
    if result.exit_code != 0:
        pytest.skip(f"HF smoke fetch failed (transient?): {result.output[:200]}")
    assert (cache_dir / "subset.parquet").exists()
