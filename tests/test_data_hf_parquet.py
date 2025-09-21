from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import Dataset

from constelx.data.constellaration import to_parquet


def _nested_sample() -> dict:
    # Minimal nested boundary/metrics record resembling HF dataset layout
    return {
        "boundary": {
            "n_field_periods": 3,
            "is_stellarator_symmetric": True,
            "r_cos": [[0.0, 0.0, 0.0, 0.0, 0.0, -0.05]],
            "z_sin": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.05]],
        },
        "metrics": {
            "placeholder_metric": 0.0025,
        },
    }


def test_to_parquet_flattens_nested(tmp_path: Path) -> None:
    ds = Dataset.from_list([_nested_sample() for _ in range(4)])
    out = to_parquet(ds, tmp_path / "subset.parquet")
    assert out.exists()
    df = pd.read_parquet(out)
    # Expect flattened dot-keys for boundary arrays
    assert any(c.startswith("boundary.r_cos.") for c in df.columns)
    assert any(c.startswith("boundary.z_sin.") for c in df.columns)
    # Metrics should be present and numeric
    assert "metrics.placeholder_metric" in df.columns
    assert pd.api.types.is_numeric_dtype(df["metrics.placeholder_metric"])  # type: ignore[index]
