"""Lightweight dataset utilities for constelx.data.

This module intentionally provides a deterministic synthetic fallback so tests
and local development do not depend on remote datasets or network access.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd
from datasets import Dataset


@dataclass
class Example:
    nfp: int
    r_cos_15: float
    z_sin_15: float


def _synthetic_examples(count: int = 64, *, nfp: int = 3, seed: int = 0) -> List[Example]:
    rng = np.random.default_rng(seed)
    xs = rng.normal(loc=-0.05, scale=0.03, size=count)
    ys = rng.normal(loc=0.05, scale=0.03, size=count)
    return [Example(nfp=nfp, r_cos_15=float(xs[i]), z_sin_15=float(ys[i])) for i in range(count)]


def _to_hf_records(examples: Sequence[Example]) -> List[Mapping[str, Any]]:
    # Shape as a simple record with explicit columns; keep names friendly for flattening
    out: List[Mapping[str, Any]] = []
    for e in examples:
        out.append(
            {
                "boundary.n_field_periods": int(e.nfp),
                # Flatten a few boundary coefficients; dot-separated to play nice with Parquet
                "boundary.r_cos.1.5": float(e.r_cos_15),
                "boundary.z_sin.1.5": float(e.z_sin_15),
                # Provide a placeholder metric compatible with downstream examples
                "metrics.placeholder_metric": float(
                    e.r_cos_15 * e.r_cos_15 + e.z_sin_15 * e.z_sin_15
                ),
            }
        )
    return out


def fetch_dataset(count: int = 128, *, nfp: int = 3, seed: int = 0) -> Dataset:
    """Return a HuggingFace ``datasets.Dataset`` with minimal columns.

    Columns:
    - ``boundary.n_field_periods``: integer NFP filter key
    - ``boundary.r_cos.1.5`` and ``boundary.z_sin.1.5``: boundary coefficients
    - ``metrics.placeholder_metric``: numeric scalar target
    """
    examples = _synthetic_examples(count, nfp=nfp, seed=seed)
    recs = _to_hf_records(examples)
    return Dataset.from_list(list(recs))


def _flatten_record(rec: Mapping[str, Any]) -> MutableMapping[str, Any]:
    # Already flat; return as-is while ensuring JSON-serializable types
    out: dict[str, Any] = {}
    for k, v in rec.items():
        if isinstance(v, (np.generic,)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def save_subset(ds: Dataset, cache_dir: Path) -> Path:
    """Save a dataset subset to Parquet under ``cache_dir/subset.parquet``.

    The Parquet file flattens records so downstream code can select columns by
    prefix (e.g., ``boundary.r_cos``).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Convert the dataset into a flat DataFrame
    records = [_flatten_record(x) for x in ds.to_list()]
    df = pd.DataFrame.from_records(records)
    out = cache_dir / "subset.parquet"
    df.to_parquet(out, index=False)
    return out


__all__ = ["fetch_dataset", "save_subset"]
