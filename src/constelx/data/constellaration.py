from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset


def load(split: str = "train") -> Dataset:
    return load_dataset("proxima-fusion/constellaration", split=split)


def select_columns(ds: Dataset) -> Dataset:
    # Keep only boundary.* and metrics.* columns for seeds/training
    cols = [c for c in ds.column_names if c.startswith("boundary.") or c.startswith("metrics.")]
    return ds.remove_columns([c for c in ds.column_names if c not in cols])


def filter_nfp(ds: Dataset, nfp: int) -> Dataset:
    return ds.filter(lambda x: int(x.get("boundary.n_field_periods", 0)) == int(nfp))


def to_parquet(ds: Dataset, out: Path) -> Path:
    df = pd.DataFrame(ds.to_list())
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def make_seeds_jsonl(ds: Dataset, out: Path, k: int = 64) -> Path:
    """Write up to k boundary dicts to JSONL as {"boundary": {...}}.

    Requires the dataset to expose a nested 'boundary' object per record.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        count = 0
        for rec in ds:
            if "boundary" in rec and isinstance(rec["boundary"], dict):
                import json

                f.write(json.dumps({"boundary": rec["boundary"]}) + "\n")
                count += 1
                if count >= int(k):
                    break
    return out
