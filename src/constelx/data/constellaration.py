from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from datasets import Dataset, load_dataset


def load(split: str = "train") -> Dataset:
    return load_dataset("proxima-fusion/constellaration", split=split)


def select_columns(ds: Dataset) -> Dataset:
    # Keep only boundary.* and metrics.* columns for seeds/training
    cols = [
        c
        for c in ds.column_names
        if c in {"boundary", "metrics"} or c.startswith("boundary.") or c.startswith("metrics.")
    ]
    return ds.remove_columns([c for c in ds.column_names if c not in cols])


def _flatten_to_dots(prefix: str, value: Any) -> dict[str, Any]:
    """Flatten nested structures under a prefix into dot-separated keys.

    - Dicts: recurse with ``{prefix}.{key}``
    - Lists/tuples: index with integer segments ``{prefix}.{i}`` (and recurse)
    - Scalars: emit ``{prefix}: value``
    """
    out: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for k, v in value.items():
            out.update(_flatten_to_dots(f"{prefix}.{k}", v))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            out.update(_flatten_to_dots(f"{prefix}.{i}", v))
    else:
        out[prefix] = value
    return out


def _flatten_record(rec: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten a record keeping only boundary.* and metrics.* as dot keys.

    This handles both dataset layouts:
    - nested objects: ``{"boundary": {...}, "metrics": {...}}``
    - already-flat: ``{"boundary.r_cos.1.5": 0.1, ...}``
    """
    out: dict[str, Any] = {}
    # Prefer nested objects when present
    b = rec.get("boundary")
    m = rec.get("metrics")
    if isinstance(b, Mapping):
        out.update(_flatten_to_dots("boundary", b))
    if isinstance(m, Mapping):
        out.update(_flatten_to_dots("metrics", m))
    # Also keep any already-flat boundary.* or metrics.* keys
    for k, v in rec.items():
        if isinstance(k, str) and (k.startswith("boundary.") or k.startswith("metrics.")):
            out[k] = v
    return out


def _safe_nfp(rec: dict[str, Any]) -> int:
    """Extract NFP from a dataset record, handling both nested and flat layouts.

    Returns -1 if unavailable or invalid.
    """
    try:
        v = rec.get("boundary.n_field_periods", None)
        if v is None and isinstance(rec.get("boundary"), dict):
            v = rec["boundary"].get("n_field_periods", None)
        return int(v) if v is not None else -1
    except Exception:
        return -1


def filter_nfp(ds: Dataset, nfp: int) -> Dataset:
    target = int(nfp)
    return ds.filter(lambda x: _safe_nfp(x) == target)


def to_parquet(ds: Dataset, out: Path) -> Path:
    """Write a thin ML slice of the HF dataset to Parquet.

    - Keeps only ``boundary.*`` and ``metrics.*`` columns
    - Flattens nested objects into dot-separated keys so downstream code can
      select columns by prefix (e.g., ``boundary.r_cos``)
    """
    # Select relevant columns when the dataset is already columnar
    try:
        ds = select_columns(ds)
    except Exception:
        # If selection fails (e.g., nested schema), proceed and flatten per-record
        pass

    records = [_flatten_record(x) for x in ds.to_list()]
    df = pd.DataFrame.from_records(records)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def make_seeds_jsonl(ds: Dataset, out: Path, k: int = 64) -> Path:
    """Write up to k boundary dicts to JSONL as {"boundary": {...}}.

    Tries nested 'boundary' first; otherwise reconstructs from flattened keys
    like 'boundary.r_cos.1.5' and 'boundary.z_sin.1.5'.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        count = 0
        for rec in ds:
            b: dict[str, Any] | None = None
            if "boundary" in rec and isinstance(rec["boundary"], dict):
                import json

                b = dict(rec["boundary"])  # copy
            else:
                # reconstruct from flattened keys
                rcos_entries: list[tuple[int, int, float]] = []
                zsin_entries: list[tuple[int, int, float]] = []
                nfp_val = rec.get("boundary.n_field_periods")
                try:
                    nfp_int = int(nfp_val) if nfp_val is not None else 3
                except Exception:
                    nfp_int = 3
                for key, val in rec.items():
                    if not key.startswith("boundary."):
                        continue
                    parts = key.split(".")
                    if len(parts) == 4 and parts[1] in {"r_cos", "z_sin"}:
                        try:
                            m = int(parts[2])
                            n = int(parts[3])
                            v = float(val) if val is not None else 0.0
                        except Exception:
                            continue
                        if parts[1] == "r_cos":
                            rcos_entries.append((m, n, v))
                        else:
                            zsin_entries.append((m, n, v))
                if rcos_entries or zsin_entries:
                    max_m = max([m for m, _, _ in (rcos_entries + zsin_entries)], default=0)
                    max_n = max([n for _, n, _ in (rcos_entries + zsin_entries)], default=0)
                    # Build arrays of size (max_m+1) x (max_n+1)
                    r_cos = [[0.0 for _ in range(max_n + 1)] for _ in range(max_m + 1)]
                    z_sin = [[0.0 for _ in range(max_n + 1)] for _ in range(max_m + 1)]
                    for m, n, v in rcos_entries:
                        if m <= max_m and n <= max_n:
                            r_cos[m][n] = v
                    for m, n, v in zsin_entries:
                        if m <= max_m and n <= max_n:
                            z_sin[m][n] = v
                    b = {
                        "r_cos": r_cos,
                        "r_sin": None,
                        "z_cos": None,
                        "z_sin": z_sin,
                        "n_field_periods": int(nfp_int),
                        "is_stellarator_symmetric": True,
                    }
            if b:
                import json

                f.write(json.dumps({"boundary": b}) + "\n")
                count += 1
                if count >= int(k):
                    break
    return out
