from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, load_dataset


def load(split: str = "train") -> Dataset:
    return load_dataset("proxima-fusion/constellaration", split=split)


def select_columns(ds: Dataset) -> Dataset:
    # Keep only boundary.* and metrics.* columns for seeds/training
    cols = [c for c in ds.column_names if c.startswith("boundary.") or c.startswith("metrics.")]
    return ds.remove_columns([c for c in ds.column_names if c not in cols])


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
    df = pd.DataFrame(ds.to_list())
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
