from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Iterable, Mapping


class ProvenanceError(RuntimeError):
    """Raised when provenance validation encounters non-real rows."""


def _normalize_source(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip().lower()
    return str(value).strip().lower()


def ensure_real_rows(rows: Iterable[Mapping[str, object]]) -> None:
    """Ensure every row comes from the real evaluator and has a numeric score."""

    empty = True
    for idx, row in enumerate(rows, start=1):
        empty = False
        src = _normalize_source(row.get("source"))
        if src != "real":
            raise ProvenanceError(
                f"metrics row {idx} has source={row.get('source')!r}; expected 'real'"
            )
        eval_score = row.get("evaluator_score")
        if eval_score in (None, ""):
            raise ProvenanceError(f"metrics row {idx} missing evaluator_score")
        try:
            numeric = float(eval_score)
        except (TypeError, ValueError):
            raise ProvenanceError(
                f"metrics row {idx} has non-numeric evaluator_score={eval_score!r}"
            ) from None
        if not math.isfinite(numeric):
            raise ProvenanceError(
                f"metrics row {idx} has non-finite evaluator_score={eval_score!r}"
            )
    if empty:
        raise ProvenanceError("metrics table is empty; expected at least one real row")


def ensure_real_metrics_csv(path: Path) -> None:
    """Read ``path`` (CSV) and ensure provenance invariants hold."""

    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        ensure_real_rows(reader)


def _main(argv: list[str]) -> int:
    if not argv:
        print("Usage: python -m constelx.metrics.provenance <metrics.csv>", file=sys.stderr)
        return 2
    path = Path(argv[0])
    try:
        ensure_real_metrics_csv(path)
    except FileNotFoundError:
        print(f"metrics file not found: {path}", file=sys.stderr)
        return 2
    except ProvenanceError as exc:
        print(f"provenance check failed: {exc}", file=sys.stderr)
        return 1
    print("metrics provenance OK")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via workflow hook
    raise SystemExit(_main(sys.argv[1:]))
