from __future__ import annotations

import csv
from pathlib import Path

import pytest

from constelx.metrics import ProvenanceError, ensure_real_metrics_csv, ensure_real_rows


def test_ensure_real_rows_accepts_real_only() -> None:
    rows = [
        {"source": "real", "evaluator_score": "0.5"},
        {"source": "REAL", "evaluator_score": 0.25},
    ]
    ensure_real_rows(rows)


@pytest.mark.parametrize(
    "row",
    [
        {"source": "placeholder", "evaluator_score": "0.5"},
        {"source": "real", "evaluator_score": ""},
        {"source": "real", "evaluator_score": None},
        {"source": "real", "evaluator_score": "nan"},
    ],
)
def test_ensure_real_rows_rejects_invalid(row: dict[str, object]) -> None:
    with pytest.raises(ProvenanceError):
        ensure_real_rows([row])


def test_ensure_real_metrics_csv(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "evaluator_score"])
        writer.writeheader()
        writer.writerow({"source": "real", "evaluator_score": "1.0"})
    ensure_real_metrics_csv(metrics_path)


def test_ensure_real_metrics_csv_raises(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "evaluator_score"])
        writer.writeheader()
        writer.writerow({"source": "synthetic", "evaluator_score": "1.0"})
    with pytest.raises(ProvenanceError):
        ensure_real_metrics_csv(metrics_path)
