from __future__ import annotations

from pathlib import Path

from constelx.data.results_db import ResultsDB


def test_results_db_persistence_and_novelty(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    db = ResultsDB(path)
    boundary = {"x": 0.1, "y": 0.2}
    assert db.is_novel(boundary)
    db.add(boundary, {"score": 1.0})
    assert not db.is_novel(boundary)
    db.save()

    # Reload and ensure persistence
    db2 = ResultsDB(path)
    assert not db2.is_novel(boundary)
    assert db2.is_novel({"x": 0.2, "y": 0.3})
    # Different keys should be considered novel
    assert db2.is_novel({"x": 0.1, "y": 0.2, "z": 0.0})
