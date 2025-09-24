"""Simple results database with novelty checks.

This module provides a very small utility for persisting optimization results
and determining whether a proposed boundary is sufficiently different from
previously seen ones.  It is intentionally lightweight so tests and examples
can run without external services.

Example
-------
>>> from pathlib import Path
>>> db = ResultsDB(Path('results.jsonl'))
>>> boundary = {'x': 0.1, 'y': 0.2}
>>> db.is_novel(boundary)
True
>>> db.add(boundary, {'score': 1.0})
>>> db.is_novel(boundary)
False
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass
class Result:
    """Container for a single optimization result."""

    boundary: Mapping[str, float]
    metrics: Mapping[str, float]

    def to_dict(self) -> Mapping[str, Mapping[str, float]]:
        return {"boundary": dict(self.boundary), "metrics": dict(self.metrics)}


class ResultsDB:
    """A tiny JSONL-backed results archive with novelty checks.

    Parameters
    ----------
    path:
        Location of the JSONL file used for persistence.  If the file exists it
        is loaded on instantiation.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.records: List[Result] = []
        if self.path.exists():
            with self.path.open() as fh:
                for line in fh:
                    data = json.loads(line)
                    self.records.append(Result(data["boundary"], data["metrics"]))

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def add(self, boundary: Mapping[str, float], metrics: Mapping[str, float]) -> None:
        """Append a result to the in-memory store."""

        self.records.append(Result(dict(boundary), dict(metrics)))

    def save(self) -> None:
        """Write all records to ``self.path`` as JSON lines."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w") as fh:
            for rec in self.records:
                json.dump(rec.to_dict(), fh)
                fh.write("\n")

    # ------------------------------------------------------------------
    # Novelty check
    # ------------------------------------------------------------------
    def _vector(self, boundary: Mapping[str, float], keys: List[str]) -> NDArray[np.float64]:
        """Return a vector of boundary values ordered by ``keys``."""

        return np.asarray([boundary[k] for k in keys], dtype=float)

    def is_novel(
        self,
        boundary: Mapping[str, float],
        *,
        atol: float = 1e-8,
        rtol: float = 0.0,
    ) -> bool:
        """Return ``True`` if ``boundary`` is novel within tolerances.

        Boundaries with different keys are treated as novel.
        """

        keys = sorted(boundary)
        cand = self._vector(boundary, keys)
        for rec in self.records:
            if set(rec.boundary) != set(boundary):
                # Different keys; consider novel
                continue
            vec = self._vector(rec.boundary, keys)
            if np.allclose(vec, cand, atol=atol, rtol=rtol):
                return False
        return True


__all__ = ["ResultsDB", "Result"]
