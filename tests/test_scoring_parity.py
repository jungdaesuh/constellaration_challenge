import json
from pathlib import Path

import pytest


px = pytest.importorskip("constellaration")  # type: ignore[assignment]


def _load_boundaries(p: Path) -> list[dict]:
    out: list[dict] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def test_eval_score_matches_official_aggregator_for_p1() -> None:
    from constelx.physics import proxima_eval
    from constelx.eval import score as eval_score

    fixtures = Path(__file__).parent / "fixtures" / "boundaries_p1_small.jsonl"
    assert fixtures.exists(), "missing fixture: boundaries_p1_small.jsonl"
    bnds = _load_boundaries(fixtures)
    assert bnds, "no boundaries loaded from fixture"

    for b in bnds:
        metrics, _info = proxima_eval.forward_metrics(b, problem="p1")
        # Official aggregator
        s_off = proxima_eval.score("p1", metrics)
        # Our eval.score delegates to official when problem is provided
        s_eval = eval_score(metrics, problem="p1")
        assert pytest.approx(s_off, rel=1e-12, abs=1e-12) == s_eval

