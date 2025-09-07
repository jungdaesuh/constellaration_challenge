import json
import os
from pathlib import Path

import pytest

# Skip entire module unless explicitly enabled to avoid heavy physics imports by default.
RUN_PHYS = os.getenv("CONSTELX_RUN_PHYSICS_TESTS", "0").lower() in {"1", "true", "yes"}
pytestmark = pytest.mark.skipif(
    not RUN_PHYS, reason="Set CONSTELX_RUN_PHYSICS_TESTS=1 to run physics parity tests"
)


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
    pytest.importorskip("constellaration")
    from constelx.eval import score as eval_score
    from constelx.physics import proxima_eval

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


def test_eval_score_matches_official_aggregator_for_p2() -> None:
    pytest.importorskip("constellaration")
    from constelx.eval import score as eval_score
    from constelx.physics import proxima_eval

    fixtures = Path(__file__).parent / "fixtures" / "boundaries_p2_small.jsonl"
    assert fixtures.exists(), "missing fixture: boundaries_p2_small.jsonl"
    bnds = _load_boundaries(fixtures)
    assert bnds, "no boundaries loaded from fixture"

    for b in bnds:
        metrics, _info = proxima_eval.forward_metrics(b, problem="p2")
        s_off = proxima_eval.score("p2", metrics)
        s_eval = eval_score(metrics, problem="p2")
        assert pytest.approx(s_off, rel=1e-12, abs=1e-12) == s_eval


def test_eval_score_matches_official_aggregator_for_p3() -> None:
    pytest.importorskip("constellaration")
    from constelx.eval import score as eval_score
    from constelx.physics import proxima_eval

    fixtures = Path(__file__).parent / "fixtures" / "boundaries_p3_small.jsonl"
    assert fixtures.exists(), "missing fixture: boundaries_p3_small.jsonl"
    bnds = _load_boundaries(fixtures)
    assert bnds, "no boundaries loaded from fixture"

    for b in bnds:
        metrics, _info = proxima_eval.forward_metrics(b, problem="p3")
        s_off = proxima_eval.score("p3", metrics)
        s_eval = eval_score(metrics, problem="p3")
        assert pytest.approx(s_off, rel=1e-12, abs=1e-12) == s_eval
