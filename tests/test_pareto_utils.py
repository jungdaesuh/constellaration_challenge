from __future__ import annotations

import pytest

from constelx.optim.pareto import (
    DEFAULT_P3_SCALARIZATION,
    ScalarizationConfig,
    dominates,
    extract_objectives,
    linspace_weights,
    pareto_front,
    pareto_indices,
    scalarize,
)


def test_extract_objectives_from_list() -> None:
    metrics = {"objectives": [1.0, 2.0, 3.0]}
    assert extract_objectives(metrics) == (1.0, 2.0, 3.0)


def test_extract_objectives_from_indexed_keys() -> None:
    metrics = {"objective_0": 1.5, "objective_1": 3.5}
    assert extract_objectives(metrics) == (1.5, 3.5)


def test_scalarize_weighted_sum() -> None:
    cfg = ScalarizationConfig(method="weighted_sum", weights=(0.25, 0.75))
    assert scalarize((4.0, 2.0), cfg) == pytest.approx(0.25 * 4.0 + 0.75 * 2.0)


def test_scalarize_weighted_chebyshev_reference() -> None:
    cfg = ScalarizationConfig(
        method="weighted_chebyshev", weights=(0.5, 0.5), reference_point=(1.0, 1.0)
    )
    expected = max(0.5 * abs(2.0 - 1.0), 0.5 * abs(3.0 - 1.0)) + cfg.rho * (
        0.5 * abs(2.0 - 1.0) + 0.5 * abs(3.0 - 1.0)
    )
    assert scalarize((2.0, 3.0), cfg) == pytest.approx(expected)


def test_dominates_and_pareto_indices() -> None:
    points = [(1.0, 1.0), (2.0, 2.0), (1.0, 3.0), (0.5, 2.0)]
    idx = pareto_indices(points)
    assert set(idx) == {0, 3}
    assert dominates(points[0], points[1])
    assert not dominates(points[2], points[0])


def test_pareto_front_records() -> None:
    records = [
        {"id": "a", "objectives": (1.0, 1.0)},
        {"id": "b", "objectives": (2.0, 2.0)},
        {"id": "c", "objectives": (0.8, 2.5)},
    ]
    front = pareto_front(records, key=lambda rec: rec["objectives"])
    assert {rec["id"] for rec in front} == {"a", "c"}


def test_linspace_weights_two_dim() -> None:
    weights = linspace_weights(dim=2, count=3)
    assert weights == [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]


def test_default_p3_scalarization_stable() -> None:
    score = scalarize((0.2, 0.4), DEFAULT_P3_SCALARIZATION)
    assert score > 0.0
