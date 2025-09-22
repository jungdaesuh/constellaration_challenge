from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from constelx.physics.constel_api import evaluate_boundary, example_boundary
from constelx.physics.proxima_eval import forward_metrics


class _BoomSurface:
    @classmethod
    def model_validate(cls, boundary: dict[str, Any]) -> dict[str, Any]:
        return dict(boundary)


class _BoomProblem:
    def evaluate(self, _: Any) -> Any:
        raise RuntimeError("boom")


class _DummySettings:
    def __init__(self) -> None:
        self.vmec_preset_settings = types.SimpleNamespace(
            verbose=False,
            hot_restart=types.SimpleNamespace(enabled=False, restart_key=None),
        )


class _ConstellarationSettings:
    default_high_fidelity = staticmethod(lambda *_, **__: _DummySettings())
    default_high_fidelity_skip_qi = staticmethod(lambda *_, **__: _DummySettings())
    default_medium_fidelity = staticmethod(lambda *_, **__: _DummySettings())
    default_low_fidelity = staticmethod(lambda *_, **__: _DummySettings())


def _install_stub_constellaration(monkeypatch: pytest.MonkeyPatch) -> None:
    geometry_surface = types.ModuleType("constellaration.geometry.surface_rz_fourier")
    geometry_surface.SurfaceRZFourier = _BoomSurface

    geometry = types.ModuleType("constellaration.geometry")
    geometry.surface_rz_fourier = geometry_surface

    problems = types.ModuleType("constellaration.problems")
    problems.GeometricalProblem = _BoomProblem
    problems.SimpleToBuildQIStellarator = _BoomProblem
    problems.MHDStableQIStellarator = _BoomProblem

    forward_model = types.ModuleType("constellaration.forward_model")
    forward_model.ConstellarationSettings = _ConstellarationSettings

    root = types.ModuleType("constellaration")
    root.__path__ = []  # mark as package for import machinery

    monkeypatch.setitem(sys.modules, "constellaration", root)
    monkeypatch.setitem(sys.modules, "constellaration.geometry", geometry)
    monkeypatch.setitem(
        sys.modules, "constellaration.geometry.surface_rz_fourier", geometry_surface
    )
    monkeypatch.setitem(sys.modules, "constellaration.problems", problems)
    monkeypatch.setitem(sys.modules, "constellaration.forward_model", forward_model)


def test_evaluate_boundary_real_failure_requires_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    boundary = example_boundary()
    monkeypatch.setenv("CONSTELX_USE_REAL_EVAL", "1")
    monkeypatch.delenv("CONSTELX_DEV", raising=False)

    def boom(_: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr("constelx.physics.constel_api._evaluate_with_real_physics", boom)

    with pytest.raises(RuntimeError) as excinfo:
        evaluate_boundary(boundary, use_real=True)

    assert "Real physics evaluation failed" in str(excinfo.value)


def test_evaluate_boundary_real_failure_dev_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    boundary = example_boundary()
    monkeypatch.setenv("CONSTELX_USE_REAL_EVAL", "1")
    monkeypatch.setenv("CONSTELX_DEV", "1")

    def boom(_: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr("constelx.physics.constel_api._evaluate_with_real_physics", boom)

    metrics = evaluate_boundary(boundary, use_real=True)

    assert metrics["source"] == "placeholder"
    assert "placeholder_metric" in metrics


def test_forward_metrics_real_failure_requires_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stub_constellaration(monkeypatch)
    monkeypatch.delenv("CONSTELX_DEV", raising=False)

    with pytest.raises(RuntimeError) as excinfo:
        forward_metrics(example_boundary(), problem="p1")

    assert "Real physics evaluation failed" in str(excinfo.value)


def test_forward_metrics_real_failure_dev_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stub_constellaration(monkeypatch)
    monkeypatch.setenv("CONSTELX_DEV", "1")

    metrics, info = forward_metrics(example_boundary(), problem="p1")

    assert metrics["source"] == "placeholder"
    assert info["source"] == "placeholder"
