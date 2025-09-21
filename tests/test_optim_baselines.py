from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from constelx.optim.baselines import BaselineConfig, run_botorch_qnei, run_ngopt


def test_run_ngopt_returns_solution(tmp_path: Path) -> None:
    pytest.importorskip("nevergrad")
    cfg = BaselineConfig(budget=5, cache_dir=tmp_path)
    x, score = run_ngopt(cfg)
    assert x.shape == (2,)
    assert np.isfinite(score)


def test_run_botorch_qnei_returns_solution(tmp_path: Path) -> None:
    pytest.importorskip("botorch")
    cfg = BaselineConfig(budget=6, cache_dir=tmp_path)
    x, score = run_botorch_qnei(cfg)
    assert x.shape == (2,)
    assert np.isfinite(score)
