from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Ensure src/ is on sys.path when running tests without installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.is_dir() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from constelx.eval.boundary_param import sample_random  # noqa: E402


@pytest.fixture
def torch_module() -> object:
    return pytest.importorskip("torch", reason="PyTorch not installed")


@pytest.fixture
def surrogate_modules(torch_module: object) -> tuple[object, type, object, type]:
    from constelx.surrogate.screen import SurrogateScreenError, load_scorer
    from constelx.surrogate.train import MLP

    return torch_module, MLP, load_scorer, SurrogateScreenError


def _build_seeds_prior_records(count: int = 96) -> List[Dict[str, Any]]:
    from constelx.eval.boundary_param import sample_random

    records: List[Dict[str, Any]] = []
    for idx in range(count):
        b = sample_random(nfp=3, seed=idx)
        # Introduce deterministic diversity across higher modes to give PCA signal
        scale = 0.01 * ((idx % 5) - 2)
        if len(b["r_cos"]) > 2 and len(b["r_cos"][2]) > 4:
            b["r_cos"][2][4] = float(scale)
            b["z_sin"][2][4] = float(-scale)
        amp = abs(b["r_cos"][1][5]) + abs(b["z_sin"][1][5])
        metric = float(amp + 0.5 * abs(scale))
        feasible = metric < 0.09
        records.append({"boundary": b, "metrics": {"feasible": feasible, "metric": metric}})
    return records


@pytest.fixture()
def seeds_prior_records() -> List[Dict[str, Any]]:
    return _build_seeds_prior_records()


@pytest.fixture()
def seeds_prior_dataset_jsonl(tmp_path: Path, seeds_prior_records: List[Dict[str, Any]]) -> Path:
    path = tmp_path / "dataset.jsonl"
    with path.open("w") as f:
        for rec in seeds_prior_records:
            f.write(json.dumps(rec) + "\n")
    return path
