from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure src/ is on sys.path when running tests without installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.is_dir() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@pytest.fixture
def torch_module() -> object:
    return pytest.importorskip("torch", reason="PyTorch not installed")


@pytest.fixture
def surrogate_modules(torch_module: object) -> tuple[object, type, object, type]:
    from constelx.surrogate.screen import SurrogateScreenError, load_scorer
    from constelx.surrogate.train import MLP

    return torch_module, MLP, load_scorer, SurrogateScreenError
