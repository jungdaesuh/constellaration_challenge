from __future__ import annotations

import pytest


@pytest.fixture
def torch_module() -> object:
    return pytest.importorskip("torch", reason="PyTorch not installed")


@pytest.fixture
def surrogate_modules(torch_module: object) -> tuple[object, type, object, type]:
    from constelx.surrogate.screen import SurrogateScreenError, load_scorer
    from constelx.surrogate.train import MLP

    return torch_module, MLP, load_scorer, SurrogateScreenError
