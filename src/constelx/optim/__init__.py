from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

from . import evolution, pareto

__all__ = ["evolution", "pareto", "desc_trust_region"]

if TYPE_CHECKING:  # pragma: no cover - for static analysis only
    from . import desc_trust_region as _desc_trust_region

    desc_trust_region = _desc_trust_region


def __getattr__(name: str) -> ModuleType:
    if name == "desc_trust_region":
        module = import_module("constelx.optim.desc_trust_region")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'constelx.optim' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
