"""Utility helpers for metrics provenance and validation."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["ProvenanceError", "ensure_real_metrics_csv", "ensure_real_rows"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module(".provenance", __name__)
        return getattr(module, name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
