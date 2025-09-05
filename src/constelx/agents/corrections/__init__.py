from __future__ import annotations

from typing import Any, Dict, Mapping, Protocol


class CorrectionHook(Protocol):
    # pragma: no cover - protocol signature only
    def __call__(self, boundary: Mapping[str, Any]) -> Dict[str, Any]: ...


__all__ = ["CorrectionHook"]
