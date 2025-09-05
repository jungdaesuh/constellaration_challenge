"""Cache backends for evaluator results.

Provides a robust disk-backed cache via `diskcache` when available, with a
JSON-file fallback that requires no extra dependencies. The API is intentionally
minimal to keep usage simple inside `eval.forward` and `forward_many`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol


class CacheBackend(Protocol):
    # pragma: no cover - protocol
    def get(self, key: str) -> Optional[Dict[str, Any]]: ...

    def set(self, key: str, value: Mapping[str, Any]) -> None: ...


@dataclass
class JsonCacheBackend:
    base_dir: Path

    def __post_init__(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.base_dir / f"{key}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            import json

            return dict(json.loads(p.read_text()))
        except Exception:
            return None

    def set(self, key: str, value: Mapping[str, Any]) -> None:
        p = self._path(key)
        try:
            import json

            p.write_text(json.dumps(dict(value)))
        except Exception:
            # Best-effort; ignore persistence errors
            pass


class DiskCacheBackend:
    def __init__(
        self,
        directory: Path,
        *,
        size_limit: Optional[int] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        # Import lazily to avoid hard dependency and keep mypy happy.
        try:
            import diskcache as _dc

            self._dc = _dc
        except Exception as e:  # pragma: no cover - protected by factory
            raise RuntimeError("diskcache is not available") from e
        self._cache = self._dc.Cache(str(directory), size_limit=size_limit)
        self._ttl = ttl_seconds

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            v = self._cache.get(key, default=None)
            if isinstance(v, dict):
                return dict(v)
            return None
        except Exception:
            return None

    def set(self, key: str, value: Mapping[str, Any]) -> None:
        try:
            self._cache.set(key, dict(value), expire=self._ttl)
        except Exception:
            pass


def get_cache_backend(
    cache_dir: Path, *, prefer: Optional[str] = None, ttl_seconds: Optional[int] = None
) -> CacheBackend:
    """Create a cache backend for a directory.

    - prefer: 'disk' or 'json' (default: try diskcache, else fallback to JSON)
    - ttl_seconds: optional expiry for diskcache values
    """

    if prefer == "json":
        return JsonCacheBackend(cache_dir)
    if prefer == "disk":
        try:
            return DiskCacheBackend(cache_dir, ttl_seconds=ttl_seconds)
        except Exception:
            return JsonCacheBackend(cache_dir)
    # auto
    try:
        return DiskCacheBackend(cache_dir, ttl_seconds=ttl_seconds)
    except Exception:
        return JsonCacheBackend(cache_dir)


__all__ = [
    "CacheBackend",
    "JsonCacheBackend",
    "DiskCacheBackend",
    "get_cache_backend",
]
