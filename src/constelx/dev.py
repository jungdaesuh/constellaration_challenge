from __future__ import annotations

import os
from typing import Optional


def _env_truthy(name: str) -> Optional[bool]:
    raw = os.getenv(name)
    if raw is None:
        return None
    v = raw.strip().lower()
    if not v:
        return None
    if v in {"1", "true", "yes", "on", "y"}:
        return True
    if v in {"0", "false", "no", "off", "n"}:
        return False
    return None


def is_dev_mode() -> bool:
    """Return True when development mode is enabled via CONSTELX_DEV=1.

    This flag is intended to allow placeholder/synthetic paths while keeping
    production and CI jobs clean by default.
    """
    val = _env_truthy("CONSTELX_DEV")
    return bool(val) if val is not None else False


def enforce_real_enabled() -> bool:
    """Return True when non-real (placeholder/synthetic) paths should be rejected.

    Enforcement activates when CONSTELX_ENFORCE_REAL=1 (or TRUE). In dev mode
    (CONSTELX_DEV=1), enforcement is always disabled.
    """
    if is_dev_mode():
        return False
    val = _env_truthy("CONSTELX_ENFORCE_REAL")
    return bool(val) if val is not None else False


def require_dev_for_placeholder(context: str) -> None:
    """Raise a RuntimeError if placeholder/synthetic usage is disallowed.

    Parameters
    - context: short description of the operation (for error message only).
    """
    if enforce_real_enabled():
        raise RuntimeError(
            (
                f"{context} is only allowed in development mode. "
                "Set CONSTELX_DEV=1 to opt in, or provide real inputs."
            )
        )
