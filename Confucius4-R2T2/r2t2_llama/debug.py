"""Shared opt-in diagnostics helpers for the r2t2 package."""

from __future__ import annotations

import os


def debug_enabled() -> bool:
    return os.getenv("DEBUG_PRINT", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def debug_print(*args, **kwargs) -> None:
    if debug_enabled():
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)
