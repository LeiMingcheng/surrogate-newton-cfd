"""Small JSONL profiling hooks for runtime paths."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
from typing import Any


_LOCK = threading.Lock()


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def profile_enabled() -> bool:
    """Return whether runtime profiling should emit JSONL events."""
    return _truthy(os.environ.get("SURROGATE_PROFILE")) and bool(
        os.environ.get("SURROGATE_PROFILE_LOG", "").strip()
    )


def emit_profile_event(event: str, **payload: Any) -> None:
    """Append one profiling event to ``SURROGATE_PROFILE_LOG`` when enabled."""
    if not profile_enabled():
        return
    log_path = Path(os.environ["SURROGATE_PROFILE_LOG"]).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.time(),
        "pid": os.getpid(),
        "thread": threading.current_thread().name,
        "event": str(event),
    }
    record.update(payload)
    with _LOCK:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


__all__ = ["emit_profile_event", "profile_enabled"]
