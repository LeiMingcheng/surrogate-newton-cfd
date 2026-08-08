"""Runtime path helpers shared by surrogate runtime modules."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_ROOT = PROJECT_ROOT / "tmp" / "surrogate_newton_runtime"


def resolve_runtime_root(env_var: str = "SURROGATE_NEWTON_RUNTIME_ROOT") -> Path:
    """Resolve and create the project runtime root."""
    raw = os.environ.get(env_var)
    root = Path(raw).expanduser() if raw else DEFAULT_RUNTIME_ROOT
    if not root.is_absolute():
        root = DEFAULT_RUNTIME_ROOT / str(root).lstrip("./")
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_runtime_dir(
    path: str | Path | None,
    *,
    env_var: str | None = None,
    default_subdir: str,
    base_dir: str | Path | None = None,
) -> Path:
    """Resolve a writable runtime directory from explicit, env, or default input."""
    raw = path or (os.environ.get(env_var) if env_var else None)
    if base_dir is None:
        base_path = resolve_runtime_root()
    else:
        base_path = Path(base_dir).expanduser()
        if not base_path.is_absolute():
            base_path = resolve_runtime_root() / str(base_path).lstrip("./")

    if raw:
        resolved = Path(raw).expanduser()
        if not resolved.is_absolute():
            resolved = base_path / str(resolved).lstrip("./")
    else:
        resolved = base_path / str(default_subdir)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


__all__ = [
    "DEFAULT_RUNTIME_ROOT",
    "PROJECT_ROOT",
    "resolve_runtime_dir",
    "resolve_runtime_root",
]
