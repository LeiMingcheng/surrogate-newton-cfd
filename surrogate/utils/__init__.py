"""Small utility helpers for surrogate runtime code."""

from surrogate.utils.tensors import detach_to_cpu, move_mapping_to_device
from surrogate.utils.runtime_paths import (
    DEFAULT_RUNTIME_ROOT,
    PROJECT_ROOT,
    resolve_runtime_dir,
    resolve_runtime_root,
)
from surrogate.utils.timing_profile import emit_profile_event, profile_enabled
from surrogate.utils.cgns_geometry import load_cgns_geometry_2d

__all__ = [
    "DEFAULT_RUNTIME_ROOT",
    "PROJECT_ROOT",
    "detach_to_cpu",
    "emit_profile_event",
    "load_cgns_geometry_2d",
    "move_mapping_to_device",
    "profile_enabled",
    "resolve_runtime_dir",
    "resolve_runtime_root",
]
