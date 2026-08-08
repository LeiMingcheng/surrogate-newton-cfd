from __future__ import annotations

import os
import re
import time
from pathlib import Path


_CGNS_PATH_CACHE: dict[tuple[str, str], str | None] = {}

_LONG_CGNS_BASENAME_RE = re.compile(
    r"^airfoil_(?P<airfoil_id>\d+)_G2_A_L0_case_(?P<case_id>\d+)_000_vol\.cgns$"
)
_SHORT_CGNS_BASENAME_RE = re.compile(
    r"^af(?P<airfoil_id>\d+)_c(?P<case_id>\d+)(?:_vol)?\.cgns$"
)


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str) -> float | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _extract_airfoil_case_ids(cgns_basename: str) -> tuple[str, str]:
    basename = Path(str(cgns_basename)).name
    match = _LONG_CGNS_BASENAME_RE.match(basename)
    if match is not None:
        return match.group("airfoil_id"), match.group("case_id")
    match = _SHORT_CGNS_BASENAME_RE.match(basename)
    if match is not None:
        return match.group("airfoil_id"), match.group("case_id")
    raise ValueError(f"Unsupported CGNS basename format: {cgns_basename}")


def resolve_cgns_path(cgns_root: str | Path, cgns_basename: str) -> str:
    if os.path.isabs(cgns_basename) or (os.sep in cgns_basename):
        if os.path.exists(cgns_basename):
            return cgns_basename

    cgns_root_abs = os.path.abspath(str(cgns_root))
    cache_key = (cgns_root_abs, cgns_basename)
    if cache_key in _CGNS_PATH_CACHE:
        cached = _CGNS_PATH_CACHE[cache_key]
        if cached is None:
            raise FileNotFoundError(
                f"CGNS file not found: {cgns_basename}\n"
                f"Searched in: {cgns_root_abs}\n"
                f"(cached negative result)"
            )
        return cached

    airfoil_id, case_id = _extract_airfoil_case_ids(cgns_basename)

    direct_dir = f"airfoil_{airfoil_id}_G2_A_L0_case_{case_id}"
    direct_path = os.path.join(cgns_root_abs, direct_dir, cgns_basename)
    if os.path.exists(direct_path):
        _CGNS_PATH_CACHE[cache_key] = direct_path
        return direct_path

    possible_dirs = [
        f"Transi_Cutout_{i}/airfoil_{airfoil_id}_G2_A_L0_case_{case_id}"
        for i in range(1, 10)
    ]
    for subdir in possible_dirs:
        full_path = os.path.join(cgns_root_abs, subdir, cgns_basename)
        if os.path.exists(full_path):
            _CGNS_PATH_CACHE[cache_key] = full_path
            return full_path

    possible_dirs_turb = [
        f"Turb_Cutout_{i}/airfoil_{airfoil_id}_G2_A_L0_case_{case_id}"
        for i in range(1, 7)
    ]
    for subdir in possible_dirs_turb:
        full_path = os.path.join(cgns_root_abs, subdir, cgns_basename)
        if os.path.exists(full_path):
            _CGNS_PATH_CACHE[cache_key] = full_path
            return full_path

    possible_dirs_transi_sup = [
        f"Transi_sup_data_Cutout_{i}/airfoil_{airfoil_id}_G2_A_L0_case_{case_id}"
        for i in range(1, 3)
    ]
    for subdir in possible_dirs_transi_sup:
        full_path = os.path.join(cgns_root_abs, subdir, cgns_basename)
        if os.path.exists(full_path):
            _CGNS_PATH_CACHE[cache_key] = full_path
            return full_path

    if not _env_flag("UNIFOIL_DISABLE_CGNS_WALK"):
        walk_timeout_s = _env_float("UNIFOIL_CGNS_WALK_TIMEOUT_S")
        t0 = time.monotonic()
        for root, _dirs, files in os.walk(cgns_root_abs):
            if cgns_basename in files:
                resolved = os.path.join(root, cgns_basename)
                _CGNS_PATH_CACHE[cache_key] = resolved
                return resolved
            if walk_timeout_s is not None and (time.monotonic() - t0) > walk_timeout_s:
                break

    _CGNS_PATH_CACHE[cache_key] = None
    raise FileNotFoundError(
        f"CGNS file not found: {cgns_basename}\n"
        f"Searched in: {cgns_root_abs}\n"
        "Please check that the CGNS file exists in the source directory."
    )


__all__ = ["resolve_cgns_path"]
