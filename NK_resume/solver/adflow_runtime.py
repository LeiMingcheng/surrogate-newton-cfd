"""ADflow runtime selection for the clean NK_resume solver path."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time

from ..exceptions import ContractError


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _normalize_candidate_root(candidate: Path) -> Path:
    candidate = candidate.expanduser().resolve()
    if (candidate / "src" / "f2py").exists():
        return candidate
    nested = candidate / "adflow"
    if (nested / "src" / "f2py").exists():
        return nested.resolve()
    return candidate


def _candidate_supports_restart_info_rebuild(candidate: Path) -> bool:
    candidate = _normalize_candidate_root(candidate)
    marker = "rebuildrestartderivedstateaftersetinfo"
    for marker_path in (
        candidate / "src" / "f2py" / "adflow.pyf",
        candidate / "src" / "f2py" / "adflow.pyf.autogen",
    ):
        if marker_path.is_file():
            text = marker_path.read_text(encoding="utf-8", errors="ignore")
            if marker in text:
                return True
    return False


def candidate_adflow_runtime_paths() -> tuple[Path, ...]:
    """Return preferred ADflow source runtimes, with restart-info capable builds first."""

    override = str(os.environ.get("SURROGATE_NEWTON_ADFLOW_ROOT") or "").strip()
    if override:
        candidate = _normalize_candidate_root(Path(override))
        if not candidate.exists():
            raise ContractError(f"SURROGATE_NEWTON_ADFLOW_ROOT does not exist: {candidate}")
        return (candidate,)

    raw_candidates = (
        PROJECT_ROOT.parent / "adflow",
        PROJECT_ROOT / "third_party" / "adflow",
    )
    normalized: list[Path] = []
    for candidate in raw_candidates:
        root = _normalize_candidate_root(candidate)
        if root not in normalized:
            normalized.append(root)
    supported = [path for path in normalized if path.exists() and _candidate_supports_restart_info_rebuild(path)]
    unsupported = [path for path in normalized if path.exists() and path not in supported]
    return tuple(supported + unsupported)


def select_adflow_runtime_path(*, require: bool = True) -> Path | None:
    for candidate in candidate_adflow_runtime_paths():
        if candidate.exists():
            return candidate.resolve()
    if require:
        raise ContractError(
            "No compatible ADflow source runtime was found. Set "
            "SURROGATE_NEWTON_ADFLOW_ROOT to the modified ADflow checkout."
        )
    return None


def _prepend_env_path(name: str, value: str) -> None:
    parts = [part for part in os.environ.get(name, "").split(os.pathsep) if part]
    parts = [part for part in parts if part != value]
    os.environ[name] = os.pathsep.join([value, *parts]) if parts else value


def _patch_adflow_root_changed_options() -> None:
    from adflow import ADFLOW

    if bool(getattr(ADFLOW, "_nk_resume_root_changed_patch", False)):
        return

    original_call = ADFLOW.__call__
    original_del = getattr(ADFLOW, "__del__", None)

    def _sanitize_root_changed_options(self) -> None:
        root_changed_options = getattr(self, "rootChangedOptions", None)
        comm = getattr(self, "comm", None)
        if comm is not None:
            root_changed_options = comm.bcast(root_changed_options, root=0)
        if not root_changed_options:
            self.rootChangedOptions = {}
            return
        dropped_names = [str(key) for key in list(root_changed_options.keys())]
        self.rootChangedOptions = {}
        existing = list(getattr(self, "_nk_resume_filtered_root_changed_options", []))
        existing.extend(dropped_names)
        self._nk_resume_filtered_root_changed_options = existing

    def wrapped_call(self, *args, **kwargs):
        _sanitize_root_changed_options(self)
        return original_call(self, *args, **kwargs)

    def _barrier_destructor_comm(self) -> None:
        comm = getattr(self, "comm", None)
        if comm is not None and int(comm.Get_size()) > 1:
            comm.Barrier()
        settle_sec = float(os.environ.get("SURROGATE_NEWTON_ADFLOW_DEL_STAGE_SETTLE_SEC", "0.05") or "0.05")
        if settle_sec > 0.0:
            time.sleep(settle_sec)

    def _manual_teardown(self):
        if bool(getattr(self, "_nk_resume_manual_teardown_done", False)):
            return None
        self._nk_resume_manual_teardown_done = True
        skip_release_memory = int(os.environ.get("SURROGATE_NEWTON_ADFLOW_DEL_SKIP_RELEASE_MEMORY", "0") or "0") > 0
        skip_destroy_nk = int(os.environ.get("SURROGATE_NEWTON_ADFLOW_DEL_SKIP_DESTROY_NK", "0") or "0") > 0
        skip_destroy_ank = int(os.environ.get("SURROGATE_NEWTON_ADFLOW_DEL_SKIP_DESTROY_ANK", "0") or "0") > 0
        _barrier_destructor_comm(self)
        release_adjoint_memory = getattr(self, "releaseAdjointMemory", None)
        if callable(release_adjoint_memory):
            release_adjoint_memory()
            _barrier_destructor_comm(self)
        adflow_obj = getattr(self, "adflow", None)
        if adflow_obj is not None:
            nk_solver = getattr(adflow_obj, "nksolver", None)
            destroy_nk = getattr(nk_solver, "destroynksolver", None)
            if callable(destroy_nk) and not skip_destroy_nk:
                destroy_nk()
                _barrier_destructor_comm(self)
            ank_solver = getattr(adflow_obj, "anksolver", None)
            destroy_ank = getattr(ank_solver, "destroyanksolver", None)
            if callable(destroy_ank) and not skip_destroy_ank:
                destroy_ank()
                _barrier_destructor_comm(self)
            utils_obj = getattr(adflow_obj, "utils", None)
            release_part1 = getattr(utils_obj, "releasememorypart1", None)
            if callable(release_part1) and not skip_release_memory:
                release_part1()
                _barrier_destructor_comm(self)
            release_part2 = getattr(utils_obj, "releasememorypart2", None)
            if callable(release_part2) and not skip_release_memory:
                release_part2()
                _barrier_destructor_comm(self)
        return None

    def wrapped_del(self):
        if bool(getattr(self, "_nk_resume_skip_destructor_teardown", False)):
            return None
        use_original_del = int(os.environ.get("SURROGATE_NEWTON_ADFLOW_USE_ORIGINAL_DEL", "0") or "0") > 0
        if use_original_del and callable(original_del):
            return original_del(self)
        return _manual_teardown(self)

    ADFLOW.__call__ = wrapped_call
    if callable(original_del):
        ADFLOW.__del__ = wrapped_del
    ADFLOW._nk_resume_manual_teardown = _manual_teardown
    ADFLOW._nk_resume_root_changed_patch = True


def ensure_adflow_runtime_on_path(path: str | os.PathLike[str] | None = None) -> Path:
    selected = _normalize_candidate_root(Path(path)) if path is not None else select_adflow_runtime_path()
    if not selected.exists():
        raise ContractError(f"ADflow runtime does not exist: {selected}")

    selected_text = str(selected.resolve())
    sys.path[:] = [entry for entry in sys.path if entry != selected_text]
    sys.path.insert(0, selected_text)
    _prepend_env_path("PYTHONPATH", selected_text)
    os.environ["SURROGATE_NEWTON_ADFLOW_ROOT"] = selected_text
    os.environ["SURROGATE_NEWTON_ADFLOW_SELECTED_PATH"] = selected_text
    _patch_adflow_root_changed_options()
    return selected.resolve()
