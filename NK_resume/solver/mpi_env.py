"""MPI launcher and runtime environment helpers for clean NK_resume workers."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import sys

from ..exceptions import ContractError
from .adflow_runtime import select_adflow_runtime_path


def python_executable() -> str:
    path = Path(os.environ.get("SURROGATE_NEWTON_PYTHON", sys.executable)).resolve()
    if not path.exists():
        raise ContractError(f"Required Python interpreter does not exist: {path}")
    return str(path)


def _thread_env_defaults(omp_threads: int) -> dict[str, str]:
    value = str(max(1, int(omp_threads)))
    return {
        "OMP_NUM_THREADS": value,
        "OPENBLAS_NUM_THREADS": value,
        "MKL_NUM_THREADS": value,
        "VECLIB_MAXIMUM_THREADS": value,
        "NUMEXPR_NUM_THREADS": value,
    }


def _sanitize_launcher_env(env: dict[str, str]) -> dict[str, str]:
    allowed_prefixed_keys = {
        "ADFLOW_DISABLE_PER_CALL_TRACE",
        "ADFLOW_ENABLE_NATIVE_FORCE_EVAL",
        "ADFLOW_MPI_PHASE_TRACE",
        "ADFLOW_REUSE_MATCHED_STATE",
        "NK_RESUME_CELL_MAPPING",
        "NK_RESUME_CELL_MAP_MAX_DIST2",
        "NK_RESUME_MPI_TRACE",
        "SURROGATE_NEWTON_ADFLOW_CLEANUP_SKIP_MANUAL_TEARDOWN",
        "SURROGATE_NEWTON_ADFLOW_DEL_SKIP_RELEASE_MEMORY",
        "SURROGATE_NEWTON_ADFLOW_DEL_STAGE_SETTLE_SEC",
        "SURROGATE_NEWTON_ADFLOW_ROOT",
        "SURROGATE_NEWTON_D4_CLEANUP_MODE",
        "SURROGATE_NEWTON_NK_BUNDLE_TRACE",
        "SURROGATE_NEWTON_NK_INTERMEDIATE_STAGE_METRICS",
    }
    sanitized = dict(env)
    for key in list(sanitized):
        if key.startswith(("SURROGATE_NEWTON_", "ADFLOW_", "NK_RESUME_")) and key not in allowed_prefixed_keys:
            sanitized.pop(key, None)
    return sanitized


def build_mpi_env(omp_threads: int, mpi_tmp_root: str | Path) -> dict[str, str]:
    root = Path(mpi_tmp_root).resolve()
    hydra_tmp_root = root / "hydra_tmp"
    ompi_tmp_root = root / "ompi_tmp"
    pmix_tmp_root = root / "pmix_tmp"
    for path in (hydra_tmp_root, ompi_tmp_root, pmix_tmp_root):
        path.mkdir(parents=True, exist_ok=True)

    env = _sanitize_launcher_env(os.environ.copy())
    env["PYTHONUNBUFFERED"] = "1"
    for key, value in _thread_env_defaults(int(omp_threads)).items():
        env[key] = value
    env["TMPDIR"] = str(hydra_tmp_root)
    env["TMP"] = str(hydra_tmp_root)
    env["TEMP"] = str(hydra_tmp_root)
    env["MP_TMPDIR"] = str(hydra_tmp_root)
    env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
    env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
    env["OMPI_MCA_mpi_warn_on_fork"] = "0"
    env["OMPI_MCA_btl_vader_single_copy_mechanism"] = "none"
    env["OMPI_MCA_orte_tmpdir_base"] = str(ompi_tmp_root)
    env["PRTE_MCA_prte_tmpdir_base"] = str(ompi_tmp_root)
    env["PMIX_SERVER_TMPDIR"] = str(pmix_tmp_root)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.setdefault("ADFLOW_DISABLE_PER_CALL_TRACE", "1")
    env.setdefault("ADFLOW_ENABLE_NATIVE_FORCE_EVAL", "0")
    env.setdefault("ADFLOW_REUSE_MATCHED_STATE", "0")
    env.setdefault("SURROGATE_NEWTON_ADFLOW_CLEANUP_SKIP_MANUAL_TEARDOWN", "1")
    adflow_runtime = select_adflow_runtime_path(require=False)
    if adflow_runtime is not None:
        runtime_text = str(adflow_runtime)
        env["SURROGATE_NEWTON_ADFLOW_ROOT"] = runtime_text
        pythonpath = str(env.get("PYTHONPATH") or "")
        entries = [entry for entry in pythonpath.split(os.pathsep) if entry]
        entries = [entry for entry in entries if entry != runtime_text]
        env["PYTHONPATH"] = os.pathsep.join([runtime_text, *entries]) if entries else runtime_text
    return env


def interesting_env_subset(env: dict[str, str]) -> dict[str, str]:
    exact_keys = {
        "PYTHONUNBUFFERED",
        "PYTHONPATH",
        "CONDA_PREFIX",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TMPDIR",
        "TMP",
        "TEMP",
        "MP_TMPDIR",
        "CUDA_VISIBLE_DEVICES",
        "OMPI_ALLOW_RUN_AS_ROOT",
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM",
        "OMPI_MCA_mpi_warn_on_fork",
        "OMPI_MCA_btl_vader_single_copy_mechanism",
        "OMPI_MCA_orte_tmpdir_base",
        "PRTE_MCA_prte_tmpdir_base",
        "PMIX_SERVER_TMPDIR",
    }
    prefix_keys = ("SURROGATE_NEWTON_", "ADFLOW_", "NK_RESUME_")
    return {
        str(key): str(env[key])
        for key in sorted(env)
        if key in exact_keys or key.startswith(prefix_keys)
    }


def resolve_mpi_launcher(text: str | None = "auto") -> list[str]:
    token = str(text or "auto").strip()
    if not token or token == "auto":
        python_bin = Path(python_executable()).parent
        candidates: list[str] = [
            str(python_bin / "mpirun"),
            str(python_bin / "mpiexec"),
        ]
        conda_prefix = str(os.environ.get("CONDA_PREFIX", "")).strip()
        if conda_prefix:
            candidates.extend(
                [
                    str(Path(conda_prefix) / "bin" / "mpirun"),
                    str(Path(conda_prefix) / "bin" / "mpiexec"),
                ]
            )
        candidates.extend(
            [
                "mpirun",
                "mpiexec",
            ]
        )
        for candidate in candidates:
            resolved = shutil.which(candidate)
            if resolved:
                cmd = [resolved]
                break
        else:
            raise ContractError("MPI launcher not found")
    else:
        cmd = shlex.split(token)
        if not cmd:
            raise ContractError("MPI launcher command is empty")
        resolved = shutil.which(cmd[0])
        if resolved:
            cmd[0] = resolved
        elif not os.path.isabs(cmd[0]):
            launcher_name = os.path.basename(cmd[0])
            candidates = [
                Path(python_executable()).parent / launcher_name,
                Path(sys.executable).resolve().parent / launcher_name,
            ]
            for candidate in candidates:
                if candidate.exists():
                    cmd[0] = str(candidate)
                    break
        elif os.path.isabs(cmd[0]) and not Path(cmd[0]).exists():
            raise ContractError(f"MPI launcher not found: {cmd[0]}")
    launcher_name = os.path.basename(cmd[0]).lower()
    launcher_resolved_name = Path(cmd[0]).resolve().name.lower()
    if (
        (launcher_name.startswith("mpirun") or launcher_name.startswith("mpiexec"))
        and "hydra" not in launcher_name
        and "hydra" not in launcher_resolved_name
        and "--bind-to" not in cmd
        and "-bind-to" not in cmd
    ):
        cmd.extend(["--bind-to", "none"])
    return [str(token) for token in cmd]


def inject_mpi_runtime_env_args(launcher_cmd: list[str], env: dict[str, str]) -> list[str]:
    cmd = list(launcher_cmd)
    if not cmd:
        return cmd
    launcher_name = os.path.basename(cmd[0]).lower()
    if not (launcher_name.startswith("mpirun") or launcher_name.startswith("mpiexec")):
        return cmd
    launcher_resolved_name = Path(cmd[0]).resolve().name.lower()
    use_hydra_env = "hydra" in launcher_name or "hydra" in launcher_resolved_name
    injected = [cmd[0]]
    for key in (
        "PYTHONUNBUFFERED",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TMPDIR",
        "TMP",
        "TEMP",
        "MP_TMPDIR",
        "OMPI_ALLOW_RUN_AS_ROOT",
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM",
        "OMPI_MCA_mpi_warn_on_fork",
        "OMPI_MCA_btl_vader_single_copy_mechanism",
        "OMPI_MCA_orte_tmpdir_base",
        "PRTE_MCA_prte_tmpdir_base",
        "PMIX_SERVER_TMPDIR",
        "CUDA_VISIBLE_DEVICES",
        "ADFLOW_DISABLE_PER_CALL_TRACE",
        "ADFLOW_ENABLE_NATIVE_FORCE_EVAL",
        "ADFLOW_MPI_PHASE_TRACE",
        "ADFLOW_REUSE_MATCHED_STATE",
        "NK_RESUME_CELL_MAPPING",
        "NK_RESUME_CELL_MAP_MAX_DIST2",
        "NK_RESUME_MPI_TRACE",
        "SURROGATE_NEWTON_ADFLOW_CLEANUP_SKIP_MANUAL_TEARDOWN",
        "SURROGATE_NEWTON_ADFLOW_DEL_SKIP_RELEASE_MEMORY",
        "SURROGATE_NEWTON_ADFLOW_DEL_STAGE_SETTLE_SEC",
        "SURROGATE_NEWTON_ADFLOW_ROOT",
        "SURROGATE_NEWTON_D4_CLEANUP_MODE",
        "SURROGATE_NEWTON_NK_BUNDLE_TRACE",
        "SURROGATE_NEWTON_NK_INTERMEDIATE_STAGE_METRICS",
    ):
        value = env.get(key)
        if value is None:
            continue
        if use_hydra_env:
            injected.extend(["-genv", key, str(value)])
        else:
            injected.extend(["-x", f"{key}={value}"])
    injected.extend(cmd[1:])
    return injected
