"""Shared pyHyp and MPI runtime used by optimization CFD workflows."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
from threading import Lock
from typing import Any, Callable, Mapping

import numpy as np

from surrogate.serving.geometry import GeometryPreparationConfig, GeometryPreparer, PreparedGeometry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PYHYP_LOCK = Lock()


def prepare_authority_mesh(
    *,
    geometry27: np.ndarray,
    t_max: float,
    tag: str,
    mesh_path: str | Path,
) -> tuple[PreparedGeometry, Path]:
    """Generate the canonical pyHyp mesh for one 27D optimization geometry."""

    target = Path(mesh_path).resolve()
    # pyHyp owns process-global native state and is not safe to enter from
    # multiple Python threads.  Candidate inference may remain concurrent, but
    # mesh construction must be serialized within the optimizer process.
    with _PYHYP_LOCK:
        prepared = GeometryPreparer(
            config=GeometryPreparationConfig(mesh_mode="pyhyp", cache_size=1)
        ).prepare(
            {
                "geometry": np.asarray(geometry27, dtype=np.float32),
                "t_max": float(t_max),
                "tag": str(tag),
                "persist_cgns_path": target,
            }
        )
    return prepared, target


def mpi_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the single MPI environment contract for ADFLOW subprocesses."""

    env = os.environ.copy()
    python_path = str(PROJECT_ROOT)
    if env.get("PYTHONPATH"):
        python_path = os.pathsep.join((python_path, env["PYTHONPATH"]))
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMPI_ALLOW_RUN_AS_ROOT": "1",
            "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
            "OMPI_MCA_mpi_warn_on_fork": "0",
            "PYTHONPATH": python_path,
        }
    )
    if extra:
        env.update({str(key): str(value) for key, value in extra.items()})
    return env


def split_jobs_for_pools(
    jobs: list[dict[str, Any]],
    requested_pool_count: int,
) -> list[list[dict[str, Any]]]:
    """Split operating points across a fixed number of MPI pools."""

    pool_count = min(int(requested_pool_count), len(jobs))
    return [jobs[index::pool_count] for index in range(pool_count)]


def run_adflow_pool(
    manifest_path: str | Path,
    pool_dir: str | Path,
    *,
    ranks_per_case: int,
    mpi_launcher: str,
    python: str,
    timeout_s: float,
    env_overrides: Mapping[str, str] | None = None,
) -> None:
    """Execute one manifest in one fixed MPI world."""

    manifest = Path(manifest_path).resolve()
    directory = Path(pool_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / "worker.log"
    command = [
        str(mpi_launcher),
        "-np",
        str(int(ranks_per_case)),
        str(python),
        "-m",
        "optimization.cfd_worker",
        "--manifest",
        str(manifest),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=directory,
            env=mpi_env(env_overrides),
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=float(timeout_s),
        )
    if completed.returncode != 0:
        raise RuntimeError(f"ADFLOW pool failed; see {log_path}")


def execute_adflow_jobs(
    jobs: list[dict[str, Any]],
    *,
    root: str | Path,
    pool_count: int,
    pool_runner: Callable[[Path, Path], None],
) -> int:
    """Write pool manifests, execute them concurrently, and return the pool count."""

    groups = split_jobs_for_pools(jobs, pool_count)
    root_path = Path(root)
    manifests: list[tuple[Path, Path]] = []
    for pool_id, group in enumerate(groups):
        pool_dir = (root_path / "pools" / f"pool_{pool_id:02d}").resolve()
        manifest_path = pool_dir / "manifest.json"
        pool_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps({"jobs": group}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifests.append((manifest_path, pool_dir))

    with ThreadPoolExecutor(max_workers=len(manifests)) as executor:
        futures = [executor.submit(pool_runner, path, directory) for path, directory in manifests]
        for future in futures:
            future.result()
    return len(manifests)


__all__ = [
    "execute_adflow_jobs",
    "mpi_env",
    "prepare_authority_mesh",
    "run_adflow_pool",
    "split_jobs_for_pools",
]
