"""MPI pool execution for clean NK_resume manifests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import importlib
import json
from pathlib import Path
from typing import Any

from ..exceptions import ContractError
from ..payload import load_manifest_dict, resume_case_from_payload
from ..plans import resume_plan_from_dict
from .adflow import ADflowBackend
from .backend import ProjectionRequest, ProjectionResult, SolverBackend
from .service import ReplayService


BackendFactory = Callable[[Any], SolverBackend]


def _default_mpi_comm() -> Any:
    try:
        module = importlib.import_module("mpi4py.MPI")
    except Exception:
        return None
    return getattr(module, "COMM_WORLD", None)


def _comm_rank(comm: Any) -> int:
    getter = getattr(comm, "Get_rank", None)
    if not callable(getter):
        return 0
    return int(getter())


def _comm_size(comm: Any) -> int:
    getter = getattr(comm, "Get_size", None)
    if not callable(getter):
        return 1
    return int(getter())


def _comm_barrier(comm: Any) -> None:
    barrier = getattr(comm, "Barrier", None)
    if callable(barrier):
        barrier()


def _comm_split(comm: Any, *, color: int, key: int) -> Any:
    splitter = getattr(comm, "Split", None)
    if not callable(splitter):
        if int(color) != 0 or int(key) != 0:
            raise ContractError("MPI pool split requires an MPI communicator")
        return comm
    return splitter(color=int(color), key=int(key))


def _comm_gather(comm: Any, payload: Any, *, root: int = 0) -> list[Any] | None:
    gather = getattr(comm, "gather", None)
    if not callable(gather):
        return [payload] if int(root) == 0 else None
    return gather(payload, root=int(root))


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _default_backend_factory(comm: Any) -> SolverBackend:
    return ADflowBackend(comm=comm)


@dataclass(frozen=True)
class MPIPoolManifestProjectionResult:
    """Projection artifacts produced by static MPI pool execution."""

    manifest_path: str
    job_count: int
    ranks_per_case: int
    pool_count: int
    result_paths: tuple[str, ...]
    status_paths: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "job_count": self.job_count,
            "ranks_per_case": self.ranks_per_case,
            "pool_count": self.pool_count,
            "result_paths": list(self.result_paths),
            "status_paths": list(self.status_paths),
            "metadata": dict(self.metadata),
        }


def _pool_status_path(manifest_path: str | Path, pool_id: int) -> Path:
    root = Path(manifest_path).resolve().parent
    return root / "pool_status" / f"pool_{int(pool_id):04d}.json"


def _validate_pool_shape(*, world_size: int, ranks_per_case: int) -> int:
    ranks_per_case = int(ranks_per_case)
    if ranks_per_case <= 0:
        raise ContractError("ranks_per_case must be positive")
    if int(world_size) < ranks_per_case:
        raise ContractError(
            f"MPI world size {world_size} is smaller than ranks_per_case={ranks_per_case}"
        )
    pool_count, remainder = divmod(int(world_size), ranks_per_case)
    if remainder:
        raise ContractError(
            f"MPI world size {world_size} must be divisible by ranks_per_case={ranks_per_case}"
        )
    if pool_count <= 0:
        raise ContractError("MPI pool count must be positive")
    return int(pool_count)


def _sort_statuses(statuses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(statuses, key=lambda item: int(item.get("pool_id", 0)))


def project_manifest_pools(
    manifest_path: str,
    *,
    ranks_per_case: int = 8,
    comm: Any = None,
    backend_factory: BackendFactory | None = None,
) -> MPIPoolManifestProjectionResult:
    """Execute a manifest as static MPI pools, one solver case per pool at a time."""

    manifest_path = str(manifest_path)
    if not manifest_path.strip():
        raise ContractError("manifest_path is required")
    world_comm = comm if comm is not None else _default_mpi_comm()
    world_rank = _comm_rank(world_comm)
    world_size = _comm_size(world_comm)
    pool_count = _validate_pool_shape(
        world_size=world_size,
        ranks_per_case=int(ranks_per_case),
    )
    pool_id = world_rank // int(ranks_per_case)
    pool_rank = world_rank % int(ranks_per_case)
    pool_comm = _comm_split(world_comm, color=pool_id, key=pool_rank)

    manifest = load_manifest_dict(manifest_path)
    plan = resume_plan_from_dict(dict(manifest.get("plan") or {}))
    replay_plan = ReplayService(
        mode="dry_run",
        ranks_per_case=int(ranks_per_case),
        pool_count=pool_count,
    ).plan(manifest)
    jobs = tuple(replay_plan.jobs)
    selected = [(index, job) for index, job in enumerate(jobs) if index % pool_count == pool_id]
    solver_backend = (backend_factory or _default_backend_factory)(pool_comm)

    pool_results: list[dict[str, Any]] = []
    result_paths_by_index: dict[int, str] = {}
    for job_index, job in selected:
        case = resume_case_from_payload(job.payload_path)
        result: ProjectionResult = solver_backend.project(
            ProjectionRequest(
                case=case,
                plan=plan,
                output_dir=job.output_dir,
                metadata={
                    "manifest_path": manifest_path,
                    "manifest_job": job.to_dict(),
                    "result_path": job.result_path,
                    "mpi_pool": {
                        "pool_id": int(pool_id),
                        "pool_rank": int(pool_rank),
                        "pool_count": int(pool_count),
                        "ranks_per_case": int(ranks_per_case),
                    },
                },
            )
        )
        result_paths_by_index[int(job_index)] = str(result.result_path)
        pool_results.append(
            {
                "job_index": int(job_index),
                "case_id": job.case_id,
                "state_name": job.state_name,
                "result_path": str(result.result_path),
                "status": result.status,
            }
        )

    status_path = _pool_status_path(manifest_path, pool_id)
    status_payload = {
        "schema_version": "mpi_pool_status_v1",
        "manifest_path": manifest_path,
        "pool_id": int(pool_id),
        "pool_count": int(pool_count),
        "pool_rank": int(pool_rank),
        "ranks_per_case": int(ranks_per_case),
        "assigned_job_count": len(selected),
        "jobs": pool_results,
        "result_paths_by_index": {
            str(index): path for index, path in sorted(result_paths_by_index.items())
        },
    }
    if pool_rank == 0:
        _write_json_atomic(status_path, status_payload)

    _comm_barrier(pool_comm)
    _comm_barrier(world_comm)

    gathered = _comm_gather(world_comm, status_payload if pool_rank == 0 else None, root=0)
    if world_rank != 0:
        return MPIPoolManifestProjectionResult(
            manifest_path=manifest_path,
            job_count=len(jobs),
            ranks_per_case=int(ranks_per_case),
            pool_count=int(pool_count),
            result_paths=(),
            status_paths=(),
            metadata={"world_rank": int(world_rank), "primary": False},
        )

    statuses = _sort_statuses(
        [
            dict(item)
            for item in (gathered or [])
            if isinstance(item, Mapping)
        ]
    )
    if len(statuses) != pool_count:
        raise ContractError(
            f"Expected {pool_count} pool statuses, collected {len(statuses)}"
        )
    result_paths: dict[int, str] = {}
    for status in statuses:
        for key, value in dict(status.get("result_paths_by_index") or {}).items():
            result_paths[int(key)] = str(value)
    missing = [index for index in range(len(jobs)) if index not in result_paths]
    if missing:
        raise ContractError(f"MPI pool execution did not produce jobs: {missing}")

    return MPIPoolManifestProjectionResult(
        manifest_path=manifest_path,
        job_count=len(jobs),
        ranks_per_case=int(ranks_per_case),
        pool_count=int(pool_count),
        result_paths=tuple(result_paths[index] for index in range(len(jobs))),
        status_paths=tuple(str(_pool_status_path(manifest_path, index)) for index in range(pool_count)),
        metadata={
            "world_size": int(world_size),
            "primary": True,
            "statuses": _jsonable(statuses),
        },
    )
