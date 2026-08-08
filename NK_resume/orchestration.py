"""Mainline orchestration for clean NK_resume manifest runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import importlib
import json
from pathlib import Path
from typing import Any

from .exceptions import ContractError
from .metrics import aggregate_results
from .pipeline import ManifestProjectionResult, create_pipeline
from .solver import (
    MPIPoolManifestProjectionResult,
    ProjectionResult,
    SolverBackend,
    WarmPoolManifestProjectionResult,
    load_projection_result_dict,
    project_manifest_pools,
    project_manifest_warm_pools,
)


MANIFEST_RUN_SUMMARY_SCHEMA = "manifest_run_summary_v1"
BackendFactory = Callable[[Any], SolverBackend]


def _comm_rank() -> int:
    try:
        module = importlib.import_module("mpi4py.MPI")
    except Exception:
        return 0
    comm = getattr(module, "COMM_WORLD", None)
    getter = getattr(comm, "Get_rank", None)
    if not callable(getter):
        return 0
    return int(getter())


def _is_primary_process() -> bool:
    return _comm_rank() == 0


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


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def _summary_path(manifest_path: str | Path, executor: str) -> Path:
    manifest = Path(manifest_path)
    stem = manifest.stem or "manifest"
    return manifest.resolve().parent / f"{stem}.{executor}.run_summary.json"


def _load_projection_results(result_paths: tuple[str, ...]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for result_path in result_paths:
        path = Path(result_path)
        if not path.exists():
            raise ContractError(f"Projection result does not exist: {path}")
        results.append(load_projection_result_dict(path))
    return results


def _execution_dict(
    execution: ManifestProjectionResult | MPIPoolManifestProjectionResult | WarmPoolManifestProjectionResult,
) -> dict[str, Any]:
    return execution.to_dict()


@dataclass(frozen=True)
class ManifestRunResult:
    """Result of one clean manifest run, including aggregate summary output."""

    manifest_path: str
    executor: str
    job_count: int
    result_paths: tuple[str, ...]
    summary_path: str = ""
    status_paths: tuple[str, ...] = field(default_factory=tuple)
    aggregate: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "executor": self.executor,
            "job_count": self.job_count,
            "result_paths": list(self.result_paths),
            "summary_path": self.summary_path,
            "status_paths": list(self.status_paths),
            "aggregate": dict(self.aggregate),
            "metadata": dict(self.metadata),
        }


def write_manifest_run_summary(
    *,
    manifest_path: str,
    executor: str,
    execution: ManifestProjectionResult | MPIPoolManifestProjectionResult | WarmPoolManifestProjectionResult,
    result_dicts: list[dict[str, Any]],
    summary_path: str | Path,
) -> ManifestRunResult:
    """Write a canonical summary for one executed manifest."""

    result_paths = tuple(str(path) for path in execution.result_paths)
    if len(result_paths) != len(result_dicts):
        raise ContractError(
            f"result_paths count {len(result_paths)} does not match loaded results {len(result_dicts)}"
        )
    aggregate = aggregate_results(result_dicts)
    status_paths = tuple(str(path) for path in getattr(execution, "status_paths", ()) or ())
    payload = {
        "schema_version": MANIFEST_RUN_SUMMARY_SCHEMA,
        "manifest_path": str(manifest_path),
        "executor": str(executor),
        "job_count": int(execution.job_count),
        "result_paths": list(result_paths),
        "status_paths": list(status_paths),
        "aggregate": aggregate,
        "results": result_dicts,
        "metadata": {
            "execution": _execution_dict(execution),
        },
    }
    path = Path(summary_path)
    _write_json_atomic(path, payload)
    return ManifestRunResult(
        manifest_path=str(manifest_path),
        executor=str(executor),
        job_count=int(execution.job_count),
        result_paths=result_paths,
        summary_path=str(path),
        status_paths=status_paths,
        aggregate=aggregate,
        metadata={"summary_schema": MANIFEST_RUN_SUMMARY_SCHEMA},
    )


def run_manifest(
    manifest_path: str,
    *,
    executor: str = "sequential",
    ranks_per_case: int = 8,
    pool_count: int | None = None,
    mpi_launcher: str = "auto",
    mpi_omp_threads: int = 1,
    runtime_output_dir: str | Path | None = None,
    ready_timeout_sec: float = 30.0,
    submit_timeout_sec: float = 60.0,
    wait_for_manifest_sec: float = 60.0,
    injection_strategy: str = "restart_info",
    summary_path: str | Path | None = None,
    backend: SolverBackend | None = None,
    pool_backend_factory: BackendFactory | None = None,
) -> ManifestRunResult:
    """Execute a clean manifest and write a compact run summary on rank 0."""

    manifest_path = str(manifest_path)
    if not manifest_path.strip():
        raise ContractError("manifest_path is required")
    executor = str(executor).strip().lower()
    if executor not in {"sequential", "pools", "warm_pools"}:
        raise ContractError("executor must be one of: sequential, pools, warm_pools")

    if executor == "sequential":
        execution: ManifestProjectionResult | MPIPoolManifestProjectionResult | WarmPoolManifestProjectionResult = (
            create_pipeline().project_manifest(manifest_path, backend=backend)
        )
    elif executor == "pools":
        execution = project_manifest_pools(
            manifest_path,
            ranks_per_case=int(ranks_per_case),
            backend_factory=pool_backend_factory,
        )
    else:
        execution = project_manifest_warm_pools(
            manifest_path,
            ranks_per_case=int(ranks_per_case),
            pool_count=pool_count,
            mpi_launcher=str(mpi_launcher),
            mpi_omp_threads=int(mpi_omp_threads),
            output_dir=runtime_output_dir,
            ready_timeout_sec=float(ready_timeout_sec),
            submit_timeout_sec=float(submit_timeout_sec),
            wait_for_manifest_sec=float(wait_for_manifest_sec),
            injection_strategy=str(injection_strategy),
        )

    result_paths = tuple(str(path) for path in execution.result_paths)
    if not _is_primary_process() and not result_paths:
        return ManifestRunResult(
            manifest_path=manifest_path,
            executor=executor,
            job_count=int(execution.job_count),
            result_paths=(),
            metadata={"primary": False},
        )
    if len(result_paths) != int(execution.job_count):
        raise ContractError(
            f"Manifest run produced {len(result_paths)} result paths for {execution.job_count} jobs"
        )
    result_dicts = _load_projection_results(result_paths)
    output_path = Path(summary_path) if summary_path else _summary_path(manifest_path, executor)
    return write_manifest_run_summary(
        manifest_path=manifest_path,
        executor=executor,
        execution=execution,
        result_dicts=result_dicts,
        summary_path=output_path,
    )
