"""Warm MPI worker service for clean NK_resume manifests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

from ..exceptions import ContractError
from ..payload import load_manifest_dict, resume_case_from_payload
from ..plans import resume_plan_from_dict
from .adflow import ADflowBackend
from .adflow_runtime import ensure_adflow_runtime_on_path
from .backend import ProjectionRequest, ProjectionResult
from .service import ReplayService
from .state import ADflowStateAdapter


_MODULE_START = time.perf_counter()
_MODULE_START_UNIX_SEC = time.time()


def _mpi_comm() -> Any:
    from mpi4py import MPI

    return MPI.COMM_WORLD


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


def _comm_bcast(comm: Any, payload: Any, *, root: int = 0) -> Any:
    broadcaster = getattr(comm, "bcast", None)
    if not callable(broadcaster):
        return payload
    return broadcaster(payload, root=int(root))


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
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except (TypeError, ValueError):
            pass
    return str(value)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _preload_runtime_imports() -> dict[str, Any]:
    t0 = time.perf_counter()
    selected_runtime = ensure_adflow_runtime_on_path()
    import baseclasses
    import adflow

    return {
        "preload_runtime_import_sec": float(time.perf_counter() - t0),
        "selected_adflow_runtime": str(selected_runtime),
        "adflow_module": str(getattr(adflow, "__file__", "")),
        "baseclasses_module": str(getattr(baseclasses, "__file__", "")),
    }


def _write_ready_file(
    path: Path,
    *,
    manifest_path: Path,
    comm: Any,
    preload_metadata: Mapping[str, Any] | None = None,
) -> float:
    ready_unix_sec = float(time.time())
    _write_json_atomic(
        path,
        {
            "event": "nk_resume_warm_worker_ready",
            "manifest_path": str(manifest_path),
            "rank_count": int(_comm_size(comm)),
            "module_start_unix_sec": float(_MODULE_START_UNIX_SEC),
            "ready_written_unix_sec": float(ready_unix_sec),
            "preload": dict(preload_metadata or {}),
        },
    )
    return ready_unix_sec


def _load_manifest_with_optional_wait(
    path: Path,
    *,
    wait_timeout_sec: float,
    poll_sec: float = 0.05,
) -> tuple[dict[str, Any], float]:
    wait_t0 = time.perf_counter()
    deadline = None
    if float(wait_timeout_sec) > 0.0:
        deadline = wait_t0 + float(wait_timeout_sec)
    while True:
        if path.exists():
            try:
                return load_manifest_dict(path), float(time.perf_counter() - wait_t0)
            except json.JSONDecodeError:
                pass
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for manifest after {float(wait_timeout_sec):.3f}s: {path}"
            )
        time.sleep(max(0.01, float(poll_sec)))


def _job_original_index(job: Any, local_index: int) -> int:
    metadata = dict(getattr(job, "metadata", {}) or {})
    value = metadata.get("original_job_index", local_index)
    return int(value)


def execute_manifest(
    *,
    manifest_path: str | Path,
    summary_path: str | Path,
    output_dir: str | Path,
    ranks_per_case: int,
    pool_id: int,
    pool_count: int,
    injection_strategy: str,
    comm: Any,
    manifest: Mapping[str, Any] | None = None,
    manifest_wait_sec: float = 0.0,
    backend: ADflowBackend | None = None,
) -> dict[str, Any]:
    """Execute one already-sharded manifest inside one MPI worker pool."""

    if int(ranks_per_case) != _comm_size(comm):
        raise ContractError(
            f"Warm worker rank count {_comm_size(comm)} does not match ranks_per_case={ranks_per_case}"
        )
    manifest_path = Path(manifest_path).resolve()
    summary_path = Path(summary_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rank = _comm_rank(comm)

    manifest_load_wait_sec = 0.0
    if rank == 0:
        if manifest is None:
            manifest_payload, manifest_load_wait_sec = _load_manifest_with_optional_wait(
                manifest_path,
                wait_timeout_sec=float(manifest_wait_sec),
            )
        else:
            manifest_payload = dict(manifest)
    else:
        manifest_payload = None
    manifest_payload = _comm_bcast(comm, manifest_payload, root=0)
    manifest_load_wait_sec = _comm_bcast(comm, manifest_load_wait_sec, root=0)
    if not isinstance(manifest_payload, Mapping):
        raise ContractError("Warm worker received an invalid manifest payload")

    plan = resume_plan_from_dict(dict(manifest_payload.get("plan") or {}))
    replay_plan = ReplayService(
        mode="dry_run",
        ranks_per_case=int(ranks_per_case),
        pool_count=1,
    ).plan(manifest_payload)
    if backend is None:
        backend = ADflowBackend(
            comm=comm,
            state_adapter=ADflowStateAdapter(injection_strategy=str(injection_strategy)),
            metadata={
                "executor": "warm_pools",
                "pool_id": int(pool_id),
                "pool_count": int(pool_count),
            },
        )

    _comm_barrier(comm)
    batch_t0 = time.perf_counter()
    batch_start_unix_sec = float(time.time())
    jobs: list[dict[str, Any]] = []
    result_paths_by_index: dict[int, str] = {}
    for local_index, job in enumerate(replay_plan.jobs):
        job_t0 = time.perf_counter()
        case = resume_case_from_payload(job.payload_path)
        result: ProjectionResult = backend.project(
            ProjectionRequest(
                case=case,
                plan=plan,
                output_dir=job.output_dir,
                metadata={
                    "manifest_path": str(manifest_path),
                    "manifest_job": job.to_dict(),
                    "result_path": job.result_path,
                    "mpi_pool": {
                        "pool_id": int(pool_id),
                        "pool_rank": int(rank),
                        "pool_count": int(pool_count),
                        "ranks_per_case": int(ranks_per_case),
                    },
                },
            )
        )
        original_index = _job_original_index(job, local_index)
        if rank == 0:
            result_paths_by_index[int(original_index)] = str(result.result_path)
            jobs.append(
                {
                    "job_index": int(original_index),
                    "local_job_index": int(local_index),
                    "case_id": job.case_id,
                    "state_name": job.state_name,
                    "result_path": str(result.result_path),
                    "status": result.status,
                    "job_wall_sec": float(time.perf_counter() - job_t0),
                    "stage_timing": result.final_stage.timing,
                }
            )

    batch_wall_sec = float(time.perf_counter() - batch_t0)
    summary_payload = {
        "schema_version": "warm_worker_summary_v1",
        "mode": "nk_resume_warm_worker",
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "pool_id": int(pool_id),
        "pool_count": int(pool_count),
        "rank_count": int(_comm_size(comm)),
        "ranks_per_case": int(ranks_per_case),
        "job_count": int(len(replay_plan.jobs)),
        "jobs": jobs,
        "result_paths_by_index": {
            str(index): path for index, path in sorted(result_paths_by_index.items())
        },
        "module_start_unix_sec": float(_MODULE_START_UNIX_SEC),
        "batch_start_unix_sec": float(batch_start_unix_sec),
        "batch_end_unix_sec": float(time.time()),
        "batch_wall_sec": float(batch_wall_sec),
        "manifest_load_wait_sec": float(manifest_load_wait_sec),
        "injection_strategy": str(injection_strategy),
    }
    if rank == 0:
        _write_json_atomic(summary_path, summary_payload)
    _comm_barrier(comm)
    return summary_payload


def execute_prepare_manifest(
    *,
    manifest_path: str | Path,
    summary_path: str | Path,
    output_dir: str | Path,
    ranks_per_case: int,
    pool_id: int,
    pool_count: int,
    comm: Any,
    backend: ADflowBackend,
) -> dict[str, Any]:
    """Construct the solver for the single geometry in a warmup manifest."""

    if int(ranks_per_case) != _comm_size(comm):
        raise ContractError(
            f"Warm worker rank count {_comm_size(comm)} does not match ranks_per_case={ranks_per_case}"
        )
    manifest_path = Path(manifest_path).resolve()
    summary_path = Path(summary_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest_dict(manifest_path)
    replay_plan = ReplayService(
        mode="dry_run",
        ranks_per_case=int(ranks_per_case),
        pool_count=1,
    ).plan(manifest)
    if len(replay_plan.jobs) != 1:
        raise ContractError("Solver preparation requires a one-job manifest")
    plan = resume_plan_from_dict(dict(manifest.get("plan") or {}))
    job = replay_plan.jobs[0]
    case = resume_case_from_payload(job.payload_path)
    _comm_barrier(comm)
    started = time.perf_counter()
    solver_warmup = backend.prepare(
        ProjectionRequest(
            case=case,
            plan=plan,
            output_dir=job.output_dir,
            metadata={"result_path": job.result_path},
        )
    )
    _comm_barrier(comm)
    summary = {
        "schema_version": "resident_solver_prepare_v1",
        "mode": "nk_resume_resident_solver_prepare",
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "pool_id": int(pool_id),
        "pool_count": int(pool_count),
        "rank_count": int(_comm_size(comm)),
        "prepare_wall_sec": float(time.perf_counter() - started),
        "solver_warmup": solver_warmup,
    }
    if _comm_rank(comm) == 0:
        _write_json_atomic(summary_path, summary)
    _comm_barrier(comm)
    return summary


def _wait_for_resident_request(
    control_dir: Path,
    *,
    last_submit_id: str,
    wait_timeout_sec: float,
    poll_sec: float = 0.05,
) -> tuple[dict[str, Any], float]:
    request_path = control_dir / "request.json"
    wait_t0 = time.perf_counter()
    deadline = None
    if float(wait_timeout_sec) > 0.0:
        deadline = wait_t0 + float(wait_timeout_sec)
    while True:
        if request_path.exists():
            try:
                request = json.loads(request_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                request = None
            if isinstance(request, Mapping):
                submit_id = str(request.get("submit_id") or "").strip()
                request_type = str(request.get("type") or "submit").strip().lower()
                if submit_id and submit_id != last_submit_id and request_type in {"submit", "prepare", "shutdown"}:
                    return {**dict(request), "submit_id": submit_id, "type": request_type}, float(
                        time.perf_counter() - wait_t0
                    )
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for resident request after {float(wait_timeout_sec):.3f}s"
            )
        time.sleep(max(0.01, float(poll_sec)))


def run_resident_service(args: argparse.Namespace, comm: Any) -> None:
    control_dir = Path(args.resident_control_dir).resolve()
    control_dir.mkdir(parents=True, exist_ok=True)
    ready_file = Path(args.ready_file).resolve() if args.ready_file else control_dir / "service.ready.json"
    preload_metadata = _preload_runtime_imports() if bool(args.preload_runtime) else {}
    backend = ADflowBackend(
        comm=comm,
        state_adapter=ADflowStateAdapter(
            injection_strategy=str(args.injection_strategy)
        ),
        metadata={
            "executor": "resident_warm_pools",
            "pool_id": int(args.pool_id),
            "pool_count": int(args.pool_count),
        },
    )
    if _comm_rank(comm) == 0:
        _write_ready_file(
            ready_file,
            manifest_path=Path(args.manifest).resolve(),
            comm=comm,
            preload_metadata=preload_metadata,
        )
    _comm_barrier(comm)

    last_submit_id = ""
    while True:
        # Poll the atomic shared request on every rank so idle MPICH workers
        # sleep here instead of spinning inside a blocking broadcast.
        request, wait_sec = _wait_for_resident_request(
            control_dir,
            last_submit_id=last_submit_id,
            wait_timeout_sec=float(args.wait_for_manifest_sec),
        )
        request_type = str(request.get("type") or "submit")
        submit_id = str(request.get("submit_id") or "")
        if request_type == "shutdown":
            if _comm_rank(comm) == 0:
                _write_json_atomic(
                    control_dir / "done.json",
                    {"submit_id": submit_id, "type": "shutdown", "status": "ok"},
                )
            break
        manifest_path = Path(str(request.get("manifest") or request.get("manifest_path") or args.manifest)).resolve()
        summary_path = Path(
            str(request.get("summary") or request.get("summary_path") or args.summary)
        ).resolve()
        output_dir = Path(str(request.get("output_dir") or args.output_dir)).resolve()
        if request_type == "prepare":
            summary = execute_prepare_manifest(
                manifest_path=manifest_path,
                summary_path=summary_path,
                output_dir=output_dir,
                ranks_per_case=int(args.ranks_per_case),
                pool_id=int(args.pool_id),
                pool_count=int(args.pool_count),
                comm=comm,
                backend=backend,
            )
        else:
            summary = execute_manifest(
                manifest_path=manifest_path,
                summary_path=summary_path,
                output_dir=output_dir,
                ranks_per_case=int(args.ranks_per_case),
                pool_id=int(args.pool_id),
                pool_count=int(args.pool_count),
                injection_strategy=str(args.injection_strategy),
                comm=comm,
                manifest_wait_sec=0.0,
                backend=backend,
            )
        if _comm_rank(comm) == 0:
            _write_json_atomic(
                control_dir / "done.json",
                {
                    "submit_id": submit_id,
                    "type": request_type,
                    "status": "ok",
                    "request_wait_sec": float(wait_sec),
                    "summary_path": str(summary_path),
                    "summary": summary,
                },
            )
        last_submit_id = submit_id


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean NK_resume warm MPI worker service.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--warmup-manifest", default="")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ready-file", default="")
    parser.add_argument("--wait-for-manifest-sec", type=float, default=60.0)
    parser.add_argument("--resident-control-dir", default="")
    parser.add_argument("--ranks-per-case", type=int, default=8)
    parser.add_argument("--pool-id", type=int, default=0)
    parser.add_argument("--pool-count", type=int, default=1)
    parser.add_argument("--injection-strategy", choices=("restart_info", "states"), default="restart_info")
    parser.add_argument("--no-preload-runtime", dest="preload_runtime", action="store_false")
    parser.set_defaults(preload_runtime=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    comm = _mpi_comm()
    manifest_path = Path(args.manifest).resolve()
    ready_file = Path(args.ready_file).resolve() if str(args.ready_file).strip() else None
    if str(args.resident_control_dir).strip():
        run_resident_service(args, comm)
        return 0
    preload_metadata = _preload_runtime_imports() if bool(args.preload_runtime) else {}
    backend = ADflowBackend(
        comm=comm,
        state_adapter=ADflowStateAdapter(injection_strategy=str(args.injection_strategy)),
        metadata={
            "executor": "warm_pools",
            "pool_id": int(args.pool_id),
            "pool_count": int(args.pool_count),
        },
    )
    warmup_manifest_text = str(args.warmup_manifest).strip()
    if warmup_manifest_text:
        warmup_manifest = load_manifest_dict(Path(warmup_manifest_text).resolve())
        warmup_plan = resume_plan_from_dict(dict(warmup_manifest.get("plan") or {}))
        warmup_replay = ReplayService(
            mode="dry_run",
            ranks_per_case=int(args.ranks_per_case),
            pool_count=1,
        ).plan(warmup_manifest)
        warmup_job = warmup_replay.jobs[0]
        warmup_case = resume_case_from_payload(warmup_job.payload_path)
        preload_metadata["solver_warmup"] = backend.prepare(
            ProjectionRequest(
                case=warmup_case,
                plan=warmup_plan,
                output_dir=warmup_job.output_dir,
                metadata={"result_path": warmup_job.result_path},
            )
        )
    if _comm_rank(comm) == 0 and ready_file is not None:
        _write_ready_file(
            ready_file,
            manifest_path=manifest_path,
            comm=comm,
            preload_metadata=preload_metadata,
        )
    _comm_barrier(comm)
    execute_manifest(
        manifest_path=manifest_path,
        summary_path=Path(args.summary).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        ranks_per_case=int(args.ranks_per_case),
        pool_id=int(args.pool_id),
        pool_count=int(args.pool_count),
        injection_strategy=str(args.injection_strategy),
        comm=comm,
        manifest_wait_sec=float(args.wait_for_manifest_sec),
        backend=backend,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
