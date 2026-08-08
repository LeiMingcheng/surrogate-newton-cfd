"""Warm MPI pool controller for clean NK_resume manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from typing import Any, Mapping

from ..exceptions import ContractError
from ..geometry import cgns_geometry_key
from ..payload import load_manifest_dict
from .mpi_env import (
    build_mpi_env,
    inject_mpi_runtime_env_args,
    interesting_env_subset,
    python_executable,
    resolve_mpi_launcher,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def _split_manifest_payloads(manifest: Mapping[str, Any], pool_count: int) -> list[dict[str, Any]]:
    jobs = list(manifest.get("jobs") or [])
    manifest_metadata = dict(manifest.get("metadata") or {})
    if not jobs:
        raise ContractError("Manifest has no jobs to split")
    if int(pool_count) <= 0:
        raise ContractError("pool_count must be positive")
    if int(pool_count) > len(jobs):
        raise ContractError(
            f"pool_count={pool_count} is larger than manifest job_count={len(jobs)}"
        )
    indexed_jobs: list[dict[str, Any]] = []
    for index, job in enumerate(jobs):
        if not isinstance(job, Mapping):
            raise ContractError("Manifest jobs must be objects")
        cloned = {str(k): v for k, v in dict(job).items()}
        metadata = dict(cloned.get("metadata") or {})
        metadata["original_job_index"] = int(index)
        cloned["metadata"] = metadata
        indexed_jobs.append(cloned)

    groups: dict[str, list[dict[str, Any]]] = {}
    for job in indexed_jobs:
        payload = dict(job.get("payload") or {})
        geometry_bundle = dict(payload.get("geometry_bundle") or {})
        basename = str(geometry_bundle.get("cgns_basename") or job.get("case_id") or "")
        groups.setdefault(cgns_geometry_key(basename), []).append(job)

    chunks: list[list[dict[str, Any]]] = [[] for _ in range(int(pool_count))]
    loads = [0 for _ in range(int(pool_count))]
    target_load = (len(indexed_jobs) + int(pool_count) - 1) // int(pool_count)
    for group in groups.values():
        start = 0
        while start < len(group):
            pool_id = min(range(int(pool_count)), key=lambda idx: (loads[idx], idx))
            available = max(1, target_load - loads[pool_id])
            stop = min(len(group), start + available)
            chunks[pool_id].extend(group[start:stop])
            loads[pool_id] += stop - start
            start = stop
    payloads: list[dict[str, Any]] = []
    for pool_id, chunk in enumerate(chunks):
        payloads.append(
            {
                "schema_version": manifest.get("schema_version"),
                "plan": dict(manifest.get("plan") or {}),
                "jobs": chunk,
                "metadata": {
                    **manifest_metadata,
                    "split_from_manifest": str(manifest_metadata.get("source_manifest", "")),
                    "pool_id": int(pool_id),
                    "pool_count": int(pool_count),
                    "executor": "warm_pools",
                },
            }
        )
    return payloads


def _split_manifest_for_pools(
    manifest_path: str | Path,
    *,
    pool_count: int,
    split_dir: str | Path,
) -> tuple[str, ...]:
    manifest = load_manifest_dict(manifest_path)
    manifest["metadata"] = {
        **dict(manifest.get("metadata") or {}),
        "source_manifest": str(Path(manifest_path).resolve()),
    }
    payloads = _split_manifest_payloads(manifest, int(pool_count))
    split_root = Path(split_dir).resolve()
    split_paths: list[str] = []
    for pool_id, payload in enumerate(payloads):
        path = split_root / f"pool_{int(pool_id):04d}.manifest.json"
        _write_json_atomic(path, payload)
        split_paths.append(str(path))
    return tuple(split_paths)


def _wait_for_ready_files(entries: list[dict[str, Any]], timeout_sec: float) -> None:
    deadline = time.perf_counter() + max(0.0, float(timeout_sec))
    pending = {str(entry["ready_file"]) for entry in entries}
    while pending:
        pending = {item for item in pending if not Path(item).exists()}
        if not pending:
            return
        exited = [
            entry
            for entry in entries
            if str(entry["ready_file"]) in pending
            and entry["proc"].poll() is not None
        ]
        if exited:
            details = {
                str(entry["pool_id"]): int(entry["proc"].returncode)
                for entry in exited
            }
            raise RuntimeError(f"Warm worker exited before ready: {details}")
        if time.perf_counter() >= deadline:
            raise TimeoutError(f"Timed out waiting for ready files: {sorted(pending)}")
        time.sleep(0.05)


def _validate_resident_results(
    jobs: list[dict[str, Any]],
    result_paths: Mapping[int, str],
) -> tuple[str, ...]:
    validated: list[str] = []
    missing: list[int] = []
    for index, job in enumerate(jobs):
        path_text = str(result_paths.get(index) or "")
        expected_path = Path(str(job["result_path"])).resolve()
        if not path_text or Path(path_text).resolve() != expected_path:
            missing.append(index)
            continue
        if not expected_path.is_file():
            missing.append(index)
            continue
        try:
            payload = json.loads(expected_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ContractError(
                f"Resident NK result is not valid JSON: {expected_path}"
            ) from error
        if str(payload.get("case_id") or "") != str(job["case_id"]):
            raise ContractError(
                "Resident NK result case_id differs from the manifest: "
                f"index={index}, expected={job['case_id']}, "
                f"actual={payload.get('case_id')}"
            )
        validated.append(str(expected_path))
    if missing:
        raise ContractError(
            f"Resident NK pools did not persist manifest jobs: {missing}"
        )
    return tuple(validated)


def _terminate_process_group(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(int(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.perf_counter() + 2.0
    while proc.poll() is None and time.perf_counter() < deadline:
        time.sleep(0.05)
    if proc.poll() is None:
        os.killpg(int(proc.pid), signal.SIGKILL)


@dataclass(frozen=True)
class WarmPoolManifestProjectionResult:
    """Projection artifacts produced by warm independent MPI pools."""

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


class ResidentWarmPoolController:
    """Keep independent MPI workers alive across manifest submissions.

    The MPI interpreter, ADFLOW imports, and backend are resident.  The backend
    reuses its solver only while the resolved CGNS root and logical geometry key
    remain unchanged; a different candidate geometry constructs a new solver.
    """

    def __init__(
        self,
        *,
        ranks_per_case: int = 8,
        pool_count: int = 3,
        mpi_launcher: str = "auto",
        mpi_omp_threads: int = 1,
        output_dir: str | Path,
        ready_timeout_sec: float = 30.0,
        submit_timeout_sec: float = 60.0,
        request_wait_timeout_sec: float = 7200.0,
        injection_strategy: str = "restart_info",
    ) -> None:
        if int(ranks_per_case) <= 0 or int(pool_count) <= 0:
            raise ContractError("Resident pool ranks and pool_count must be positive")
        self.ranks_per_case = int(ranks_per_case)
        self.pool_count = int(pool_count)
        self.mpi_launcher = str(mpi_launcher)
        self.mpi_omp_threads = int(mpi_omp_threads)
        self.output_root = Path(output_dir).resolve()
        self.ready_timeout_sec = float(ready_timeout_sec)
        self.submit_timeout_sec = float(submit_timeout_sec)
        self.request_wait_timeout_sec = float(request_wait_timeout_sec)
        self.injection_strategy = str(injection_strategy)
        self._entries: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._launched = False
        self._launch_count = 0
        self._submit_count = 0
        self._prepare_count = 0
        self._ready_wait_sec = 0.0
        self._ready_unix_sec: float | None = None
        self._poisoned = False

    def _poison(self) -> None:
        self._poisoned = True
        self.abort()

    def _check_workers(self) -> None:
        if self._poisoned:
            raise RuntimeError(
                "Resident NK pool controller is not reusable after failure"
            )
        for entry in self._entries:
            proc = entry["proc"]
            returncode = proc.poll()
            if returncode is not None:
                self._poison()
                raise RuntimeError(
                    f"Resident NK pool {entry['pool_id']} exited with code {returncode}; "
                    f"see {entry['log_path']}"
                )

    def _launch(self) -> None:
        if self._launched:
            self._check_workers()
            return
        self.output_root.mkdir(parents=True, exist_ok=True)
        env = build_mpi_env(
            self.mpi_omp_threads,
            self.output_root / "mpi_tmp",
        )
        project_root_text = str(PROJECT_ROOT)
        pythonpath = str(env.get("PYTHONPATH") or "")
        env["PYTHONPATH"] = (
            project_root_text
            if not pythonpath
            else f"{project_root_text}{os.pathsep}{pythonpath}"
        )
        launcher = inject_mpi_runtime_env_args(
            resolve_mpi_launcher(self.mpi_launcher),
            env,
        )
        launcher_name = os.path.basename(launcher[0]).lower() if launcher else ""
        launch_t0 = time.perf_counter()
        try:
            for pool_id in range(self.pool_count):
                pool_dir = self.output_root / f"pool_{pool_id:04d}"
                control_dir = pool_dir / "control"
                control_dir.mkdir(parents=True, exist_ok=True)
                ready_file = control_dir / "service.ready.json"
                request_path = control_dir / "request.json"
                done_path = control_dir / "done.json"
                for stale_path in (ready_file, request_path, done_path):
                    if stale_path.exists():
                        stale_path.unlink()
                log_path = pool_dir / "resident_worker.log"
                default_summary_path = pool_dir / "resident_summary.json"
                placeholder_manifest = pool_dir / "manifest.placeholder.json"
                pool_launcher = list(launcher)
                if (
                    (
                        launcher_name.startswith("mpirun")
                        or launcher_name.startswith("mpiexec")
                    )
                    and "-wdir" not in pool_launcher
                    and "--wdir" not in pool_launcher
                ):
                    pool_launcher.extend(["-wdir", str(pool_dir)])
                command = [
                    *pool_launcher,
                    "-np",
                    str(self.ranks_per_case),
                    python_executable(),
                    "-m",
                    "NK_resume.solver.worker_service",
                    "--manifest",
                    str(placeholder_manifest),
                    "--summary",
                    str(default_summary_path),
                    "--output-dir",
                    str(pool_dir),
                    "--ready-file",
                    str(ready_file),
                    "--resident-control-dir",
                    str(control_dir),
                    "--wait-for-manifest-sec",
                    str(self.request_wait_timeout_sec),
                    "--ranks-per-case",
                    str(self.ranks_per_case),
                    "--pool-id",
                    str(pool_id),
                    "--pool-count",
                    str(self.pool_count),
                    "--injection-strategy",
                    self.injection_strategy,
                ]
                log_handle = log_path.open("w", encoding="utf-8")
                proc = subprocess.Popen(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=str(PROJECT_ROOT),
                    start_new_session=True,
                )
                self._entries.append(
                    {
                        "pool_id": int(pool_id),
                        "pool_dir": pool_dir,
                        "control_dir": control_dir,
                        "ready_file": ready_file,
                        "request_path": request_path,
                        "done_path": done_path,
                        "log_path": log_path,
                        "default_summary_path": default_summary_path,
                        "launch_cmd": [str(token) for token in command],
                        "proc": proc,
                        "log_handle": log_handle,
                    }
                )
            _wait_for_ready_files(self._entries, self.ready_timeout_sec)
        except BaseException:
            for entry in self._entries:
                _terminate_process_group(entry["proc"])
                entry["log_handle"].close()
            self._entries.clear()
            raise
        self._ready_wait_sec = float(time.perf_counter() - launch_t0)
        self._ready_unix_sec = float(time.time())
        self._launch_count += 1
        self._launched = True

    def _submit(
        self,
        entry: Mapping[str, Any],
        *,
        submit_id: str,
        manifest_path: str | Path,
        summary_path: str | Path,
        output_dir: str | Path,
        request_type: str = "submit",
    ) -> None:
        done_path = Path(entry["done_path"])
        if done_path.exists():
            done_path.unlink()
        _write_json_atomic(
            Path(entry["request_path"]),
            {
                "type": str(request_type),
                "submit_id": str(submit_id),
                "manifest_path": str(Path(manifest_path).resolve()),
                "summary_path": str(Path(summary_path).resolve()),
                "output_dir": str(Path(output_dir).resolve()),
            },
        )

    def start(self) -> dict[str, Any]:
        """Launch the resident MPI workers and wait for their ready signals."""

        with self._lock:
            already_running = self._launched
            self._launch()
            return {
                "resident": True,
                "pool_controller_reused": bool(already_running),
                "pool_ready_wait_sec": 0.0 if already_running else float(self._ready_wait_sec),
                "ranks_per_case": self.ranks_per_case,
                "pool_count": self.pool_count,
            }

    def prepare(
        self,
        manifest_path: str | Path,
        *,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        """Construct and retain one geometry-matched ADflow solver."""

        with self._lock:
            manifest = load_manifest_dict(manifest_path)
            jobs = list(manifest.get("jobs") or [])
            if len(jobs) != 1:
                raise ContractError("Resident solver preparation requires one manifest job")
            already_running = self._launched
            self._launch()
            output_root = Path(output_dir).resolve()
            output_root.mkdir(parents=True, exist_ok=True)
            submit_id = (
                f"prepare_{self._prepare_count:06d}_{int(time.time() * 1_000_000)}"
            )
            summary_path = output_root / "resident_solver_prepare.summary.json"
            entry = self._entries[0]
            self._submit(
                entry,
                submit_id=submit_id,
                manifest_path=manifest_path,
                summary_path=summary_path,
                output_dir=output_root,
                request_type="prepare",
            )
            done = self._wait_for_done(
                entry,
                submit_id=submit_id,
                timeout_sec=self.submit_timeout_sec,
            )
            if str(done.get("status") or "") != "ok":
                self._poison()
                raise RuntimeError(f"Resident ADflow preparation failed: {done}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self._prepare_count += 1
            return {
                **summary,
                "resident": True,
                "ranks_per_case": self.ranks_per_case,
                "pool_controller_reused": bool(already_running),
                "summary_path": str(summary_path),
            }

    def _wait_for_done(
        self,
        entry: Mapping[str, Any],
        *,
        submit_id: str,
        timeout_sec: float,
    ) -> dict[str, Any]:
        deadline = time.perf_counter() + max(0.0, float(timeout_sec))
        done_path = Path(entry["done_path"])
        while True:
            self._check_workers()
            if done_path.exists():
                payload = json.loads(done_path.read_text(encoding="utf-8"))
                if str(payload.get("submit_id") or "") == str(submit_id):
                    return payload
            if time.perf_counter() >= deadline:
                self._poison()
                raise TimeoutError(
                    f"Timed out waiting for resident NK pool {entry['pool_id']} "
                    f"submission {submit_id}"
                )
            time.sleep(0.01)

    def project(
        self,
        manifest_path: str | Path,
        *,
        output_dir: str | Path,
    ) -> WarmPoolManifestProjectionResult:
        """Project one manifest using the already-running worker pools."""

        with self._lock:
            manifest = load_manifest_dict(manifest_path)
            jobs = list(manifest.get("jobs") or [])
            if not jobs:
                raise ContractError("Manifest has no jobs")
            active_pool_count = min(self.pool_count, len(jobs))
            self._launch()
            reused = self._submit_count > 0 or self._prepare_count > 0
            output_root = Path(output_dir).resolve()
            output_root.mkdir(parents=True, exist_ok=True)
            split_paths = _split_manifest_for_pools(
                manifest_path,
                pool_count=active_pool_count,
                split_dir=output_root / "split_manifests",
            )
            submit_id = (
                f"submit_{self._submit_count:06d}_{int(time.time() * 1_000_000)}"
            )
            records: list[dict[str, Any]] = []
            controller_t0 = time.perf_counter()
            for entry, split_path in zip(self._entries, split_paths):
                summary_path = (
                    output_root
                    / f"resident_pool_{int(entry['pool_id']):04d}.summary.json"
                )
                self._submit(
                    entry,
                    submit_id=submit_id,
                    manifest_path=split_path,
                    summary_path=summary_path,
                    output_dir=output_root / f"pool_{int(entry['pool_id']):04d}",
                )
                records.append(
                    {
                        "pool_id": int(entry["pool_id"]),
                        "manifest_path": str(Path(split_path).resolve()),
                        "summary_path": str(summary_path),
                        "log_path": str(entry["log_path"]),
                    }
                )

            deadline = time.perf_counter() + self.submit_timeout_sec
            result_paths: dict[int, str] = {}
            for record, entry in zip(records, self._entries):
                done = self._wait_for_done(
                    entry,
                    submit_id=submit_id,
                    timeout_sec=max(0.0, deadline - time.perf_counter()),
                )
                if str(done.get("status") or "") != "ok":
                    self._poison()
                    raise RuntimeError(
                        f"Resident NK pool {entry['pool_id']} failed submission "
                        f"{submit_id}: {done}"
                    )
                summary_path = Path(record["summary_path"])
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except (FileNotFoundError, json.JSONDecodeError) as error:
                    self._poison()
                    raise ContractError(
                        f"Resident NK pool summary is not durable: {summary_path}"
                    ) from error
                record["request_wait_sec"] = float(done.get("request_wait_sec", 0.0))
                record["batch_wall_sec"] = float(summary.get("batch_wall_sec", 0.0))
                for key, value in dict(
                    summary.get("result_paths_by_index") or {}
                ).items():
                    result_paths[int(key)] = str(value)
            missing = [index for index in range(len(jobs)) if index not in result_paths]
            if missing:
                self._poison()
                raise ContractError(
                    f"Resident NK pools did not produce manifest jobs: {missing}"
                )
            try:
                validated_result_paths = _validate_resident_results(jobs, result_paths)
            except ContractError:
                self._poison()
                raise

            controller_wall_sec = float(time.perf_counter() - controller_t0)
            controller_summary = {
                "schema_version": "resident_warm_pool_summary_v1",
                "mode": "nk_resume_resident_warm_pools",
                "manifest_path": str(Path(manifest_path).resolve()),
                "output_dir": str(output_root),
                "submit_id": submit_id,
                "job_count": len(jobs),
                "active_pool_count": active_pool_count,
                "resident_pool_count": self.pool_count,
                "ranks_per_case": self.ranks_per_case,
                "pool_controller_reused": bool(reused),
                "pool_ready_wait_sec": (
                    0.0 if reused else float(self._ready_wait_sec)
                ),
                "controller_wall_sec": controller_wall_sec,
                "all_ready_unix_sec": self._ready_unix_sec,
                "launch_count": self._launch_count,
                "submit_count": self._submit_count + 1,
                "solver_lifecycle": "geometry_scoped_solver_cache",
                "pools": records,
            }
            controller_summary_path = output_root / "resident_pool_summary.json"
            _write_json_atomic(controller_summary_path, controller_summary)
            self._submit_count += 1
            return WarmPoolManifestProjectionResult(
                manifest_path=str(Path(manifest_path).resolve()),
                job_count=len(jobs),
                ranks_per_case=self.ranks_per_case,
                pool_count=active_pool_count,
                result_paths=validated_result_paths,
                status_paths=tuple(
                    [str(controller_summary_path)]
                    + [str(record["summary_path"]) for record in records]
                ),
                metadata={
                    "primary": True,
                    "resident": True,
                    "pool_controller_reused": bool(reused),
                    "pool_ready_wait_sec": (
                        0.0 if reused else float(self._ready_wait_sec)
                    ),
                    "controller_wall_sec": controller_wall_sec,
                    "controller_summary_path": str(controller_summary_path),
                    "solver_lifecycle": "geometry_scoped_solver_cache",
                },
            )

    def close(self) -> None:
        with self._lock:
            if not self._launched:
                return
            submit_id = f"shutdown_{int(time.time() * 1_000_000)}"
            for entry in self._entries:
                done_path = Path(entry["done_path"])
                if done_path.exists():
                    done_path.unlink()
                _write_json_atomic(
                    Path(entry["request_path"]),
                    {
                        "type": "shutdown",
                        "submit_id": submit_id,
                    },
                )
            deadline = time.perf_counter() + min(30.0, self.submit_timeout_sec)
            for entry in self._entries:
                remaining = max(0.0, deadline - time.perf_counter())
                try:
                    self._wait_for_done(
                        entry,
                        submit_id=submit_id,
                        timeout_sec=remaining,
                    )
                    entry["proc"].wait(timeout=max(0.0, remaining))
                except (RuntimeError, TimeoutError, subprocess.TimeoutExpired):
                    _terminate_process_group(entry["proc"])
                entry["log_handle"].close()
            self._entries.clear()
            self._launched = False
            self._poisoned = False

    def abort(self) -> None:
        """Immediately terminate this controller's worker process groups."""

        for entry in tuple(self._entries):
            _terminate_process_group(entry["proc"])

    def __enter__(self) -> "ResidentWarmPoolController":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def project_manifest_warm_pools(
    manifest_path: str,
    *,
    ranks_per_case: int = 8,
    pool_count: int | None = None,
    mpi_launcher: str = "auto",
    mpi_omp_threads: int = 1,
    output_dir: str | Path | None = None,
    ready_timeout_sec: float = 30.0,
    submit_timeout_sec: float = 60.0,
    wait_for_manifest_sec: float = 60.0,
    injection_strategy: str = "restart_info",
) -> WarmPoolManifestProjectionResult:
    """Execute a clean manifest with pre-started independent MPI worker pools."""

    manifest_path = str(manifest_path)
    if not manifest_path.strip():
        raise ContractError("manifest_path is required")
    manifest = load_manifest_dict(manifest_path)
    jobs = list(manifest.get("jobs") or [])
    job_count = len(jobs)
    if job_count <= 0:
        raise ContractError("Manifest has no jobs")
    ranks_per_case = int(ranks_per_case)
    if ranks_per_case <= 0:
        raise ContractError("ranks_per_case must be positive")
    if pool_count is None:
        resolved_pool_count = min(5, job_count)
        pool_count_source = "default_min_5_job_count"
    else:
        resolved_pool_count = int(pool_count)
        pool_count_source = "explicit"
    if resolved_pool_count <= 0:
        raise ContractError("pool_count must be positive")
    if resolved_pool_count > job_count:
        raise ContractError(
            f"pool_count={resolved_pool_count} is larger than manifest job_count={job_count}"
        )

    output_root = (
        Path(output_dir).resolve()
        if output_dir is not None and str(output_dir).strip()
        else Path(manifest_path).resolve().parent / "warm_pools_runtime"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    split_root = output_root / "split_manifests"
    split_root.mkdir(parents=True, exist_ok=True)
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    manifest_for_split = {
        **manifest,
        "metadata": {
            **dict(manifest.get("metadata") or {}),
            "source_manifest": str(Path(manifest_path).resolve()),
        },
    }
    split_payloads = _split_manifest_payloads(manifest_for_split, resolved_pool_count)

    env = build_mpi_env(int(mpi_omp_threads), output_root / "mpi_tmp")
    project_root_text = str(PROJECT_ROOT)
    pythonpath = str(env.get("PYTHONPATH") or "")
    env["PYTHONPATH"] = (
        project_root_text if not pythonpath else f"{project_root_text}{os.pathsep}{pythonpath}"
    )
    launcher = inject_mpi_runtime_env_args(resolve_mpi_launcher(mpi_launcher), env)
    launcher_name = os.path.basename(launcher[0]).lower() if launcher else ""

    entries: list[dict[str, Any]] = []
    controller_t0 = time.perf_counter()
    for pool_id, split_payload in enumerate(split_payloads):
        pool_dir = output_root / f"pool_{int(pool_id):04d}"
        pool_dir.mkdir(parents=True, exist_ok=True)
        submit_manifest_path = split_root / f"pool_{int(pool_id):04d}.manifest.json"
        warmup_manifest_path = split_root / f"pool_{int(pool_id):04d}.warmup.json"
        _write_json_atomic(warmup_manifest_path, dict(split_payload))
        ready_file = pool_dir / "worker.ready.json"
        worker_summary_path = pool_dir / "worker_summary.json"
        log_path = log_root / f"pool_{int(pool_id):04d}.log"
        pool_launcher = list(launcher)
        if (
            (launcher_name.startswith("mpirun") or launcher_name.startswith("mpiexec"))
            and "-wdir" not in pool_launcher
            and "--wdir" not in pool_launcher
        ):
            pool_launcher.extend(["-wdir", str(pool_dir)])
        cmd = [
            *pool_launcher,
            "-np",
            str(ranks_per_case),
            python_executable(),
            "-m",
            "NK_resume.solver.worker_service",
            "--manifest",
            str(submit_manifest_path),
            "--warmup-manifest",
            str(warmup_manifest_path),
            "--summary",
            str(worker_summary_path),
            "--output-dir",
            str(pool_dir),
            "--ready-file",
            str(ready_file),
            "--wait-for-manifest-sec",
            str(float(wait_for_manifest_sec)),
            "--ranks-per-case",
            str(ranks_per_case),
            "--pool-id",
            str(int(pool_id)),
            "--pool-count",
            str(int(resolved_pool_count)),
            "--injection-strategy",
            str(injection_strategy),
        ]
        log_handle = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(PROJECT_ROOT),
            start_new_session=True,
        )
        entries.append(
            {
                "pool_id": int(pool_id),
                "pool_dir": str(pool_dir),
                "submit_manifest_path": str(submit_manifest_path),
                "warmup_manifest_path": str(warmup_manifest_path),
                "ready_file": str(ready_file),
                "worker_summary_path": str(worker_summary_path),
                "log_path": str(log_path),
                "launch_cmd": [str(token) for token in cmd],
                "launch_unix_sec": float(time.time()),
                "proc": proc,
                "log_handle": log_handle,
                "split_payload": split_payload,
            }
        )

    try:
        _wait_for_ready_files(entries, float(ready_timeout_sec))
        all_ready_unix_sec = float(time.time())
        for entry in entries:
            _write_json_atomic(Path(entry["submit_manifest_path"]), dict(entry["split_payload"]))
            entry["submit_unix_sec"] = float(time.time())
            entry["job_count"] = int(len(list(entry["split_payload"].get("jobs") or [])))
    except BaseException:
        for entry in entries:
            proc = entry.get("proc")
            if isinstance(proc, subprocess.Popen):
                _terminate_process_group(proc)
            handle = entry.get("log_handle")
            if handle is not None:
                handle.close()
        raise

    submit_deadline = time.perf_counter() + max(0.0, float(submit_timeout_sec))
    failed = False
    for entry in entries:
        proc = entry["proc"]
        remaining = max(0.0, submit_deadline - time.perf_counter())
        try:
            returncode = proc.wait(timeout=remaining if remaining > 0.0 else 0.0)
        except subprocess.TimeoutExpired:
            _terminate_process_group(proc)
            returncode = proc.wait()
            entry["timed_out"] = True
        else:
            entry["timed_out"] = False
        entry["returncode"] = int(returncode)
        entry["exit_unix_sec"] = float(time.time())
        entry["log_handle"].close()
        if int(returncode) != 0:
            failed = True
        summary_path = Path(entry["worker_summary_path"])
        if summary_path.exists():
            entry["worker_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            failed = True
        ready_path = Path(entry["ready_file"])
        if ready_path.exists():
            entry["ready_payload"] = json.loads(ready_path.read_text(encoding="utf-8"))
        entry.pop("proc", None)
        entry.pop("log_handle", None)
        entry.pop("split_payload", None)

    if failed:
        failure_summary = {
            str(entry["pool_id"]): {
                "returncode": entry.get("returncode"),
                "timed_out": entry.get("timed_out"),
                "log_path": entry.get("log_path"),
                "worker_summary_path": entry.get("worker_summary_path"),
            }
            for entry in entries
            if int(entry.get("returncode", 1)) != 0
            or not Path(str(entry.get("worker_summary_path"))).exists()
        }
        raise RuntimeError(f"Warm pool execution failed: {failure_summary}")

    result_paths: dict[int, str] = {}
    for entry in entries:
        summary = dict(entry.get("worker_summary") or {})
        for key, value in dict(summary.get("result_paths_by_index") or {}).items():
            result_paths[int(key)] = str(value)
    missing = [index for index in range(job_count) if index not in result_paths]
    if missing:
        raise ContractError(f"Warm pool execution did not produce jobs: {missing}")

    total_wall_sec = float(time.perf_counter() - controller_t0)
    submit_to_finish_wall_sec = 0.0
    if entries:
        submit_to_finish_wall_sec = max(
            float(entry.get("exit_unix_sec", all_ready_unix_sec)) - float(all_ready_unix_sec)
            for entry in entries
        )
    controller_summary = {
        "schema_version": "warm_pool_controller_summary_v1",
        "mode": "nk_resume_warm_pools",
        "manifest_path": manifest_path,
        "output_dir": str(output_root),
        "job_count": int(job_count),
        "pool_count": int(resolved_pool_count),
        "pool_count_source": pool_count_source,
        "ranks_per_case": int(ranks_per_case),
        "injection_strategy": str(injection_strategy),
        "launcher_env": interesting_env_subset(env),
        "all_ready_unix_sec": float(all_ready_unix_sec),
        "total_wall_sec": float(total_wall_sec),
        "submit_to_finish_wall_sec": float(submit_to_finish_wall_sec),
        "pools": entries,
    }
    controller_summary_path = output_root / "warm_pool_controller_summary.json"
    _write_json_atomic(controller_summary_path, controller_summary)

    return WarmPoolManifestProjectionResult(
        manifest_path=manifest_path,
        job_count=int(job_count),
        ranks_per_case=int(ranks_per_case),
        pool_count=int(resolved_pool_count),
        result_paths=tuple(result_paths[index] for index in range(job_count)),
        status_paths=tuple(
            [str(controller_summary_path)]
            + [str(Path(entry["worker_summary_path"])) for entry in entries]
        ),
        metadata={
            "primary": True,
            "output_dir": str(output_root),
            "controller_summary_path": str(controller_summary_path),
            "total_wall_sec": float(total_wall_sec),
            "submit_to_finish_wall_sec": float(submit_to_finish_wall_sec),
            "pool_count_source": pool_count_source,
        },
    )
