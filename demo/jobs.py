"""Durable single-worker scheduling for heavyweight demo computations."""

from __future__ import annotations

import json
import hashlib
import math
import re
import shutil
import sqlite3
import struct
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

JOB_ACTIONS = frozenset({"mesh", "predict", "recover", "reference"})
TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled", "expired"})
JOB_ID_PATTERN = re.compile(r"^job_[0-9a-f]{32}$")
CASE_ID_PATTERN = re.compile(r"^case_[A-Za-z0-9_]+$")


class JobError(RuntimeError):
    """Base class for scheduler errors that are safe to map to an API response."""


class JobNotFoundError(JobError):
    """Raised when a job identifier is unknown."""


class JobStateError(JobError):
    """Raised when an operation is incompatible with the current job state."""


class QueueCapacityError(JobError):
    """Raised when a global or per-client pending-job limit is reached."""


def _require_number(
    payload: dict[str, Any],
    name: str,
    *,
    minimum: float,
    maximum: float,
    integer: bool = False,
    default: float | int | None = None,
) -> float | int:
    value = payload.get(name, default)
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} is required.")
    try:
        converted = int(value) if integer else float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not minimum <= converted <= maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}.")
    return converted


def _geometry27(payload: dict[str, Any]) -> list[float]:
    values = payload.get("geometry27")
    if not isinstance(values, list) or len(values) != 27:
        raise ValueError("geometry27 must contain exactly 27 values.")
    try:
        converted = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("geometry27 must contain only numeric values.") from exc
    if not all(math.isfinite(value) for value in converted):
        raise ValueError("geometry27 must contain only finite values.")
    return converted


def validate_job_payload(action: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize and constrain a public job request before it reaches the engine."""

    if action not in JOB_ACTIONS:
        raise ValueError(f"Unknown job action: {action}.")
    if not isinstance(payload, dict):
        raise ValueError("Job payload must be an object.")
    normalized: dict[str, Any]
    if action in {"mesh", "predict"}:
        name = str(payload.get("name") or "Demo airfoil").strip()
        if not name or len(name) > 128:
            raise ValueError("name must contain between 1 and 128 characters.")
        normalized = {"geometry27": _geometry27(payload), "name": name}
        if action == "predict":
            normalized.update(
                {
                    "mach": _require_number(payload, "mach", minimum=0.2, maximum=0.9),
                    "aoa": _require_number(payload, "aoa", minimum=-5.0, maximum=10.0),
                    "n_inference_steps": _require_number(
                        payload,
                        "n_inference_steps",
                        minimum=1,
                        maximum=20,
                        integer=True,
                        default=5,
                    ),
                }
            )
        return normalized

    case_id = str(payload.get("case_id") or "")
    if not CASE_ID_PATTERN.fullmatch(case_id):
        raise ValueError("case_id is invalid.")
    normalized = {"case_id": case_id}
    if action == "recover":
        if "cycles" in payload:
            raise ValueError(
                "The NK compute budget is fixed by the server; cycles is not configurable."
            )
        normalized["residual_exponent"] = _require_number(
            payload,
            "residual_exponent",
            minimum=4,
            maximum=10,
            integer=True,
            default=6,
        )
    else:
        normalized["max_cycles"] = _require_number(
            payload,
            "max_cycles",
            minimum=25,
            maximum=3000,
            integer=True,
            default=3000,
        )
    return normalized


def _public_error(exc: Exception) -> str:
    text = str(exc)
    contains_absolute_path = any(
        Path(part.strip("'\"(),:;")).is_absolute() for part in text.split()
    )
    if isinstance(exc, ValueError) and not contains_absolute_path:
        return text
    if isinstance(exc, TimeoutError):
        return "The compute job exceeded its time limit."
    if isinstance(exc, (FileNotFoundError, KeyError)):
        return "The requested compute resource was not found."
    if isinstance(exc, (ConnectionError, TimeoutError, RuntimeError)):
        return "The requested compute runtime is unavailable."
    return "The compute job failed."


class JobScheduler:
    """Persist jobs in SQLite and execute them on dedicated engine workers."""

    def __init__(
        self,
        engine: Any | list[Any] | tuple[Any, ...],
        *,
        runtime_root: Path,
        max_pending_jobs: int = 64,
        max_pending_jobs_per_client: int = 4,
        result_ttl_sec: float = 86_400.0,
        cleanup_interval_sec: float = 60.0,
        max_payload_bytes: int = 256_000,
        max_result_bytes: int = 64 * 1024 * 1024,
        action_timeouts: dict[str, float] | None = None,
        nk_burst_limit: int = 3,
        cold_start_max_wait_sec: float = 300.0,
        case_root: Path | None = None,
        mesh_root: Path | None = None,
        solver_prepare_root: Path | None = None,
        case_ttl_sec: float | None = None,
        enforce_case_ownership: bool = False,
        hard_timeout_handler: Callable[[dict[str, Any]], None] | None = None,
        cancel_grace_sec: float = 30.0,
        autostart: bool = True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.engines = list(engine) if isinstance(engine, (list, tuple)) else [engine]
        if not self.engines or len(self.engines) > 2:
            raise ValueError("Heavy-job concurrency must be either 1 or 2.")
        self.engine = self.engines[0]
        self.root = Path(runtime_root).expanduser().resolve()
        if not self.root.is_absolute():
            raise ValueError("The scheduler runtime root must be absolute.")
        self.job_root = self.root / "results"
        self.db_path = self.root / "jobs.sqlite3"
        self.max_pending_jobs = int(max_pending_jobs)
        self.max_pending_jobs_per_client = int(max_pending_jobs_per_client)
        self.result_ttl_sec = float(result_ttl_sec)
        self.cleanup_interval_sec = float(cleanup_interval_sec)
        self.max_payload_bytes = int(max_payload_bytes)
        self.max_result_bytes = int(max_result_bytes)
        self.nk_burst_limit = int(nk_burst_limit)
        self.cold_start_max_wait_sec = float(cold_start_max_wait_sec)
        self.case_root = (
            None if case_root is None else Path(case_root).expanduser().resolve()
        )
        self.mesh_root = (
            None if mesh_root is None else Path(mesh_root).expanduser().resolve()
        )
        self.solver_prepare_root = (
            None
            if solver_prepare_root is None
            else Path(solver_prepare_root).expanduser().resolve()
        )
        self.case_ttl_sec = float(case_ttl_sec or result_ttl_sec)
        self.enforce_case_ownership = bool(enforce_case_ownership)
        self.hard_timeout_handler = hard_timeout_handler
        self.cancel_grace_sec = float(cancel_grace_sec)
        self.action_timeouts = {
            "mesh": 600.0,
            "predict": 600.0,
            "recover": 7200.0,
            "reference": 7200.0,
            **(action_timeouts or {}),
        }
        if self.max_pending_jobs < 1 or self.max_pending_jobs_per_client < 1:
            raise ValueError("Scheduler queue limits must be positive.")
        if (
            self.result_ttl_sec <= 0
            or self.case_ttl_sec <= 0
            or self.cleanup_interval_sec <= 0
        ):
            raise ValueError("Scheduler TTL and cleanup interval must be positive.")
        if self.max_payload_bytes < 1024 or self.max_result_bytes < 1024:
            raise ValueError("Scheduler payload and result limits are too small.")
        if self.nk_burst_limit < 1 or self.cold_start_max_wait_sec <= 0:
            raise ValueError("Scheduler priority limits must be positive.")
        if self.cancel_grace_sec <= 0:
            raise ValueError("The running-job cancellation grace must be positive.")
        if set(self.action_timeouts) != set(JOB_ACTIONS) or any(
            float(value) <= 0 for value in self.action_timeouts.values()
        ):
            raise ValueError("Every job action must have a positive timeout.")
        self._clock = clock
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.root.mkdir(parents=True, exist_ok=True)
        self.job_root.mkdir(parents=True, exist_ok=True)
        if self.case_root is not None:
            self.case_root.mkdir(parents=True, exist_ok=True)
        if self.mesh_root is not None:
            self.mesh_root.mkdir(parents=True, exist_ok=True)
        if self.solver_prepare_root is not None:
            self.solver_prepare_root.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
        self._recover_interrupted_jobs()
        if autostart:
            self.start()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def _connection(self) -> Any:
        """Commit or roll back and always close a short-lived SQLite connection."""

        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize_database(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL UNIQUE,
                    action TEXT NOT NULL,
                    state TEXT NOT NULL,
                    client_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    timeout_sec REAL NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    expires_at REAL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    result_path TEXT,
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS scheduler_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                INSERT OR IGNORE INTO scheduler_state(key, value)
                    VALUES ('nk_streak', '0');
                CREATE INDEX IF NOT EXISTS jobs_state_sequence
                    ON jobs(state, sequence);
                CREATE INDEX IF NOT EXISTS jobs_client_state
                    ON jobs(client_id, state);
                CREATE INDEX IF NOT EXISTS jobs_expiry
                    ON jobs(expires_at, state);
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    geometry_key TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS cases_expiry
                    ON cases(expires_at);
                CREATE TABLE IF NOT EXISTS geometries (
                    geometry_key TEXT PRIMARY KEY,
                    geometry_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS geometries_expiry
                    ON geometries(expires_at);
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
            }
            if "timeout_sec" not in columns:
                connection.execute(
                    "ALTER TABLE jobs ADD COLUMN timeout_sec REAL NOT NULL DEFAULT 7200"
                )
            if "resource_key" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN resource_key TEXT")
            case_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(cases)").fetchall()
            }
            if "geometry_key" not in case_columns:
                connection.execute("ALTER TABLE cases ADD COLUMN geometry_key TEXT")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS jobs_running_resource "
                "ON jobs(state, resource_key)"
            )

    def _recover_interrupted_jobs(self) -> None:
        now = self._clock()
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE jobs
                   SET state = 'failed', updated_at = ?, finished_at = ?, expires_at = ?,
                       error = 'The service restarted while this job was running.'
                 WHERE state = 'running'
                """,
                (now, now, now + self.result_ttl_sec),
            )

    def start(self) -> None:
        if self._threads and any(thread.is_alive() for thread in self._threads):
            return
        if self._stop.is_set():
            raise RuntimeError("A closed scheduler cannot be restarted.")
        self._threads = [
            threading.Thread(
                target=self._worker_loop,
                args=(worker_id, engine),
                name=f"demo-heavy-job-worker-{worker_id}",
                daemon=True,
            )
            for worker_id, engine in enumerate(self.engines)
        ]
        if self.hard_timeout_handler is not None:
            self._threads.append(
                threading.Thread(
                    target=self._watchdog_loop,
                    name="demo-heavy-job-watchdog",
                    daemon=True,
                )
            )
        for thread in self._threads:
            thread.start()
        self._wake.set()

    def submit(
        self,
        action: str,
        payload: dict[str, Any],
        *,
        client_id: str,
    ) -> dict[str, Any]:
        action = str(action)
        normalized = validate_job_payload(action, payload)
        encoded = json.dumps(normalized, separators=(",", ":"), allow_nan=False)
        if len(encoded.encode("utf-8")) > self.max_payload_bytes:
            raise ValueError("Job payload exceeds the server limit.")
        safe_client_id = str(client_id).strip()[:128] or "unknown"
        job_id = f"job_{uuid.uuid4().hex}"
        if action in {"recover", "reference"}:
            resource_key = f"case:{normalized['case_id']}"
        else:
            geometry_bytes = struct.pack("<27f", *normalized["geometry27"])
            resource_key = f"geometry:{hashlib.sha256(geometry_bytes).hexdigest()[:12]}"
        now = self._clock()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if action in {"recover", "reference"} and self.enforce_case_ownership:
                self._authorize_case_in_connection(
                    str(normalized["case_id"]), safe_client_id, connection, touch=True
                )
            pending = int(
                connection.execute(
                    "SELECT COUNT(*) FROM jobs WHERE state IN ('queued', 'running')"
                ).fetchone()[0]
            )
            if pending >= self.max_pending_jobs:
                raise QueueCapacityError("The compute queue is full; please retry later.")
            client_pending = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM jobs
                     WHERE client_id = ? AND state IN ('queued', 'running')
                    """,
                    (safe_client_id,),
                ).fetchone()[0]
            )
            if client_pending >= self.max_pending_jobs_per_client:
                raise QueueCapacityError(
                    "This client already has the maximum number of pending jobs."
                )
            connection.execute(
                """
                INSERT INTO jobs(
                    job_id, action, state, client_id, payload_json, timeout_sec,
                    created_at, updated_at, resource_key
                ) VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    action,
                    safe_client_id,
                    encoded,
                    float(self.action_timeouts[action]),
                    now,
                    now,
                    resource_key,
                ),
            )
        self._wake.set()
        return self.get(job_id, client_id=safe_client_id)

    def _row(self, job_id: str, connection: sqlite3.Connection) -> sqlite3.Row:
        if not JOB_ID_PATTERN.fullmatch(job_id):
            raise JobNotFoundError("Job not found.")
        row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise JobNotFoundError("Job not found.")
        return row

    def _serialize_row(
        self, row: sqlite3.Row, connection: sqlite3.Connection
    ) -> dict[str, Any]:
        queue_position: int | None = None
        if row["state"] == "queued":
            queue_position = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM jobs
                     WHERE state = 'queued' AND sequence <= ?
                    """,
                    (row["sequence"],),
                ).fetchone()[0]
            )
        result_url = (
            f"/api/jobs/{row['job_id']}/result" if row["state"] == "succeeded" else None
        )
        return {
            "job_id": row["job_id"],
            "action": row["action"],
            "state": row["state"],
            "queue_position": queue_position,
            "cancel_requested": bool(row["cancel_requested"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "expires_at": row["expires_at"],
            "timeout_sec": row["timeout_sec"],
            "result_url": result_url,
            "error": row["error"],
        }

    @staticmethod
    def _authorize_job(row: sqlite3.Row, client_id: str | None) -> None:
        if client_id is not None and str(row["client_id"]) != str(client_id):
            raise JobNotFoundError("Job not found.")

    def get(self, job_id: str, *, client_id: str | None = None) -> dict[str, Any]:
        with self._connection() as connection:
            row = self._row(job_id, connection)
            self._authorize_job(row, client_id)
            return self._serialize_row(row, connection)

    def result(self, job_id: str, *, client_id: str | None = None) -> Any:
        with self._connection() as connection:
            row = self._row(job_id, connection)
            self._authorize_job(row, client_id)
            if row["state"] != "succeeded":
                raise JobStateError(f"Job result is unavailable while state is {row['state']}.")
        result_path = self._result_path(job_id)
        if not result_path.is_file():
            raise JobStateError("Job result is no longer available.")
        if result_path.stat().st_size > self.max_result_bytes:
            raise JobStateError("Job result exceeds the server limit.")
        return json.loads(result_path.read_text(encoding="utf-8"))

    def cancel(self, job_id: str, *, client_id: str | None = None) -> dict[str, Any]:
        now = self._clock()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._row(job_id, connection)
            self._authorize_job(row, client_id)
            if row["state"] == "queued":
                connection.execute(
                    """
                    UPDATE jobs
                       SET state = 'cancelled', cancel_requested = 1,
                           updated_at = ?, finished_at = ?, expires_at = ?
                     WHERE job_id = ?
                    """,
                    (now, now, now + self.result_ttl_sec, job_id),
                )
            elif row["state"] == "running":
                connection.execute(
                    "UPDATE jobs SET cancel_requested = 1, updated_at = ? WHERE job_id = ?",
                    (now, job_id),
                )
            updated = self._row(job_id, connection)
            summary = self._serialize_row(updated, connection)
        self._wake.set()
        return summary

    def stats(self) -> dict[str, Any]:
        with self._connection() as connection:
            counts = {
                row["state"]: int(row["count"])
                for row in connection.execute(
                    "SELECT state, COUNT(*) AS count FROM jobs GROUP BY state"
                ).fetchall()
            }
            running_rows = connection.execute(
                "SELECT job_id, action, started_at FROM jobs "
                "WHERE state = 'running' ORDER BY started_at"
            ).fetchall()
            running_jobs = [dict(row) for row in running_rows]
            case_count = int(connection.execute("SELECT COUNT(*) FROM cases").fetchone()[0])
            geometry_count = int(
                connection.execute("SELECT COUNT(*) FROM geometries").fetchone()[0]
            )
        return {
            "concurrency_limit": len(self.engines),
            "queue_depth": counts.get("queued", 0),
            "running": running_jobs[0] if running_jobs else None,
            "running_jobs": running_jobs,
            "counts": counts,
            "max_pending_jobs": self.max_pending_jobs,
            "max_pending_jobs_per_client": self.max_pending_jobs_per_client,
            "result_ttl_sec": self.result_ttl_sec,
            "case_ttl_sec": self.case_ttl_sec,
            "retained_cases": case_count,
            "retained_geometries": geometry_count,
            "hard_timeout_watchdog": self.hard_timeout_handler is not None,
            "priority_policy": {
                "nk_burst_limit": self.nk_burst_limit,
                "cold_start_max_wait_sec": self.cold_start_max_wait_sec,
            },
        }

    def _claim_next(self) -> sqlite3.Row | None:
        now = self._clock()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            eligible = connection.execute(
                """
                SELECT queued.* FROM jobs AS queued
                 WHERE queued.state = 'queued'
                   AND NOT EXISTS (
                       SELECT 1 FROM jobs AS running
                        WHERE running.state = 'running'
                          AND running.resource_key IS NOT NULL
                          AND running.resource_key = queued.resource_key
                   )
                 ORDER BY queued.sequence
                """
            ).fetchall()
            if not eligible:
                return None
            nk_streak = int(
                connection.execute(
                    "SELECT value FROM scheduler_state WHERE key = 'nk_streak'"
                ).fetchone()[0]
            )
            references = [row for row in eligible if row["action"] == "reference"]
            oldest_reference_wait = (
                0.0 if not references else now - float(references[0]["created_at"])
            )
            if references and (
                nk_streak >= self.nk_burst_limit
                or oldest_reference_wait >= self.cold_start_max_wait_sec
            ):
                row = references[0]
            else:
                priority = {"recover": 0, "mesh": 1, "predict": 2, "reference": 3}
                row = min(
                    eligible,
                    key=lambda candidate: (
                        priority[str(candidate["action"])], int(candidate["sequence"])
                    ),
                )
            updated = connection.execute(
                """
                UPDATE jobs
                   SET state = 'running', started_at = ?, updated_at = ?
                 WHERE job_id = ? AND state = 'queued'
                """,
                (now, now, row["job_id"]),
            )
            if updated.rowcount != 1:
                return None
            if row["action"] == "recover":
                nk_streak += 1
            elif row["action"] == "reference":
                nk_streak = 0
            connection.execute(
                "UPDATE scheduler_state SET value = ? WHERE key = 'nk_streak'",
                (str(nk_streak),),
            )
            return self._row(row["job_id"], connection)

    def _dispatch(self, engine: Any, action: str, payload: dict[str, Any]) -> Any:
        if action == "mesh":
            return engine.prepare_mesh(**payload)
        if action == "predict":
            return engine.predict(**payload)
        case_id = str(payload.pop("case_id"))
        if action == "recover":
            return engine.recover(case_id, **payload)
        if action == "reference":
            return engine.reference(case_id, **payload)
        raise ValueError(f"Unsupported job action: {action}.")

    def _job_directory(self, job_id: str) -> Path:
        if not JOB_ID_PATTERN.fullmatch(job_id):
            raise JobNotFoundError("Job not found.")
        path = (self.job_root / job_id).resolve()
        if path.parent != self.job_root.resolve():
            raise RuntimeError("Unsafe job result path.")
        return path

    def _result_path(self, job_id: str) -> Path:
        return self._job_directory(job_id) / "result.json"

    def _store_result(self, job_id: str, result: Any) -> None:
        encoded = json.dumps(result, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(encoded) > self.max_result_bytes:
            raise RuntimeError("The job result exceeds the configured disk limit.")
        directory = self._job_directory(job_id)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "result.json.tmp"
        temporary.write_bytes(encoded)
        temporary.replace(directory / "result.json")

    def _finish_success(self, job_id: str, result: Any) -> None:
        self._store_result(job_id, result)
        now = self._clock()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._row(job_id, connection)
            if row["cancel_requested"]:
                geometry_key, geometry_id = self._result_geometry(result)
                if geometry_key is not None and geometry_id is not None:
                    self._register_geometry_in_connection(
                        geometry_key, geometry_id, connection, now=now
                    )
                connection.execute(
                    """
                    UPDATE jobs SET state = 'cancelled', updated_at = ?, finished_at = ?,
                                    expires_at = ?, result_path = NULL
                     WHERE job_id = ?
                    """,
                    (now, now, now + self.result_ttl_sec, job_id),
                )
                self._delete_job_directory(job_id)
                case_id = result.get("case_id") if isinstance(result, dict) else None
                if row["action"] == "predict" and isinstance(case_id, str):
                    self._delete_case_directory(case_id)
            else:
                connection.execute(
                    """
                    UPDATE jobs SET state = 'succeeded', updated_at = ?, finished_at = ?,
                                    expires_at = ?, result_path = 'result.json'
                     WHERE job_id = ?
                    """,
                    (now, now, now + self.result_ttl_sec, job_id),
                )
                geometry_key, geometry_id = self._result_geometry(result)
                if geometry_key is not None and geometry_id is not None:
                    self._register_geometry_in_connection(
                        geometry_key, geometry_id, connection, now=now
                    )
                case_id = result.get("case_id") if isinstance(result, dict) else None
                if row["action"] == "predict" and isinstance(case_id, str):
                    self._register_case_in_connection(
                        case_id,
                        str(row["client_id"]),
                        connection,
                        geometry_key=geometry_key,
                        now=now,
                    )

    def _finish_failure(self, job_id: str, exc: Exception) -> None:
        now = self._clock()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._row(job_id, connection)
            cancelled = bool(row["cancel_requested"])
            connection.execute(
                """
                UPDATE jobs SET state = ?, updated_at = ?, finished_at = ?, expires_at = ?,
                                error = ?
                 WHERE job_id = ?
                """,
                (
                    "cancelled" if cancelled else "failed",
                    now,
                    now,
                    now + self.result_ttl_sec,
                    None if cancelled else _public_error(exc),
                    job_id,
                ),
            )
        self._delete_job_directory(job_id)

    def _run_job(self, engine: Any, row: sqlite3.Row) -> None:
        job_id = str(row["job_id"])
        try:
            payload = json.loads(row["payload_json"])
            result = self._dispatch(engine, str(row["action"]), payload)
            if self._clock() - float(row["started_at"]) > float(row["timeout_sec"]):
                raise TimeoutError("The compute job exceeded its time limit.")
            self._finish_success(job_id, result)
        except Exception as exc:
            self._finish_failure(job_id, exc)

    def _worker_loop(self, _worker_id: int, engine: Any) -> None:
        while not self._stop.is_set():
            self.cleanup_expired()
            row = self._claim_next()
            if row is None:
                self._wake.wait(self.cleanup_interval_sec)
                self._wake.clear()
                continue
            self._run_job(engine, row)

    def _watchdog_loop(self) -> None:
        """Escalate a stuck native/MPI call to the process supervisor."""

        escalated: set[str] = set()
        while not self._stop.wait(1.0):
            now = self._clock()
            with self._connection() as connection:
                rows = connection.execute(
                    "SELECT job_id, action, started_at, updated_at, timeout_sec, "
                    "cancel_requested FROM jobs WHERE state = 'running'"
                ).fetchall()
            for row in rows:
                timed_out = now >= float(row["started_at"]) + float(row["timeout_sec"])
                cancelled_stuck = bool(row["cancel_requested"]) and (
                    now >= float(row["updated_at"]) + self.cancel_grace_sec
                )
                job_id = str(row["job_id"])
                if not (timed_out or cancelled_stuck) or job_id in escalated:
                    continue
                escalated.add(job_id)
                assert self.hard_timeout_handler is not None
                self.hard_timeout_handler(
                    {
                        "job_id": job_id,
                        "action": str(row["action"]),
                        "reason": "timeout" if timed_out else "cancel_timeout",
                    }
                )

    def _delete_job_directory(self, job_id: str) -> None:
        directory = self._job_directory(job_id)
        if directory.is_dir():
            shutil.rmtree(directory)

    def cleanup_expired(self) -> int:
        now = self._clock()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT job_id FROM jobs
                 WHERE state IN ('succeeded', 'failed', 'cancelled')
                   AND expires_at IS NOT NULL AND expires_at <= ?
                """,
                (now,),
            ).fetchall()
            job_ids = [str(row["job_id"]) for row in rows]
            if job_ids:
                for job_id in job_ids:
                    self._delete_job_directory(job_id)
                connection.executemany(
                    """
                    UPDATE jobs SET state = 'expired', updated_at = ?, result_path = NULL
                     WHERE job_id = ?
                    """,
                    [(now, job_id) for job_id in job_ids],
                )
            expired_cases = connection.execute(
                """
                SELECT case_id FROM cases
                 WHERE expires_at <= ?
                   AND NOT EXISTS (
                       SELECT 1 FROM jobs
                        WHERE state IN ('queued', 'running')
                          AND resource_key = 'case:' || cases.case_id
                   )
                """,
                (now,),
            ).fetchall()
            case_ids = [str(row["case_id"]) for row in expired_cases]
            for case_id in case_ids:
                self._delete_case_directory(case_id)
            if case_ids:
                connection.executemany(
                    "DELETE FROM cases WHERE case_id = ?",
                    [(case_id,) for case_id in case_ids],
                )
            expired_geometries = connection.execute(
                """
                SELECT geometry_key, geometry_id FROM geometries
                 WHERE expires_at <= ?
                   AND NOT EXISTS (
                       SELECT 1 FROM cases
                        WHERE cases.geometry_key = geometries.geometry_key
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM jobs
                        WHERE state IN ('queued', 'running')
                          AND resource_key = 'geometry:' || geometries.geometry_key
                   )
                """,
                (now,),
            ).fetchall()
            geometry_keys = [str(row["geometry_key"]) for row in expired_geometries]
            for row in expired_geometries:
                self._delete_geometry_artifacts(
                    str(row["geometry_key"]), str(row["geometry_id"])
                )
            if geometry_keys:
                connection.executemany(
                    "DELETE FROM geometries WHERE geometry_key = ?",
                    [(geometry_key,) for geometry_key in geometry_keys],
                )
        return len(job_ids) + len(case_ids) + len(geometry_keys)

    @staticmethod
    def _result_geometry(result: Any) -> tuple[str | None, str | None]:
        if not isinstance(result, dict):
            return None, None
        geometry_id = result.get("geometry_id")
        mesh = result.get("mesh") if isinstance(result.get("mesh"), dict) else result
        geometry_key = mesh.get("geometry_key") if isinstance(mesh, dict) else None
        if not isinstance(geometry_key, str) or not re.fullmatch(r"[0-9a-f]{12}", geometry_key):
            return None, None
        if not isinstance(geometry_id, str) or not re.fullmatch(r"[0-9a-f]{16}", geometry_id):
            return None, None
        return geometry_key, geometry_id

    def _register_geometry_in_connection(
        self,
        geometry_key: str,
        geometry_id: str,
        connection: sqlite3.Connection,
        *,
        now: float,
    ) -> None:
        connection.execute(
            """
            INSERT INTO geometries(geometry_key, geometry_id, created_at, updated_at, expires_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(geometry_key) DO UPDATE SET
                geometry_id = excluded.geometry_id,
                updated_at = excluded.updated_at,
                expires_at = excluded.expires_at
            """,
            (geometry_key, geometry_id, now, now, now + self.case_ttl_sec),
        )

    def _register_case_in_connection(
        self,
        case_id: str,
        client_id: str,
        connection: sqlite3.Connection,
        *,
        geometry_key: str | None = None,
        now: float | None = None,
    ) -> None:
        if not CASE_ID_PATTERN.fullmatch(case_id):
            raise ValueError("case_id is invalid.")
        timestamp = self._clock() if now is None else now
        connection.execute(
            """
            INSERT INTO cases(
                case_id, client_id, geometry_key, created_at, updated_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(case_id) DO UPDATE SET
                geometry_key = excluded.geometry_key,
                updated_at = excluded.updated_at,
                expires_at = excluded.expires_at
            """,
            (
                case_id,
                client_id,
                geometry_key,
                timestamp,
                timestamp,
                timestamp + self.case_ttl_sec,
            ),
        )

    def register_case(self, case_id: str, client_id: str) -> None:
        with self._connection() as connection:
            self._register_case_in_connection(case_id, client_id, connection)

    def _authorize_case_in_connection(
        self,
        case_id: str,
        client_id: str,
        connection: sqlite3.Connection,
        *,
        touch: bool,
    ) -> None:
        if not CASE_ID_PATTERN.fullmatch(case_id):
            raise FileNotFoundError("Case not found.")
        row = connection.execute(
            "SELECT client_id FROM cases WHERE case_id = ?", (case_id,)
        ).fetchone()
        if row is None or str(row["client_id"]) != str(client_id):
            raise FileNotFoundError("Case not found.")
        if touch:
            now = self._clock()
            connection.execute(
                "UPDATE cases SET updated_at = ?, expires_at = ? WHERE case_id = ?",
                (now, now + self.case_ttl_sec, case_id),
            )

    def authorize_case(self, case_id: str, client_id: str, *, touch: bool = True) -> None:
        with self._connection() as connection:
            self._authorize_case_in_connection(
                case_id, client_id, connection, touch=touch
            )

    def _delete_case_directory(self, case_id: str) -> None:
        if self.case_root is None or not CASE_ID_PATTERN.fullmatch(case_id):
            return
        directory = (self.case_root / case_id).resolve()
        if directory.parent != self.case_root.resolve():
            raise RuntimeError("Unsafe case cleanup path.")
        if directory.is_dir():
            shutil.rmtree(directory)

    @staticmethod
    def _safe_cache_path(root: Path, name: str) -> Path:
        path = (root / name).resolve()
        if path.parent != root.resolve():
            raise RuntimeError("Unsafe geometry cleanup path.")
        return path

    def _delete_geometry_artifacts(self, geometry_key: str, geometry_id: str) -> None:
        if not re.fullmatch(r"[0-9a-f]{12}", geometry_key) or not re.fullmatch(
            r"[0-9a-f]{16}", geometry_id
        ):
            return
        if self.mesh_root is not None:
            for name in (
                f"{geometry_key}.demo.json",
                f"{geometry_id}.cgns",
                f"{geometry_id}.cgns.lock",
            ):
                path = self._safe_cache_path(self.mesh_root, name)
                if path.is_file():
                    path.unlink()
        if self.solver_prepare_root is not None:
            directory = self._safe_cache_path(self.solver_prepare_root, geometry_key)
            if directory.is_dir():
                shutil.rmtree(directory)

    def close(self, *, timeout: float | None = 10.0) -> bool:
        self._stop.set()
        self._wake.set()
        if not self._threads:
            return True
        deadline = None if timeout is None else time.monotonic() + timeout
        for thread in self._threads:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            thread.join(timeout=remaining)
        return not any(thread.is_alive() for thread in self._threads)
