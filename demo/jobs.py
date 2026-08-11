"""Durable single-worker scheduling for heavyweight demo computations."""

from __future__ import annotations

import json
import math
import re
import shutil
import sqlite3
import threading
import time
import uuid
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
        normalized.update(
            {
                "cycles": _require_number(
                    payload, "cycles", minimum=1, maximum=20, integer=True, default=6
                ),
                "residual_exponent": _require_number(
                    payload,
                    "residual_exponent",
                    minimum=2,
                    maximum=12,
                    integer=True,
                    default=8,
                ),
            }
        )
    else:
        normalized["max_cycles"] = _require_number(
            payload,
            "max_cycles",
            minimum=1,
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
    """Persist jobs in SQLite and execute heavyweight engine calls one at a time."""

    def __init__(
        self,
        engine: Any,
        *,
        runtime_root: Path,
        max_pending_jobs: int = 64,
        max_pending_jobs_per_client: int = 4,
        result_ttl_sec: float = 86_400.0,
        cleanup_interval_sec: float = 60.0,
        max_payload_bytes: int = 256_000,
        max_result_bytes: int = 64 * 1024 * 1024,
        action_timeouts: dict[str, float] | None = None,
        autostart: bool = True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.engine = engine
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
        self.action_timeouts = {
            "mesh": 600.0,
            "predict": 600.0,
            "recover": 7200.0,
            "reference": 7200.0,
            **(action_timeouts or {}),
        }
        if self.max_pending_jobs < 1 or self.max_pending_jobs_per_client < 1:
            raise ValueError("Scheduler queue limits must be positive.")
        if self.result_ttl_sec <= 0 or self.cleanup_interval_sec <= 0:
            raise ValueError("Scheduler TTL and cleanup interval must be positive.")
        if self.max_payload_bytes < 1024 or self.max_result_bytes < 1024:
            raise ValueError("Scheduler payload and result limits are too small.")
        if set(self.action_timeouts) != set(JOB_ACTIONS) or any(
            float(value) <= 0 for value in self.action_timeouts.values()
        ):
            raise ValueError("Every job action must have a positive timeout.")
        self._clock = clock
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.root.mkdir(parents=True, exist_ok=True)
        self.job_root.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
        self._recover_interrupted_jobs()
        if autostart:
            self.start()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
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
                CREATE INDEX IF NOT EXISTS jobs_state_sequence
                    ON jobs(state, sequence);
                CREATE INDEX IF NOT EXISTS jobs_client_state
                    ON jobs(client_id, state);
                CREATE INDEX IF NOT EXISTS jobs_expiry
                    ON jobs(expires_at, state);
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

    def _recover_interrupted_jobs(self) -> None:
        now = self._clock()
        with self._connect() as connection:
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
        if self._thread is not None and self._thread.is_alive():
            return
        if self._stop.is_set():
            raise RuntimeError("A closed scheduler cannot be restarted.")
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="demo-heavy-job-worker",
            daemon=True,
        )
        self._thread.start()
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
        now = self._clock()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
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
                    created_at, updated_at
                ) VALUES (?, ?, 'queued', ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    action,
                    safe_client_id,
                    encoded,
                    float(self.action_timeouts[action]),
                    now,
                    now,
                ),
            )
        self._wake.set()
        return self.get(job_id)

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

    def get(self, job_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            return self._serialize_row(self._row(job_id, connection), connection)

    def result(self, job_id: str) -> Any:
        with self._connect() as connection:
            row = self._row(job_id, connection)
            if row["state"] != "succeeded":
                raise JobStateError(f"Job result is unavailable while state is {row['state']}.")
        result_path = self._result_path(job_id)
        if not result_path.is_file():
            raise JobStateError("Job result is no longer available.")
        if result_path.stat().st_size > self.max_result_bytes:
            raise JobStateError("Job result exceeds the server limit.")
        return json.loads(result_path.read_text(encoding="utf-8"))

    def cancel(self, job_id: str) -> dict[str, Any]:
        now = self._clock()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._row(job_id, connection)
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
        with self._connect() as connection:
            counts = {
                row["state"]: int(row["count"])
                for row in connection.execute(
                    "SELECT state, COUNT(*) AS count FROM jobs GROUP BY state"
                ).fetchall()
            }
            running = connection.execute(
                "SELECT job_id, action, started_at FROM jobs WHERE state = 'running' LIMIT 1"
            ).fetchone()
        return {
            "concurrency_limit": 1,
            "queue_depth": counts.get("queued", 0),
            "running": None if running is None else dict(running),
            "counts": counts,
            "max_pending_jobs": self.max_pending_jobs,
            "max_pending_jobs_per_client": self.max_pending_jobs_per_client,
            "result_ttl_sec": self.result_ttl_sec,
        }

    def _claim_next(self) -> sqlite3.Row | None:
        now = self._clock()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM jobs WHERE state = 'queued' ORDER BY sequence LIMIT 1"
            ).fetchone()
            if row is None:
                return None
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
            return self._row(row["job_id"], connection)

    def _dispatch(self, action: str, payload: dict[str, Any]) -> Any:
        if action == "mesh":
            return self.engine.prepare_mesh(**payload)
        if action == "predict":
            return self.engine.predict(**payload)
        case_id = str(payload.pop("case_id"))
        if action == "recover":
            return self.engine.recover(case_id, **payload)
        if action == "reference":
            return self.engine.reference(case_id, **payload)
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
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._row(job_id, connection)
            if row["cancel_requested"]:
                connection.execute(
                    """
                    UPDATE jobs SET state = 'cancelled', updated_at = ?, finished_at = ?,
                                    expires_at = ?, result_path = NULL
                     WHERE job_id = ?
                    """,
                    (now, now, now + self.result_ttl_sec, job_id),
                )
                self._delete_job_directory(job_id)
            else:
                connection.execute(
                    """
                    UPDATE jobs SET state = 'succeeded', updated_at = ?, finished_at = ?,
                                    expires_at = ?, result_path = 'result.json'
                     WHERE job_id = ?
                    """,
                    (now, now, now + self.result_ttl_sec, job_id),
                )

    def _finish_failure(self, job_id: str, exc: Exception) -> None:
        now = self._clock()
        with self._connect() as connection:
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

    def _run_job(self, row: sqlite3.Row) -> None:
        job_id = str(row["job_id"])
        try:
            payload = json.loads(row["payload_json"])
            result = self._dispatch(str(row["action"]), payload)
            if self._clock() - float(row["started_at"]) > float(row["timeout_sec"]):
                raise TimeoutError("The compute job exceeded its time limit.")
            self._finish_success(job_id, result)
        except Exception as exc:
            self._finish_failure(job_id, exc)

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            self.cleanup_expired()
            row = self._claim_next()
            if row is None:
                self._wake.wait(self.cleanup_interval_sec)
                self._wake.clear()
                continue
            self._run_job(row)

    def _delete_job_directory(self, job_id: str) -> None:
        directory = self._job_directory(job_id)
        if directory.is_dir():
            shutil.rmtree(directory)

    def cleanup_expired(self) -> int:
        now = self._clock()
        with self._connect() as connection:
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
        return len(job_ids)

    def close(self, *, timeout: float | None = 10.0) -> bool:
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()
