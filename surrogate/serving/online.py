"""Online serving sample records and a small persistent buffer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import sqlite3
import threading
import time
from typing import Any, Iterable, Optional

import numpy as np

from surrogate.serving.contracts import OnlineSample


_RECORD_COLUMNS = (
    "sample_id, geometry_id, source, cfd_status, model_version, npz_path, "
    "priority_score, branch_id, generation, optimizer_sample_id, created_at, updated_at, "
    "metadata_json"
)


@dataclass(frozen=True)
class OnlineSampleRecord:
    """Indexed metadata for one persistent online sample."""

    sample_id: str
    geometry_id: str
    source: str
    cfd_status: str
    model_version: Optional[str]
    npz_path: str
    priority_score: float
    branch_id: Optional[str]
    generation: Optional[int]
    optimizer_sample_id: Optional[str]
    created_at: float
    updated_at: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_numpy(value: Any, *, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def compute_sample_id(
    geometry: Any,
    flow_conditions: Any,
    *,
    target_cl: Optional[float] = None,
) -> str:
    """Compute a stable case id from geometry and flow conditions."""
    geometry_arr = _to_numpy(geometry, dtype=np.float32)
    flow_arr = _to_numpy(flow_conditions, dtype=np.float32).reshape(-1)
    if target_cl is not None and np.isfinite(float(target_cl)) and flow_arr.size >= 3:
        identity_flow = np.asarray([flow_arr[0], flow_arr[2], float(target_cl)], dtype=np.float32)
    else:
        identity_flow = flow_arr.astype(np.float32, copy=False)
    digest = hashlib.sha256(geometry_arr.tobytes() + identity_flow.tobytes()).hexdigest()
    return digest[:16]


def compute_geometry_id(geometry: Any) -> str:
    """Compute a stable geometry-only id."""
    digest = hashlib.sha256(_to_numpy(geometry, dtype=np.float32).tobytes()).hexdigest()
    return digest[:16]


class SQLiteOnlineBuffer:
    """SQLite + NPZ online sample buffer used by serving/runtime code."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.samples_dir = self.root / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "metadata.db"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / ".metadata.init.lock"
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with self._connect() as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS samples (
                        sample_id TEXT PRIMARY KEY,
                        geometry_id TEXT,
                        source TEXT,
                        cfd_status TEXT,
                        model_version TEXT,
                        npz_path TEXT NOT NULL,
                        priority_score REAL DEFAULT 0.0,
                        branch_id TEXT,
                        generation INTEGER,
                        optimizer_sample_id TEXT,
                        created_at REAL,
                        updated_at REAL,
                        metadata_json TEXT
                    )
                    """
                )
                columns = {
                    row[1]
                    for row in conn.execute("PRAGMA table_info(samples)").fetchall()
                }
                for column_name, ddl in (
                    (
                        "priority_score",
                        "ALTER TABLE samples ADD COLUMN priority_score REAL DEFAULT 0.0",
                    ),
                    ("branch_id", "ALTER TABLE samples ADD COLUMN branch_id TEXT"),
                    ("generation", "ALTER TABLE samples ADD COLUMN generation INTEGER"),
                    (
                        "optimizer_sample_id",
                        "ALTER TABLE samples ADD COLUMN optimizer_sample_id TEXT",
                    ),
                ):
                    if column_name not in columns:
                        conn.execute(ddl)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_samples_updated "
                    "ON samples(updated_at DESC)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_samples_status ON samples(cfd_status)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_samples_branch_generation "
                    "ON samples(branch_id, generation, priority_score DESC, updated_at DESC)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_samples_optimizer "
                    "ON samples(optimizer_sample_id, updated_at DESC)"
                )

    def register(self, sample: OnlineSample) -> Path:
        """Persist a sample and upsert its metadata."""
        sample_id = str(sample.sample_id)
        path = self.samples_dir / f"{sample_id}.npz"
        payload: dict[str, Any] = {
            "geometry": _to_numpy(sample.geometry, dtype=np.float32),
            "flow_conditions": _to_numpy(sample.flow_conditions, dtype=np.float32),
            "coords_vertex": _to_numpy(sample.coords_vertex, dtype=np.float64),
            "coords": _to_numpy(sample.coords, dtype=np.float32),
            "source": np.asarray(str(sample.source)),
            "cfd_status": np.asarray(str(sample.cfd_status)),
            "metadata_json": np.asarray(
                json.dumps(dict(sample.metadata), sort_keys=True, default=str)
            ),
        }
        optional_arrays = {
            "fields": sample.fields,
            "pred_fields": sample.pred_fields,
            "wall_distance": sample.wall_distance,
        }
        for key, value in optional_arrays.items():
            if value is not None:
                payload[key] = _to_numpy(value)
        temporary = self.samples_dir / (
            f".{sample_id}.{os.getpid()}.{threading.get_ident()}.writing.npz"
        )
        np.savez(temporary, **payload)
        temporary.replace(path)

        now = time.time()
        metadata = dict(sample.metadata)
        priority_score = float(metadata.get("priority_score", 0.0))
        branch_id = metadata.get("branch_id")
        generation = metadata.get("generation")
        optimizer_sample_id = metadata.get("optimizer_sample_id")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO samples (
                    sample_id, geometry_id, source, cfd_status, model_version,
                    npz_path, priority_score, branch_id, generation,
                    optimizer_sample_id, created_at, updated_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sample_id) DO UPDATE SET
                    geometry_id=excluded.geometry_id,
                    source=excluded.source,
                    cfd_status=excluded.cfd_status,
                    model_version=excluded.model_version,
                    npz_path=excluded.npz_path,
                    priority_score=excluded.priority_score,
                    branch_id=excluded.branch_id,
                    generation=excluded.generation,
                    optimizer_sample_id=excluded.optimizer_sample_id,
                    updated_at=excluded.updated_at,
                    metadata_json=excluded.metadata_json
                """,
                (
                    sample_id,
                    compute_geometry_id(sample.geometry),
                    str(sample.source),
                    str(sample.cfd_status),
                    sample.model_version,
                    str(path),
                    priority_score,
                    None if branch_id is None else str(branch_id),
                    None if generation is None else int(generation),
                    None if optimizer_sample_id is None else str(optimizer_sample_id),
                    now,
                    now,
                    json.dumps(metadata, sort_keys=True, default=str),
                ),
            )
        return path

    def delete_ids(self, sample_ids: Iterable[str]) -> dict[str, int]:
        """Delete exact indexed samples and their NPZ payloads."""

        identifiers = tuple(dict.fromkeys(str(value) for value in sample_ids))
        if not identifiers:
            return {"records": 0, "files": 0, "bytes": 0}
        placeholders = ",".join("?" for _ in identifiers)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT sample_id, npz_path FROM samples "
                f"WHERE sample_id IN ({placeholders})",
                identifiers,
            ).fetchall()
        files = 0
        removed_bytes = 0
        for _, npz_path in rows:
            path = Path(str(npz_path))
            if path.is_file():
                removed_bytes += path.stat().st_size
                path.unlink()
                files += 1
        with self._connect() as conn:
            conn.execute(
                f"DELETE FROM samples WHERE sample_id IN ({placeholders})",
                identifiers,
            )
        return {
            "records": len(rows),
            "files": files,
            "bytes": removed_bytes,
        }

    def delete_orphans(self) -> dict[str, int]:
        """Delete NPZ files that have no metadata row."""

        with self._connect() as conn:
            indexed = {
                Path(str(row[0])).resolve()
                for row in conn.execute("SELECT npz_path FROM samples").fetchall()
            }
        orphans = [
            path
            for path in self.samples_dir.glob("*.npz")
            if path.resolve() not in indexed
        ]
        removed_bytes = sum(path.stat().st_size for path in orphans)
        for path in orphans:
            path.unlink()
        return {
            "files": len(orphans),
            "bytes": removed_bytes,
        }

    def load(self, sample_id: str) -> OnlineSample:
        """Load a persisted sample."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT npz_path, model_version FROM samples WHERE sample_id = ?",
                (str(sample_id),),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown online sample id: {sample_id}")
        npz_path, model_version = row
        with np.load(npz_path, allow_pickle=False) as data:
            metadata = (
                json.loads(str(data["metadata_json"].item()))
                if "metadata_json" in data
                else {}
            )
            return OnlineSample(
                sample_id=str(sample_id),
                geometry=np.array(data["geometry"], copy=True),
                flow_conditions=np.array(data["flow_conditions"], copy=True),
                coords_vertex=np.array(data["coords_vertex"], copy=True),
                coords=np.array(data["coords"], copy=True),
                fields=np.array(data["fields"], copy=True) if "fields" in data else None,
                pred_fields=(
                    np.array(data["pred_fields"], copy=True)
                    if "pred_fields" in data
                    else None
                ),
                wall_distance=(
                    np.array(data["wall_distance"], copy=True)
                    if "wall_distance" in data
                    else None
                ),
                source=str(data["source"].item()) if "source" in data else "optimization",
                cfd_status=str(data["cfd_status"].item()) if "cfd_status" in data else "none",
                model_version=model_version,
                metadata=metadata,
            )

    def list_ids(
        self,
        *,
        limit: Optional[int] = None,
        cfd_status: Optional[str] = None,
    ) -> list[str]:
        """List sample ids ordered by most recent update."""
        query = "SELECT sample_id FROM samples"
        args: list[Any] = []
        if cfd_status is not None:
            query += " WHERE cfd_status = ?"
            args.append(str(cfd_status))
        query += " ORDER BY updated_at DESC"
        if limit is not None:
            query += " LIMIT ?"
            args.append(int(limit))
        with self._connect() as conn:
            return [str(row[0]) for row in conn.execute(query, args).fetchall()]

    def query_generation_candidates(
        self,
        *,
        branch_id: str,
        generation: int,
        excluded_statuses: Iterable[str] = ("queued", "running"),
        limit: Optional[int] = None,
    ) -> list[OnlineSampleRecord]:
        """Return one generation ordered by residual priority and recency."""

        excluded = tuple(str(value) for value in excluded_statuses)
        query = (
            f"SELECT {_RECORD_COLUMNS} FROM samples "
            "WHERE branch_id = ? AND generation = ?"
        )
        args: list[Any] = [str(branch_id), int(generation)]
        if excluded:
            query += f" AND COALESCE(cfd_status, 'none') NOT IN ({','.join('?' for _ in excluded)})"
            args.extend(excluded)
        query += " ORDER BY priority_score DESC, updated_at DESC, sample_id ASC"
        if limit is not None:
            query += " LIMIT ?"
            args.append(int(limit))
        return self._query_records(query, args)

    def query_recent_by_optimizer(
        self,
        optimizer_sample_ids: Iterable[str],
        *,
        branch_id: Optional[str] = None,
        limit: int = 256,
    ) -> list[OnlineSampleRecord]:
        """Return recent records for a set of optimizer sample identifiers."""

        identifiers = tuple(str(value) for value in optimizer_sample_ids)
        if not identifiers:
            return []
        query = (
            f"SELECT {_RECORD_COLUMNS} FROM samples "
            f"WHERE optimizer_sample_id IN ({','.join('?' for _ in identifiers)})"
        )
        args: list[Any] = list(identifiers)
        if branch_id is not None:
            query += " AND branch_id = ?"
            args.append(str(branch_id))
        query += " ORDER BY updated_at DESC, created_at DESC, sample_id ASC LIMIT ?"
        args.append(int(limit))
        return self._query_records(query, args)

    def _query_records(self, query: str, args: Iterable[Any]) -> list[OnlineSampleRecord]:
        with self._connect() as conn:
            rows = conn.execute(query, tuple(args)).fetchall()
        return [
            OnlineSampleRecord(
                sample_id=str(row[0]),
                geometry_id=str(row[1]),
                source=str(row[2]),
                cfd_status=str(row[3]),
                model_version=None if row[4] is None else str(row[4]),
                npz_path=str(row[5]),
                priority_score=float(row[6] or 0.0),
                branch_id=None if row[7] is None else str(row[7]),
                generation=None if row[8] is None else int(row[8]),
                optimizer_sample_id=None if row[9] is None else str(row[9]),
                created_at=float(row[10] or 0.0),
                updated_at=float(row[11] or 0.0),
                metadata=json.loads(str(row[12] or "{}")),
            )
            for row in rows
        ]

    def update_status(
        self,
        sample_id: str,
        *,
        cfd_status: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Update a sample status while preserving the NPZ payload."""
        sample = self.load(sample_id)
        sample.cfd_status = str(cfd_status)
        if metadata:
            sample.metadata = {**dict(sample.metadata), **metadata}
        self.register(sample)


class AsyncOnlineSampleWriter:
    """Write online samples off the inference thread and surface writer errors."""

    def __init__(self, buffer: SQLiteOnlineBuffer, *, max_queue_size: int = 4096) -> None:
        if int(max_queue_size) <= 0:
            raise ValueError("max_queue_size must be positive")
        self.buffer = buffer
        self._queue: "queue.Queue[OnlineSample | None]" = queue.Queue(maxsize=int(max_queue_size))
        self._thread = threading.Thread(
            target=self._run,
            name="surrogate-online-writer",
            daemon=True,
        )
        self._error: Optional[Exception] = None
        self._submitted = 0
        self._written = 0
        self._closed = False
        self._thread.start()

    def submit(self, sample: OnlineSample) -> None:
        self._raise_if_failed()
        if self._closed:
            raise RuntimeError("AsyncOnlineSampleWriter is closed")
        self._queue.put(sample)
        self._submitted += 1

    def stats(self) -> dict[str, Any]:
        self._raise_if_failed()
        return {
            "submitted": int(self._submitted),
            "written": int(self._written),
            "pending": int(self._queue.qsize()),
        }

    def close(self) -> None:
        if self._closed:
            self._raise_if_failed()
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join()
        self._raise_if_failed()

    def _run(self) -> None:
        while True:
            sample = self._queue.get()
            if sample is None:
                self._queue.task_done()
                return
            if self._error is not None:
                self._queue.task_done()
                continue
            try:
                self.buffer.register(sample)
                self._written += 1
            except Exception as exc:
                self._error = exc
            self._queue.task_done()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("Online sample writer failed") from self._error


__all__ = [
    "AsyncOnlineSampleWriter",
    "OnlineSampleRecord",
    "SQLiteOnlineBuffer",
    "compute_geometry_id",
    "compute_sample_id",
]
