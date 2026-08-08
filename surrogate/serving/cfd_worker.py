"""External CFD worker boundary for online surrogate serving."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from surrogate.serving.contracts import OnlineSample


@dataclass
class CFDJobConfig:
    """Configuration for external CFD job execution."""

    command: tuple[str, ...] = ()
    timeout_s: Optional[float] = None
    payload_only: bool = True
    result_filename: str = "cfd_result.json"
    payload_filename: str = "cfd_payload.npz"
    env: Mapping[str, str] = field(default_factory=dict)


@dataclass
class CFDJob:
    """One CFD job prepared from an online sample."""

    sample_id: str
    geometry: Any
    flow_conditions: Any
    coords_vertex: Any
    coords: Any
    work_dir: Path
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class CFDResult:
    """External CFD execution result."""

    sample_id: str
    status: str
    payload_path: str
    result_path: Optional[str] = None
    fields: Any = None
    coefficients: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _to_numpy(value: Any, *, dtype: Any = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


class ExternalCFDWorker:
    """Prepare online CFD payloads and optionally launch an external command."""

    def __init__(
        self,
        *,
        work_root: str | Path,
        config: Optional[CFDJobConfig] = None,
        runner: Optional[Callable[[CFDJob, Path, Path], CFDResult | Mapping[str, Any]]] = None,
    ) -> None:
        self.work_root = Path(work_root)
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.config = config or CFDJobConfig()
        self.runner = runner

    def job_from_sample(self, sample: OnlineSample) -> CFDJob:
        work_dir = self.work_root / str(sample.sample_id)
        return CFDJob(
            sample_id=str(sample.sample_id),
            geometry=sample.geometry,
            flow_conditions=sample.flow_conditions,
            coords_vertex=sample.coords_vertex,
            coords=sample.coords,
            work_dir=work_dir,
            metadata=dict(sample.metadata),
        )

    def write_payload(self, job: CFDJob) -> Path:
        job.work_dir.mkdir(parents=True, exist_ok=True)
        path = job.work_dir / self.config.payload_filename
        np.savez(
            path,
            sample_id=np.asarray(str(job.sample_id)),
            geometry=_to_numpy(job.geometry, dtype=np.float32),
            flow_conditions=_to_numpy(job.flow_conditions, dtype=np.float64),
            coords_vertex=_to_numpy(job.coords_vertex, dtype=np.float64),
            coords=_to_numpy(job.coords, dtype=np.float32),
            metadata_json=np.asarray(json.dumps(_to_jsonable(job.metadata), sort_keys=True)),
        )
        return path

    def run_sample(self, sample: OnlineSample) -> CFDResult:
        return self.run_job(self.job_from_sample(sample))

    def run_job(self, job: CFDJob) -> CFDResult:
        payload_path = self.write_payload(job)
        result_path = job.work_dir / self.config.result_filename
        start = time.perf_counter()
        if self.runner is not None:
            result = self.runner(job, payload_path, result_path)
            return result if isinstance(result, CFDResult) else CFDResult(**dict(result))
        if self.config.payload_only or not self.config.command:
            return CFDResult(
                sample_id=job.sample_id,
                status="payload_ready",
                payload_path=str(payload_path),
                result_path=str(result_path),
                metadata={"work_dir": str(job.work_dir)},
            )

        command = self._format_command(
            self.config.command,
            payload_path=payload_path,
            result_path=result_path,
            job=job,
        )
        env = None
        if self.config.env:
            import os

            env = os.environ.copy()
            env.update({str(key): str(value) for key, value in self.config.env.items()})
        completed = subprocess.run(
            command,
            cwd=str(job.work_dir),
            env=env,
            timeout=self.config.timeout_s,
            check=False,
        )
        if result_path.exists():
            return self.load_result(result_path, fallback_payload_path=payload_path)
        return CFDResult(
            sample_id=job.sample_id,
            status="done" if int(completed.returncode) == 0 else "failed",
            payload_path=str(payload_path),
            result_path=str(result_path),
            metadata={
                "returncode": int(completed.returncode),
                "elapsed_s": float(time.perf_counter() - start),
                "command": command,
            },
        )

    def load_result(
        self,
        result_path: str | Path,
        *,
        fallback_payload_path: str | Path,
    ) -> CFDResult:
        path = Path(result_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("result_path", str(path))
        data.setdefault("payload_path", str(fallback_payload_path))
        return CFDResult(**data)

    @staticmethod
    def _format_command(
        command: Sequence[str],
        *,
        payload_path: Path,
        result_path: Path,
        job: CFDJob,
    ) -> list[str]:
        values = {
            "payload": str(payload_path),
            "result": str(result_path),
            "work_dir": str(job.work_dir),
            "sample_id": str(job.sample_id),
        }
        return [str(part).format(**values) for part in command]


__all__ = [
    "CFDJob",
    "CFDJobConfig",
    "CFDResult",
    "ExternalCFDWorker",
]
