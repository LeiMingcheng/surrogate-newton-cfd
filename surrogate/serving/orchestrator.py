"""Online CFD orchestration for serving-generated samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from surrogate.serving.cfd_worker import CFDResult, ExternalCFDWorker
from surrogate.serving.contracts import OnlineSample
from surrogate.serving.online import SQLiteOnlineBuffer


@dataclass
class OnlineOrchestratorConfig:
    """Controls for one online CFD orchestration pass."""

    candidate_status: str = "queued"
    running_status: str = "running"
    success_status: str = "done"
    failed_status: str = "failed"
    max_cases: int = 1


class OnlineCFDOrchestrator:
    """Move queued online samples through an external CFD worker."""

    def __init__(
        self,
        *,
        buffer: SQLiteOnlineBuffer,
        worker: ExternalCFDWorker,
        config: Optional[OnlineOrchestratorConfig] = None,
    ) -> None:
        self.buffer = buffer
        self.worker = worker
        self.config = config or OnlineOrchestratorConfig()

    def select_sample_ids(self) -> list[str]:
        return self.buffer.list_ids(
            limit=int(self.config.max_cases),
            cfd_status=self.config.candidate_status,
        )

    def run_once(self) -> list[CFDResult]:
        results: list[CFDResult] = []
        for sample_id in self.select_sample_ids():
            sample = self.buffer.load(sample_id)
            self._register_status(sample, self.config.running_status)
            try:
                result = self.worker.run_sample(sample)
                self._apply_result(sample, result)
            except Exception as exc:
                result = CFDResult(
                    sample_id=sample.sample_id,
                    status=self.config.failed_status,
                    payload_path="",
                    metadata={"error": str(exc), "type": type(exc).__name__},
                )
                self._register_status(sample, self.config.failed_status, error=str(exc))
            results.append(result)
        return results

    def _register_status(self, sample: OnlineSample, status: str, **metadata: str) -> None:
        sample.cfd_status = str(status)
        sample.metadata = {**dict(sample.metadata), **metadata}
        self.buffer.register(sample)

    def _apply_result(self, sample: OnlineSample, result: CFDResult) -> None:
        if result.fields is not None:
            sample.fields = np.asarray(result.fields)
        if result.status in {"done", "completed", "success"}:
            sample.cfd_status = self.config.success_status
        elif result.status == "payload_ready":
            sample.cfd_status = "payload_ready"
        else:
            sample.cfd_status = (
                self.config.failed_status
                if result.status == "failed"
                else str(result.status)
            )
        sample.metadata = {
            **dict(sample.metadata),
            "cfd_result": {
                "payload_path": result.payload_path,
                "result_path": result.result_path,
                "coefficients": dict(result.coefficients),
                "metadata": dict(result.metadata),
            },
        }
        self.buffer.register(sample)


__all__ = [
    "OnlineCFDOrchestrator",
    "OnlineOrchestratorConfig",
]
