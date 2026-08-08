"""Dynamic request batching primitives for serving adapters."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
import queue
import threading
import time
from typing import Any, Callable, Iterable, Optional


BatchHandler = Callable[[list[Any]], list[Any]]


@dataclass
class BatchingConfig:
    """Controls for dynamic request batching."""

    max_batch_size: int = 8
    timeout_s: float = 0.01

    def validate(self) -> None:
        self.max_batch_size = int(self.max_batch_size)
        self.timeout_s = float(self.timeout_s)
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")


@dataclass
class _QueuedItem:
    request: Any
    future: Future


class DynamicBatcher:
    """Threaded dynamic batcher around a transport-neutral batch handler."""

    def __init__(
        self,
        handler: BatchHandler,
        *,
        config: Optional[BatchingConfig] = None,
        name: str = "surrogate-serving-batcher",
    ) -> None:
        self.handler = handler
        self.config = config or BatchingConfig()
        self.config.validate()
        self.name = str(name)
        self._queue: "queue.Queue[_QueuedItem | None]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._closed = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._thread.start()

    def close(self, *, wait: bool = True) -> None:
        self._closed = True
        self._queue.put(None)
        if wait and self._thread is not None:
            self._thread.join()

    def submit(self, request: Any) -> Future:
        if self._closed:
            raise RuntimeError("DynamicBatcher is closed")
        self.start()
        future: Future = Future()
        self._queue.put(_QueuedItem(request=request, future=future))
        return future

    def map(self, requests: Iterable[Any], *, timeout: Optional[float] = None) -> list[Any]:
        futures = [self.submit(request) for request in requests]
        return [future.result(timeout=timeout) for future in futures]

    def _collect_batch(self, first: _QueuedItem) -> list[_QueuedItem]:
        batch = [first]
        deadline = time.perf_counter() + float(self.config.timeout_s)
        while len(batch) < int(self.config.max_batch_size):
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                item = self._queue.get(timeout=remaining)
            except queue.Empty:
                break
            if item is None:
                self._queue.put(None)
                break
            batch.append(item)
        return batch

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            batch = self._collect_batch(item)
            try:
                responses = self.handler([queued.request for queued in batch])
                if len(responses) != len(batch):
                    raise RuntimeError(
                        f"Batch handler returned {len(responses)} responses "
                        f"for {len(batch)} requests"
                    )
            except Exception as exc:
                for queued in batch:
                    queued.future.set_exception(exc)
                continue
            for queued, response in zip(batch, responses):
                queued.future.set_result(response)


__all__ = [
    "BatchingConfig",
    "DynamicBatcher",
]
