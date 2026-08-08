"""Canonical client for the native surrogate socket service."""

from __future__ import annotations

from dataclasses import dataclass
import socket
import time
from typing import Any, Mapping

from surrogate.serving.server import recv_pickle, send_pickle
from surrogate.utils.timing_profile import emit_profile_event


@dataclass(frozen=True)
class SurrogateClientConfig:
    host: str = "127.0.0.1"
    port: int = 65432
    timeout_s: float = 120.0
    model_key: str = ""

    def __post_init__(self) -> None:
        if not 0 < int(self.port) <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if float(self.timeout_s) <= 0.0:
            raise ValueError("timeout_s must be positive")


class SurrogateClient:
    """Send one transport-neutral mapping per TCP connection."""

    def __init__(self, config: SurrogateClientConfig | None = None) -> None:
        self.config = config or SurrogateClientConfig()

    def request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        with socket.create_connection(
            (self.config.host, int(self.config.port)),
            timeout=float(self.config.timeout_s),
        ) as sock:
            sock.settimeout(float(self.config.timeout_s))
            send_pickle(sock, dict(payload))
            response = recv_pickle(sock)
        emit_profile_event(
            "serving_client_request",
            command=payload.get("command"),
            wall_time_s=float(time.perf_counter() - started),
        )
        if not isinstance(response, Mapping):
            raise TypeError(f"Serving response must be a mapping, got {type(response)!r}")
        result = dict(response)
        if result.get("ok") is False or result.get("error") is not None:
            raise RuntimeError(
                f"Surrogate service error ({result.get('type', 'RuntimeError')}): "
                f"{result.get('error', 'unknown error')}"
            )
        return result

    def ping(self) -> dict[str, Any]:
        result = self.request({"command": "ping"})
        if int(result.get("protocol_version", 0)) != 1:
            raise RuntimeError(
                f"Unsupported surrogate protocol: {result.get('protocol_version')!r}"
            )
        expected = str(self.config.model_key).strip()
        actual = str(result.get("model_key") or "").strip()
        if expected and actual != expected:
            raise RuntimeError(
                f"Surrogate model mismatch: expected {expected!r}, server reports {actual!r}"
            )
        return result


__all__ = ["SurrogateClient", "SurrogateClientConfig"]
