"""Socket serving adapter built on transport-neutral surrogate contracts."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import pickle
import socketserver
import struct
import time
from typing import Any, Mapping, Optional

import numpy as np
import torch

from surrogate.serving.aoa import AoASolverConfig, SurrogateAoASolver
from surrogate.serving.batching import BatchingConfig, DynamicBatcher
from surrogate.serving.contracts import AoARequest, AoAResult, OnlineSample, PredictionRequest
from surrogate.serving.geometry import GeometryPreparationConfig, GeometryPreparer, PreparedGeometry
from surrogate.serving.online import AsyncOnlineSampleWriter, compute_sample_id
from surrogate.utils.timing_profile import emit_profile_event


@dataclass
class SocketServingConfig:
    """Configuration for the pickle-over-TCP serving adapter."""

    host: str = "127.0.0.1"
    port: int = 65432
    max_batch_size: int = 8
    batch_timeout_s: float = 0.01
    mesh_mode: str = "pyhyp"
    mesh_cache_size: int = 4096
    device: str = "cuda"
    request_timeout_s: Optional[float] = None

    def validate(self) -> None:
        if not 0 <= int(self.port) <= 65535:
            raise ValueError("port must be between 0 and 65535")
        if int(self.max_batch_size) <= 0:
            raise ValueError("max_batch_size must be positive")
        if float(self.batch_timeout_s) < 0:
            raise ValueError("batch_timeout_s must be non-negative")
        if self.mesh_mode != "pyhyp":
            raise ValueError("mesh_mode must be 'pyhyp'")
        if int(self.mesh_cache_size) <= 0:
            raise ValueError("mesh_cache_size must be positive")
        if self.request_timeout_s is not None and float(self.request_timeout_s) <= 0:
            raise ValueError("request_timeout_s must be positive when set")


def recvall(sock: Any, n_bytes: int) -> bytes | None:
    """Read exactly ``n_bytes`` from a socket-like object."""
    chunks = bytearray()
    while len(chunks) < int(n_bytes):
        packet = sock.recv(int(n_bytes) - len(chunks))
        if not packet:
            return None
        chunks.extend(packet)
    return bytes(chunks)


def recv_pickle(sock: Any) -> Any:
    """Receive one length-prefixed pickle payload."""
    header = recvall(sock, 4)
    if header is None:
        raise EOFError("Socket closed while reading payload header")
    size = struct.unpack("!I", header)[0]
    payload = recvall(sock, size)
    if payload is None:
        raise EOFError("Socket closed while reading payload body")
    return pickle.loads(payload)


def send_pickle(sock: Any, payload: Any) -> None:
    """Send one length-prefixed pickle payload."""
    body = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!I", len(body)) + body)


def _result_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: _result_to_dict(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _result_to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_result_to_dict(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.generic):
        return value.item()
    return value


class SurrogateServingApp:
    """Transport-independent serving application behind the socket adapter."""

    def __init__(
        self,
        predictor: Any,
        *,
        geometry_preparer: Optional[GeometryPreparer] = None,
        aoa_solver: Optional[SurrogateAoASolver] = None,
        config: Optional[SocketServingConfig] = None,
        runtime_metadata: Optional[Mapping[str, Any]] = None,
        online_writer: Optional[AsyncOnlineSampleWriter] = None,
        residual_calculator: Any = None,
    ) -> None:
        self.predictor = predictor
        self.config = config or SocketServingConfig()
        self.config.validate()
        self.runtime_metadata = dict(runtime_metadata or {})
        self.online_writer = online_writer
        self.residual_calculator = residual_calculator
        self.geometry_preparer = geometry_preparer or GeometryPreparer(
            config=GeometryPreparationConfig(
                mesh_mode=self.config.mesh_mode,
                cache_size=self.config.mesh_cache_size,
            )
        )
        self.aoa_solver = aoa_solver or SurrogateAoASolver(
            predictor,
            config=AoASolverConfig(device=self.config.device),
        )

    def process_batch(self, payloads: list[Any]) -> list[Any]:
        """Tensor-batch compatible AoA requests and preserve response order."""

        started = time.perf_counter()
        responses: list[Any] = [None] * len(payloads)
        groups: dict[
            tuple[Any, ...],
            list[tuple[int, AoARequest, bool, bool, bool]],
        ] = {}
        fallback: list[tuple[int, Any]] = []
        for index, payload in enumerate(payloads):
            if not isinstance(payload, Mapping) or payload.get("command"):
                fallback.append((index, payload))
                continue
            mode = self._aoa_mode(payload)
            if mode is None:
                fallback.append((index, payload))
                continue
            request = self._prepare_aoa_request(payload)
            record_sample = bool(payload.get("record_sample", True))
            return_fields = bool(payload.get("return_fields", True))
            return_geometry_context = bool(
                payload.get("return_geometry_context", False)
            )
            initial_shape = None
            if request.initial_field is not None:
                shape = getattr(request.initial_field, "shape", None)
                if shape is None:
                    shape = np.asarray(request.initial_field).shape
                initial_shape = tuple(int(value) for value in shape)
            key = (
                mode,
                tuple(int(value) for value in request.geometry.shape),
                tuple(int(value) for value in request.coords.shape),
                tuple(int(value) for value in request.coords_vertex.shape),
                initial_shape,
                request.metadata.get("n_inference_steps"),
            )
            groups.setdefault(key, []).append(
                (
                    index,
                    request,
                    record_sample,
                    return_fields,
                    return_geometry_context,
                )
            )

        for key, entries in groups.items():
            mode = str(key[0])
            requests = [request for _, request, _, _, _ in entries]
            if len(requests) == 1:
                results = [self._solve_aoa_request(mode, requests[0])]
            elif mode == "target_cl":
                results = self.aoa_solver.solve_target_cl_batch(requests)
            else:
                results = self.aoa_solver.evaluate_fixed_batch(requests)
            for (
                index,
                request,
                record_sample,
                return_fields,
                return_geometry_context,
            ), result in zip(entries, results):
                responses[index] = self._finalize_aoa_result(
                    request,
                    result,
                    record_sample=record_sample,
                    return_fields=return_fields,
                    return_geometry_context=return_geometry_context,
                )

        for index, payload in fallback:
            responses[index] = self.process_payload(payload)
        emit_profile_event(
            "serving_batch",
            requests=int(len(payloads)),
            grouped_requests=int(sum(len(entries) for entries in groups.values())),
            fallback_requests=int(len(fallback)),
            group_sizes=[int(len(entries)) for entries in groups.values()],
            wall_time_s=float(time.perf_counter() - started),
        )
        return responses

    def process_payload(self, payload: Any) -> Any:
        if not isinstance(payload, Mapping):
            raise TypeError(f"Serving payload must be a mapping, got {type(payload)}")

        command = payload.get("command")
        if command:
            return self._handle_command(str(command), payload)

        mode = self._aoa_mode(payload)
        if mode is not None:
            request = self._prepare_aoa_request(payload)
            return self._finalize_aoa_result(
                request,
                self._solve_aoa_request(mode, request),
                record_sample=bool(payload.get("record_sample", True)),
                return_fields=bool(payload.get("return_fields", True)),
                return_geometry_context=bool(
                    payload.get("return_geometry_context", False)
                ),
            )

        flow_conditions = payload.get("flow_conditions")
        if flow_conditions is None:
            raise ValueError("Raw prediction requires flow_conditions")
        prepared, metadata = self._prepare_geometry(payload)
        response = self.predictor.predict(
            PredictionRequest(
                geometry=prepared.geometry,
                flow_conditions=flow_conditions,
                coords=prepared.coords,
                initial_field=payload.get("initial_field"),
                metadata=metadata,
            )
        )
        return _result_to_dict(response)

    @staticmethod
    def _aoa_mode(payload: Mapping[str, Any]) -> Optional[str]:
        if payload.get("target_cl") is not None:
            return "target_cl"
        if payload.get("aoa") is not None and payload.get("mach") is not None:
            return "fixed_aoa"
        return None

    def _prepare_aoa_request(self, payload: Mapping[str, Any]) -> AoARequest:
        prepared, metadata = self._prepare_geometry(payload)
        return AoARequest(
            geometry=prepared.geometry,
            coords=prepared.coords,
            coords_vertex=prepared.coords_vertex,
            mach=payload["mach"],
            reynolds=payload.get("reynolds", 20.0e6),
            target_cl=payload.get("target_cl"),
            aoa=payload.get("aoa"),
            initial_field=payload.get("initial_field"),
            metadata=metadata,
        )

    def _prepare_geometry(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[PreparedGeometry, dict[str, Any]]:
        prepared = self.geometry_preparer.prepare(payload)
        metadata = {
            "geometry_id": prepared.geometry_id,
            **dict(prepared.metadata),
            **dict(payload.get("metadata", {}) or {}),
        }
        if payload.get("n_inference_steps") is not None:
            n_inference_steps = int(payload["n_inference_steps"])
            if n_inference_steps < 1 or n_inference_steps > 20:
                raise ValueError("n_inference_steps must be between 1 and 20")
            metadata["n_inference_steps"] = n_inference_steps
        return prepared, metadata

    def _solve_aoa_request(self, mode: str, request: AoARequest) -> Any:
        if mode == "target_cl":
            return self.aoa_solver.solve_target_cl(request)
        return self.aoa_solver.evaluate_fixed(request)

    def _finalize_aoa_result(
        self,
        request: AoARequest,
        result: AoAResult,
        *,
        record_sample: bool = True,
        return_fields: bool = True,
        return_geometry_context: bool = False,
    ) -> dict[str, Any]:
        residual_score = None
        residual_components = None
        if self.residual_calculator is not None:
            residual_score, residual_components = self._compute_residual(request, result)
        needs_fields = bool(return_fields or (self.online_writer is not None and record_sample))
        if needs_fields:
            response = _result_to_dict(result)
        else:
            response = {
                field.name: _result_to_dict(getattr(result, field.name))
                for field in fields(result)
                if field.name != "fields"
            }
        if self.online_writer is not None and record_sample:
            self._record_online_samples(
                request,
                result_payload=response,
                residual_score=residual_score,
                residual_components=residual_components,
            )
        if residual_score is not None:
            response["residual_score"] = _result_to_dict(residual_score)
            response["residual_components"] = _result_to_dict(residual_components)
        if not return_fields:
            response.pop("fields", None)
        if return_geometry_context:
            response["geometry_context"] = _result_to_dict(
                {
                    "geometry": request.geometry,
                    "coords": request.coords,
                    "coords_vertex": request.coords_vertex,
                }
            )
        return response

    def _compute_residual(self, request: AoARequest, result: AoAResult) -> tuple[Any, Any]:
        device = torch.device(self.config.device)
        fields = torch.as_tensor(result.fields, device=device)
        count = int(fields.shape[0])
        coords_vertex = torch.as_tensor(request.coords_vertex, dtype=torch.float64, device=device)
        if coords_vertex.ndim == 3:
            coords_vertex = coords_vertex.unsqueeze(0).expand(count, -1, -1, -1).contiguous()
        coords = torch.as_tensor(request.coords, dtype=torch.float64, device=device)
        if coords.ndim == 3:
            coords = coords.unsqueeze(0).expand(count, -1, -1, -1).contiguous()
        mach = torch.as_tensor(request.mach, dtype=torch.float32, device=device).reshape(-1)
        aoa = torch.as_tensor(result.aoa, dtype=torch.float32, device=device).reshape(-1)
        reynolds = torch.as_tensor(request.reynolds, dtype=torch.float32, device=device).reshape(-1)
        if int(reynolds.numel()) == 1:
            reynolds = reynolds.repeat(count)
        flow_conditions = torch.stack([mach, aoa, reynolds], dim=1)
        return self.residual_calculator.compute_residual(
            fields=fields,
            coords={"vertex": coords_vertex, "center": coords[:, :2]},
            flow_conditions=flow_conditions,
            return_spatial=False,
            return_components=True,
        )

    def _record_online_samples(
        self,
        request: AoARequest,
        *,
        result_payload: Mapping[str, Any],
        residual_score: Any,
        residual_components: Any,
    ) -> None:
        geometry = np.asarray(_result_to_dict(request.geometry), dtype=np.float32)
        coords = np.asarray(_result_to_dict(request.coords), dtype=np.float32)
        coords_vertex = np.asarray(_result_to_dict(request.coords_vertex), dtype=np.float64)
        fields = np.asarray(result_payload["fields"])
        mach = np.asarray(_result_to_dict(request.mach), dtype=np.float32).reshape(-1)
        aoa = np.asarray(result_payload["aoa"], dtype=np.float32).reshape(-1)
        reynolds = np.asarray(_result_to_dict(request.reynolds), dtype=np.float32).reshape(-1)
        if reynolds.size == 1:
            reynolds = np.repeat(reynolds, mach.size)
        target_cl = None
        if request.target_cl is not None:
            target_cl = np.asarray(_result_to_dict(request.target_cl), dtype=np.float32).reshape(-1)
        cl = np.asarray(result_payload["cl"], dtype=np.float32).reshape(-1)
        cd = np.asarray(result_payload["cd"], dtype=np.float32).reshape(-1)
        cm = np.asarray(result_payload["cm"], dtype=np.float32).reshape(-1)
        residual_scores = None
        if residual_score is not None:
            residual_scores = np.asarray(
                _result_to_dict(residual_score),
                dtype=np.float64,
            ).reshape(-1)
        residual_values = _result_to_dict(residual_components)
        base_metadata = dict(request.metadata)
        model_version = self.runtime_metadata.get("model_version")
        for index in range(int(mach.size)):
            geometry_item = self._condition_item(
                geometry,
                index=index,
                count=int(mach.size),
                unbatched_ndim=1,
            )
            coords_item = self._condition_item(
                coords,
                index=index,
                count=int(mach.size),
                unbatched_ndim=3,
            )
            coords_vertex_item = self._condition_item(
                coords_vertex,
                index=index,
                count=int(mach.size),
                unbatched_ndim=3,
            )
            flow_conditions = np.asarray(
                [mach[index], aoa[index], reynolds[index]],
                dtype=np.float32,
            )
            target_value = None if target_cl is None else float(target_cl[index])
            metadata = {
                **base_metadata,
                "target_cl": target_value,
                "aoa": float(aoa[index]),
                "cl": float(cl[index]),
                "cd": float(cd[index]),
                "cm": float(cm[index]),
            }
            if residual_scores is not None:
                score_index = index if residual_scores.size > 1 else 0
                score = float(residual_scores[score_index])
                metadata["residual_score"] = score
                metadata["priority_score"] = float(-score)
                metadata["residual_components"] = self._batch_item(
                    residual_values,
                    index=index,
                    count=int(mach.size),
                )
            sample = OnlineSample(
                sample_id=compute_sample_id(
                    geometry_item,
                    flow_conditions,
                    target_cl=target_value,
                ),
                geometry=geometry_item,
                flow_conditions=flow_conditions,
                coords_vertex=coords_vertex_item,
                coords=coords_item,
                pred_fields=fields[index],
                source=str(base_metadata.get("source", "optimization")),
                cfd_status=str(base_metadata.get("cfd_status", "none")),
                model_version=None if model_version is None else str(model_version),
                metadata=metadata,
            )
            self.online_writer.submit(sample)

    @staticmethod
    def _condition_item(
        value: np.ndarray,
        *,
        index: int,
        count: int,
        unbatched_ndim: int,
    ) -> np.ndarray:
        if value.ndim == unbatched_ndim + 1 and int(value.shape[0]) == count:
            return value[index]
        return value

    @classmethod
    def _batch_item(cls, value: Any, *, index: int, count: int) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): cls._batch_item(item, index=index, count=count)
                for key, item in value.items()
            }
        if isinstance(value, np.ndarray):
            if value.ndim > 0 and int(value.shape[0]) == count:
                return _result_to_dict(value[index])
            return _result_to_dict(value)
        if isinstance(value, list) and len(value) == count:
            return cls._batch_item(value[index], index=index, count=count)
        return _result_to_dict(value)

    def close(self) -> None:
        if self.online_writer is not None:
            self.online_writer.close()

    def _handle_command(self, command: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if command == "ping":
            return {
                "ok": True,
                "message": "pong",
                "protocol_version": 1,
                **self.runtime_metadata,
                "serving": self.serving_metadata(),
            }
        if command == "clear_cache":
            self.geometry_preparer.clear_cache()
            return {"ok": True, "cache_size": 0}
        if command == "stats":
            response = {
                "ok": True,
                "geometry_cache_size": self.geometry_preparer.cache_size,
                "serving": self.serving_metadata(),
            }
            if self.online_writer is not None:
                response["online_writer"] = self.online_writer.stats()
            return response
        raise ValueError(f"Unknown serving command: {command}")

    def serving_metadata(self) -> dict[str, Any]:
        """Return the socket and batching capabilities advertised to clients."""

        return {
            "host": self.config.host,
            "port": int(self.config.port),
            "device": self.config.device,
            "mesh_mode": self.config.mesh_mode,
            "mesh_cache_size": int(self.config.mesh_cache_size),
            "max_batch_size": int(self.config.max_batch_size),
            "batch_timeout_s": float(self.config.batch_timeout_s),
            "batch_strategy": "compatible_aoa_tensor_batching",
            "cross_request_tensor_batching": True,
            "cross_request_tensor_batching_modes": ["target_cl", "fixed_aoa"],
            "intra_request_tensor_batching": True,
            "online_buffer_enabled": self.online_writer is not None,
            "pde_residual_enabled": self.residual_calculator is not None,
        }


class SurrogateSocketServer:
    """Length-prefixed pickle TCP server for surrogate serving."""

    def __init__(self, app: SurrogateServingApp) -> None:
        self.app = app
        self.config = app.config
        self.batcher = DynamicBatcher(
            self.app.process_batch,
            config=BatchingConfig(
                max_batch_size=self.config.max_batch_size,
                timeout_s=self.config.batch_timeout_s,
            ),
            name="surrogate-serving-batcher",
        )
        self._server: Optional[socketserver.ThreadingTCPServer] = None

    def serve_forever(self) -> None:
        parent = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                try:
                    payload = recv_pickle(self.request)
                    future = parent.batcher.submit(payload)
                    response = future.result(timeout=parent.config.request_timeout_s)
                    send_pickle(self.request, response)
                except Exception as exc:
                    send_pickle(
                        self.request,
                        {
                            "ok": False,
                            "error": str(exc),
                            "type": type(exc).__name__,
                        },
                    )

        class TCPServer(socketserver.ThreadingTCPServer):
            allow_reuse_address = True

        self.batcher.start()
        self._server = TCPServer((self.config.host, int(self.config.port)), Handler)
        self._server.serve_forever()

    def shutdown(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        self.batcher.close(wait=True)
        self.app.close()


__all__ = [
    "SocketServingConfig",
    "SurrogateServingApp",
    "SurrogateSocketServer",
    "recvall",
    "recv_pickle",
    "send_pickle",
]
