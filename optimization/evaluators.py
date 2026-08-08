"""The three evaluator implementations behind the unified optimizer."""

from __future__ import annotations

from concurrent.futures import Executor
import json
from pathlib import Path
import threading
import time
from typing import Any, Protocol

import numpy as np

from NK_resume import (
    ResidentWarmPoolController,
    build_direct_case,
    build_fsb_case,
    create_pipeline,
    finalonly_plan,
)
from optimization.config import OptimizationConfig
from optimization.contracts import CandidateEvaluation, OperatingPointResult
from optimization.cfd_runtime import (
    execute_adflow_jobs,
    mpi_env,
    prepare_authority_mesh,
    run_adflow_pool,
    split_jobs_for_pools,
)
from optimization.geometry_parameterization import PreparedCandidateGeometry
from surrogate.serving.client import SurrogateClient, SurrogateClientConfig
from surrogate.utils.mesh_generation import (
    generate_mesh_from_cst27,
    silence_native_output,
)
from surrogate.utils.timing_profile import emit_profile_event


class CandidateEvaluator(Protocol):
    name: str

    def evaluate(
        self,
        geometry: PreparedCandidateGeometry,
        workdir: Path,
    ) -> CandidateEvaluation: ...


def _client(config: OptimizationConfig) -> SurrogateClient:
    return SurrogateClient(
        SurrogateClientConfig(
            host=config.serving.host,
            port=config.serving.port,
            timeout_s=config.serving.timeout_s,
            model_key=config.serving.model_key,
        )
    )


def _as_vector(payload: dict[str, Any], key: str, count: int) -> np.ndarray:
    values = np.asarray(payload[key], dtype=np.float64).reshape(-1)
    if values.size != count:
        raise ValueError(f"Serving field {key!r} has {values.size} values; expected {count}")
    return values


def _surrogate_points(
    response: dict[str, Any],
    config: OptimizationConfig,
    *,
    wall_time_s: float,
    provenance: dict[str, Any],
) -> tuple[OperatingPointResult, ...]:
    count = len(config.task.mach)
    aoa = _as_vector(response, "aoa", count)
    cl = _as_vector(response, "cl", count)
    cd = _as_vector(response, "cd", count)
    cm = _as_vector(response, "cm", count)
    mask_value = response.get("converged_mask")
    if mask_value is None:
        mask = np.full(count, bool(response.get("converged", True)), dtype=bool)
    else:
        mask = np.asarray(mask_value, dtype=bool).reshape(-1)
    residual = response.get("residual_score")
    residual_values = None if residual is None else np.asarray(residual, dtype=np.float64).reshape(-1)
    return tuple(
        OperatingPointResult(
            mach=float(mach),
            target_cl=float(config.task.target_cl),
            reynolds=config.task.reynolds_for(float(mach)),
            aoa=float(aoa[index]),
            cl=float(cl[index]),
            cd=float(cd[index]),
            cm=float(cm[index]),
            converged=bool(mask[index]),
            n_iter=int(response.get("n_iter", 0)),
            residual=(
                None
                if residual_values is None
                else float(residual_values[index if residual_values.size > 1 else 0])
            ),
            wall_time_s=float(wall_time_s / count),
            provenance=provenance,
        )
        for index, mach in enumerate(config.task.mach)
    )


def prepare_surrogate_mesh(
    geometry27: np.ndarray,
    t_max: float,
    tag: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one transient inference mesh in a reusable worker process."""

    with silence_native_output():
        return generate_mesh_from_cst27(
            np.asarray(geometry27, dtype=np.float32),
            t_max=float(t_max),
            tag=str(tag),
            persist_cgns_path=None,
        )

class SurrogateEvaluator:
    name = "surrogate"

    def __init__(
        self,
        config: OptimizationConfig,
        *,
        mesh_executor: Executor | None = None,
    ) -> None:
        self.config = config
        self.mesh_executor = mesh_executor
        self.client: SurrogateClient | None = None
        self.server: dict[str, Any] | None = None
        self._client_lock = threading.Lock()

    def _connected_client(self) -> tuple[SurrogateClient, dict[str, Any]]:
        with self._client_lock:
            if self.client is None:
                self.client = _client(self.config)
                self.server = self.client.ping()
            return self.client, dict(self.server or {})

    def evaluate(
        self,
        geometry: PreparedCandidateGeometry,
        workdir: Path,
    ) -> CandidateEvaluation:
        client, server = self._connected_client()
        mesh_wall_time_s = 0.0
        prepared_mesh: dict[str, np.ndarray] = {}
        if self.mesh_executor is not None:
            mesh_start = time.perf_counter()
            coords_vertex, coords = self.mesh_executor.submit(
                prepare_surrogate_mesh,
                geometry.geometry27,
                geometry.t_max,
                workdir.name,
            ).result()
            mesh_wall_time_s = time.perf_counter() - mesh_start
            prepared_mesh = {
                "coords": coords,
                "coords_vertex": coords_vertex,
            }
        mach = np.asarray(self.config.task.mach, dtype=np.float32)
        start = time.perf_counter()
        response = client.request(
            {
                "geometry": geometry.geometry27,
                "t_max": geometry.t_max,
                **prepared_mesh,
                "mach": mach,
                "target_cl": np.full(mach.shape, self.config.task.target_cl, dtype=np.float32),
                "reynolds": np.asarray(
                    [self.config.task.reynolds_for(value) for value in mach],
                    dtype=np.float32,
                ),
                "aoa": np.full(mach.shape, np.mean(self.config.task.aoa_bounds), dtype=np.float32),
                "record_sample": False,
                "return_fields": False,
                "persist_cgns": False,
                "metadata": {"source": "optimization", "sample": workdir.name},
            }
        )
        elapsed = time.perf_counter() - start
        provenance = {
            "protocol_version": server["protocol_version"],
            "model_key": server.get("model_key"),
            "model_version": server.get("model_version"),
            "mesh_prepared_by_optimizer": self.mesh_executor is not None,
            "mesh_wall_time_s": float(mesh_wall_time_s),
            "inference_wall_time_s": float(elapsed),
        }
        total_elapsed = mesh_wall_time_s + elapsed
        emit_profile_event(
            "optimization_candidate_surrogate",
            sample=workdir.name,
            mesh_wall_time_s=float(mesh_wall_time_s),
            inference_wall_time_s=float(elapsed),
            wall_time_s=float(total_elapsed),
        )
        return CandidateEvaluation(
            evaluator=self.name,
            points=_surrogate_points(
                response,
                self.config,
                wall_time_s=total_elapsed,
                provenance=provenance,
            ),
            wall_time_s=float(total_elapsed),
            provenance=provenance,
        )


def _prepare_authority_mesh(
    geometry: PreparedCandidateGeometry,
    workdir: Path,
) -> tuple[Any, Path]:
    return prepare_authority_mesh(
        geometry27=geometry.geometry27,
        t_max=geometry.t_max,
        tag=workdir.name,
        mesh_path=workdir / "mesh" / "airfoil.cgns",
    )


def _mpi_env() -> dict[str, str]:
    return mpi_env()


def _split_jobs_for_pools(
    jobs: list[dict[str, Any]],
    requested_pool_count: int,
) -> list[list[dict[str, Any]]]:
    return split_jobs_for_pools(jobs, requested_pool_count)


class CFDEvaluator:
    name = "cfd"

    def __init__(self, config: OptimizationConfig) -> None:
        self.config = config

    def _run_pool(self, manifest_path: Path, pool_dir: Path) -> None:
        run_adflow_pool(
            manifest_path,
            pool_dir,
            ranks_per_case=self.config.cfd.ranks_per_case,
            mpi_launcher=self.config.cfd.mpi_launcher,
            python=self.config.cfd.python,
            timeout_s=self.config.cfd.timeout_s,
        )

    def evaluate(
        self,
        geometry: PreparedCandidateGeometry,
        workdir: Path,
    ) -> CandidateEvaluation:
        _, mesh_path = _prepare_authority_mesh(geometry, workdir)
        root = workdir / "cfd"
        jobs: list[dict[str, Any]] = []
        for index, mach in enumerate(self.config.task.mach):
            result_path = (root / f"mach_{index:02d}" / "result.json").resolve()
            jobs.append(
                {
                    "mesh_path": str(mesh_path),
                    "output_dir": str(result_path.parent),
                    "result_path": str(result_path),
                    "mach": float(mach),
                    "target_cl": float(self.config.task.target_cl),
                    "reynolds": self.config.task.reynolds_for(float(mach)),
                    "aoa_init": float(np.mean(self.config.task.aoa_bounds)),
                    "max_iterations": self.config.cfd.max_iterations,
                    "max_aoa_iterations": self.config.cfd.max_aoa_iterations,
                    "cl_tolerance": self.config.cfd.cl_tolerance,
                    "l2_convergence": self.config.cfd.l2_convergence,
                    "options_version": self.config.cfd.options_version,
                    "reference_state_mode": self.config.cfd.reference_state_mode,
                }
            )
        start = time.perf_counter()
        pool_count = execute_adflow_jobs(
            jobs,
            root=root,
            pool_count=self.config.cfd.pool_count,
            pool_runner=self._run_pool,
        )
        elapsed = time.perf_counter() - start
        points: list[OperatingPointResult] = []
        for job in jobs:
            payload = json.loads(Path(job["result_path"]).read_text(encoding="utf-8"))
            points.append(OperatingPointResult(**payload))
        return CandidateEvaluation(
            evaluator=self.name,
            points=tuple(points),
            wall_time_s=float(elapsed),
            provenance={
                "solver": "ADFLOW.solveCL",
                "pool_count": pool_count,
                "ranks_per_case": self.config.cfd.ranks_per_case,
                "mesh_path": str(mesh_path),
            },
        )


def _predictor_kind(model_key: str) -> str:
    if model_key.startswith("direct_"):
        return "direct"
    if model_key.startswith("fsb_"):
        return "fsb"
    raise ValueError(f"NK optimization requires a canonical serving model_key, got {model_key!r}")


def _fixed_request(
    prepared: Any,
    config: OptimizationConfig,
    indices: list[int],
    aoa: np.ndarray,
) -> dict[str, Any]:
    mach = np.asarray([config.task.mach[index] for index in indices], dtype=np.float32)
    return {
        "geometry": prepared.geometry,
        "coords": prepared.coords,
        "coords_vertex": prepared.coords_vertex,
        "mach": mach,
        "aoa": np.asarray([aoa[index] for index in indices], dtype=np.float32),
        "reynolds": np.asarray(
            [config.task.reynolds_for(value) for value in mach], dtype=np.float32
        ),
        "metadata": {"source": "optimization_nk"},
    }


def _find_cl_bracket(
    history: list[dict[str, float]],
    *,
    aoa_epsilon: float,
) -> tuple[dict[str, float], dict[str, float]] | None:
    ordered = sorted(history, key=lambda item: float(item["aoa"]))
    candidates: list[
        tuple[float, dict[str, float], dict[str, float]]
    ] = []
    for left, right in zip(ordered, ordered[1:]):
        aoa_delta = abs(float(right["aoa"]) - float(left["aoa"]))
        if aoa_delta < float(aoa_epsilon):
            continue
        left_error = float(left["error"])
        right_error = float(right["error"])
        if left_error == 0.0 or right_error == 0.0 or left_error * right_error < 0.0:
            candidates.append((aoa_delta, left, right))
    if not candidates:
        return None
    _, left, right = min(candidates, key=lambda item: item[0])
    return left, right


def _propose_fixed_lift_aoa(
    history: list[dict[str, float]],
    config: OptimizationConfig,
) -> tuple[float | None, str]:
    """Return the next bounded AoA or a terminal reason."""

    if not history:
        raise ValueError("Fixed-lift AoA proposal requires at least one NK result")
    current = history[-1]
    current_aoa = float(current["aoa"])
    current_error = float(current["error"])
    aoa_bounds = config.task.aoa_bounds
    lower, upper = aoa_bounds
    nk = config.nk
    if abs(current_error) <= nk.cl_tolerance:
        return None, "accepted"
    if current_error > 0.0 and current_aoa <= lower + nk.aoa_epsilon:
        return None, "aoa_bound_reached"
    if current_error < 0.0 and current_aoa >= upper - nk.aoa_epsilon:
        return None, "aoa_bound_reached"

    bracket = _find_cl_bracket(history, aoa_epsilon=nk.aoa_epsilon)
    proposal: float | None = None
    method = ""
    if bracket is not None:
        left, right = bracket
        left_aoa = float(left["aoa"])
        right_aoa = float(right["aoa"])
        left_error = float(left["error"])
        right_error = float(right["error"])
        denominator = right_error - left_error
        if abs(denominator) > 1.0e-8:
            proposal = left_aoa - left_error * (right_aoa - left_aoa) / denominator
            method = "bracket_regula_falsi"
        else:
            proposal = 0.5 * (left_aoa + right_aoa)
            method = "bracket_midpoint"
        inner_lower = min(left_aoa, right_aoa) + nk.aoa_epsilon
        inner_upper = max(left_aoa, right_aoa) - nk.aoa_epsilon
        if inner_lower < inner_upper:
            proposal = float(np.clip(proposal, inner_lower, inner_upper))
    else:
        previous = next(
            (
                item
                for item in reversed(history[:-1])
                if abs(float(item["aoa"]) - current_aoa) >= nk.aoa_epsilon
            ),
            None,
        )
        if previous is not None:
            previous_aoa = float(previous["aoa"])
            previous_error = float(previous["error"])
            denominator = current_error - previous_error
            if abs(denominator) > 1.0e-6:
                raw = current_aoa - current_error * (
                    current_aoa - previous_aoa
                ) / denominator
                proposal = current_aoa + nk.secant_aoa_damping * (
                    raw - current_aoa
                )
                method = "damped_secant"
        if proposal is None:
            slope = float(nk.initial_cl_alpha)
            if abs(slope) < 1.0e-3:
                slope = 0.1
            raw = current_aoa - current_error / slope
            proposal = current_aoa + nk.initial_aoa_damping * (
                raw - current_aoa
            )
            method = "damped_initial_slope"

    max_step = (
        nk.max_aoa_step
        if len(history) <= 1
        else min(nk.max_aoa_step, nk.late_max_aoa_step)
    )
    delta = float(np.clip(proposal - current_aoa, -max_step, max_step))
    if abs(delta) < nk.aoa_epsilon:
        direction = -1.0 if current_error > 0.0 else 1.0
        delta = direction * min(nk.minimum_aoa_step, max_step)
        method = f"{method}_minimum_step"
    proposal = float(np.clip(current_aoa + delta, *aoa_bounds))
    if abs(proposal - current_aoa) < nk.aoa_epsilon:
        return None, "stagnated"
    if any(
        abs(float(item["aoa"]) - proposal) < nk.repeated_aoa_tolerance
        for item in history
    ):
        bracket = _find_cl_bracket(history, aoa_epsilon=nk.aoa_epsilon)
        if bracket is None:
            return None, "repeated_aoa"
        proposal = 0.5 * (float(bracket[0]["aoa"]) + float(bracket[1]["aoa"]))
        if any(
            abs(float(item["aoa"]) - proposal) < nk.aoa_epsilon
            for item in history
        ):
            return None, "repeated_aoa"
        method = "bracket_midpoint_fallback"
    return proposal, method


class SurrogateNKEvaluator:
    name = "surrogate_nk"

    def __init__(
        self,
        config: OptimizationConfig,
        *,
        resident_pool: ResidentWarmPoolController | None = None,
    ) -> None:
        self.config = config
        self.resident_pool = resident_pool

    def evaluate(
        self,
        geometry: PreparedCandidateGeometry,
        workdir: Path,
    ) -> CandidateEvaluation:
        prepared, mesh_path = _prepare_authority_mesh(geometry, workdir)
        client = _client(self.config)
        server = client.ping()
        model_key = str(server.get("model_key") or "")
        predictor_kind = _predictor_kind(model_key)
        count = len(self.config.task.mach)
        mach = np.asarray(self.config.task.mach, dtype=np.float32)
        start = time.perf_counter()
        response = client.request(
            {
                "geometry": prepared.geometry,
                "coords": prepared.coords,
                "coords_vertex": prepared.coords_vertex,
                "mach": mach,
                "target_cl": np.full(count, self.config.task.target_cl, dtype=np.float32),
                "reynolds": np.asarray(
                    [self.config.task.reynolds_for(value) for value in mach], dtype=np.float32
                ),
                "aoa": np.full(count, np.mean(self.config.task.aoa_bounds), dtype=np.float32),
                "metadata": {"source": "optimization_nk", "sample": workdir.name},
            }
        )
        aoa = _as_vector(response, "aoa", count)
        fields = np.asarray(response["fields"])
        active = list(range(count))
        histories: list[list[dict[str, float]]] = [[] for _ in range(count)]
        best_payloads: list[dict[str, Any] | None] = [None] * count
        terminal_reasons: list[str] = [""] * count
        pipeline = create_pipeline()
        plan = finalonly_plan(
            predictor_kind,
            adaptive_cycles=self.config.nk.adaptive_cycles,
            adaptive_threshold=self.config.nk.residual_tolerance,
        )

        for correction in range(self.config.nk.max_corrections + 1):
            cases = []
            for local_index, point_index in enumerate(active):
                builder = build_direct_case if predictor_kind == "direct" else build_fsb_case
                cases.append(
                    builder(
                        case_id=f"{workdir.name}_mach{point_index:02d}_correction{correction:02d}",
                        cgns_basename=mesh_path.name,
                        cgns_root=mesh_path.parent,
                        prediction_field=fields[local_index],
                        flow_conditions=(
                            self.config.task.mach[point_index],
                            aoa[point_index],
                            self.config.task.reynolds_for(self.config.task.mach[point_index]),
                        ),
                        flow_conditions_dict={
                            "mach": self.config.task.mach[point_index],
                            "aoa": aoa[point_index],
                            "reynolds": self.config.task.reynolds_for(
                                self.config.task.mach[point_index]
                            ),
                            "temperature": 300.0,
                            "x_ref": 0.25,
                        },
                        coords_center=prepared.coords,
                        coords_vertex=prepared.coords_vertex,
                        output_dir=workdir / "nk",
                        options_version=self.config.nk.options_version,
                        l2conv=self.config.nk.residual_tolerance,
                        ranks_per_case=self.config.nk.ranks_per_case,
                        mpi_launcher=self.config.nk.mpi_launcher,
                        mpi_omp_threads=self.config.nk.mpi_omp_threads,
                        model_metadata={"model_key": model_key},
                        runtime_metadata={"optimization_sample": workdir.name},
                    )
                )
            iteration_root = workdir / "nk" / f"correction_{correction:02d}"
            export = pipeline.export_cases(cases, plan, output_dir=str(iteration_root))
            if self.resident_pool is None:
                projected = pipeline.project_manifest_warm_pools(
                    export.manifest_path,
                    ranks_per_case=self.config.nk.ranks_per_case,
                    pool_count=min(self.config.nk.pool_count, len(cases)),
                    mpi_launcher=self.config.nk.mpi_launcher,
                    mpi_omp_threads=self.config.nk.mpi_omp_threads,
                    output_dir=iteration_root / "runtime",
                    ready_timeout_sec=self.config.nk.timeout_s,
                    submit_timeout_sec=self.config.nk.timeout_s,
                    wait_for_manifest_sec=self.config.nk.timeout_s,
                )
            else:
                projected = self.resident_pool.project(
                    export.manifest_path,
                    output_dir=iteration_root / "runtime",
                )
            next_active: list[int] = []
            next_aoa: list[float] = []
            for local_index, point_index in enumerate(active):
                payload = json.loads(
                    Path(projected.result_paths[local_index]).read_text(encoding="utf-8")
                )
                stage = payload["stages"][-1]
                force = dict(stage["metrics"].get("force_coefficients") or {})
                if not force:
                    raise RuntimeError(
                        f"NK result has no corrected force coefficients: {projected.result_paths[local_index]}"
                    )
                residual_summary = dict(
                    dict(stage["metrics"].get("residual") or {}).get("summary") or {}
                )
                cl = float(force["cl"])
                error = cl - self.config.task.target_cl
                histories[point_index].append(
                    {
                        "aoa": float(aoa[point_index]),
                        "cl": cl,
                        "error": float(error),
                    }
                )
                candidate_payload = {
                    "force": force,
                    "stage": stage,
                    "result_path": projected.result_paths[local_index],
                    "correction": correction,
                    "residual": residual_summary.get("final"),
                    "aoa": float(aoa[point_index]),
                }
                best_payload = best_payloads[point_index]
                if (
                    best_payload is None
                    or abs(error)
                    < abs(
                        float(best_payload["force"]["cl"])
                        - self.config.task.target_cl
                    )
                ):
                    best_payloads[point_index] = candidate_payload
                if abs(error) <= self.config.nk.cl_tolerance:
                    terminal_reasons[point_index] = "accepted"
                    continue
                if correction == self.config.nk.max_corrections:
                    terminal_reasons[point_index] = "max_corrections"
                    continue
                proposed_aoa, reason = _propose_fixed_lift_aoa(
                    histories[point_index],
                    self.config,
                )
                if proposed_aoa is None:
                    terminal_reasons[point_index] = reason
                    continue
                next_active.append(point_index)
                next_aoa.append(proposed_aoa)

            if not next_active:
                break
            for point_index, proposed_aoa in zip(next_active, next_aoa):
                aoa[point_index] = proposed_aoa
            active = next_active
            fixed = client.request(_fixed_request(prepared, self.config, active, aoa))
            fields = np.asarray(fixed["fields"])

        elapsed = time.perf_counter() - start
        points: list[OperatingPointResult] = []
        for index, payload_value in enumerate(best_payloads):
            if payload_value is None:
                raise RuntimeError(f"Missing NK result for operating point {index}")
            force = payload_value["force"]
            cl = float(force["cl"])
            points.append(
                OperatingPointResult(
                    mach=float(self.config.task.mach[index]),
                    target_cl=float(self.config.task.target_cl),
                    reynolds=self.config.task.reynolds_for(self.config.task.mach[index]),
                    aoa=float(payload_value["aoa"]),
                    cl=cl,
                    cd=float(force["cd"]),
                    cm=float(force["cm"]),
                    converged=abs(cl - self.config.task.target_cl)
                    <= self.config.nk.cl_tolerance,
                    n_iter=int(payload_value["correction"] + 1),
                    residual=payload_value["residual"],
                    wall_time_s=float(elapsed / count),
                    field_path=str(
                        payload_value["stage"].get("output_paths", {}).get("post_field", "")
                    ),
                    provenance={
                        "model_key": model_key,
                        "result_path": payload_value["result_path"],
                        "plan": "finalonly",
                        "fixed_lift_history": histories[index],
                        "fixed_lift_stop_reason": terminal_reasons[index],
                    },
                )
            )
        return CandidateEvaluation(
            evaluator=self.name,
            points=tuple(points),
            wall_time_s=float(elapsed),
            provenance={
                "model_key": model_key,
                "plan": "finalonly",
                "mesh_path": str(mesh_path),
                "pool_count": min(self.config.nk.pool_count, count),
                "ranks_per_case": self.config.nk.ranks_per_case,
                "resident_pool": self.resident_pool is not None,
            },
        )


def create_evaluator(
    config: OptimizationConfig,
    *,
    refinement: bool = False,
) -> CandidateEvaluator:
    if config.mode == "surrogate":
        return SurrogateEvaluator(config)
    if config.mode == "cfd":
        return CFDEvaluator(config)
    if config.mode == "surrogate_nk":
        if config.nk.selection == "topk" and not refinement:
            return SurrogateEvaluator(config)
        return SurrogateNKEvaluator(config)
    raise ValueError(f"Unsupported evaluator mode: {config.mode}")


__all__ = [
    "CFDEvaluator",
    "CandidateEvaluator",
    "SurrogateEvaluator",
    "SurrogateNKEvaluator",
    "create_evaluator",
]
