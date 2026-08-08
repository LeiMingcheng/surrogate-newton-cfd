"""Full-schedule alternating FSB/NK orchestration for clean NK_resume.

The surrogate side owns the FSB scheduler state. Each selected FSB transition
exports a canonical `NK_resume` manifest, runs the solver correction, and feeds
the corrected physical field back into the next bridge scheduler step.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import importlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from NK_resume import (
    ContractError,
    ExportResult,
    ManifestRunResult,
    NKWorkPlan,
    ResumeCase,
    SolverPreset,
    build_fsb_case,
    create_pipeline,
    load_projection_result_dict,
    normalize_force_coefficients,
    run_manifest,
    alternating_plan,
)
from surrogate.data import UniformFlowInitializer
from surrogate.inference.backends import FSBPredictorBackend
from surrogate.inference.contracts import FSBPredictorConfig
from surrogate.nk_resume.collectors import FSBOrdinalModelBatch, load_fsb_ordinal_model_batch
from surrogate.nk_resume.alternating_state import (
    FSBAlternatingSchedulerState,
    write_alternating_scheduler_state,
)


ALTERNATING_FSB_NK_EXPERIMENT_SUMMARY_SCHEMA = "alternating_fsb_nk_experiment_summary_v1"


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return {str(key): item for key, item in dict(value or {}).items()}


def _int_tuple(values: Iterable[int] | None) -> tuple[int, ...]:
    if values is None:
        return ()
    return tuple(int(value) for value in values)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def _mpi_comm() -> Any:
    try:
        module = importlib.import_module("mpi4py.MPI")
    except Exception:
        return None
    return getattr(module, "COMM_WORLD", None)


def _comm_rank(comm: Any) -> int:
    getter = getattr(comm, "Get_rank", None)
    if not callable(getter):
        return 0
    return int(getter())


def _comm_bcast(comm: Any, value: Any, *, root: int = 0) -> Any:
    broadcaster = getattr(comm, "bcast", None)
    if not callable(broadcaster):
        return value
    return broadcaster(value, root=int(root))


def _comm_barrier(comm: Any) -> None:
    barrier = getattr(comm, "Barrier", None)
    if callable(barrier):
        barrier()


def _parse_ordinals(text: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in str(text).split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ContractError(f"ordinal range must be increasing: {item}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(item))
    if not values:
        raise ContractError("ordinals must not be empty")
    if any(value < 0 for value in values):
        raise ContractError("ordinals must be non-negative")
    return tuple(values)


def _parse_int_csv(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in str(text).split(",") if item.strip())


def _parse_step_cycles(text: str) -> tuple[tuple[int, tuple[int, ...]], ...]:
    entries: list[tuple[int, tuple[int, ...]]] = []
    for raw_entry in str(text).split(";"):
        entry = raw_entry.strip()
        if not entry:
            continue
        step_text, cycles_text = entry.split("=", 1)
        entries.append((int(step_text.strip()), _parse_int_csv(cycles_text)))
    return tuple(entries)


def _path_or_none(value: str | Path | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _tensor(value: Any, *, name: str, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.numel() == 0:
        raise ContractError(f"{name} must not be empty")
    if not torch.isfinite(tensor).all():
        raise ContractError(f"{name} must contain only finite values")
    return tensor


def _batched_tensor(value: Any, *, name: str, device: torch.device) -> torch.Tensor:
    tensor = _tensor(value, name=name, device=device)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ContractError(f"{name} must have shape (B,C,H,W) or (C,H,W)")
    return tensor


def _field_array(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim == 4:
        if int(array.shape[0]) != 1:
            raise ContractError(f"{name} batch dimension must be 1")
        array = array[0]
    if array.ndim != 3:
        raise ContractError(f"{name} must have shape (C,H,W) or (1,C,H,W)")
    if int(array.shape[0]) not in {4, 5}:
        raise ContractError(f"{name} must have 4 or 5 channels")
    array = np.asarray(array, dtype=np.float64)
    if array.size == 0:
        raise ContractError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ContractError(f"{name} must contain only finite values")
    return array


def _coords_array(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim == 4:
        if int(array.shape[0]) != 1:
            raise ContractError(f"{name} batch dimension must be 1")
        array = array[0]
    if array.ndim != 3:
        raise ContractError(f"{name} must have shape (C,H,W) or (1,C,H,W)")
    return np.asarray(array, dtype=np.float64)


def _flow_tuple(value: Any) -> tuple[float, ...]:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim >= 2 and int(array.shape[0]) == 1:
        array = array[0]
    array = np.asarray(array, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ContractError("flow_conditions must not be empty")
    return tuple(float(item) for item in array.tolist())


def _benchmark_case_references(
    *,
    cgns_root: str | Path,
    cgns_basename: str,
) -> tuple[str, dict[str, float], dict[str, float], dict[str, Any]]:
    """Resolve a nested Strict Benchmark case and load its formal references."""

    root = Path(cgns_root)
    basename = Path(str(cgns_basename).strip())
    direct = root / basename
    if direct.is_file():
        cgns_path = direct
        relative = basename
    else:
        suffix = "_000_vol.cgns"
        if basename.name.endswith(suffix):
            case_dir = basename.name[: -len(suffix)]
            nested_relative = Path(case_dir) / basename.name
            nested_path = root / nested_relative
        else:
            nested_relative = basename
            nested_path = direct
        if nested_path.is_file():
            relative = nested_relative
            cgns_path = nested_path
        else:
            # Non-benchmark callers may resolve geometry later through another
            # runtime adapter. Keep the generic alternating contract intact.
            return str(basename), {}, {}, {}

    forces_path = cgns_path.parent / "forces.json"
    if not forces_path.is_file():
        return str(relative), {}, {}, {}
    payload = json.loads(forces_path.read_text(encoding="utf-8"))
    flow = dict(payload.get("flow_conditions") or {})
    flow_conditions = {
        "mach": float(flow["Mach"]),
        "alpha": float(flow["AoA"]),
        "reynolds": float(flow["Reynolds"]),
        "temperature": 300.0,
        "area_ref": 1.0,
        "chord_ref": 1.0,
        "reynolds_length": 1.0,
    }
    force_coefficients = normalize_force_coefficients(
        dict(payload.get("force_coefficients") or {})
    )
    final_total_residual = float(payload["final_total_residual"])
    final_l2_ratio = float(payload["l2_ratio"])
    if final_l2_ratio <= 0.0:
        raise ContractError(
            f"Benchmark forces.json l2_ratio must be positive: {forces_path}"
        )
    residual_reference = {
        "reference_totalr0": final_total_residual / final_l2_ratio,
        "final_total_residual": final_total_residual,
        "final_l2_ratio": final_l2_ratio,
        "source": "benchmark_forces_json",
        "forces_json_path": str(forces_path.resolve()),
    }
    return str(relative), flow_conditions, force_coefficients, residual_reference


def _target_field(batch: FSBOrdinalModelBatch, *, normalizer: Any) -> np.ndarray | None:
    value = batch.target_field_physical(normalizer=normalizer)
    if value is None:
        return None
    return _field_array(value, name="target_field")


def _field_tensor(
    engine: Any,
    field_value: Any,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = torch.as_tensor(field_value, dtype=dtype, device=engine.device)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ContractError(f"field must have shape (B,C,H,W) or (C,H,W), got {tuple(tensor.shape)}")
    return tensor


@dataclass(frozen=True)
class AlternatingFSBNKExperimentRequest:
    """Request for a clean full-schedule alternating FSB/NK run."""

    config_path: str | Path
    ordinals: tuple[int, ...]
    output_dir: str | Path
    index_path: str | Path | None = None
    stats_path: str | Path | None = None
    checkpoint_path: str | Path | None = None
    device: str = "cuda"
    use_ema: bool = True
    n_inference_steps: int | None = None
    custom_timesteps: tuple[int, ...] = ()
    eta: float = 0.0
    noise_mode: str = "zeros"
    correction_steps: tuple[int, ...] | str = "all"
    final_correction: bool = True
    cgns_root: str | Path = ""
    ranks_per_case: int = 8
    mpi_launcher: str = "auto"
    mpi_omp_threads: int = 1
    transition_solver_preset: str = "nk"
    transition_fixed_cycles: int = 6
    transition_adaptive_cycles: tuple[int, ...] = ()
    transition_adaptive_cycles_by_step: tuple[
        tuple[int, tuple[int, ...]], ...
    ] = ()
    transition_adaptive_threshold: float = 1.0e-4
    final_solver_preset: str = "nk"
    final_fixed_cycles: int = 6
    final_adaptive_cycles: tuple[int, ...] = ()
    final_adaptive_threshold: float = 1.0e-8
    l2conv: float = 1.0e-8
    executor: str = "sequential"
    pool_count: int = 0
    ready_timeout_sec: float = 300.0
    submit_timeout_sec: float = 1800.0
    wait_for_manifest_sec: float = 120.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ordinals = _int_tuple(self.ordinals)
        if not ordinals:
            raise ContractError("AlternatingFSBNKExperimentRequest.ordinals must not be empty")
        if any(value < 0 for value in ordinals):
            raise ContractError("AlternatingFSBNKExperimentRequest.ordinals must be non-negative")
        executor = str(self.executor).strip().lower()
        if executor not in {"sequential", "pools", "warm_pools"}:
            raise ContractError(
                "alternating executor must be one of: sequential, pools, warm_pools"
            )
        if int(self.transition_fixed_cycles) <= 0:
            raise ContractError("transition_fixed_cycles must be positive")
        if int(self.final_fixed_cycles) <= 0:
            raise ContractError("final_fixed_cycles must be positive")
        transition_adaptive_cycles = _int_tuple(self.transition_adaptive_cycles)
        transition_adaptive_cycles_by_step = tuple(
            (int(step), _int_tuple(cycles))
            for step, cycles in self.transition_adaptive_cycles_by_step
        )
        final_adaptive_cycles = _int_tuple(self.final_adaptive_cycles)
        for name, cycles in (
            ("transition_adaptive_cycles", transition_adaptive_cycles),
            ("final_adaptive_cycles", final_adaptive_cycles),
        ):
            if any(value <= 0 for value in cycles):
                raise ContractError(f"{name} values must be positive")
            if tuple(sorted(cycles)) != cycles or len(set(cycles)) != len(cycles):
                raise ContractError(f"{name} must be strictly increasing")
        transition_steps = [step for step, _cycles in transition_adaptive_cycles_by_step]
        if any(step <= 0 for step in transition_steps):
            raise ContractError(
                "transition_adaptive_cycles_by_step keys must be positive"
            )
        if len(set(transition_steps)) != len(transition_steps):
            raise ContractError(
                "transition_adaptive_cycles_by_step keys must be unique"
            )
        for step, cycles in transition_adaptive_cycles_by_step:
            if not cycles:
                raise ContractError(
                    f"transition_adaptive_cycles_by_step[{step}] must not be empty"
                )
            if any(value <= 0 for value in cycles):
                raise ContractError(
                    f"transition_adaptive_cycles_by_step[{step}] values must be positive"
                )
            if tuple(sorted(cycles)) != cycles or len(set(cycles)) != len(cycles):
                raise ContractError(
                    f"transition_adaptive_cycles_by_step[{step}] must be strictly increasing"
                )
        if float(self.transition_adaptive_threshold) <= 0.0:
            raise ContractError("transition_adaptive_threshold must be positive")
        if float(self.final_adaptive_threshold) <= 0.0:
            raise ContractError("final_adaptive_threshold must be positive")
        if int(self.ranks_per_case) <= 0:
            raise ContractError("ranks_per_case must be positive")
        if int(self.pool_count) < 0:
            raise ContractError("pool_count must be non-negative")
        if float(self.ready_timeout_sec) <= 0.0:
            raise ContractError("ready_timeout_sec must be positive")
        if float(self.submit_timeout_sec) <= 0.0:
            raise ContractError("submit_timeout_sec must be positive")
        if float(self.wait_for_manifest_sec) <= 0.0:
            raise ContractError("wait_for_manifest_sec must be positive")
        if float(self.l2conv) <= 0.0:
            raise ContractError("l2conv must be positive")
        correction_steps = self.correction_steps
        if not isinstance(correction_steps, str):
            correction_steps = tuple(int(value) for value in correction_steps)
            if any(value <= 0 for value in correction_steps):
                raise ContractError("correction_steps are 1-based and must be positive")
            if len(set(correction_steps)) != len(correction_steps):
                raise ContractError("correction_steps must be unique")
        object.__setattr__(self, "ordinals", ordinals)
        object.__setattr__(self, "custom_timesteps", _int_tuple(self.custom_timesteps))
        object.__setattr__(self, "transition_adaptive_cycles", transition_adaptive_cycles)
        object.__setattr__(
            self,
            "transition_adaptive_cycles_by_step",
            transition_adaptive_cycles_by_step,
        )
        object.__setattr__(self, "final_adaptive_cycles", final_adaptive_cycles)
        object.__setattr__(self, "correction_steps", correction_steps)
        object.__setattr__(self, "executor", executor)
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True)
class AlternatingFSBNKCaseArtifact:
    """Per-case artifact for one alternating NK correction stage."""

    ordinal: int
    case_id: str
    step_number: int
    step_index: int
    state_name: str
    t_current: int
    t_next: int
    scheduler_state_path: str = ""
    pre_field_path: str = ""
    post_field_path: str = ""
    payload_path: str = ""
    result_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "case_id": self.case_id,
            "step_number": self.step_number,
            "step_index": self.step_index,
            "state_name": self.state_name,
            "t_current": self.t_current,
            "t_next": self.t_next,
            "scheduler_state_path": self.scheduler_state_path,
            "pre_field_path": self.pre_field_path,
            "post_field_path": self.post_field_path,
            "payload_path": self.payload_path,
            "result_path": self.result_path,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AlternatingFSBNKStageArtifact:
    """One exported and optionally executed alternating correction stage."""

    stage_kind: str
    step_number: int
    step_index: int
    state_name: str
    t_current: int
    t_next: int
    manifest_path: str
    export: ExportResult | None = None
    run: ManifestRunResult | None = None
    cases: tuple[AlternatingFSBNKCaseArtifact, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_kind": self.stage_kind,
            "step_number": self.step_number,
            "step_index": self.step_index,
            "state_name": self.state_name,
            "t_current": self.t_current,
            "t_next": self.t_next,
            "manifest_path": self.manifest_path,
            "export": None if self.export is None else self.export.to_dict(),
            "run": None if self.run is None else self.run.to_dict(),
            "cases": [case.to_dict() for case in self.cases],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AlternatingFSBNKExperimentResult:
    """Artifacts produced by clean alternating FSB/NK orchestration."""

    output_dir: str
    ordinals: tuple[int, ...]
    executor: str
    transition_count: int
    correction_steps: tuple[int, ...]
    final_correction: bool
    summary_path: str
    stages: tuple[AlternatingFSBNKStageArtifact, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "ordinals": list(self.ordinals),
            "executor": self.executor,
            "transition_count": self.transition_count,
            "correction_steps": list(self.correction_steps),
            "final_correction": self.final_correction,
            "summary_path": self.summary_path,
            "stages": [stage.to_dict() for stage in self.stages],
            "metadata": dict(self.metadata),
        }


@dataclass
class _AlternatingContext:
    batch: FSBOrdinalModelBatch
    geometry: torch.Tensor
    flow_conditions: torch.Tensor
    coords: torch.Tensor
    x1: torch.Tensor
    x_t: torch.Tensor
    latest_field: np.ndarray | None = None


def _build_backend(request: AlternatingFSBNKExperimentRequest) -> FSBPredictorBackend:
    return FSBPredictorBackend(
        FSBPredictorConfig(
            config_path=str(request.config_path),
            checkpoint_path=_path_or_none(request.checkpoint_path),
            device=request.device,
            use_ema=bool(request.use_ema),
            n_inference_steps=request.n_inference_steps,
            custom_timesteps=list(request.custom_timesteps) if request.custom_timesteps else None,
            eta=float(request.eta),
            noise_mode=str(request.noise_mode),
        )
    )


def _load_batches(request: AlternatingFSBNKExperimentRequest) -> tuple[FSBOrdinalModelBatch, ...]:
    return tuple(
        load_fsb_ordinal_model_batch(
            config_path=request.config_path,
            ordinal=ordinal,
            index_path=request.index_path,
            stats_path=request.stats_path,
            checkpoint_path=request.checkpoint_path,
        )
        for ordinal in request.ordinals
    )


def _initial_field(
    *,
    backend: FSBPredictorBackend,
    batch: FSBOrdinalModelBatch,
) -> torch.Tensor:
    device = torch.device(backend.device)
    flow_conditions = _tensor(batch.flow_conditions, name="flow_conditions", device=device)
    if flow_conditions.ndim == 1:
        flow_conditions = flow_conditions.unsqueeze(0)
    coords = _batched_tensor(batch.coords, name="coords", device=device)
    initializer = UniformFlowInitializer(
        normalizer=backend.normalizer,
        device=device,
    )
    return initializer.generate_uniform_field(
        flow_conditions=flow_conditions,
        spatial_shape=(int(coords.shape[-2]), int(coords.shape[-1])),
        coords=coords,
    )


def _plan(
    request: AlternatingFSBNKExperimentRequest,
    *,
    transition_adaptive_cycles: tuple[int, ...] | None = None,
):
    transition_cycles = (
        request.transition_adaptive_cycles
        if transition_adaptive_cycles is None
        else transition_adaptive_cycles
    )
    if transition_cycles:
        transition_work = NKWorkPlan.adaptive(
            transition_cycles,
            threshold=float(request.transition_adaptive_threshold),
            name="alternating_transition",
            solver_preset=SolverPreset(request.transition_solver_preset),
        )
    else:
        transition_work = NKWorkPlan.fixed(
            int(request.transition_fixed_cycles),
            solver_preset=SolverPreset(request.transition_solver_preset),
        )
    if request.final_adaptive_cycles:
        final_work = NKWorkPlan.adaptive(
            request.final_adaptive_cycles,
            threshold=float(request.final_adaptive_threshold),
            name="alternating_final",
            solver_preset=SolverPreset(request.final_solver_preset),
        )
    else:
        final_work = NKWorkPlan.fixed(
            int(request.final_fixed_cycles),
            solver_preset=SolverPreset(request.final_solver_preset),
        )
    return alternating_plan(
        transition_work=transition_work,
        final_work=final_work,
    )


def _transition_cycles_for_step(
    request: AlternatingFSBNKExperimentRequest,
    step_number: int,
) -> tuple[int, ...]:
    cycles_by_step = dict(request.transition_adaptive_cycles_by_step)
    return cycles_by_step.get(int(step_number), request.transition_adaptive_cycles)


def _resolve_timesteps(
    backend: FSBPredictorBackend,
    request: AlternatingFSBNKExperimentRequest,
) -> torch.Tensor:
    timesteps = backend.engine._resolve_timesteps(
        list(request.custom_timesteps) if request.custom_timesteps else None
    )
    if int(timesteps.numel()) < 2:
        raise ContractError("alternating FSB/NK requires at least one FSB transition")
    return timesteps.to(device=torch.device(backend.device), dtype=torch.long)


def _resolve_correction_steps(
    value: tuple[int, ...] | str,
    *,
    transition_count: int,
) -> tuple[int, ...]:
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "all":
            steps = tuple(range(1, int(transition_count) + 1))
        elif not text:
            steps = ()
        else:
            steps = _parse_int_csv(text)
    else:
        steps = tuple(int(item) for item in value)
    bad = [item for item in steps if item < 1 or item > int(transition_count)]
    if bad:
        raise ContractError(
            "correction_steps are 1-based FSB transition numbers; "
            f"out of range for {transition_count} transitions: {bad}"
        )
    return tuple(sorted(set(steps)))


def _prepare_contexts(
    *,
    backend: FSBPredictorBackend,
    batches: tuple[FSBOrdinalModelBatch, ...],
    timesteps: torch.Tensor,
) -> tuple[_AlternatingContext, ...]:
    device = torch.device(backend.device)
    contexts: list[_AlternatingContext] = []
    for batch in batches:
        geometry = _tensor(batch.geometry, name="geometry", device=device)
        if geometry.ndim == 1:
            geometry = geometry.unsqueeze(0)
        flow_conditions = _tensor(batch.flow_conditions, name="flow_conditions", device=device)
        if flow_conditions.ndim == 1:
            flow_conditions = flow_conditions.unsqueeze(0)
        coords = _batched_tensor(batch.coords, name="coords", device=device)
        x1 = _initial_field(backend=backend, batch=batch)
        x_t = backend.engine._initial_sample(x1, int(timesteps[0].item()))
        contexts.append(
            _AlternatingContext(
                batch=batch,
                geometry=geometry,
                flow_conditions=flow_conditions,
                coords=coords,
                x1=x1,
                x_t=x_t,
            )
        )
    return tuple(contexts)


def _build_case(
    *,
    request: AlternatingFSBNKExperimentRequest,
    batch: FSBOrdinalModelBatch,
    target_normalizer: Any,
    field_value: Any,
    state_name: str,
    step_index: int,
    step_number: int,
    stage_kind: str,
) -> ResumeCase:
    if not str(batch.cgns_basename).strip():
        raise ContractError("alternating case collection requires cgns_basename")
    (
        cgns_basename,
        flow_conditions_dict,
        force_coefficients,
        residual_reference,
    ) = _benchmark_case_references(
        cgns_root=request.cgns_root,
        cgns_basename=batch.cgns_basename,
    )
    suffix = "final" if state_name == "final" else f"step{int(step_number):03d}"
    case_id = f"nk_resume_ordinal{batch.ordinal:04d}_fsb_alternating_{suffix}"
    return build_fsb_case(
        case_id=case_id,
        cgns_basename=cgns_basename,
        prediction_field=_field_array(field_value, name=f"{stage_kind} prediction"),
        state_name=state_name,
        step_index=int(step_index),
        cgns_root=request.cgns_root,
        flow_conditions=_flow_tuple(batch.flow_conditions),
        flow_conditions_dict=flow_conditions_dict,
        source_info={
            **batch.source_info,
            "collector": "run_alternating_fsb_nk_experiment",
        },
        options_version=2,
        l2conv=float(request.l2conv),
        ranks_per_case=int(request.ranks_per_case),
        mpi_launcher=request.mpi_launcher,
        mpi_omp_threads=int(request.mpi_omp_threads),
        ground_truth_field=_target_field(batch, normalizer=target_normalizer),
        coords_center=_coords_array(batch.coords_center, name="coords_center"),
        coords_vertex=_coords_array(batch.coords_vertex, name="coords_vertex"),
        force_coefficients=force_coefficients,
        residual_reference=residual_reference,
        output_dir=Path(request.output_dir) / "cases",
        ordinal=batch.ordinal,
        dataset_index=batch.ordinal,
        config_path=batch.config_path,
        checkpoint_path=batch.checkpoint_path,
        stats_path=batch.stats_path,
        device=request.device,
        inference_steps=request.n_inference_steps,
        custom_timesteps=request.custom_timesteps,
        model_metadata={
            **batch.metadata,
            "collector": "run_alternating_fsb_nk_experiment",
        },
        prediction_metadata={
            "collector": "run_alternating_fsb_nk_experiment",
            "stage_kind": stage_kind,
            "state": state_name,
            "step_index": int(step_index),
            "step_number": int(step_number),
        },
        runtime_metadata={
            **request.metadata,
            "entrypoint": "run_alternating_fsb_nk_experiment",
        },
    )


def _predict_transition(
    *,
    engine: Any,
    context: _AlternatingContext,
    timesteps: torch.Tensor,
    step_index: int,
) -> tuple[np.ndarray, torch.Tensor, FSBAlternatingSchedulerState]:
    t_current = int(timesteps[step_index].item())
    t_next = int(timesteps[step_index + 1].item())
    t_batch = torch.full(
        (context.x_t.shape[0],),
        t_current,
        device=engine.device,
        dtype=torch.long,
    )
    with torch.no_grad():
        model_output = engine._predict_model(
            noisy_fields=context.x_t,
            timesteps=t_batch,
            geometry=context.geometry,
            flow_conditions=context.flow_conditions,
            coords=context.coords,
        )
        pred_x0 = engine.i2sb_scheduler.reconstruct_x0(context.x_t, model_output, t_batch)
        if engine.project_x0_to_physical_bounds:
            pred_x0 = engine._project_x0_to_physical_bounds(pred_x0)
        physical = engine._physical_from_norm(pred_x0)
        scheduler_state = FSBAlternatingSchedulerState(
            resolved_timesteps=timesteps.detach().cpu(),
            target_step=int(step_index),
            x_t_before_step=context.x_t.detach().cpu(),
            x1_norm=context.x1.detach().cpu(),
            t_current=t_current,
            t_next=t_next,
            eta=float(engine.eta),
            noise_mode=str(engine.noise_mode),
            metadata={
                "ordinal": context.batch.ordinal,
                "step_number": int(step_index) + 1,
                "stage_kind": "transition",
            },
        )
    return _field_array(physical, name="transition physical x0"), pred_x0, scheduler_state


def _advance_context(
    *,
    engine: Any,
    context: _AlternatingContext,
    timesteps: torch.Tensor,
    step_index: int,
    x0_norm: torch.Tensor,
    field_physical: np.ndarray,
) -> None:
    t_current = int(timesteps[step_index].item())
    t_next = int(timesteps[step_index + 1].item())
    with torch.no_grad():
        context.x_t = engine.i2sb_scheduler.step_from_x0(
            timestep=t_current,
            x0=x0_norm,
            x1=context.x1,
            sample=context.x_t,
            timestep_next=t_next,
            eta=float(engine.eta),
        )
    context.latest_field = np.asarray(field_physical, dtype=np.float64)


def _write_field(path: Path, field_value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(field_value, dtype=np.float64))
    return str(path)


def _run_manifest_for_stage(
    *,
    request: AlternatingFSBNKExperimentRequest,
    comm: Any,
    manifest_path: str,
    summary_path: Path,
) -> ManifestRunResult:
    manifest_path = str(_comm_bcast(comm, manifest_path, root=0))
    _comm_barrier(comm)
    kwargs: dict[str, Any] = {
        "executor": request.executor,
        "ranks_per_case": int(request.ranks_per_case),
        "summary_path": summary_path,
    }
    if request.executor == "warm_pools":
        kwargs.update(
            {
                "pool_count": int(request.pool_count) or None,
                "mpi_launcher": str(request.mpi_launcher),
                "mpi_omp_threads": int(request.mpi_omp_threads),
                "runtime_output_dir": summary_path.parent
                / f"{summary_path.stem}.warm_runtime",
                "ready_timeout_sec": float(request.ready_timeout_sec),
                "submit_timeout_sec": float(request.submit_timeout_sec),
                "wait_for_manifest_sec": float(request.wait_for_manifest_sec),
            }
        )
    return run_manifest(manifest_path, **kwargs)


def _wait_for_nonprimary_stages(
    *,
    request: AlternatingFSBNKExperimentRequest,
    comm: Any,
    stage_specs: tuple[dict[str, Any], ...],
    output_dir: Path,
) -> AlternatingFSBNKExperimentResult:
    stages: list[AlternatingFSBNKStageArtifact] = []
    for spec in stage_specs:
        step_number = int(spec["step_number"])
        state_name = str(spec["state_name"])
        summary_path = output_dir / f"{state_name}_step_{step_number:03d}.{request.executor}.run_summary.json"
        run_result = _run_manifest_for_stage(
            request=request,
            comm=comm,
            manifest_path="",
            summary_path=summary_path,
        )
        stages.append(
            AlternatingFSBNKStageArtifact(
                stage_kind=str(spec["stage_kind"]),
                step_number=step_number,
                step_index=int(spec["step_index"]),
                state_name=state_name,
                t_current=int(spec["t_current"]),
                t_next=int(spec["t_next"]),
                manifest_path=run_result.manifest_path,
                run=run_result,
                metadata={"primary": False},
            )
        )
    summary_path = output_dir / "alternating_fsb_nk_experiment_summary.json"
    return AlternatingFSBNKExperimentResult(
        output_dir=str(output_dir),
        ordinals=request.ordinals,
        executor=request.executor,
        transition_count=int(stage_specs[-1]["transition_count"]) if stage_specs else 0,
        correction_steps=tuple(int(v) for v in stage_specs[0].get("correction_steps", ())) if stage_specs else (),
        final_correction=bool(request.final_correction),
        summary_path=str(summary_path),
        stages=tuple(stages),
        metadata={"primary": False},
    )


def _post_fields_from_run(
    *,
    cases: tuple[ResumeCase, ...],
    run_result: ManifestRunResult,
    state_name: str,
) -> tuple[tuple[np.ndarray, ...], tuple[str, ...]]:
    if len(cases) != len(run_result.result_paths):
        raise ContractError(
            f"alternating run result count mismatch: cases={len(cases)}, "
            f"results={len(run_result.result_paths)}"
        )
    fields: list[np.ndarray] = []
    post_paths: list[str] = []
    for case, result_path in zip(cases, run_result.result_paths):
        result = load_projection_result_dict(result_path)
        if str(result.get("case_id") or "") != case.case_id:
            raise ContractError(
                f"alternating result case_id does not match {case.case_id!r}: "
                f"{result.get('case_id')!r}"
            )
        stage = next(
            item
            for item in result["stages"]
            if str(item.get("name") or "").strip().lower() == state_name
        )
        post_field_path = str(stage["output_paths"]["post_field"])
        field_value = np.load(post_field_path, allow_pickle=False)
        fields.append(_field_array(field_value, name=f"{state_name} post field"))
        post_paths.append(post_field_path)
    return tuple(fields), tuple(post_paths)


def _merge_case_artifacts(
    artifacts: tuple[AlternatingFSBNKCaseArtifact, ...],
    *,
    export: ExportResult,
    run_result: ManifestRunResult,
    post_field_paths: tuple[str, ...],
) -> tuple[AlternatingFSBNKCaseArtifact, ...]:
    out: list[AlternatingFSBNKCaseArtifact] = []
    for index, artifact in enumerate(artifacts):
        out.append(
            AlternatingFSBNKCaseArtifact(
                ordinal=artifact.ordinal,
                case_id=artifact.case_id,
                step_number=artifact.step_number,
                step_index=artifact.step_index,
                state_name=artifact.state_name,
                t_current=artifact.t_current,
                t_next=artifact.t_next,
                scheduler_state_path=artifact.scheduler_state_path,
                pre_field_path=artifact.pre_field_path,
                post_field_path=post_field_paths[index] if index < len(post_field_paths) else "",
                payload_path=export.payload_paths[index] if index < len(export.payload_paths) else "",
                result_path=run_result.result_paths[index] if index < len(run_result.result_paths) else "",
                metadata=artifact.metadata,
            )
        )
    return tuple(out)


def _stage_specs(
    *,
    timesteps: torch.Tensor,
    correction_steps: tuple[int, ...],
    final_correction: bool,
) -> tuple[dict[str, Any], ...]:
    transition_count = int(timesteps.numel()) - 1
    specs: list[dict[str, Any]] = []
    for step_number in correction_steps:
        step_index = int(step_number) - 1
        specs.append(
            {
                "stage_kind": "transition",
                "step_number": int(step_number),
                "step_index": step_index,
                "state_name": "denoise",
                "t_current": int(timesteps[step_index].item()),
                "t_next": int(timesteps[step_index + 1].item()),
                "transition_count": transition_count,
                "correction_steps": list(correction_steps),
            }
        )
    if final_correction:
        specs.append(
            {
                "stage_kind": "final",
                "step_number": transition_count + 1,
                "step_index": transition_count,
                "state_name": "final",
                "t_current": 0,
                "t_next": 0,
                "transition_count": transition_count,
                "correction_steps": list(correction_steps),
            }
        )
    return tuple(specs)


def run_alternating_fsb_nk_experiment(
    request: AlternatingFSBNKExperimentRequest,
) -> AlternatingFSBNKExperimentResult:
    """Run a clean FSB schedule with NK feedback after selected transitions."""

    output_dir = Path(request.output_dir)
    comm = _mpi_comm() if request.executor == "pools" else None
    is_primary = _comm_rank(comm) == 0
    if is_primary:
        output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "alternating_fsb_nk_experiment_summary.json"

    default_plan = _plan(request)
    stage_specs: tuple[dict[str, Any], ...] = ()
    if is_primary:
        backend = _build_backend(request)
        batches = _load_batches(request)
        timesteps = _resolve_timesteps(backend, request)
        transition_count = int(timesteps.numel()) - 1
        correction_steps = _resolve_correction_steps(
            request.correction_steps,
            transition_count=transition_count,
        )
        contexts = _prepare_contexts(backend=backend, batches=batches, timesteps=timesteps)
        stage_specs = _stage_specs(
            timesteps=timesteps,
            correction_steps=correction_steps,
            final_correction=bool(request.final_correction),
        )
    else:
        backend = None
        batches = ()
        timesteps = None
        transition_count = 0
        correction_steps = ()
        contexts = ()

    stage_specs = tuple(_comm_bcast(comm, stage_specs, root=0))
    if not is_primary:
        return _wait_for_nonprimary_stages(
            request=request,
            comm=comm,
            stage_specs=stage_specs,
            output_dir=output_dir,
        )

    if backend is None or timesteps is None:
        raise ContractError("primary alternating run has no backend or timesteps")
    stage_by_step = {
        (str(spec["stage_kind"]), int(spec["step_number"])): spec for spec in stage_specs
    }
    state_dir = output_dir / "scheduler_state"
    field_dir = output_dir / "fields"
    stages: list[AlternatingFSBNKStageArtifact] = []

    for step_index in range(transition_count):
        step_number = step_index + 1
        predicted: list[tuple[np.ndarray, torch.Tensor, FSBAlternatingSchedulerState]] = []
        for context in contexts:
            predicted.append(
                _predict_transition(
                    engine=backend.engine,
                    context=context,
                    timesteps=timesteps,
                    step_index=step_index,
                )
            )

        spec = stage_by_step.get(("transition", step_number))
        if spec is None:
            for context, (field_value, x0_norm, _state) in zip(contexts, predicted):
                _advance_context(
                    engine=backend.engine,
                    context=context,
                    timesteps=timesteps,
                    step_index=step_index,
                    x0_norm=x0_norm,
                    field_physical=field_value,
                )
            continue

        cases: list[ResumeCase] = []
        artifacts: list[AlternatingFSBNKCaseArtifact] = []
        for context, (field_value, _x0_norm, scheduler_state) in zip(contexts, predicted):
            batch = context.batch
            state_path = write_alternating_scheduler_state(
                scheduler_state,
                state_dir
                / f"ordinal{batch.ordinal:04d}.step{step_number:03d}.scheduler_state.npz",
            )
            pre_path = _write_field(
                field_dir / f"ordinal{batch.ordinal:04d}.step{step_number:03d}.pre.npy",
                field_value,
            )
            case = _build_case(
                request=request,
                batch=batch,
                target_normalizer=backend.normalizer,
                field_value=field_value,
                state_name="denoise",
                step_index=step_index,
                step_number=step_number,
                stage_kind="transition",
            )
            cases.append(case)
            artifacts.append(
                AlternatingFSBNKCaseArtifact(
                    ordinal=batch.ordinal,
                    case_id=case.case_id,
                    step_number=step_number,
                    step_index=step_index,
                    state_name="denoise",
                    t_current=int(spec["t_current"]),
                    t_next=int(spec["t_next"]),
                    scheduler_state_path=state_path,
                    pre_field_path=pre_path,
                    metadata={
                        "stage_kind": "transition",
                        "resolved_timesteps": [int(v) for v in timesteps.detach().cpu().tolist()],
                    },
                )
            )

        transition_plan = _plan(
            request,
            transition_adaptive_cycles=_transition_cycles_for_step(
                request, step_number
            ),
        )
        export = create_pipeline().export_cases(
            tuple(cases),
            transition_plan,
            output_dir=str(output_dir / f"transition_step_{step_number:03d}_export"),
        )
        run_result = _run_manifest_for_stage(
            request=request,
            comm=comm,
            manifest_path=export.manifest_path,
            summary_path=output_dir
            / f"denoise_step_{step_number:03d}.{request.executor}.run_summary.json",
        )
        post_fields, post_paths = _post_fields_from_run(
            cases=tuple(cases),
            run_result=run_result,
            state_name="denoise",
        )
        for context, field_value in zip(contexts, post_fields):
            x0_physical = _field_tensor(
                backend.engine,
                field_value,
                dtype=context.x_t.dtype,
            )
            x0_norm = backend.engine._norm_from_physical(x0_physical)
            _advance_context(
                engine=backend.engine,
                context=context,
                timesteps=timesteps,
                step_index=step_index,
                x0_norm=x0_norm,
                field_physical=field_value,
            )
        merged_artifacts = _merge_case_artifacts(
            tuple(artifacts),
            export=export,
            run_result=run_result,
            post_field_paths=post_paths,
        )
        stages.append(
            AlternatingFSBNKStageArtifact(
                stage_kind="transition",
                step_number=step_number,
                step_index=step_index,
                state_name="denoise",
                t_current=int(spec["t_current"]),
                t_next=int(spec["t_next"]),
                manifest_path=export.manifest_path,
                export=export,
                run=run_result,
                cases=merged_artifacts,
            )
        )

    final_spec = stage_by_step.get(("final", transition_count + 1))
    if final_spec is not None:
        final_cases: list[ResumeCase] = []
        final_artifacts: list[AlternatingFSBNKCaseArtifact] = []
        for context in contexts:
            with torch.no_grad():
                final_field = backend.engine._physical_from_norm(context.x_t)
            field_value = _field_array(final_field, name="alternating final field")
            context.latest_field = field_value
            pre_path = _write_field(
                field_dir / f"ordinal{context.batch.ordinal:04d}.final.pre.npy",
                field_value,
            )
            case = _build_case(
                request=request,
                batch=context.batch,
                target_normalizer=backend.normalizer,
                field_value=field_value,
                state_name="final",
                step_index=transition_count,
                step_number=transition_count + 1,
                stage_kind="final",
            )
            final_cases.append(case)
            final_artifacts.append(
                AlternatingFSBNKCaseArtifact(
                    ordinal=context.batch.ordinal,
                    case_id=case.case_id,
                    step_number=transition_count + 1,
                    step_index=transition_count,
                    state_name="final",
                    t_current=0,
                    t_next=0,
                    pre_field_path=pre_path,
                    metadata={"stage_kind": "final"},
                )
            )
        final_export = create_pipeline().export_cases(
            tuple(final_cases),
            default_plan,
            output_dir=str(output_dir / "final_export"),
        )
        final_run = _run_manifest_for_stage(
            request=request,
            comm=comm,
            manifest_path=final_export.manifest_path,
            summary_path=output_dir / f"final.{request.executor}.run_summary.json",
        )
        final_post_fields, final_post_paths = _post_fields_from_run(
            cases=tuple(final_cases),
            run_result=final_run,
            state_name="final",
        )
        for context, field_value in zip(contexts, final_post_fields):
            context.latest_field = np.asarray(field_value, dtype=np.float64)
        final_artifacts_tuple = _merge_case_artifacts(
            tuple(final_artifacts),
            export=final_export,
            run_result=final_run,
            post_field_paths=final_post_paths,
        )
        stages.append(
            AlternatingFSBNKStageArtifact(
                stage_kind="final",
                step_number=transition_count + 1,
                step_index=transition_count,
                state_name="final",
                t_current=0,
                t_next=0,
                manifest_path=final_export.manifest_path,
                export=final_export,
                run=final_run,
                cases=final_artifacts_tuple,
            )
        )

    result = AlternatingFSBNKExperimentResult(
        output_dir=str(output_dir),
        ordinals=request.ordinals,
        executor=request.executor,
        transition_count=transition_count,
        correction_steps=correction_steps,
        final_correction=bool(request.final_correction),
        summary_path=str(summary_path),
        stages=tuple(stages),
        metadata={
            "schema_version": ALTERNATING_FSB_NK_EXPERIMENT_SUMMARY_SCHEMA,
            "config_path": str(request.config_path),
            "checkpoint_path": "" if request.checkpoint_path is None else str(request.checkpoint_path),
            "device": request.device,
            "custom_timesteps": [int(v) for v in request.custom_timesteps],
            "transition_solver_preset": str(request.transition_solver_preset),
            "transition_fixed_cycles": int(request.transition_fixed_cycles),
            "transition_adaptive_cycles": [
                int(value) for value in request.transition_adaptive_cycles
            ],
            "transition_adaptive_cycles_by_step": {
                str(step): [int(value) for value in cycles]
                for step, cycles in request.transition_adaptive_cycles_by_step
            },
            "transition_adaptive_threshold": float(
                request.transition_adaptive_threshold
            ),
            "final_solver_preset": str(request.final_solver_preset),
            "final_fixed_cycles": int(request.final_fixed_cycles),
            "final_adaptive_cycles": [int(value) for value in request.final_adaptive_cycles],
            "final_adaptive_threshold": float(request.final_adaptive_threshold),
        },
    )
    _write_json_atomic(
        summary_path,
        {
            "schema_version": ALTERNATING_FSB_NK_EXPERIMENT_SUMMARY_SCHEMA,
            **result.to_dict(),
        },
    )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run clean alternating FSB/NK experiments.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ordinals", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--index")
    parser.add_argument("--stats")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--n-inference-steps", type=int)
    parser.add_argument("--custom-timesteps", default="")
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--noise-mode", default="zeros")
    parser.add_argument("--correction-steps", default="all")
    parser.add_argument("--no-final-correction", action="store_true")
    parser.add_argument("--cgns-root", default="")
    parser.add_argument("--ranks-per-case", type=int, default=8)
    parser.add_argument("--mpi-launcher", default="auto")
    parser.add_argument("--mpi-omp-threads", type=int, default=1)
    parser.add_argument("--pool-count", type=int, default=0)
    parser.add_argument("--ready-timeout-sec", type=float, default=300.0)
    parser.add_argument("--submit-timeout-sec", type=float, default=1800.0)
    parser.add_argument("--wait-for-manifest-sec", type=float, default=120.0)
    parser.add_argument("--transition-solver-preset", choices=("none", "nk", "prod", "pseudo"), default="nk")
    parser.add_argument("--transition-fixed-cycles", type=int, default=6)
    parser.add_argument("--transition-adaptive-cycles", default="")
    parser.add_argument(
        "--transition-adaptive-cycles-by-step",
        default="",
        help="Per-step adaptive schedules, for example '5=1,2;6=1,2,3'",
    )
    parser.add_argument("--transition-adaptive-threshold", type=float, default=1.0e-4)
    parser.add_argument("--final-solver-preset", choices=("none", "nk", "prod", "pseudo"), default="nk")
    parser.add_argument("--final-fixed-cycles", type=int, default=6)
    parser.add_argument("--final-adaptive-cycles", default="")
    parser.add_argument("--final-adaptive-threshold", type=float, default=1.0e-8)
    parser.add_argument(
        "--executor",
        choices=("sequential", "pools", "warm_pools"),
        default="sequential",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    custom_timesteps = _parse_int_csv(args.custom_timesteps)
    result = run_alternating_fsb_nk_experiment(
        AlternatingFSBNKExperimentRequest(
            config_path=args.config,
            ordinals=_parse_ordinals(args.ordinals),
            output_dir=args.output_dir,
            index_path=args.index,
            stats_path=args.stats,
            checkpoint_path=args.checkpoint,
            device=args.device,
            use_ema=not bool(args.no_ema),
            n_inference_steps=args.n_inference_steps,
            custom_timesteps=custom_timesteps,
            eta=float(args.eta),
            noise_mode=args.noise_mode,
            correction_steps=args.correction_steps,
            final_correction=not bool(args.no_final_correction),
            cgns_root=args.cgns_root,
            ranks_per_case=int(args.ranks_per_case),
            mpi_launcher=args.mpi_launcher,
            mpi_omp_threads=int(args.mpi_omp_threads),
            transition_solver_preset=args.transition_solver_preset,
            transition_fixed_cycles=int(args.transition_fixed_cycles),
            transition_adaptive_cycles=_parse_int_csv(args.transition_adaptive_cycles),
            transition_adaptive_cycles_by_step=_parse_step_cycles(
                args.transition_adaptive_cycles_by_step
            ),
            transition_adaptive_threshold=float(args.transition_adaptive_threshold),
            final_solver_preset=args.final_solver_preset,
            final_fixed_cycles=int(args.final_fixed_cycles),
            final_adaptive_cycles=_parse_int_csv(args.final_adaptive_cycles),
            final_adaptive_threshold=float(args.final_adaptive_threshold),
            executor=args.executor,
            pool_count=int(args.pool_count),
            ready_timeout_sec=float(args.ready_timeout_sec),
            submit_timeout_sec=float(args.submit_timeout_sec),
            wait_for_manifest_sec=float(args.wait_for_manifest_sec),
        )
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALTERNATING_FSB_NK_EXPERIMENT_SUMMARY_SCHEMA",
    "AlternatingFSBNKCaseArtifact",
    "AlternatingFSBNKExperimentRequest",
    "AlternatingFSBNKExperimentResult",
    "AlternatingFSBNKStageArtifact",
    "run_alternating_fsb_nk_experiment",
]
