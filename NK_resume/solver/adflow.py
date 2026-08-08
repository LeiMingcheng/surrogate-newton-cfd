"""Single-case ADflow/NK backend for the clean NK_resume contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import importlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from ..exceptions import ContractError, NotMigratedError
from ..geometry import cgns_geometry_key, resolve_cgns_ref
from ..metrics import (
    compute_field_force_coefficients,
    field_metrics,
    force_metrics,
    normalize_force_coefficients,
    residual_metrics,
    trajectory_summary,
)
from ..plans import CyclePolicy, SolverPreset, StagePlan
from .adflow_options import build_adflow_options_for_stage
from .backend import (
    ProjectionRequest,
    ProjectionResult,
    ProjectionStageResult,
    write_projection_result,
)
from .adflow_runtime import ensure_adflow_runtime_on_path
from .state import ADflowStateAdapter


SolverFactory = Callable[..., Any]
AeroProblemFactory = Callable[..., Any]
StateInjector = Callable[..., Mapping[str, Any] | None]
StateExtractor = Callable[..., Any]
ForceCoefficientCalculator = Callable[[Any, Any, Any], Mapping[str, Any]]


def _default_mpi_comm() -> Any:
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


def _comm_size(comm: Any) -> int:
    getter = getattr(comm, "Get_size", None)
    if not callable(getter):
        return 1
    return int(getter())


def _comm_barrier(comm: Any) -> None:
    barrier = getattr(comm, "Barrier", None)
    if callable(barrier):
        barrier()


def _comm_bcast(comm: Any, value: Any, *, root: int = 0) -> Any:
    broadcast = getattr(comm, "bcast", None)
    if callable(broadcast):
        return broadcast(value, root=root)
    return value


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return {str(k): _jsonable(v) for k, v in dict(value or {}).items()}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except (TypeError, ValueError):
            pass
    return str(value)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _numeric_series(value: Any, *, name: str) -> tuple[float, ...]:
    if value is None:
        return ()
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be numeric") from exc
    if array.size == 0:
        return ()
    if not np.all(np.isfinite(array)):
        raise ContractError(f"{name} must contain only finite values")
    return tuple(float(entry) for entry in array)


def _stage_for_request(request: ProjectionRequest) -> StagePlan:
    state_name = request.case.prediction.state_name
    for stage in request.plan.stages:
        if stage.name == state_name:
            return stage
    raise ContractError(f"ProjectionRequest has no stage for state {state_name!r}")


def _cycle_budgets(stage: StagePlan) -> tuple[int, ...]:
    work = stage.work
    if work.cycle_policy == CyclePolicy.FIXED:
        return (int(work.fixed_cycles),)
    if work.cycle_policy == CyclePolicy.ADAPTIVE:
        if work.adaptive_schedule is None:
            raise ContractError("adaptive stage has no schedule")
        return tuple(int(value) for value in work.adaptive_schedule.cumulative_cycles)
    raise ContractError(f"Unsupported cycle policy: {work.cycle_policy}")


def _thresholds(stage: StagePlan) -> tuple[float | None, ...]:
    work = stage.work
    if work.cycle_policy != CyclePolicy.ADAPTIVE or work.adaptive_schedule is None:
        return tuple(None for _ in _cycle_budgets(stage))
    return tuple(float(value) for value in work.adaptive_schedule.thresholds)


def _flow_values(request: ProjectionRequest) -> dict[str, float]:
    values = tuple(request.case.solver_context.flow_conditions)
    mapping = request.case.solver_context.flow_conditions_dict
    if len(values) >= 3:
        mach, alpha, reynolds = values[:3]
    else:
        mach = mapping.get("mach", mapping.get("ma"))
        alpha = mapping.get("alpha", mapping.get("aoa"))
        reynolds = mapping.get("reynolds", mapping.get("re"))
    mach = _finite_float(mach)
    alpha = _finite_float(alpha)
    reynolds = _finite_float(reynolds)
    if mach is None or alpha is None or reynolds is None:
        raise ContractError(
            "ADflow projection requires mach, alpha, and reynolds in SolverContext"
        )
    return {"mach": mach, "alpha": alpha, "reynolds": reynolds}


def _residual_reference_totalr(request: ProjectionRequest) -> float | None:
    reference = request.case.ground_truth.residual_reference
    for key in (
        "reference_totalr0",
        "totalr0",
        "reference_totalr",
        "l2_reference",
        "residual_vector_l2_reference",
    ):
        value = _finite_float(reference.get(key))
        if value is not None and value > 0.0:
            return value
    return None


def _aero_problem_kwargs(request: ProjectionRequest) -> dict[str, Any]:
    flow = _flow_values(request)
    mapping = request.case.solver_context.flow_conditions_dict
    kwargs = {
        "name": request.case.case_id,
        "mach": flow["mach"],
        "alpha": flow["alpha"],
        "T": float(mapping.get("temperature", mapping.get("T", 300.0))),
        "areaRef": float(mapping.get("area_ref", mapping.get("areaRef", 1.0))),
        "chordRef": float(mapping.get("chord_ref", mapping.get("chordRef", 1.0))),
        "xRef": float(mapping.get("x_ref", mapping.get("xRef", 0.0))),
        "yRef": float(mapping.get("y_ref", mapping.get("yRef", 0.0))),
        "zRef": float(mapping.get("z_ref", mapping.get("zRef", 0.0))),
        "evalFuncs": ["cl", "cd", "cmz", "cdp", "cdv"],
    }
    if mapping.get("reference_state_mode") == "dataset_unified":
        kwargs["P"] = float(mapping.get("pressure", mapping.get("P", 101325.0)))
    else:
        kwargs["reynolds"] = flow["reynolds"]
        kwargs["reynoldsLength"] = float(mapping.get("reynolds_length", 1.0))
    return kwargs


def _load_default_solver_factory() -> SolverFactory:
    ensure_adflow_runtime_on_path()
    try:
        module = importlib.import_module("adflow")
    except Exception as exc:  # pragma: no cover - depends on external install.
        raise NotMigratedError(
            "ADflow runtime dependency",
            detail="Install/import adflow or pass solver_factory to ADflowBackend.",
        ) from exc
    try:
        return module.ADFLOW
    except AttributeError as exc:  # pragma: no cover - depends on external install.
        raise NotMigratedError(
            "ADflow runtime dependency",
            detail="adflow.ADFLOW is not available.",
        ) from exc


def _load_default_aero_problem_factory() -> AeroProblemFactory:
    ensure_adflow_runtime_on_path()
    try:
        module = importlib.import_module("baseclasses")
    except Exception as exc:  # pragma: no cover - depends on external install.
        raise NotMigratedError(
            "AeroProblem runtime dependency",
            detail="Install/import baseclasses or pass aero_problem_factory.",
        ) from exc
    try:
        return module.AeroProblem
    except AttributeError as exc:  # pragma: no cover - depends on external install.
        raise NotMigratedError(
            "AeroProblem runtime dependency",
            detail="baseclasses.AeroProblem is not available.",
        ) from exc


def _make_solver(factory: SolverFactory, options: Mapping[str, Any], comm: Any) -> Any:
    try:
        return factory(options=dict(options), comm=comm, debug=False)
    except TypeError:
        try:
            return factory(options=dict(options), comm=comm)
        except TypeError:
            return factory(options=dict(options))


def _set_aero_problem(solver: Any, aero_problem: Any) -> None:
    setter = getattr(solver, "setAeroProblem", None)
    if callable(setter):
        try:
            setter(aero_problem, releaseAdjointMemory=True)
        except TypeError:
            setter(aero_problem)


def _call_state_injector(
    injector: StateInjector,
    *,
    solver: Any,
    request: ProjectionRequest,
    aero_problem: Any,
    comm: Any,
) -> dict[str, Any]:
    payload = injector(
        solver=solver,
        case=request.case,
        aero_problem=aero_problem,
        field=request.case.prediction.field,
        comm=comm,
    )
    return _metadata(payload)


def _extract_field(
    extractor: StateExtractor | None,
    *,
    solver: Any,
    request: ProjectionRequest,
    aero_problem: Any,
    comm: Any,
) -> Any | None:
    if extractor is None:
        return None
    return extractor(
        solver=solver,
        case=request.case,
        aero_problem=aero_problem,
        comm=comm,
    )


def _call_solver_once(solver: Any, aero_problem: Any) -> None:
    try:
        solver(aero_problem, writeSolution=False, releaseAdjointMemory=False)
    except TypeError:
        try:
            solver(aero_problem, writeSolution=False, relaseAdjointMemory=False)
        except TypeError:
            solver(aero_problem)


def _clear_root_changed_options(solver: Any) -> None:
    try:
        solver.rootChangedOptions = {}
    except (AttributeError, TypeError):
        solver.__dict__["rootChangedOptions"] = {}
    else:
        solver.__dict__["rootChangedOptions"] = {}
    normalized = getattr(solver, "rootChangedOptions", None)
    if not hasattr(normalized, "items"):
        solver.__dict__["rootChangedOptions"] = {}
        normalized = getattr(solver, "rootChangedOptions", None)
    if not hasattr(normalized, "items"):
        raise ContractError(
            "ADflow solver.rootChangedOptions could not be normalized to a mapping"
        )


def _field_force_coefficients(
    request: ProjectionRequest,
    post_field: Any | None,
    *,
    calculator: ForceCoefficientCalculator | None,
) -> dict[str, float]:
    if post_field is None:
        raise ContractError("force metrics require an extracted post_field")
    if request.case.geometry.coords_vertex is None:
        raise ContractError("force metrics require GeometryContext.coords_vertex")

    flow_conditions: Any = request.case.solver_context.flow_conditions_dict
    if not flow_conditions:
        flow_conditions = request.case.solver_context.flow_conditions
    compute_coefficients = calculator or compute_field_force_coefficients
    raw = compute_coefficients(
        np.asarray(post_field, dtype=np.float64),
        np.asarray(request.case.geometry.coords_vertex, dtype=np.float64),
        flow_conditions,
    )
    out: dict[str, float] = {}
    for key, value in dict(raw or {}).items():
        scalar = _finite_float(value)
        if scalar is not None:
            out[str(key)] = scalar
    if not out:
        raise ContractError("force coefficient computation returned no numeric values")
    return out


def _solver_force_coefficients(
    solver: Any,
    aero_problem: Any,
    comm: Any,
) -> dict[str, float]:
    """Read the corrected-state coefficients from ADflow itself."""

    evaluate = getattr(solver, "evalFunctions", None)
    if not callable(evaluate):
        return {}
    raw: dict[str, Any] = {}
    evaluate(aero_problem, raw)
    candidate: dict[str, float] = {}
    if _comm_rank(comm) == 0:
        name = str(getattr(aero_problem, "name", ""))
        aliases = {
            "cl": (f"{name}_cl", "cl"),
            "cd": (f"{name}_cd", "cd"),
            "cm": (f"{name}_cmz", f"{name}_cm", "cmz", "cm"),
            "cdp": (f"{name}_cdp", "cdp"),
            "cdv": (f"{name}_cdv", "cdv"),
        }
        for target, keys in aliases.items():
            for key in keys:
                value = _finite_float(raw.get(key))
                if value is not None:
                    candidate[target] = value
                    break
    return normalize_force_coefficients(_comm_bcast(comm, candidate, root=0))


def _refresh_solver_force_state(solver: Any, aero_problem: Any) -> None:
    """Refresh ADFLOW derived wall quantities without advancing the state."""
    if not callable(getattr(solver, "evalFunctions", None)):
        return
    get_residual = getattr(solver, "getResidual", None)
    if not callable(get_residual):
        raise ContractError(
            "ADflow native force evaluation requires getResidual to refresh wall traction"
        )
    get_residual(aero_problem)


def _residual_snapshot(solver: Any) -> tuple[dict[str, Any], tuple[float, ...]]:
    getter = getattr(solver, "getResNorms", None)
    if not callable(getter):
        raise ContractError("ADflow solver does not expose getResNorms")
    values = _numeric_series(getter(), name="ADflow getResNorms")
    if len(values) < 3:
        raise ContractError(
            "ADflow getResNorms must expose totalR0, totalRStart, and totalRFinal"
        )
    series = values[-2:]
    payload = {
        "source": "adflow_getResNorms",
        "reference_totalr0": float(values[0]),
        "values": list(series),
    }
    return payload, series


def _residual_snapshot_mpi(solver: Any, comm: Any) -> tuple[dict[str, Any], tuple[float, ...]]:
    rank = _comm_rank(comm)
    size = _comm_size(comm)
    if size <= 1:
        return _residual_snapshot(solver)
    if rank == 0:
        payload, series = _residual_snapshot(solver)
    else:
        payload = None
        series = None
    broadcaster = getattr(comm, "bcast", None)
    if not callable(broadcaster):
        raise ContractError("distributed residual snapshots require MPI bcast")
    payload = broadcaster(payload, root=0)
    series = broadcaster(series, root=0)
    if not isinstance(payload, Mapping):
        raise ContractError("broadcast ADflow residual payload must be a mapping")
    residual_series = _numeric_series(series, name="broadcast ADflow residual series")
    if not residual_series:
        raise ContractError("broadcast ADflow residual series is empty")
    return _metadata(payload), residual_series


def _nk_residual_contract(
    cycle_residuals: Sequence[Mapping[str, Any]],
    *,
    requested_cycles: int,
    reference_totalr0: float | None,
) -> dict[str, Any]:
    if not cycle_residuals:
        return {}
    residual_pre = float(cycle_residuals[0]["start_totalr"])
    residual_post = float(cycle_residuals[-1]["final_totalr"])
    if not math.isfinite(residual_pre) or residual_pre <= 0.0:
        raise ContractError("NK entry residual must be finite and positive")
    if not math.isfinite(residual_post) or residual_post < 0.0:
        raise ContractError("NK terminal residual must be finite and non-negative")
    payload: dict[str, Any] = {
        "schema_version": "adflow_nk_residual_v1",
        "source": "adflow_getResNorms_per_solver_call",
        "requested_cycles": int(requested_cycles),
        "executed_cycles": len(cycle_residuals),
        "pre_nk_totalr": residual_pre,
        "post_nk_totalr": residual_post,
        "post_over_pre": float(residual_post / residual_pre),
        "cycles": [_metadata(value) for value in cycle_residuals],
    }
    if reference_totalr0 is not None:
        payload["reference_totalr0"] = float(reference_totalr0)
        payload["post_over_reference_totalr0"] = float(
            residual_post / reference_totalr0
        )
    return payload


def _write_field(path: Path, field_value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(field_value))
    return str(path)


def _budget_metric_record(
    metrics: Mapping[str, Any],
    *,
    budget: int,
    delta_cycles: int,
    residual_totalr: float,
    residual_ratio: float | None,
    solver_wall_sec: float,
    solver_wall_cumulative_sec: float,
    diagnostic_wall_sec: float,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "budget": int(budget),
        "delta_cycles": int(delta_cycles),
        "residual_totalr": float(residual_totalr),
        "solver_wall_sec": float(solver_wall_sec),
        "solver_wall_cumulative_sec": float(solver_wall_cumulative_sec),
        "diagnostic_wall_sec": float(diagnostic_wall_sec),
    }
    if residual_ratio is not None:
        record["residual_ratio"] = float(residual_ratio)
    force_coefficients = metrics.get("force_coefficients")
    if isinstance(force_coefficients, Mapping):
        record["force_coefficients"] = _metadata(force_coefficients)
    force = metrics.get("force")
    if isinstance(force, Mapping):
        abs_delta = force.get("abs_delta")
        if isinstance(abs_delta, Mapping):
            record["force_abs_delta"] = _metadata(abs_delta)
    field_value = metrics.get("field")
    if isinstance(field_value, Mapping):
        candidate_vs_reference = field_value.get("candidate_vs_reference")
        if isinstance(candidate_vs_reference, Mapping):
            summary = candidate_vs_reference.get("summary")
            if isinstance(summary, Mapping):
                record["field_vs_reference"] = _metadata(summary)
        candidate_vs_initial = field_value.get("candidate_vs_initial")
        if isinstance(candidate_vs_initial, Mapping):
            summary = candidate_vs_initial.get("summary")
            if isinstance(summary, Mapping):
                record["field_vs_initial"] = _metadata(summary)
    return record


@dataclass
class ADflowBackend:
    """Execute one clean projection request through ADflow when dependencies exist."""

    solver_factory: SolverFactory | None = None
    aero_problem_factory: AeroProblemFactory | None = None
    state_adapter: ADflowStateAdapter | None = field(default_factory=ADflowStateAdapter)
    state_injector: StateInjector | None = None
    state_extractor: StateExtractor | None = None
    force_coefficient_calculator: ForceCoefficientCalculator | None = None
    comm: Any = None
    print_iterations: bool = False
    write_result: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    _active_solver: Any = field(default=None, init=False, repr=False)
    _active_solver_key: str = field(default="", init=False, repr=False)
    _active_solver_grid_path: str = field(default="", init=False, repr=False)

    name = "adflow"

    def _solver_factory(self) -> SolverFactory:
        return self.solver_factory or _load_default_solver_factory()

    def _aero_problem_factory(self) -> AeroProblemFactory:
        return self.aero_problem_factory or _load_default_aero_problem_factory()

    def _comm(self) -> Any:
        return self.comm if self.comm is not None else _default_mpi_comm()

    def _state_injector(self) -> StateInjector | None:
        if self.state_injector is not None:
            return self.state_injector
        if self.state_adapter is not None:
            return self.state_adapter.inject
        return None

    def _state_extractor(self) -> StateExtractor | None:
        if self.state_extractor is not None:
            return self.state_extractor
        if self.state_adapter is not None:
            return self.state_adapter.extract
        return None

    def _build_aero_problem(self, request: ProjectionRequest) -> Any:
        return self._aero_problem_factory()(**_aero_problem_kwargs(request))

    def _solver_key(
        self,
        cgns_basename: str,
        cgns_path: str,
        options: Mapping[str, Any],
    ) -> str:
        stable_options = {
            str(key): value
            for key, value in options.items()
            if str(key) not in {"gridFile", "outputDirectory"}
        }
        return json.dumps(
            [
                str(Path(cgns_path).resolve().parent),
                cgns_geometry_key(cgns_basename),
                _jsonable(stable_options),
            ],
            sort_keys=True,
            separators=(",", ":"),
        )

    def _get_or_create_solver(
        self,
        *,
        cgns_basename: str,
        cgns_path: str,
        options: Mapping[str, Any],
        comm: Any,
    ) -> tuple[Any, bool, float]:
        key = self._solver_key(cgns_basename, cgns_path, options)
        if self._active_solver is not None and self._active_solver_key == key:
            return self._active_solver, True, 0.0
        if self._active_solver is not None:
            previous_solver = self._active_solver
            self._active_solver = None
            self._active_solver_key = ""
            self._active_solver_grid_path = ""
            del previous_solver
        t0 = time.perf_counter()
        solver = _make_solver(self._solver_factory(), options, comm)
        ctor_wall_sec = float(time.perf_counter() - t0)
        self._active_solver = solver
        self._active_solver_key = key
        self._active_solver_grid_path = str(cgns_path)
        return solver, False, ctor_wall_sec

    def _request_solver_options(
        self, request: ProjectionRequest
    ) -> tuple[StagePlan, Any, dict[str, Any]]:
        stage = _stage_for_request(request)
        cgns_ref = resolve_cgns_ref(
            request.case.solver_context.cgns_root,
            request.case.solver_context.cgns_basename,
            require_exists=True,
        )
        if stage.work.solver_preset == SolverPreset.NONE:
            option_cycles = 0
        elif stage.work.solver_preset == SolverPreset.PROD:
            option_cycles = max(_cycle_budgets(stage))
        else:
            option_cycles = 1
        options = build_adflow_options_for_stage(
            stage,
            cgns_path=cgns_ref.path,
            output_dir=str(Path(request.output_dir) / stage.name),
            options_version=request.case.solver_context.options_version,
            l2conv=request.case.solver_context.l2conv,
            cycles=option_cycles,
            print_iterations=self.print_iterations,
        )
        return stage, cgns_ref, options

    def prepare(self, request: ProjectionRequest) -> dict[str, Any]:
        """Construct the geometry-matched solver before a warm worker is ready."""

        stage, cgns_ref, options = self._request_solver_options(request)
        solver, reused, ctor_wall_sec = self._get_or_create_solver(
            cgns_basename=cgns_ref.basename,
            cgns_path=cgns_ref.path,
            options=options,
            comm=self._comm(),
        )
        del solver
        return {
            "stage": stage.name,
            "geometry_key": cgns_geometry_key(cgns_ref.basename),
            "solver_grid_path": self._active_solver_grid_path,
            "solver_reused": bool(reused),
            "solver_ctor_wall_sec": float(ctor_wall_sec),
        }

    def _execute_budget(
        self,
        *,
        solver: Any,
        aero_problem: Any,
        delta_cycles: int,
        comm: Any,
        single_call: bool = False,
    ) -> tuple[float, list[dict[str, Any]]]:
        if delta_cycles <= 0:
            return 0.0, []
        solver_wall_sec = 0.0
        cycle_residuals: list[dict[str, Any]] = []
        call_cycles = (int(delta_cycles),) if single_call else range(1, int(delta_cycles) + 1)
        for cycle_index in call_cycles:
            _clear_root_changed_options(solver)
            t0 = time.perf_counter()
            _call_solver_once(solver, aero_problem)
            solver_wall_sec += float(time.perf_counter() - t0)
            _, series = _residual_snapshot_mpi(solver, comm)
            if len(series) < 2:
                raise ContractError(
                    "ADflow getResNorms must expose start and final residuals"
                )
            cycle_residuals.append(
                {
                    "cycle": int(cycle_index),
                    "start_totalr": float(series[0]),
                    "final_totalr": float(series[-1]),
                }
            )
        return solver_wall_sec, cycle_residuals

    def _stage_metrics(
        self,
        *,
        request: ProjectionRequest,
        post_field: Any | None,
        residual_values: tuple[float, ...],
        residual_ratio_values: tuple[float, ...] = (),
        residual_budgets: tuple[int, ...] = (),
        solver_force: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        candidate_force = normalize_force_coefficients(solver_force)
        force_source = "adflow_evalFunctions"
        if (
            not candidate_force
            and post_field is not None
            and request.case.geometry.coords_vertex is not None
        ):
            candidate_force = normalize_force_coefficients(
                _field_force_coefficients(
                    request,
                    post_field,
                    calculator=self.force_coefficient_calculator,
                )
            )
            force_source = "post_field"
        if candidate_force:
            metrics["force_coefficients"] = candidate_force
        if request.case.ground_truth.force_coefficients:
            reference_force = normalize_force_coefficients(
                request.case.ground_truth.force_coefficients
            )
            missing_force = sorted(set(reference_force) - set(candidate_force))
            if missing_force:
                raise ContractError(
                    "force coefficient computation is missing reference keys: "
                    + ", ".join(missing_force)
                )
            metrics["force"] = force_metrics(
                candidate_force,
                reference_force,
                metadata={"source": force_source},
            )
        metric_residual_values = residual_ratio_values or residual_values
        if metric_residual_values:
            budgets = residual_budgets or tuple(range(len(metric_residual_values)))
            metrics["residual"] = residual_metrics(
                metric_residual_values,
                budgets=budgets,
                threshold=request.case.solver_context.l2conv,
                metadata={
                    "source": "adflow_getResNorms",
                    "value_kind": "ratio_to_reference_totalr0"
                    if residual_ratio_values
                    else "raw_totalr",
                },
            )
            metrics["trajectory"] = trajectory_summary(
                metric_residual_values,
                budgets=budgets,
                thresholds=(request.case.solver_context.l2conv,),
            )
        if post_field is not None:
            reference = request.case.ground_truth.field
            if reference is not None:
                metrics["field"] = field_metrics(
                    post_field,
                    reference,
                    initial=request.case.prediction.field,
                    metadata={"source": "adflow_state_extractor"},
                )
        return metrics

    def project(self, request: ProjectionRequest) -> ProjectionResult:
        stage, cgns_ref, adflow_options = self._request_solver_options(request)
        budgets = _cycle_budgets(stage)
        thresholds = _thresholds(stage)
        output_root = Path(request.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        stage_dir = output_root / stage.name
        stage_dir.mkdir(parents=True, exist_ok=True)

        solver_preset = stage.work.solver_preset
        if solver_preset == SolverPreset.PROD:
            budgets = (max(budgets),)
            thresholds = (thresholds[-1],)
        state_injector = self._state_injector()
        state_extractor = self._state_extractor()
        if solver_preset != SolverPreset.NONE and state_injector is None:
            raise NotMigratedError(
                "ADflow state injection",
                detail="Pass a clean state_injector before executing ADflow/NK resume.",
            )

        comm = self._comm()
        rank = _comm_rank(comm)
        size = _comm_size(comm)
        t_total = time.perf_counter()
        phase_timing: dict[str, float] = {}
        solver, solver_reused, solver_ctor_wall_sec = self._get_or_create_solver(
            cgns_basename=cgns_ref.basename,
            cgns_path=cgns_ref.path,
            options=adflow_options,
            comm=comm,
        )
        phase_timing["solver_ctor_wall_sec"] = float(solver_ctor_wall_sec)
        t_phase = time.perf_counter()
        aero_problem = self._build_aero_problem(request)
        phase_timing["aero_problem_ctor_wall_sec"] = float(time.perf_counter() - t_phase)
        t_phase = time.perf_counter()
        _set_aero_problem(solver, aero_problem)
        phase_timing["set_aero_problem_wall_sec"] = float(time.perf_counter() - t_phase)

        inject_metadata: dict[str, Any] = {}
        if state_injector is not None:
            t_phase = time.perf_counter()
            inject_metadata = _call_state_injector(
                state_injector,
                solver=solver,
                request=request,
                aero_problem=aero_problem,
                comm=comm,
            )
            phase_timing["state_injection_wall_sec"] = float(time.perf_counter() - t_phase)
        else:
            phase_timing["state_injection_wall_sec"] = 0.0

        residual_values: list[float] = []
        residual_ratio_values: list[float] = []
        cycle_residuals: list[dict[str, Any]] = []
        budget_values: list[int] = []
        budget_metrics: list[dict[str, Any]] = []
        cumulative = 0
        solver_wall_sec = 0.0
        budget_diagnostic_wall_sec = 0.0
        state_extraction_wall_sec = 0.0
        force_evaluation_wall_sec = 0.0
        stopped_at_budget: int | None = None
        residual_snapshot: dict[str, Any] = {}
        reference_totalr0 = _residual_reference_totalr(request)
        post_field = None
        solver_force: dict[str, float] = {}
        for index, budget in enumerate(budgets):
            delta = int(budget) - cumulative
            if delta < 0:
                raise ContractError("ADflow budgets must be monotone")
            delta_solver_wall_sec = 0.0
            if solver_preset != SolverPreset.NONE:
                delta_solver_wall_sec, delta_cycle_residuals = self._execute_budget(
                    solver=solver,
                    aero_problem=aero_problem,
                    delta_cycles=delta,
                    comm=comm,
                    single_call=solver_preset == SolverPreset.PROD,
                )
                cycle_offset = len(cycle_residuals)
                for value in delta_cycle_residuals:
                    value["cycle"] = int(value["cycle"]) + cycle_offset
                cycle_residuals.extend(delta_cycle_residuals)
                solver_wall_sec += delta_solver_wall_sec
            cumulative = int(budget)
            if solver_preset == SolverPreset.NONE:
                continue
            residual_snapshot, series = _residual_snapshot_mpi(solver, comm)
            if reference_totalr0 is None:
                reference_totalr0 = float(
                    residual_snapshot["reference_totalr0"]
                )
            current_totalr = float(series[-1])
            residual_values.append(current_totalr)
            if reference_totalr0 is not None:
                residual_ratio_values.append(current_totalr / reference_totalr0)
            budget_values.append(cumulative)
            t_diagnostic = time.perf_counter()
            t_snapshot = time.perf_counter()
            post_field = _extract_field(
                state_extractor,
                solver=solver,
                request=request,
                aero_problem=aero_problem,
                comm=comm,
            )
            state_extraction_wall_sec += float(time.perf_counter() - t_snapshot)
            t_snapshot = time.perf_counter()
            solver_force = _solver_force_coefficients(solver, aero_problem, comm)
            force_evaluation_wall_sec += float(time.perf_counter() - t_snapshot)
            snapshot_metrics = self._stage_metrics(
                request=request,
                post_field=post_field,
                residual_values=(current_totalr,),
                residual_ratio_values=(residual_ratio_values[-1],)
                if residual_ratio_values
                else (),
                residual_budgets=(cumulative,),
                solver_force=solver_force,
            )
            diagnostic_wall_sec = float(time.perf_counter() - t_diagnostic)
            budget_diagnostic_wall_sec += diagnostic_wall_sec
            budget_metrics.append(
                _budget_metric_record(
                    snapshot_metrics,
                    budget=cumulative,
                    delta_cycles=delta,
                    residual_totalr=current_totalr,
                    residual_ratio=residual_ratio_values[-1]
                    if residual_ratio_values
                    else None,
                    solver_wall_sec=delta_solver_wall_sec,
                    solver_wall_cumulative_sec=solver_wall_sec,
                    diagnostic_wall_sec=diagnostic_wall_sec,
                )
            )
            threshold = thresholds[index] if index < len(thresholds) else None
            threshold_series = residual_ratio_values if reference_totalr0 is not None else residual_values
            if threshold is not None and threshold_series and threshold_series[-1] <= threshold:
                stopped_at_budget = cumulative
                break

        if solver_preset == SolverPreset.NONE:
            t_phase = time.perf_counter()
            _refresh_solver_force_state(solver, aero_problem)
            post_field = _extract_field(
                state_extractor,
                solver=solver,
                request=request,
                aero_problem=aero_problem,
                comm=comm,
            )
            state_extraction_wall_sec += float(time.perf_counter() - t_phase)
            t_phase = time.perf_counter()
            solver_force = _solver_force_coefficients(solver, aero_problem, comm)
            force_evaluation_wall_sec += float(time.perf_counter() - t_phase)
        phase_timing["state_extraction_wall_sec"] = float(state_extraction_wall_sec)
        phase_timing["force_evaluation_wall_sec"] = float(force_evaluation_wall_sec)
        phase_timing["budget_diagnostic_wall_sec"] = float(budget_diagnostic_wall_sec)
        output_paths: dict[str, str] = {}
        if post_field is not None and rank == 0:
            t_phase = time.perf_counter()
            output_paths["post_field"] = _write_field(stage_dir / "post_field.npy", post_field)
            phase_timing["post_field_write_wall_sec"] = float(time.perf_counter() - t_phase)
        elif post_field is not None:
            output_paths["post_field"] = str(stage_dir / "post_field.npy")
            phase_timing["post_field_write_wall_sec"] = 0.0
        else:
            phase_timing["post_field_write_wall_sec"] = 0.0
        t_phase = time.perf_counter()
        _comm_barrier(comm)
        phase_timing["post_field_barrier_wall_sec"] = float(time.perf_counter() - t_phase)

        t_phase = time.perf_counter()
        metrics = self._stage_metrics(
            request=request,
            post_field=post_field,
            residual_values=tuple(residual_values),
            residual_ratio_values=tuple(residual_ratio_values),
            residual_budgets=tuple(budget_values),
            solver_force=solver_force,
        )
        nk_residual = (
            _nk_residual_contract(
                cycle_residuals,
                requested_cycles=max(budgets, default=0),
                reference_totalr0=reference_totalr0,
            )
            if solver_preset == SolverPreset.NK
            else {}
        )
        if nk_residual:
            metrics["nk_residual_contract"] = nk_residual
        phase_timing["metrics_wall_sec"] = float(time.perf_counter() - t_phase)
        if residual_values:
            residual_payload: dict[str, Any] = {
                "values": list(residual_values),
                "budgets": list(budget_values),
                "stopped_at_budget": stopped_at_budget,
            }
            if reference_totalr0 is not None:
                residual_payload["reference_totalr0"] = reference_totalr0
                residual_payload["ratio_values"] = list(residual_ratio_values)
            metrics["adflow_residual_by_budget"] = residual_payload
            metrics["terminal_metrics_by_budget"] = budget_metrics

        stage_result = ProjectionStageResult(
            name=stage.name,
            source_state=stage.source_state,
            status="ok",
            metrics=metrics,
            solver_options={
                "adflow": adflow_options,
                "stage": stage.to_dict(),
            },
            timing={
                "solver_wall_sec": solver_wall_sec,
                "total_wall_sec": float(time.perf_counter() - t_total),
                **phase_timing,
            },
            output_paths=output_paths,
            metadata={
                "backend": self.name,
                "cgns_ref": cgns_ref.to_dict(),
                "injector": inject_metadata,
                "residual_snapshot": residual_snapshot,
                "budgets": list(budgets),
                "thresholds": list(thresholds),
                "stopped_at_budget": stopped_at_budget,
                "reference_totalr0": reference_totalr0,
                "nk_residual_contract": nk_residual,
                "solver_geometry_key": cgns_geometry_key(cgns_ref.basename),
                "solver_grid_path": self._active_solver_grid_path,
                "solver_reused": bool(solver_reused),
                "mpi": {
                    "rank": rank,
                    "size": size,
                    "result_writer": rank == 0,
                },
                **_metadata(self.metadata),
            },
        )
        requested_result_path = str(request.metadata.get("result_path") or "").strip()
        result_path = requested_result_path or str(
            output_root / f"{request.case.case_id}.{stage.name}.result.json"
        )
        result = ProjectionResult(
            case_id=request.case.case_id,
            stages=(stage_result,),
            status="ok",
            result_path=result_path,
            metadata={
                "backend": self.name,
                "mode": "single_case",
            },
        )
        if self.write_result and rank == 0:
            t_phase = time.perf_counter()
            write_projection_result(result, result_path)
            phase_timing["result_json_write_wall_sec"] = float(time.perf_counter() - t_phase)
            stage_result.timing["result_json_write_wall_sec"] = phase_timing[
                "result_json_write_wall_sec"
            ]
        elif self.write_result:
            stage_result.timing["result_json_write_wall_sec"] = 0.0
        t_phase = time.perf_counter()
        _comm_barrier(comm)
        stage_result.timing["result_barrier_wall_sec"] = float(time.perf_counter() - t_phase)
        stage_result.timing["total_wall_sec"] = float(time.perf_counter() - t_total)
        return result
