"""Canonical data contract for NK_resume.

This module is deliberately independent from historical runtime packages.  It
defines the data boundary that new migration work must target.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .exceptions import ContractError


ArrayLike = Any
JsonDict = dict[str, Any]


def _as_tuple(values: Sequence[Any] | None, *, name: str) -> tuple[Any, ...]:
    if values is None:
        return ()
    try:
        return tuple(values)
    except TypeError as exc:
        raise ContractError(f"{name} must be a sequence") from exc


def _as_metadata(value: Mapping[str, Any] | None) -> JsonDict:
    return {str(key): item for key, item in dict(value or {}).items()}


def _path_text(value: str | Path | None) -> str:
    if value is None:
        return ""
    return str(value)


@dataclass(frozen=True)
class FixedLiftContext:
    """Native ADFLOW solveCL controls for one resume case."""

    target_cl: float
    cl_tolerance: float
    max_aoa_solves: int
    cl_alpha_guess: float = 0.1
    delta_alpha: float = 0.5
    total_time_limit_s: float = 30.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_cl", float(self.target_cl))
        object.__setattr__(self, "cl_tolerance", float(self.cl_tolerance))
        object.__setattr__(self, "max_aoa_solves", int(self.max_aoa_solves))
        object.__setattr__(self, "cl_alpha_guess", float(self.cl_alpha_guess))
        object.__setattr__(self, "delta_alpha", float(self.delta_alpha))
        object.__setattr__(self, "total_time_limit_s", float(self.total_time_limit_s))
        if self.cl_tolerance <= 0.0:
            raise ContractError("FixedLiftContext.cl_tolerance must be positive")
        if self.max_aoa_solves <= 0:
            raise ContractError("FixedLiftContext.max_aoa_solves must be positive")
        if self.cl_alpha_guess == 0.0 or self.delta_alpha == 0.0:
            raise ContractError("FixedLiftContext slope guess and alpha delta must be nonzero")
        if self.total_time_limit_s <= 0.0:
            raise ContractError("FixedLiftContext.total_time_limit_s must be positive")

    def to_dict(self) -> JsonDict:
        return {
            "target_cl": self.target_cl,
            "cl_tolerance": self.cl_tolerance,
            "max_aoa_solves": self.max_aoa_solves,
            "cl_alpha_guess": self.cl_alpha_guess,
            "delta_alpha": self.delta_alpha,
            "total_time_limit_s": self.total_time_limit_s,
        }


@dataclass(frozen=True)
class ModelInputs:
    """Surrogate-side inputs needed to reproduce or describe a prediction."""

    predictor_kind: str
    config_path: str = ""
    checkpoint_path: str = ""
    stats_path: str = ""
    device: str = ""
    inference_steps: int | None = None
    custom_timesteps: tuple[int, ...] = ()
    metadata: JsonDict = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        predictor_kind = str(self.predictor_kind).strip().lower()
        if not predictor_kind:
            raise ContractError("ModelInputs.predictor_kind is required")
        object.__setattr__(self, "predictor_kind", predictor_kind)
        object.__setattr__(self, "config_path", _path_text(self.config_path))
        object.__setattr__(self, "checkpoint_path", _path_text(self.checkpoint_path))
        object.__setattr__(self, "stats_path", _path_text(self.stats_path))
        object.__setattr__(self, "custom_timesteps", tuple(int(v) for v in self.custom_timesteps))
        object.__setattr__(self, "metadata", _as_metadata(self.metadata))
        if self.inference_steps is not None and int(self.inference_steps) < 0:
            raise ContractError("ModelInputs.inference_steps must be non-negative")


@dataclass(frozen=True)
class SolverContext:
    """ADflow/NK-side context independent from surrogate internals."""

    cgns_basename: str
    cgns_root: str = ""
    flow_conditions: tuple[float, ...] = ()
    flow_conditions_dict: JsonDict = dc_field(default_factory=dict)
    source_info: JsonDict = dc_field(default_factory=dict)
    wall_layers: int | None = None
    options_version: int = 2
    l2conv: float = 1.0e-8
    ranks_per_case: int = 1
    mpi_launcher: str = "auto"
    mpi_omp_threads: int = 1
    geometry_bundle_path: str = ""
    fixed_lift: FixedLiftContext | None = None
    metadata: JsonDict = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        cgns_basename = str(self.cgns_basename).strip()
        if not cgns_basename:
            raise ContractError("SolverContext.cgns_basename is required")
        object.__setattr__(self, "cgns_basename", cgns_basename)
        object.__setattr__(self, "cgns_root", _path_text(self.cgns_root))
        object.__setattr__(self, "flow_conditions", tuple(float(v) for v in self.flow_conditions))
        object.__setattr__(self, "flow_conditions_dict", _as_metadata(self.flow_conditions_dict))
        object.__setattr__(self, "source_info", _as_metadata(self.source_info))
        object.__setattr__(self, "options_version", int(self.options_version))
        object.__setattr__(self, "l2conv", float(self.l2conv))
        object.__setattr__(self, "ranks_per_case", int(self.ranks_per_case))
        object.__setattr__(self, "mpi_omp_threads", int(self.mpi_omp_threads))
        object.__setattr__(self, "geometry_bundle_path", _path_text(self.geometry_bundle_path))
        if self.fixed_lift is not None and not isinstance(self.fixed_lift, FixedLiftContext):
            object.__setattr__(self, "fixed_lift", FixedLiftContext(**dict(self.fixed_lift)))
        object.__setattr__(self, "metadata", _as_metadata(self.metadata))
        if self.wall_layers is not None and int(self.wall_layers) <= 0:
            raise ContractError("SolverContext.wall_layers must be positive when set")
        if self.options_version <= 0:
            raise ContractError("SolverContext.options_version must be positive")
        if self.ranks_per_case <= 0:
            raise ContractError("SolverContext.ranks_per_case must be positive")
        if self.mpi_omp_threads <= 0:
            raise ContractError("SolverContext.mpi_omp_threads must be positive")
        if self.l2conv <= 0.0:
            raise ContractError("SolverContext.l2conv must be positive")


@dataclass(frozen=True)
class GeometryContext:
    """Solver-side geometry arrays used by ADflow mapping and force metrics."""

    coords_center: ArrayLike | None = None
    coords_vertex: ArrayLike | None = None
    metadata: JsonDict = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _as_metadata(self.metadata))


@dataclass(frozen=True)
class GroundTruth:
    """Reference data used for metrics and projection validation."""

    field: ArrayLike | None = None
    # Compatibility mirrors. New runtime code should use ResumeCase.geometry.
    coords_center: ArrayLike | None = None
    coords_vertex: ArrayLike | None = None
    force_coefficients: JsonDict = dc_field(default_factory=dict)
    residual_reference: JsonDict = dc_field(default_factory=dict)
    metadata: JsonDict = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "force_coefficients", _as_metadata(self.force_coefficients))
        object.__setattr__(self, "residual_reference", _as_metadata(self.residual_reference))
        object.__setattr__(self, "metadata", _as_metadata(self.metadata))


@dataclass(frozen=True)
class PredictionState:
    """The state that will be injected into ADflow or analyzed as a terminal prediction."""

    field: ArrayLike
    state_name: str = "final"
    step_index: int | None = None
    metadata: JsonDict = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        state_name = str(self.state_name).strip().lower()
        if not state_name:
            raise ContractError("PredictionState.state_name is required")
        object.__setattr__(self, "state_name", state_name)
        if self.step_index is not None and int(self.step_index) < 0:
            raise ContractError("PredictionState.step_index must be non-negative")
        object.__setattr__(self, "metadata", _as_metadata(self.metadata))


@dataclass(frozen=True)
class RuntimeState:
    """Runtime identity and output placement for one resume case."""

    case_id: str
    output_dir: str = ""
    ordinal: int | None = None
    dataset_index: int | None = None
    created_by: str = "NK_resume"
    metadata: JsonDict = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        case_id = str(self.case_id).strip()
        if not case_id:
            raise ContractError("RuntimeState.case_id is required")
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "output_dir", _path_text(self.output_dir))
        if self.ordinal is not None and int(self.ordinal) < 0:
            raise ContractError("RuntimeState.ordinal must be non-negative")
        if self.dataset_index is not None and int(self.dataset_index) < 0:
            raise ContractError("RuntimeState.dataset_index must be non-negative")
        object.__setattr__(self, "metadata", _as_metadata(self.metadata))


@dataclass(frozen=True)
class ResumeCase:
    """Canonical case passed across predictor, payload, replay, and solver layers."""

    model_inputs: ModelInputs
    solver_context: SolverContext
    ground_truth: GroundTruth
    prediction: PredictionState
    runtime: RuntimeState
    geometry: GeometryContext = dc_field(default_factory=GeometryContext)

    def __post_init__(self) -> None:
        geometry = self.geometry
        if not isinstance(geometry, GeometryContext):
            raise TypeError(f"Expected GeometryContext, got {type(geometry)!r}")
        if geometry.coords_center is None and self.ground_truth.coords_center is not None:
            geometry = GeometryContext(
                coords_center=self.ground_truth.coords_center,
                coords_vertex=geometry.coords_vertex,
                metadata=geometry.metadata,
            )
        if geometry.coords_vertex is None and self.ground_truth.coords_vertex is not None:
            geometry = GeometryContext(
                coords_center=geometry.coords_center,
                coords_vertex=self.ground_truth.coords_vertex,
                metadata=geometry.metadata,
            )
        object.__setattr__(self, "geometry", geometry)

        if (
            self.ground_truth.coords_center is None
            and self.ground_truth.coords_vertex is None
            and (geometry.coords_center is not None or geometry.coords_vertex is not None)
        ):
            object.__setattr__(
                self,
                "ground_truth",
                GroundTruth(
                    field=self.ground_truth.field,
                    coords_center=geometry.coords_center,
                    coords_vertex=geometry.coords_vertex,
                    force_coefficients=self.ground_truth.force_coefficients,
                    residual_reference=self.ground_truth.residual_reference,
                    metadata=self.ground_truth.metadata,
                ),
            )

    @property
    def case_id(self) -> str:
        return self.runtime.case_id

    def require_geometry_bundle(self) -> None:
        if not str(self.solver_context.geometry_bundle_path).strip():
            raise ContractError(
                f"ResumeCase {self.case_id} does not reference a geometry bundle"
            )

    def summary(self) -> JsonDict:
        return {
            "case_id": self.case_id,
            "predictor_kind": self.model_inputs.predictor_kind,
            "state_name": self.prediction.state_name,
            "ordinal": self.runtime.ordinal,
            "dataset_index": self.runtime.dataset_index,
            "cgns_basename": self.solver_context.cgns_basename,
            "options_version": self.solver_context.options_version,
            "ranks_per_case": self.solver_context.ranks_per_case,
            "geometry_bundle_path": self.solver_context.geometry_bundle_path,
            "has_coords_center": self.geometry.coords_center is not None,
            "has_coords_vertex": self.geometry.coords_vertex is not None,
        }
