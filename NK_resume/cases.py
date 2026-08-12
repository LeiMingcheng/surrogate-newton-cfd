"""Canonical case builders for the clean NK_resume contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .exceptions import ContractError
from .plans import PredictorKind
from .schema import (
    ArrayLike,
    FixedLiftContext,
    GeometryContext,
    GroundTruth,
    ModelInputs,
    PredictionState,
    ResumeCase,
    RuntimeState,
    SolverContext,
)


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return {str(key): item for key, item in dict(value or {}).items()}


def _path_text(value: str | Path | None) -> str:
    if value is None:
        return ""
    return str(value)


def _int_tuple(values: Sequence[int] | None) -> tuple[int, ...]:
    if values is None:
        return ()
    try:
        return tuple(int(value) for value in values)
    except TypeError as exc:
        raise ContractError("custom_timesteps must be a sequence") from exc


def _float_tuple(values: Sequence[float] | None) -> tuple[float, ...]:
    if values is None:
        return ()
    try:
        return tuple(float(value) for value in values)
    except TypeError as exc:
        raise ContractError("flow_conditions must be a sequence") from exc


def _predictor_kind(value: PredictorKind | str) -> str:
    if isinstance(value, PredictorKind):
        return value.value
    try:
        return PredictorKind(str(value).strip().lower()).value
    except ValueError as exc:
        allowed = ", ".join(kind.value for kind in PredictorKind)
        raise ContractError(f"predictor_kind must be one of: {allowed}") from exc


def validate_case(case: ResumeCase) -> ResumeCase:
    """Return `case` after confirming it is already canonical."""

    if not isinstance(case, ResumeCase):
        raise TypeError(f"Expected ResumeCase, got {type(case)!r}")
    return case


def build_resume_case(
    *,
    case_id: str,
    predictor_kind: PredictorKind | str,
    cgns_basename: str,
    prediction_field: ArrayLike,
    state_name: str = "final",
    step_index: int | None = None,
    cgns_root: str | Path = "",
    flow_conditions: Sequence[float] | None = None,
    flow_conditions_dict: Mapping[str, Any] | None = None,
    source_info: Mapping[str, Any] | None = None,
    wall_layers: int | None = None,
    options_version: int = 2,
    l2conv: float = 1.0e-8,
    ranks_per_case: int = 1,
    mpi_launcher: str = "auto",
    mpi_omp_threads: int = 1,
    geometry_bundle_path: str | Path = "",
    fixed_lift: FixedLiftContext | Mapping[str, Any] | None = None,
    ground_truth_field: ArrayLike | None = None,
    coords_center: ArrayLike | None = None,
    coords_vertex: ArrayLike | None = None,
    force_coefficients: Mapping[str, Any] | None = None,
    residual_reference: Mapping[str, Any] | None = None,
    output_dir: str | Path = "",
    ordinal: int | None = None,
    dataset_index: int | None = None,
    created_by: str = "NK_resume",
    config_path: str | Path = "",
    checkpoint_path: str | Path = "",
    stats_path: str | Path = "",
    device: str = "",
    inference_steps: int | None = None,
    custom_timesteps: Sequence[int] | None = None,
    model_metadata: Mapping[str, Any] | None = None,
    solver_metadata: Mapping[str, Any] | None = None,
    ground_truth_metadata: Mapping[str, Any] | None = None,
    prediction_metadata: Mapping[str, Any] | None = None,
    runtime_metadata: Mapping[str, Any] | None = None,
) -> ResumeCase:
    """Build one canonical case from explicit predictor and solver fields."""

    if prediction_field is None:
        raise ContractError("prediction_field is required")

    return ResumeCase(
        model_inputs=ModelInputs(
            predictor_kind=_predictor_kind(predictor_kind),
            config_path=_path_text(config_path),
            checkpoint_path=_path_text(checkpoint_path),
            stats_path=_path_text(stats_path),
            device=str(device),
            inference_steps=inference_steps,
            custom_timesteps=_int_tuple(custom_timesteps),
            metadata=_metadata(model_metadata),
        ),
        solver_context=SolverContext(
            cgns_basename=cgns_basename,
            cgns_root=_path_text(cgns_root),
            flow_conditions=_float_tuple(flow_conditions),
            flow_conditions_dict=_metadata(flow_conditions_dict),
            source_info=_metadata(source_info),
            wall_layers=wall_layers,
            options_version=options_version,
            l2conv=l2conv,
            ranks_per_case=ranks_per_case,
            mpi_launcher=str(mpi_launcher),
            mpi_omp_threads=mpi_omp_threads,
            geometry_bundle_path=_path_text(geometry_bundle_path),
            fixed_lift=(
                None
                if fixed_lift is None
                else fixed_lift
                if isinstance(fixed_lift, FixedLiftContext)
                else FixedLiftContext(**dict(fixed_lift))
            ),
            metadata=_metadata(solver_metadata),
        ),
        ground_truth=GroundTruth(
            field=ground_truth_field,
            force_coefficients=_metadata(force_coefficients),
            residual_reference=_metadata(residual_reference),
            metadata=_metadata(ground_truth_metadata),
        ),
        prediction=PredictionState(
            field=prediction_field,
            state_name=state_name,
            step_index=step_index,
            metadata=_metadata(prediction_metadata),
        ),
        runtime=RuntimeState(
            case_id=case_id,
            output_dir=_path_text(output_dir),
            ordinal=ordinal,
            dataset_index=dataset_index,
            created_by=str(created_by),
            metadata=_metadata(runtime_metadata),
        ),
        geometry=GeometryContext(
            coords_center=coords_center,
            coords_vertex=coords_vertex,
        ),
    )


def build_direct_case(**fields: Any) -> ResumeCase:
    """Build a final-only direct-surrogate case."""

    return build_resume_case(predictor_kind=PredictorKind.DIRECT, **fields)


def build_fsb_case(**fields: Any) -> ResumeCase:
    """Build an FSB case for an alternating transition or the final state."""

    return build_resume_case(predictor_kind=PredictorKind.FSB, **fields)
