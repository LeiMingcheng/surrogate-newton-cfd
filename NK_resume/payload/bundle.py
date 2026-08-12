"""Geometry-bundle and light payload contract.

This module implements only the clean NK_resume payload boundary.  It does not
read from or call historical `nk_resume` code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..exceptions import ContractError
from ..geometry import cgns_ref_from_solver_context, wall_distance_ref_from_solver_context
from ..schema import (
    GeometryContext,
    GroundTruth,
    ModelInputs,
    PredictionState,
    ResumeCase,
    RuntimeState,
    SolverContext,
)


GEOMETRY_BUNDLE_SCHEMA = "geometry_bundle_v1"
CASE_PAYLOAD_SCHEMA = "case_payload_v1"


def _metadata(value: dict[str, Any] | None) -> dict[str, Any]:
    return {str(k): _to_jsonable(v) for k, v in dict(value or {}).items()}


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _to_jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return _to_jsonable(value.tolist())
        except (TypeError, ValueError):
            pass
    return str(value)


def _output_file(path_text: str, *, name: str) -> Path:
    path = Path(path_text)
    if not str(path).strip():
        raise ContractError(f"{name} is required")
    if path.exists() and path.is_dir():
        raise ContractError(f"{name} must be a file path, got directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def _read_json(path: str | Path, *, expected_schema: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != expected_schema:
        raise ContractError(
            f"Expected {expected_schema}, got {payload.get('schema_version')!r}"
        )
    return payload


def _numeric_array(value: Any, *, name: str) -> np.ndarray:
    if value is None:
        raise ContractError(f"{name} is required")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} cannot be converted to a numeric array") from exc
    if array.size == 0:
        raise ContractError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ContractError(f"{name} must contain only finite values")
    return array


def _field_array(value: Any, *, name: str) -> np.ndarray:
    array = _numeric_array(value, name=name)
    if array.ndim != 3:
        raise ContractError(f"{name} must have shape (C,H,W), got {array.shape}")
    if array.shape[0] not in {4, 5}:
        raise ContractError(
            f"{name} must be channel-first with 4 or 5 channels, got {array.shape}"
        )
    if array.shape[1] <= 0 or array.shape[2] <= 0:
        raise ContractError(f"{name} spatial dimensions must be positive")
    return array


def _coords_center_array(value: Any, *, field_shape: tuple[int, int, int]) -> np.ndarray:
    array = _numeric_array(value, name="coords_center")
    _, height, width = field_shape
    if array.ndim != 3 or array.shape[0] < 2:
        raise ContractError(f"coords_center must have shape (C,H,W), got {array.shape}")
    if tuple(array.shape[-2:]) != (height, width):
        raise ContractError(
            "coords_center spatial shape must match prediction field: "
            f"coords_center={array.shape}, prediction_field={field_shape}"
        )
    return array


def _coords_vertex_array(value: Any, *, field_shape: tuple[int, int, int]) -> np.ndarray:
    array = _numeric_array(value, name="coords_vertex")
    _, height, width = field_shape
    if array.ndim != 3 or array.shape[0] < 2:
        raise ContractError(f"coords_vertex must have shape (C,H+1,W+1), got {array.shape}")
    if tuple(array.shape[-2:]) != (height + 1, width + 1):
        raise ContractError(
            "coords_vertex spatial shape must be one larger than prediction field: "
            f"coords_vertex={array.shape}, prediction_field={field_shape}"
        )
    return array


def _case_arrays(case: ResumeCase) -> tuple[dict[str, np.ndarray], list[str]]:
    prediction = _field_array(case.prediction.field, name="prediction.field")
    arrays: dict[str, np.ndarray] = {"prediction_field": prediction}
    array_keys = ["prediction_field"]
    field_shape = tuple(int(v) for v in prediction.shape)

    if case.ground_truth.field is not None:
        ground_truth = _field_array(case.ground_truth.field, name="ground_truth.field")
        if ground_truth.shape != prediction.shape:
            raise ContractError(
                "ground_truth.field shape must match prediction.field: "
                f"ground_truth={ground_truth.shape}, prediction={prediction.shape}"
            )
        arrays["ground_truth_field"] = ground_truth
        array_keys.append("ground_truth_field")
    if case.geometry.coords_center is not None:
        arrays["coords_center"] = _coords_center_array(
            case.geometry.coords_center,
            field_shape=field_shape,
        )
        array_keys.append("coords_center")
    if case.geometry.coords_vertex is not None:
        arrays["coords_vertex"] = _coords_vertex_array(
            case.geometry.coords_vertex,
            field_shape=field_shape,
        )
        array_keys.append("coords_vertex")
    if case.ground_truth.force_coefficients and "coords_vertex" not in arrays:
        raise ContractError("force coefficients require coords_vertex in the clean payload")
    return arrays, array_keys


def _manifest_array_keys(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    raw = manifest.get("array_keys")
    if not isinstance(raw, list):
        raise ContractError("case payload manifest array_keys must be a list")
    keys = tuple(str(key) for key in raw)
    if "prediction_field" not in keys:
        raise ContractError("case payload manifest array_keys must include prediction_field")
    if len(set(keys)) != len(keys):
        raise ContractError("case payload manifest array_keys must be unique")
    return keys


def _validate_loaded_array_key_set(
    manifest: Mapping[str, Any],
    array_keys: set[str],
) -> set[str]:
    expected_keys = set(_manifest_array_keys(manifest))
    if array_keys != expected_keys:
        raise ContractError(
            "case payload arrays do not match manifest array_keys: "
            f"missing={sorted(expected_keys - array_keys)}, extra={sorted(array_keys - expected_keys)}"
        )
    return expected_keys


def _validate_loaded_arrays(
    manifest: Mapping[str, Any],
    arrays: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    actual_keys = set(str(key) for key in arrays)
    _validate_loaded_array_key_set(manifest, actual_keys)

    out: dict[str, np.ndarray] = {}
    prediction = _field_array(arrays["prediction_field"], name="prediction_field")
    out["prediction_field"] = prediction
    field_shape = tuple(int(v) for v in prediction.shape)

    ground_truth = manifest.get("ground_truth")
    if not isinstance(ground_truth, Mapping):
        raise ContractError("case payload manifest missing ground_truth object")
    geometry_context = manifest.get("geometry_context")
    if not isinstance(geometry_context, Mapping):
        geometry_context = ground_truth
    if bool(ground_truth.get("has_field")) != ("ground_truth_field" in arrays):
        raise ContractError("ground_truth.has_field does not match ground_truth_field array")
    if bool(geometry_context.get("has_coords_center")) != ("coords_center" in arrays):
        raise ContractError("geometry_context.has_coords_center does not match coords_center array")
    if bool(geometry_context.get("has_coords_vertex")) != ("coords_vertex" in arrays):
        raise ContractError("geometry_context.has_coords_vertex does not match coords_vertex array")

    if "ground_truth_field" in arrays:
        gt_field = _field_array(arrays["ground_truth_field"], name="ground_truth_field")
        if gt_field.shape != prediction.shape:
            raise ContractError(
                "ground_truth_field shape must match prediction_field: "
                f"ground_truth={gt_field.shape}, prediction={prediction.shape}"
            )
        out["ground_truth_field"] = gt_field
    if "coords_center" in arrays:
        out["coords_center"] = _coords_center_array(
            arrays["coords_center"],
            field_shape=field_shape,
        )
    if "coords_vertex" in arrays:
        out["coords_vertex"] = _coords_vertex_array(
            arrays["coords_vertex"],
            field_shape=field_shape,
        )
    if dict(ground_truth.get("force_coefficients") or {}) and "coords_vertex" not in out:
        raise ContractError("force coefficients require coords_vertex in the clean payload")
    return out


def _model_inputs_payload(case: ResumeCase) -> dict[str, Any]:
    model = case.model_inputs
    return {
        "predictor_kind": model.predictor_kind,
        "config_path": model.config_path,
        "checkpoint_path": model.checkpoint_path,
        "stats_path": model.stats_path,
        "device": model.device,
        "inference_steps": model.inference_steps,
        "custom_timesteps": list(model.custom_timesteps),
        "metadata": _metadata(model.metadata),
    }


def _solver_context_payload(case: ResumeCase) -> dict[str, Any]:
    solver = case.solver_context
    return {
        "cgns_basename": solver.cgns_basename,
        "cgns_root": solver.cgns_root,
        "flow_conditions": list(solver.flow_conditions),
        "flow_conditions_dict": _metadata(solver.flow_conditions_dict),
        "source_info": _metadata(solver.source_info),
        "wall_layers": solver.wall_layers,
        "options_version": solver.options_version,
        "l2conv": solver.l2conv,
        "ranks_per_case": solver.ranks_per_case,
        "mpi_launcher": solver.mpi_launcher,
        "mpi_omp_threads": solver.mpi_omp_threads,
        "geometry_bundle_path": solver.geometry_bundle_path,
        "fixed_lift": None if solver.fixed_lift is None else solver.fixed_lift.to_dict(),
        "metadata": _metadata(solver.metadata),
    }


def _ground_truth_payload(case: ResumeCase) -> dict[str, Any]:
    gt = case.ground_truth
    return {
        "has_field": gt.field is not None,
        "has_coords_center": case.geometry.coords_center is not None,
        "has_coords_vertex": case.geometry.coords_vertex is not None,
        "force_coefficients": _metadata(gt.force_coefficients),
        "residual_reference": _metadata(gt.residual_reference),
        "metadata": _metadata(gt.metadata),
    }


def _geometry_context_payload(case: ResumeCase) -> dict[str, Any]:
    geometry = case.geometry
    return {
        "has_coords_center": geometry.coords_center is not None,
        "has_coords_vertex": geometry.coords_vertex is not None,
        "metadata": _metadata(geometry.metadata),
    }


def _prediction_payload(case: ResumeCase) -> dict[str, Any]:
    pred = case.prediction
    return {
        "state_name": pred.state_name,
        "step_index": pred.step_index,
        "metadata": _metadata(pred.metadata),
    }


def _runtime_payload(case: ResumeCase) -> dict[str, Any]:
    runtime = case.runtime
    return {
        "case_id": runtime.case_id,
        "output_dir": runtime.output_dir,
        "ordinal": runtime.ordinal,
        "dataset_index": runtime.dataset_index,
        "created_by": runtime.created_by,
        "metadata": _metadata(runtime.metadata),
    }


@dataclass(frozen=True)
class GeometryBundleRef:
    path: str
    cgns_basename: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.path).strip():
            raise ContractError("GeometryBundleRef.path is required")
        if not str(self.cgns_basename).strip():
            raise ContractError("GeometryBundleRef.cgns_basename is required")
        object.__setattr__(self, "metadata", {str(k): v for k, v in dict(self.metadata).items()})

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "cgns_basename": self.cgns_basename,
            "metadata": _metadata(self.metadata),
        }


@dataclass(frozen=True)
class PayloadRef:
    path: str
    geometry_bundle: GeometryBundleRef
    case_id: str
    state_name: str = "final"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.path).strip():
            raise ContractError("PayloadRef.path is required")
        if not str(self.case_id).strip():
            raise ContractError("PayloadRef.case_id is required")
        object.__setattr__(self, "metadata", {str(k): v for k, v in dict(self.metadata).items()})

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "geometry_bundle": self.geometry_bundle.to_dict(),
            "case_id": self.case_id,
            "state_name": self.state_name,
            "metadata": _metadata(self.metadata),
        }


def write_geometry_bundle(case: ResumeCase, output_path: str) -> GeometryBundleRef:
    """Write a clean geometry-bundle manifest for one canonical case."""

    path = _output_file(output_path, name="output_path")
    cgns_ref = cgns_ref_from_solver_context(case.solver_context)
    wall_distance_ref = wall_distance_ref_from_solver_context(case.solver_context)
    payload = {
        "schema_version": GEOMETRY_BUNDLE_SCHEMA,
        "case_id": case.case_id,
        "cgns_ref": cgns_ref.to_dict(),
        "wall_distance_ref": None if wall_distance_ref is None else wall_distance_ref.to_dict(),
        "solver_context": _solver_context_payload(case),
        "runtime": _runtime_payload(case),
        "metadata": {
            "writer": "NK_resume.payload.bundle.write_geometry_bundle",
        },
    }
    _write_json_atomic(path, payload)
    return GeometryBundleRef(
        path=str(path),
        cgns_basename=case.solver_context.cgns_basename,
        metadata={
            "schema_version": GEOMETRY_BUNDLE_SCHEMA,
            "case_id": case.case_id,
            "cgns_ref": cgns_ref.to_dict(),
        },
    )


def load_geometry_bundle(path: str | Path) -> dict[str, Any]:
    """Load and validate a clean geometry-bundle manifest."""

    return _read_json(path, expected_schema=GEOMETRY_BUNDLE_SCHEMA)


def write_case_payload(
    case: ResumeCase,
    output_path: str,
    *,
    geometry_bundle: GeometryBundleRef | None = None,
) -> PayloadRef:
    """Write a clean light payload with arrays in NPZ and metadata in JSON."""

    if geometry_bundle is None:
        case.require_geometry_bundle()
        geometry_bundle = GeometryBundleRef(
            path=case.solver_context.geometry_bundle_path,
            cgns_basename=case.solver_context.cgns_basename,
            metadata={"schema_version": GEOMETRY_BUNDLE_SCHEMA},
        )

    path = _output_file(output_path, name="output_path")
    arrays, array_keys = _case_arrays(case)

    manifest = {
        "schema_version": CASE_PAYLOAD_SCHEMA,
        "case": case.summary(),
        "geometry_bundle": geometry_bundle.to_dict(),
        "model_inputs": _model_inputs_payload(case),
        "solver_context": _solver_context_payload(case),
        "geometry_context": _geometry_context_payload(case),
        "ground_truth": _ground_truth_payload(case),
        "prediction": _prediction_payload(case),
        "runtime": _runtime_payload(case),
        "array_keys": array_keys,
        "metadata": {
            "writer": "NK_resume.payload.bundle.write_case_payload",
        },
    }
    arrays["manifest_json"] = np.asarray(json.dumps(manifest, sort_keys=True))

    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)

    return PayloadRef(
        path=str(path),
        geometry_bundle=geometry_bundle,
        case_id=case.case_id,
        state_name=case.prediction.state_name,
        metadata={
            "schema_version": CASE_PAYLOAD_SCHEMA,
            "array_keys": array_keys,
        },
    )


def load_case_payload(path: str | Path, *, load_arrays: bool = True) -> dict[str, Any]:
    """Load and validate a clean case payload."""

    with np.load(Path(path), allow_pickle=False) as data:
        if "manifest_json" not in data.files:
            raise ContractError("clean case payload is missing manifest_json")
        manifest = json.loads(str(data["manifest_json"].item()))
        if manifest.get("schema_version") != CASE_PAYLOAD_SCHEMA:
            raise ContractError(
                f"Expected {CASE_PAYLOAD_SCHEMA}, got {manifest.get('schema_version')!r}"
            )
        if load_arrays:
            arrays = {
                key: data[key].copy()
                for key in data.files
                if key != "manifest_json"
            }
        else:
            arrays = {key: None for key in data.files if key != "manifest_json"}
    if load_arrays:
        arrays = _validate_loaded_arrays(manifest, arrays)
    else:
        _validate_loaded_array_key_set(manifest, set(str(key) for key in arrays))
    return {"manifest": manifest, "arrays": arrays}


def _mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ContractError(f"case payload manifest missing object: {key}")
    return {str(k): v for k, v in value.items()}


def _array_or_none(arrays: Mapping[str, Any], key: str) -> Any | None:
    value = arrays.get(key)
    return None if value is None else value


def resume_case_from_payload(path: str | Path) -> ResumeCase:
    """Reconstruct a canonical ResumeCase from a clean case payload."""

    payload = load_case_payload(path, load_arrays=True)
    manifest = payload["manifest"]
    arrays = payload["arrays"]
    if not isinstance(manifest, Mapping) or not isinstance(arrays, Mapping):
        raise ContractError("Invalid clean case payload structure")
    if "prediction_field" not in arrays:
        raise ContractError("Clean case payload is missing prediction_field")

    model = _mapping(manifest, "model_inputs")
    solver = _mapping(manifest, "solver_context")
    ground_truth = _mapping(manifest, "ground_truth")
    raw_geometry = manifest.get("geometry_context")
    geometry_context = (
        {str(k): v for k, v in raw_geometry.items()}
        if isinstance(raw_geometry, Mapping)
        else ground_truth
    )
    prediction = _mapping(manifest, "prediction")
    runtime = _mapping(manifest, "runtime")
    geometry_bundle = _mapping(manifest, "geometry_bundle")
    bundle_path = str(geometry_bundle.get("path") or "").strip()
    bundle_basename = str(geometry_bundle.get("cgns_basename") or "").strip()
    solver_basename = str(solver.get("cgns_basename") or "").strip()
    if bundle_basename and solver_basename and bundle_basename != solver_basename:
        raise ContractError(
            "case payload geometry bundle basename does not match solver context: "
            f"geometry_bundle={bundle_basename!r}, solver_context={solver_basename!r}"
        )
    geometry_bundle_path = str(solver.get("geometry_bundle_path") or "").strip() or bundle_path

    return ResumeCase(
        model_inputs=ModelInputs(
            predictor_kind=str(model.get("predictor_kind") or ""),
            config_path=str(model.get("config_path") or ""),
            checkpoint_path=str(model.get("checkpoint_path") or ""),
            stats_path=str(model.get("stats_path") or ""),
            device=str(model.get("device") or ""),
            inference_steps=model.get("inference_steps"),
            custom_timesteps=tuple(model.get("custom_timesteps") or ()),
            metadata=dict(model.get("metadata") or {}),
        ),
        solver_context=SolverContext(
            cgns_basename=str(solver.get("cgns_basename") or ""),
            cgns_root=str(solver.get("cgns_root") or ""),
            flow_conditions=tuple(solver.get("flow_conditions") or ()),
            flow_conditions_dict=dict(solver.get("flow_conditions_dict") or {}),
            source_info=dict(solver.get("source_info") or {}),
            wall_layers=solver.get("wall_layers"),
            options_version=int(solver.get("options_version") or 2),
            l2conv=float(solver.get("l2conv") or 1.0e-8),
            ranks_per_case=int(solver.get("ranks_per_case") or 1),
            mpi_launcher=str(solver.get("mpi_launcher") or "auto"),
            mpi_omp_threads=int(solver.get("mpi_omp_threads") or 1),
            geometry_bundle_path=geometry_bundle_path,
            fixed_lift=solver.get("fixed_lift"),
            metadata=dict(solver.get("metadata") or {}),
        ),
        ground_truth=GroundTruth(
            field=_array_or_none(arrays, "ground_truth_field"),
            force_coefficients=dict(ground_truth.get("force_coefficients") or {}),
            residual_reference=dict(ground_truth.get("residual_reference") or {}),
            metadata=dict(ground_truth.get("metadata") or {}),
        ),
        prediction=PredictionState(
            field=arrays["prediction_field"],
            state_name=str(prediction.get("state_name") or "final"),
            step_index=prediction.get("step_index"),
            metadata=dict(prediction.get("metadata") or {}),
        ),
        runtime=RuntimeState(
            case_id=str(runtime.get("case_id") or ""),
            output_dir=str(runtime.get("output_dir") or ""),
            ordinal=runtime.get("ordinal"),
            dataset_index=runtime.get("dataset_index"),
            created_by=str(runtime.get("created_by") or "NK_resume"),
            metadata=dict(runtime.get("metadata") or {}),
        ),
        geometry=GeometryContext(
            coords_center=_array_or_none(arrays, "coords_center"),
            coords_vertex=_array_or_none(arrays, "coords_vertex"),
            metadata=dict(geometry_context.get("metadata") or {}),
        ),
    )
