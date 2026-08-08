"""Model-side readers for clean NK_resume payloads.

Payload writing is owned by `NK_resume` and works from canonical `ResumeCase`
objects. This module only exposes read helpers for model-side inspection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from NK_resume.payload import load_geometry_bundle as _load_geometry_bundle
from NK_resume.payload import resume_case_from_payload


def load_geometry_bundle(bundle_path: str | Path) -> dict[str, Any]:
    """Load a clean NK_resume geometry-bundle manifest."""

    return _load_geometry_bundle(Path(bundle_path))


def load_payload(path: str | Path) -> dict[str, Any]:
    """Load a clean NK_resume case payload with model-facing aliases."""

    case = resume_case_from_payload(Path(path))
    return {
        "case": case,
        "case_id": case.case_id,
        "predictor_kind": case.model_inputs.predictor_kind,
        "state_name": case.prediction.state_name,
        "predicted_fields": np.asarray(case.prediction.field, dtype=np.float64),
        "target_fields": None
        if case.ground_truth.field is None
        else np.asarray(case.ground_truth.field, dtype=np.float64),
        "flow_conditions": np.asarray(case.solver_context.flow_conditions, dtype=np.float64),
        "flow_conditions_metadata": dict(case.solver_context.flow_conditions_dict),
        "target_coefficients": dict(case.ground_truth.force_coefficients),
        "coords": None
        if case.geometry.coords_center is None
        else np.asarray(case.geometry.coords_center, dtype=np.float64),
        "coords_pde": None
        if case.geometry.coords_center is None
        else np.asarray(case.geometry.coords_center, dtype=np.float64),
        "coords_vertex": None
        if case.geometry.coords_vertex is None
        else np.asarray(case.geometry.coords_vertex, dtype=np.float64),
        "geometry_bundle_path": case.solver_context.geometry_bundle_path,
        "cgns_root": case.solver_context.cgns_root,
        "cgns_basename": case.solver_context.cgns_basename,
        "source_info": dict(case.solver_context.source_info),
    }


__all__ = [
    "load_geometry_bundle",
    "load_payload",
]
