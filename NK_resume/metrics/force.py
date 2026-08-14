"""Force-coefficient metrics independent from solver internals."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..exceptions import ContractError
from .moment import STANDARD_MOMENT_REFERENCE, right_hand_cmz_to_standard_cm


FORCE_METRICS_SCHEMA = "force_metrics_v1"

_FORCE_ALIASES = {
    "cl": "cl",
    "c_l": "cl",
    "lift": "cl",
    "lift_coefficient": "cl",
    "cd": "cd",
    "c_d": "cd",
    "drag": "cd",
    "drag_coefficient": "cd",
    "cm": "cm",
    "c_m": "cm",
    "cmz": "cmz",
    "moment": "cm",
    "moment_coefficient": "cm",
}


def _array(value: Any, *, name: str) -> np.ndarray:
    try:
        out = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(out)):
        raise ContractError(f"{name} must contain only finite values")
    return out


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return {str(k): _jsonable(v) for k, v in dict(value or {}).items()}


def _force_key(key: str) -> str:
    normalized = str(key).strip().lower().replace("-", "_").replace(" ", "_")
    return _FORCE_ALIASES.get(normalized, normalized)


def _finite_float(value: Any, *, name: str) -> float:
    if hasattr(value, "item"):
        value = value.item()
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be numeric") from exc
    if not math.isfinite(out):
        raise ContractError(f"{name} must be finite")
    return out


def _flow_scalar(
    values: Mapping[str, Any],
    keys: Sequence[str],
    *,
    name: str,
    default: float | None = None,
) -> float:
    lowered = {str(key).strip().lower(): value for key, value in values.items()}
    for key in keys:
        if key.lower() in lowered:
            return _finite_float(lowered[key.lower()], name=name)
    if default is None:
        raise ContractError(f"flow_conditions must contain {name}")
    return float(default)


def _flow_scalars(flow_conditions: Any) -> tuple[float, float, float]:
    if isinstance(flow_conditions, Mapping):
        mach = _flow_scalar(flow_conditions, ("ma", "mach"), name="mach")
        alpha = _flow_scalar(
            flow_conditions,
            ("aoa", "alpha"),
            name="alpha",
            default=0.0,
        )
        reynolds = _flow_scalar(
            flow_conditions,
            ("re", "reynolds"),
            name="reynolds",
            default=1.0e6,
        )
        return mach, alpha, reynolds

    values = _array(flow_conditions, name="flow_conditions").reshape(-1)
    if values.size < 2:
        raise ContractError("flow_conditions array must contain at least mach and alpha")
    mach = _finite_float(values[0], name="mach")
    alpha = _finite_float(values[1], name="alpha")
    reynolds = (
        _finite_float(values[2], name="reynolds")
        if values.size > 2
        else 1.0e6
    )
    return mach, alpha, reynolds


def _wall_segment_mask(x_wall_v: np.ndarray, y_wall_v: np.ndarray) -> np.ndarray:
    ds_size = max(int(x_wall_v.size) - 1, 0)
    seg_mask = np.ones(ds_size, dtype=bool)
    y_eps = 1.0e-6
    surface_vmask = np.abs(y_wall_v) > y_eps
    if not surface_vmask.any():
        return seg_mask

    x_max_surface = float(np.max(x_wall_v[surface_vmask]))
    x_min_surface = float(np.min(x_wall_v[surface_vmask]))
    chord_like = max(x_max_surface - x_min_surface, 1.0e-6)
    x_margin = max(1.0e-3, 0.01 * chord_like)
    x_thresh = x_max_surface + x_margin
    has_cut = bool(((x_wall_v > x_thresh) & (np.abs(y_wall_v) <= y_eps)).any())
    if has_cut:
        seg_mask = (x_wall_v[:-1] <= x_thresh) & (x_wall_v[1:] <= x_thresh)
    return seg_mask


def normalize_force_coefficients(values: Mapping[str, Any] | None) -> dict[str, float]:
    """Normalize common force coefficient names to stable lowercase keys."""

    out: dict[str, float] = {}
    for key, value in dict(values or {}).items():
        normalized_key = _force_key(str(key))
        normalized_value = _finite_float(value, name=f"force coefficient {key!r}")
        if normalized_key == "cmz":
            normalized_key = "cm"
            normalized_value = float(right_hand_cmz_to_standard_cm(normalized_value))
        out[normalized_key] = normalized_value
    return out


def compute_field_force_coefficients(
    fields: Any,
    coords_vertex: Any,
    flow_conditions: Any,
    *,
    gamma: float = 1.4,
    chord_ref: float = 1.0,
    area_ref: float | None = None,
    moment_center: tuple[float, float] = STANDARD_MOMENT_REFERENCE,
    viscous_force: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Compute field-based 2D force coefficients without solver internals.

    The built-in path performs the same wall pressure integration used by the
    project force post-processing stack.  A caller may supply already-computed
    viscous body-force components through ``viscous_force``; this keeps
    ``NK_resume`` independent from model-side PDE/runtime packages.
    """

    field = _array(fields, name="fields")
    coords = _array(coords_vertex, name="coords_vertex")
    if field.ndim != 3 or field.shape[0] < 4:
        raise ContractError("fields must have shape (C, H, W) with at least 4 channels")
    if coords.ndim != 3 or coords.shape[0] < 2:
        raise ContractError("coords_vertex must have shape (2, H+1, W+1)")

    mach, alpha_deg, _reynolds = _flow_scalars(flow_conditions)
    alpha_rad = math.radians(alpha_deg)

    p_wall = field[3, 0, :]
    x_wall_v = coords[0, 0, :]
    y_wall_v = coords[1, 0, :]
    if p_wall.size != x_wall_v.size - 1 or p_wall.size != y_wall_v.size - 1:
        raise ContractError(
            "wall pressure width must match coords_vertex wall segment count"
        )

    x_wall_c = 0.5 * (x_wall_v[:-1] + x_wall_v[1:])
    y_wall_c = 0.5 * (y_wall_v[:-1] + y_wall_v[1:])
    dx = x_wall_v[1:] - x_wall_v[:-1]
    dy = y_wall_v[1:] - y_wall_v[:-1]
    ds = np.sqrt(dx**2 + dy**2)

    signed_area = 0.5 * np.sum(
        x_wall_v[:-1] * y_wall_v[1:] - x_wall_v[1:] * y_wall_v[:-1]
    )
    orient = 1.0 if signed_area > 0.0 else -1.0
    nx = orient * dy / (ds + 1.0e-12)
    ny = orient * (-dx) / (ds + 1.0e-12)

    seg_mask = _wall_segment_mask(x_wall_v, y_wall_v)
    q_nondim = 0.5 * float(gamma) * mach**2
    cp_wall = (p_wall - 1.0) / (q_nondim + 1.0e-12)

    fx_pressure = -np.sum(cp_wall[seg_mask] * nx[seg_mask] * ds[seg_mask])
    fy_pressure = -np.sum(cp_wall[seg_mask] * ny[seg_mask] * ds[seg_mask])

    cos_a = math.cos(alpha_rad)
    sin_a = math.sin(alpha_rad)
    s_ref = float(chord_ref) if area_ref is None else float(area_ref)
    clp = (fy_pressure * cos_a - fx_pressure * sin_a) / s_ref
    cdp = (fx_pressure * cos_a + fy_pressure * sin_a) / s_ref

    fx_viscous = 0.0
    fy_viscous = 0.0
    if viscous_force:
        fx_viscous = _finite_float(
            viscous_force.get("Fx_v", viscous_force.get("fx_v", 0.0)),
            name="viscous Fx",
        )
        fy_viscous = _finite_float(
            viscous_force.get("Fy_v", viscous_force.get("fy_v", 0.0)),
            name="viscous Fy",
        )
    clv = (fy_viscous * cos_a - fx_viscous * sin_a) / (q_nondim * s_ref + 1.0e-12)
    cdv = (fx_viscous * cos_a + fy_viscous * sin_a) / (q_nondim * s_ref + 1.0e-12)
    cl = clp + clv
    cd = cdp + cdv

    x_ref = float(moment_center[0])
    y_ref = float(moment_center[1])
    pressure_moment = np.sum(
        (-cp_wall[seg_mask] * ds[seg_mask])
        * (
            (x_wall_c[seg_mask] - x_ref) * ny[seg_mask]
            - (y_wall_c[seg_mask] - y_ref) * nx[seg_mask]
        )
    )
    viscous_moment = 0.0
    if viscous_force:
        viscous_moment = _finite_float(
            viscous_force.get("M_v", viscous_force.get("m_v", 0.0)),
            name="viscous moment",
        )
    cmp = right_hand_cmz_to_standard_cm(
        pressure_moment / (s_ref * float(chord_ref))
    )
    cmv = right_hand_cmz_to_standard_cm(
        viscous_moment / (q_nondim * s_ref * float(chord_ref) + 1.0e-12)
    )
    cm = cmp + cmv
    return normalize_force_coefficients(
        {
            "CL": cl,
            "CLp": clp,
            "CLv": clv,
            "CD": cd,
            "CDp": cdp,
            "CDv": cdv,
            "Cm": cm,
            "Cmp": cmp,
            "Cmv": cmv,
        }
    )


@dataclass(frozen=True)
class ForceMetric:
    """Force metric result for one candidate/reference pair."""

    candidate: dict[str, float]
    reference: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        candidate = normalize_force_coefficients(self.candidate)
        reference = normalize_force_coefficients(self.reference)
        object.__setattr__(self, "candidate", candidate)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @property
    def common_keys(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.candidate) & set(self.reference)))

    @property
    def deltas(self) -> dict[str, float]:
        return {
            key: self.candidate[key] - self.reference[key]
            for key in self.common_keys
        }

    @property
    def abs_deltas(self) -> dict[str, float]:
        return {key: abs(value) for key, value in self.deltas.items()}

    def to_dict(self) -> dict[str, Any]:
        abs_values = tuple(self.abs_deltas.values())
        max_abs = max(abs_values) if abs_values else None
        mean_abs = sum(abs_values) / len(abs_values) if abs_values else None
        return {
            "schema_version": FORCE_METRICS_SCHEMA,
            "candidate": dict(self.candidate),
            "reference": dict(self.reference),
            "common_keys": list(self.common_keys),
            "delta": self.deltas,
            "abs_delta": self.abs_deltas,
            "summary": {
                "max_abs_delta": max_abs,
                "mean_abs_delta": mean_abs,
                "compared_count": len(abs_values),
            },
            "metadata": dict(self.metadata),
        }


def force_metrics(
    candidate: Mapping[str, Any] | None,
    reference: Mapping[str, Any] | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute clean force coefficient deltas."""

    return ForceMetric(
        candidate=normalize_force_coefficients(candidate),
        reference=normalize_force_coefficients(reference),
        metadata=_metadata(metadata),
    ).to_dict()
