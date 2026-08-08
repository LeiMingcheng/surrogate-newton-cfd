"""Read-only migration boundary for historical payloads.

This module is allowed to inspect historical NPZ artifacts as data.  It must
not import or execute historical runtime packages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..exceptions import ContractError, LegacyBoundaryError, NotMigratedError


LEGACY_PROJECTION_REFERENCE_SCHEMA = "legacy_projection_reference_v1"


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
        return _jsonable(value.item())
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return str(value)


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return {str(k): _jsonable(v) for k, v in dict(value or {}).items()}


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


def _optional_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _stage_metric(stage: Mapping[str, Any], section: str, key: str) -> float | None:
    value = stage.get(section)
    if not isinstance(value, Mapping):
        return None
    return _optional_finite_float(value.get(key))


@dataclass(frozen=True)
class LegacyProjectionReference:
    """Reference scalars read from a historical projection result NPZ."""

    path: str
    reference_totalr0: float
    stage_cycles: tuple[int, ...]
    stage_post_totalr: tuple[float, ...]
    stage_post_l2_ratio_ref: tuple[float, ...]
    final_mse_vs_gt: float | None = None
    final_force_l1_vs_gt: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        path = str(self.path).strip()
        if not path:
            raise ContractError("LegacyProjectionReference.path is required")
        reference_totalr0 = _finite_float(
            self.reference_totalr0,
            name="reference_totalr0",
        )
        if reference_totalr0 <= 0.0:
            raise ContractError("reference_totalr0 must be positive")
        cycles = tuple(int(value) for value in self.stage_cycles)
        totalr = tuple(float(value) for value in self.stage_post_totalr)
        ratios = tuple(float(value) for value in self.stage_post_l2_ratio_ref)
        if not cycles:
            raise ContractError("stage_cycles must not be empty")
        if len(cycles) != len(totalr) or len(cycles) != len(ratios):
            raise ContractError("stage cycle/residual series lengths must match")
        if any(value <= 0 for value in cycles):
            raise ContractError("stage_cycles must be positive")
        if any(not math.isfinite(value) or value < 0.0 for value in totalr):
            raise ContractError("stage_post_totalr values must be finite and non-negative")
        if any(not math.isfinite(value) or value < 0.0 for value in ratios):
            raise ContractError(
                "stage_post_l2_ratio_ref values must be finite and non-negative"
            )
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "reference_totalr0", reference_totalr0)
        object.__setattr__(self, "stage_cycles", cycles)
        object.__setattr__(self, "stage_post_totalr", totalr)
        object.__setattr__(self, "stage_post_l2_ratio_ref", ratios)
        object.__setattr__(
            self,
            "final_mse_vs_gt",
            None
            if self.final_mse_vs_gt is None
            else _finite_float(self.final_mse_vs_gt, name="final_mse_vs_gt"),
        )
        object.__setattr__(
            self,
            "final_force_l1_vs_gt",
            None
            if self.final_force_l1_vs_gt is None
            else _finite_float(self.final_force_l1_vs_gt, name="final_force_l1_vs_gt"),
        )
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @property
    def final_totalr(self) -> float:
        return self.stage_post_totalr[-1]

    @property
    def final_l2_ratio_ref(self) -> float:
        return self.stage_post_l2_ratio_ref[-1]

    def residual_reference(self) -> dict[str, Any]:
        return {
            "schema_version": LEGACY_PROJECTION_REFERENCE_SCHEMA,
            "source_path": self.path,
            "reference_totalr0": self.reference_totalr0,
            "stage_cycles": list(self.stage_cycles),
            "stage_post_totalr": list(self.stage_post_totalr),
            "stage_post_l2_ratio_ref": list(self.stage_post_l2_ratio_ref),
            "final_totalr": self.final_totalr,
            "final_l2_ratio_ref": self.final_l2_ratio_ref,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.residual_reference(),
            "final_mse_vs_gt": self.final_mse_vs_gt,
            "final_force_l1_vs_gt": self.final_force_l1_vs_gt,
            "metadata": _metadata(self.metadata),
        }


def read_legacy_projection_reference(path: str | Path) -> LegacyProjectionReference:
    """Read residual reference scalars from a historical final_result.npz."""

    result_path = Path(path)
    if not str(result_path).strip():
        raise ContractError("path is required")
    if not result_path.is_file():
        raise ContractError(f"legacy projection result not found: {result_path}")

    with np.load(result_path, allow_pickle=False) as data:
        if "bundle_json" not in data.files:
            raise ContractError("legacy projection result is missing bundle_json")
        bundle = json.loads(str(data["bundle_json"].item()))

    if not isinstance(bundle, Mapping):
        raise ContractError("legacy bundle_json must be a JSON object")
    stages = bundle.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ContractError("legacy bundle_json must contain non-empty stages")

    reference_totalr0 = None
    stage_cycles: list[int] = []
    stage_post_totalr: list[float] = []
    stage_post_l2_ratio_ref: list[float] = []
    for index, stage in enumerate(stages):
        if not isinstance(stage, Mapping):
            raise ContractError(f"legacy stage {index} must be an object")
        if reference_totalr0 is None:
            reference_totalr0 = _stage_metric(stage, "pre", "reference_totalr0")
        if reference_totalr0 is None:
            reference_totalr0 = _stage_metric(stage, "post", "reference_totalr0")
        cycle_value = stage.get("cycles_cumulative", stage.get("cycles"))
        totalr = _stage_metric(stage, "post", "current_totalr")
        ratio = _stage_metric(stage, "post", "l2_ratio_ref")
        if totalr is None:
            totalr = _stage_metric(stage, "post", "residual_vector_l2")
        if ratio is None:
            ratio = _stage_metric(stage, "post", "residual_vector_l2_ratio_ref")
        if cycle_value is None or totalr is None or ratio is None:
            raise ContractError(f"legacy stage {index} is missing residual scalars")
        stage_cycles.append(int(cycle_value))
        stage_post_totalr.append(float(totalr))
        stage_post_l2_ratio_ref.append(float(ratio))

    if reference_totalr0 is None:
        raise ContractError("legacy bundle_json is missing reference_totalr0")
    final_stage = stages[-1]
    final_mse = _stage_metric(final_stage, "post", "mse_vs_gt")
    final_force_l1 = _stage_metric(final_stage, "post", "force_l1_vs_gt")
    return LegacyProjectionReference(
        path=str(result_path),
        reference_totalr0=float(reference_totalr0),
        stage_cycles=tuple(stage_cycles),
        stage_post_totalr=tuple(stage_post_totalr),
        stage_post_l2_ratio_ref=tuple(stage_post_l2_ratio_ref),
        final_mse_vs_gt=final_mse,
        final_force_l1_vs_gt=final_force_l1,
        metadata={
            "legacy_mode": str(bundle.get("mode") or ""),
            "legacy_plan_name": str(bundle.get("plan_name") or ""),
            "legacy_adaptive_execution_mode": str(
                bundle.get("adaptive_execution_mode") or ""
            ),
        },
    )


def load_legacy_payload(path: str) -> object:
    if not str(path).strip():
        raise ValueError("path is required")
    raise NotMigratedError(
        "legacy payload reader",
        detail="Legacy reads require explicit migration approval and must never be the default write path.",
    )


def write_legacy_payload(*args: object, **kwargs: object) -> None:
    raise LegacyBoundaryError("Writing historical payload formats is forbidden in NK_resume.")
