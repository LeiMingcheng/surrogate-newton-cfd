"""Transport-neutral serving contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass
class PredictionRequest:
    """A transport-neutral surrogate prediction request."""

    geometry: Any
    flow_conditions: Any
    coords: Any
    initial_field: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResponse:
    """A transport-neutral surrogate prediction response."""

    fields: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class AoARequest:
    """Request for fixed-AoA evaluation or target-CL AoA solving."""

    geometry: Any
    coords: Any
    coords_vertex: Any
    mach: Any
    reynolds: Any = 20.0e6
    target_cl: Any = None
    aoa: Any = None
    initial_field: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class AoAResult:
    """Result from fixed-AoA evaluation or target-CL solving."""

    aoa: Any
    fields: Any
    cl: Any
    cd: Any
    cm: Any
    converged: bool
    n_iter: int
    converged_mask: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class OnlineSample:
    """Online sample record shared by serving and teacher-data pipelines."""

    sample_id: str
    geometry: Any
    flow_conditions: Any
    coords_vertex: Any
    coords: Any
    fields: Any = None
    pred_fields: Any = None
    wall_distance: Any = None
    source: str = "optimization"
    cfd_status: str = "none"
    model_version: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = [
    "AoARequest",
    "AoAResult",
    "OnlineSample",
    "PredictionRequest",
    "PredictionResponse",
]
