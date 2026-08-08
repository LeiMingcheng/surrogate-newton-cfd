"""Small model-side contracts for NK resume surrogate predictions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class ResumeRequest:
    """Input tensors needed to produce one resume seed field."""

    geometry: Any
    flow_conditions: Any
    coords: Any
    initial_field: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ResumePrediction:
    """Surrogate prediction prepared for canonical NK_resume export."""

    fields: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = [
    "ResumePrediction",
    "ResumeRequest",
]
