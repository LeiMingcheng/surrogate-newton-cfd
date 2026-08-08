"""Contracts shared by every optimization evaluator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class OperatingPointResult:
    mach: float
    target_cl: float
    reynolds: float
    aoa: float
    cl: float
    cd: float
    cm: float
    converged: bool
    n_iter: int = 0
    residual: float | None = None
    wall_time_s: float = 0.0
    field_path: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateEvaluation:
    evaluator: str
    points: tuple[OperatingPointResult, ...]
    wall_time_s: float
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("CandidateEvaluation requires at least one operating point")

    @property
    def converged(self) -> bool:
        return all(point.converged for point in self.points)

    @property
    def cd_average(self) -> float:
        return sum(point.cd for point in self.points) / len(self.points)

    @property
    def cl_error_max(self) -> float:
        return max(abs(point.cl - point.target_cl) for point in self.points)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator": self.evaluator,
            "points": [point.to_dict() for point in self.points],
            "converged": self.converged,
            "cd_average": self.cd_average,
            "cl_error_max": self.cl_error_max,
            "wall_time_s": self.wall_time_s,
            "provenance": dict(self.provenance),
        }


__all__ = ["CandidateEvaluation", "OperatingPointResult"]
