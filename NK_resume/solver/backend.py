"""Generic solver backend contract."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from ..exceptions import ContractError
from ..plans import ResumePlan
from ..schema import ResumeCase


PROJECTION_STAGE_RESULT_SCHEMA = "projection_stage_result_v1"
PROJECTION_RESULT_SCHEMA = "projection_result_v1"


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
    return str(value)


def _path_text(value: str | Path | None) -> str:
    return "" if value is None else str(value)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


@dataclass(frozen=True)
class ProjectionRequest:
    """Request passed to a solver backend for one canonical case."""

    case: ResumeCase
    plan: ResumePlan
    output_dir: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        output_dir = _path_text(self.output_dir).strip()
        if not output_dir:
            raise ContractError("ProjectionRequest.output_dir is required")
        plan_dict = self.plan.to_dict()
        predictor_kind = str(plan_dict["predictor_kind"]).lower()
        if self.case.model_inputs.predictor_kind != predictor_kind:
            raise ContractError(
                f"ProjectionRequest predictor mismatch: case={self.case.model_inputs.predictor_kind!r}, "
                f"plan={predictor_kind!r}"
            )
        stage_names = {stage["name"] for stage in plan_dict["stages"]}
        if self.case.prediction.state_name not in stage_names:
            raise ContractError(
                f"ProjectionRequest state {self.case.prediction.state_name!r} "
                f"is not in plan stages {sorted(stage_names)}"
            )
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case.summary(),
            "plan": self.plan.to_dict(),
            "output_dir": self.output_dir,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ProjectionStageResult:
    """Result for one interaction stage such as `denoise` or `final`."""

    name: str
    source_state: str = ""
    status: str = "ok"
    metrics: dict[str, Any] = field(default_factory=dict)
    solver_options: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)
    output_paths: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip().lower()
        if not name:
            raise ContractError("ProjectionStageResult.name is required")
        source_state = str(self.source_state or name).strip().lower()
        status = str(self.status).strip().lower()
        if status not in {"ok", "failed", "skipped"}:
            raise ContractError("ProjectionStageResult.status must be ok, failed, or skipped")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "source_state", source_state)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "metrics", _metadata(self.metrics))
        object.__setattr__(self, "solver_options", _metadata(self.solver_options))
        object.__setattr__(self, "timing", _metadata(self.timing))
        object.__setattr__(
            self,
            "output_paths",
            {str(k): str(v) for k, v in dict(self.output_paths).items()},
        )
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROJECTION_STAGE_RESULT_SCHEMA,
            "name": self.name,
            "source_state": self.source_state,
            "status": self.status,
            "metrics": dict(self.metrics),
            "solver_options": dict(self.solver_options),
            "timing": dict(self.timing),
            "output_paths": dict(self.output_paths),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ProjectionResult:
    """Clean projection result for one case."""

    case_id: str
    stages: tuple[ProjectionStageResult | Mapping[str, Any], ...]
    status: str = "ok"
    result_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        case_id = str(self.case_id).strip()
        if not case_id:
            raise ContractError("ProjectionResult.case_id is required")
        stages = tuple(_stage_result(stage) for stage in self.stages)
        if not stages:
            raise ContractError("ProjectionResult requires at least one stage")
        status = str(self.status).strip().lower()
        if status not in {"ok", "failed", "partial"}:
            raise ContractError("ProjectionResult.status must be ok, failed, or partial")
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "result_path", _path_text(self.result_path))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @property
    def final_stage(self) -> ProjectionStageResult:
        for stage in reversed(self.stages):
            if stage.name == "final":
                return stage
        return self.stages[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROJECTION_RESULT_SCHEMA,
            "case_id": self.case_id,
            "status": self.status,
            "result_path": self.result_path,
            "stages": [stage.to_dict() for stage in self.stages],
            "metadata": dict(self.metadata),
        }


def _stage_result(stage: ProjectionStageResult | Mapping[str, Any]) -> ProjectionStageResult:
    if isinstance(stage, ProjectionStageResult):
        return stage
    if not isinstance(stage, Mapping):
        raise ContractError(f"Unsupported projection stage type: {type(stage).__name__}")
    return ProjectionStageResult(
        name=str(stage.get("name") or stage.get("stage_name") or ""),
        source_state=str(stage.get("source_state") or ""),
        status=str(stage.get("status") or "ok"),
        metrics=dict(stage.get("metrics") or {}),
        solver_options=dict(stage.get("solver_options") or {}),
        timing=dict(stage.get("timing") or {}),
        output_paths=dict(stage.get("output_paths") or {}),
        metadata=dict(stage.get("metadata") or {}),
    )


def write_projection_result(result: ProjectionResult, output_path: str | Path) -> str:
    """Write a clean projection result JSON file."""

    path = Path(output_path)
    if not str(path).strip():
        raise ContractError("output_path is required")
    _write_json_atomic(path, result.to_dict())
    return str(path)


def load_projection_result_dict(path: str | Path) -> dict[str, Any]:
    """Load and validate a clean projection result JSON file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != PROJECTION_RESULT_SCHEMA:
        raise ContractError(
            f"Expected {PROJECTION_RESULT_SCHEMA}, got {payload.get('schema_version')!r}"
        )
    stages = payload.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ContractError("Projection result JSON must contain at least one stage")
    for stage in stages:
        if not isinstance(stage, Mapping):
            raise ContractError("Projection result stages must be objects")
        if stage.get("schema_version") != PROJECTION_STAGE_RESULT_SCHEMA:
            raise ContractError(
                f"Expected stage schema {PROJECTION_STAGE_RESULT_SCHEMA}, "
                f"got {stage.get('schema_version')!r}"
            )
    return payload


class SolverBackend(Protocol):
    name: str

    def project(self, request: ProjectionRequest) -> ProjectionResult:
        """Execute one canonical projection request."""
