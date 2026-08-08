"""Canonical execution plans for NK_resume.

There are two separate layers:

- `ResumePlan.kind` describes the surrogate/NK interaction form
  (`finalonly` or `alternating`).
- `NKWorkPlan` describes how one solver correction stage runs internally
  (`fixed` or `adaptive`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from .exceptions import ContractError


class PlanKind(str, Enum):
    """Surrogate-to-NK interaction form."""

    FINALONLY = "finalonly"
    ALTERNATING = "alternating"


class PredictorKind(str, Enum):
    DIRECT = "direct"
    FSB = "fsb"


class SolverPreset(str, Enum):
    NONE = "none"
    NK = "nk"
    ANK = "ank"
    PROD = "prod"
    PSEUDO = "pseudo"


class CyclePolicy(str, Enum):
    """Solver work policy inside one correction stage."""

    FIXED = "fixed"
    ADAPTIVE = "adaptive"


def _positive_int_tuple(values: Iterable[int], *, name: str) -> tuple[int, ...]:
    out = tuple(int(v) for v in values)
    if not out:
        raise ContractError(f"{name} must not be empty")
    if any(v <= 0 for v in out):
        raise ContractError(f"{name} values must be positive")
    return out


@dataclass(frozen=True)
class AdaptiveSchedule:
    """Monotone cumulative-cycle schedule for one solver stage."""

    cumulative_cycles: tuple[int, ...]
    thresholds: tuple[float, ...] = ()
    name: str = "adaptive"

    def __post_init__(self) -> None:
        cycles = _positive_int_tuple(self.cumulative_cycles, name="cumulative_cycles")
        if tuple(sorted(cycles)) != cycles or len(set(cycles)) != len(cycles):
            raise ContractError("AdaptiveSchedule.cumulative_cycles must be strictly increasing")
        thresholds = tuple(float(v) for v in self.thresholds)
        if not thresholds:
            thresholds = tuple(0.0 for _ in cycles)
        if len(thresholds) == 1 and len(cycles) > 1:
            thresholds = thresholds * len(cycles)
        if len(thresholds) != len(cycles):
            raise ContractError("AdaptiveSchedule.thresholds length must match cumulative_cycles")
        object.__setattr__(self, "cumulative_cycles", cycles)
        object.__setattr__(self, "thresholds", thresholds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cumulative_cycles": list(self.cumulative_cycles),
            "thresholds": list(self.thresholds),
        }


@dataclass(frozen=True)
class NKWorkPlan:
    """How one NK/ADflow correction stage should execute."""

    cycle_policy: CyclePolicy
    solver_preset: SolverPreset = SolverPreset.NK
    fixed_cycles: int = 0
    adaptive_schedule: AdaptiveSchedule | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        policy = CyclePolicy(self.cycle_policy)
        solver = SolverPreset(self.solver_preset)
        object.__setattr__(self, "cycle_policy", policy)
        object.__setattr__(self, "solver_preset", solver)
        object.__setattr__(self, "fixed_cycles", int(self.fixed_cycles))
        object.__setattr__(self, "metadata", {str(k): v for k, v in dict(self.metadata).items()})

        if policy == CyclePolicy.FIXED:
            if self.fixed_cycles <= 0:
                raise ContractError("fixed NK work requires fixed_cycles > 0")
            if self.adaptive_schedule is not None:
                raise ContractError("fixed NK work cannot also define an adaptive schedule")
        elif policy == CyclePolicy.ADAPTIVE:
            if self.adaptive_schedule is None:
                raise ContractError("adaptive NK work requires an AdaptiveSchedule")
            if self.fixed_cycles:
                raise ContractError("adaptive NK work cannot also define fixed cycles")

    @classmethod
    def fixed(
        cls,
        cycles: int,
        *,
        solver_preset: SolverPreset | str = SolverPreset.NK,
    ) -> "NKWorkPlan":
        return cls(
            cycle_policy=CyclePolicy.FIXED,
            solver_preset=SolverPreset(solver_preset),
            fixed_cycles=int(cycles),
        )

    @classmethod
    def adaptive(
        cls,
        cycles: Iterable[int],
        *,
        threshold: float = 1.0e-8,
        name: str = "adaptive",
        solver_preset: SolverPreset | str = SolverPreset.NK,
    ) -> "NKWorkPlan":
        cycle_tuple = _positive_int_tuple(cycles, name="cycles")
        return cls(
            cycle_policy=CyclePolicy.ADAPTIVE,
            solver_preset=SolverPreset(solver_preset),
            adaptive_schedule=AdaptiveSchedule(
                cumulative_cycles=cycle_tuple,
                thresholds=tuple(float(threshold) for _ in cycle_tuple),
                name=name,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "solver_preset": self.solver_preset.value,
            "cycle_policy": self.cycle_policy.value,
            "fixed_cycles": self.fixed_cycles,
            "metadata": dict(self.metadata),
        }
        if self.adaptive_schedule is not None:
            payload["adaptive"] = self.adaptive_schedule.to_dict()
        return payload


@dataclass(frozen=True)
class StagePlan:
    """One interaction stage, such as `denoise` or `final`."""

    name: str
    work: NKWorkPlan
    source_state: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip().lower()
        if not name:
            raise ContractError("StagePlan.name is required")
        object.__setattr__(self, "name", name)
        source_state = str(self.source_state or name).strip().lower()
        object.__setattr__(self, "source_state", source_state)
        object.__setattr__(self, "metadata", {str(k): v for k, v in dict(self.metadata).items()})

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_state": self.source_state,
            "work": self.work.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ResumePlan:
    """High-level plan consumed by export/replay backends."""

    kind: PlanKind
    predictor_kind: PredictorKind
    stages: tuple[StagePlan, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = PlanKind(self.kind)
        predictor_kind = PredictorKind(self.predictor_kind)
        stages = tuple(self.stages)
        if not stages:
            raise ContractError("ResumePlan requires at least one stage")
        names = [stage.name for stage in stages]
        if len(set(names)) != len(names):
            raise ContractError(f"ResumePlan stage names must be unique: {names}")
        if kind == PlanKind.FINALONLY and names != ["final"]:
            raise ContractError("finalonly plan must contain exactly one `final` stage")
        if kind == PlanKind.ALTERNATING:
            if predictor_kind != PredictorKind.FSB:
                raise ContractError("alternating plan is only valid for FSB predictors")
            if "final" not in names or len(stages) < 2:
                raise ContractError(
                    "alternating plan must contain a transition stage and final"
                )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "predictor_kind", predictor_kind)
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "metadata", {str(k): v for k, v in dict(self.metadata).items()})

    @property
    def final_stage(self) -> StagePlan:
        for stage in self.stages:
            if stage.name == "final":
                return stage
        raise ContractError("ResumePlan has no final stage")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "predictor_kind": self.predictor_kind.value,
            "stages": [stage.to_dict() for stage in self.stages],
            "metadata": dict(self.metadata),
        }


def finalonly_plan(
    predictor_kind: PredictorKind | str,
    *,
    work: NKWorkPlan | None = None,
    fixed_cycles: int | None = None,
    adaptive_cycles: Iterable[int] = (6, 8, 10),
    adaptive_threshold: float = 1.0e-8,
    solver_preset: SolverPreset | str = SolverPreset.NK,
) -> ResumePlan:
    if work is None:
        if fixed_cycles is not None:
            work = NKWorkPlan.fixed(fixed_cycles, solver_preset=solver_preset)
        else:
            work = NKWorkPlan.adaptive(
                adaptive_cycles,
                threshold=adaptive_threshold,
                name="finalonly",
                solver_preset=solver_preset,
            )
    return ResumePlan(
        kind=PlanKind.FINALONLY,
        predictor_kind=PredictorKind(predictor_kind),
        stages=(StagePlan(name="final", source_state="final", work=work),),
    )


def alternating_plan(
    *,
    transition_work: NKWorkPlan | None = None,
    final_work: NKWorkPlan | None = None,
    transition_cycles: Iterable[int] = (2, 4, 6),
    final_cycles: Iterable[int] = (4, 6, 8, 10, 12),
    transition_threshold: float = 1.0e-4,
    final_threshold: float = 1.0e-8,
) -> ResumePlan:
    if transition_work is None:
        transition_work = NKWorkPlan.adaptive(
            transition_cycles,
            threshold=transition_threshold,
            name="alternating_transition",
        )
    if final_work is None:
        final_work = NKWorkPlan.adaptive(
            final_cycles,
            threshold=final_threshold,
            name="alternating_final",
        )
    return ResumePlan(
        kind=PlanKind.ALTERNATING,
        predictor_kind=PredictorKind.FSB,
        stages=(
            StagePlan(name="denoise", source_state="denoise", work=transition_work),
            StagePlan(name="final", source_state="final", work=final_work),
        ),
    )


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be an object")
    return {str(k): v for k, v in value.items()}


def adaptive_schedule_from_dict(payload: dict[str, Any]) -> AdaptiveSchedule:
    """Reconstruct an AdaptiveSchedule from a clean plan payload."""

    data = _mapping(payload, name="adaptive")
    return AdaptiveSchedule(
        cumulative_cycles=tuple(data.get("cumulative_cycles") or ()),
        thresholds=tuple(data.get("thresholds") or ()),
        name=str(data.get("name") or "adaptive"),
    )


def nk_work_plan_from_dict(payload: dict[str, Any]) -> NKWorkPlan:
    """Reconstruct an NKWorkPlan from a clean plan payload."""

    data = _mapping(payload, name="work")
    policy = CyclePolicy(str(data.get("cycle_policy") or ""))
    adaptive = data.get("adaptive")
    return NKWorkPlan(
        cycle_policy=policy,
        solver_preset=SolverPreset(str(data.get("solver_preset") or SolverPreset.NK.value)),
        fixed_cycles=int(data.get("fixed_cycles") or 0),
        adaptive_schedule=adaptive_schedule_from_dict(adaptive)
        if isinstance(adaptive, dict)
        else None,
        metadata=dict(data.get("metadata") or {}),
    )


def stage_plan_from_dict(payload: dict[str, Any]) -> StagePlan:
    """Reconstruct a StagePlan from a clean plan payload."""

    data = _mapping(payload, name="stage")
    return StagePlan(
        name=str(data.get("name") or ""),
        source_state=str(data.get("source_state") or ""),
        work=nk_work_plan_from_dict(_mapping(data.get("work"), name="stage.work")),
        metadata=dict(data.get("metadata") or {}),
    )


def resume_plan_from_dict(payload: dict[str, Any]) -> ResumePlan:
    """Reconstruct a ResumePlan from its clean dictionary form."""

    data = _mapping(payload, name="plan")
    stages = data.get("stages")
    if not isinstance(stages, list):
        raise ContractError("plan.stages must be a list")
    return ResumePlan(
        kind=PlanKind(str(data.get("kind") or "")),
        predictor_kind=PredictorKind(str(data.get("predictor_kind") or "")),
        stages=tuple(stage_plan_from_dict(stage) for stage in stages),
        metadata=dict(data.get("metadata") or {}),
    )
