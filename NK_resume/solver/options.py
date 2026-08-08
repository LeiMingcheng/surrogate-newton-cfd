"""Solver option contract for the clean NK_resume runtime.

This module defines solver *semantics* shared by plans, replay manifests, and
future ADflow backends.  It intentionally does not copy historical ADflow
option dictionaries; backend-specific translation belongs in the migrated
solver backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..exceptions import ContractError
from ..plans import ResumePlan, SolverPreset, StagePlan


SOLVER_OPTIONS_SCHEMA = "solver_options_v1"


def _metadata(value: dict[str, Any] | None) -> dict[str, Any]:
    return {str(k): v for k, v in dict(value or {}).items()}


@dataclass(frozen=True)
class SolverOptions:
    """Backend-neutral solver option request.

    `solver_preset` is the solver method family (`nk`, `ank`, `prod`, or `pseudo`),
    not a resume interaction plan.  Cycle budgets still live in `NKWorkPlan`.
    """

    options_version: int = 2
    l2conv: float = 1.0e-8
    solver_preset: SolverPreset | str = SolverPreset.NK
    backend: str = "adflow"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        options_version = int(self.options_version)
        l2conv = float(self.l2conv)
        try:
            preset = SolverPreset(self.solver_preset)
        except ValueError as exc:
            raise ContractError(f"Unsupported solver preset: {self.solver_preset!r}") from exc
        backend = str(self.backend).strip().lower()
        if options_version <= 0:
            raise ContractError("SolverOptions.options_version must be positive")
        if l2conv <= 0.0:
            raise ContractError("SolverOptions.l2conv must be positive")
        if not backend:
            raise ContractError("SolverOptions.backend is required")
        object.__setattr__(self, "options_version", options_version)
        object.__setattr__(self, "l2conv", l2conv)
        object.__setattr__(self, "solver_preset", preset)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        semantic = _semantic_preset(self.solver_preset)
        return {
            "schema_version": SOLVER_OPTIONS_SCHEMA,
            "backend": self.backend,
            "options_version": self.options_version,
            "l2conv": self.l2conv,
            "solver_preset": self.solver_preset.value,
            "semantic_preset": semantic,
            "metadata": dict(self.metadata),
        }


def _semantic_preset(preset: SolverPreset) -> dict[str, Any]:
    if preset == SolverPreset.NONE:
        return {
            "executes_solver": False,
            "nonlinear_method": "none",
            "description": "No solver correction is requested.",
        }
    if preset == SolverPreset.NK:
        return {
            "executes_solver": True,
            "nonlinear_method": "nk",
            "description": "Newton-Krylov correction stage.",
        }
    if preset == SolverPreset.ANK:
        return {
            "executes_solver": True,
            "nonlinear_method": "ank",
            "description": "Approximate Newton-Krylov-only correction stage.",
        }
    if preset == SolverPreset.PROD:
        return {
            "executes_solver": True,
            "nonlinear_method": "prod",
            "description": "Production ADflow correction preset.",
        }
    if preset == SolverPreset.PSEUDO:
        return {
            "executes_solver": True,
            "nonlinear_method": "pseudo",
            "description": "Pseudo-time correction preset.",
        }
    raise ContractError(f"Unsupported solver preset: {preset!r}")


def build_solver_options(options: SolverOptions) -> dict[str, Any]:
    """Build a validated backend-neutral solver option payload."""

    return options.to_dict()


def solver_options_for_stage(
    stage: StagePlan,
    *,
    options_version: int = 2,
    l2conv: float = 1.0e-8,
    backend: str = "adflow",
) -> dict[str, Any]:
    """Build solver option payload for one interaction stage."""

    payload = build_solver_options(
        SolverOptions(
            options_version=options_version,
            l2conv=l2conv,
            solver_preset=stage.work.solver_preset,
            backend=backend,
            metadata={
                "stage_name": stage.name,
                "cycle_policy": stage.work.cycle_policy.value,
            },
        )
    )
    payload["stage"] = {
        "name": stage.name,
        "source_state": stage.source_state,
        "work": stage.work.to_dict(),
    }
    return payload


def solver_options_for_plan(
    plan: ResumePlan,
    *,
    options_version: int = 2,
    l2conv: float = 1.0e-8,
    backend: str = "adflow",
) -> dict[str, Any]:
    """Build solver option payloads for all stages in a resume plan."""

    return {
        "schema_version": SOLVER_OPTIONS_SCHEMA,
        "plan": plan.to_dict(),
        "stages": [
            solver_options_for_stage(
                stage,
                options_version=options_version,
                l2conv=l2conv,
                backend=backend,
            )
            for stage in plan.stages
        ],
    }
