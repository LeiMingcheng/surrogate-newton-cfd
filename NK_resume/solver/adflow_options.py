"""Clean ADflow option translation for NK_resume."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..exceptions import ContractError
from ..plans import SolverPreset, StagePlan


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return {str(k): v for k, v in dict(value or {}).items()}


def _path_text(value: str | Path) -> str:
    text = str(value).strip()
    if not text:
        raise ContractError("ADflow path value is required")
    return text


@dataclass(frozen=True)
class ADflowOptionRequest:
    """Inputs needed to build one ADflow solver option dictionary."""

    cgns_path: str | Path
    output_dir: str | Path
    options_version: int = 2
    l2conv: float = 1.0e-8
    cycles: int = 1
    solver_preset: SolverPreset | str = SolverPreset.NK
    turbulence_model: str = "SA"
    print_iterations: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        options_version = int(self.options_version)
        l2conv = float(self.l2conv)
        cycles = int(self.cycles)
        try:
            preset = SolverPreset(self.solver_preset)
        except ValueError as exc:
            raise ContractError(f"Unsupported solver preset: {self.solver_preset!r}") from exc
        if options_version <= 0:
            raise ContractError("ADflowOptionRequest.options_version must be positive")
        if l2conv <= 0.0:
            raise ContractError("ADflowOptionRequest.l2conv must be positive")
        if cycles < 0:
            raise ContractError("ADflowOptionRequest.cycles must be non-negative")
        object.__setattr__(self, "cgns_path", _path_text(self.cgns_path))
        object.__setattr__(self, "output_dir", _path_text(self.output_dir))
        object.__setattr__(self, "options_version", options_version)
        object.__setattr__(self, "l2conv", l2conv)
        object.__setattr__(self, "cycles", cycles)
        object.__setattr__(self, "solver_preset", preset)
        object.__setattr__(self, "turbulence_model", str(self.turbulence_model))
        object.__setattr__(self, "print_iterations", bool(self.print_iterations))
        object.__setattr__(self, "metadata", _metadata(self.metadata))


def _base_adflow_options(request: ADflowOptionRequest) -> dict[str, Any]:
    options: dict[str, Any] = {
        "gridFile": request.cgns_path,
        "outputDirectory": request.output_dir,
        "equationType": "RANS",
        "turbulenceModel": request.turbulence_model,
        "CFL": 1.0,
        "CFLCoarse": 1.25,
        "MGCycle": "sg",
        "firstRun": True,
        "useBlockettes": False,
        "storeRindLayer": True,
        "useANKSolver": True,
        "ANKNSubiterTurb": 5,
        "ANKSecondOrdSwitchTol": 1.0e-3,
        "ANKStepFactor": 0.5,
        "ANKMaxIter": 40,
        "useNKSolver": True,
        "NKSwitchTol": 1.0e-5,
        "NKADPC": True,
        "NKInnerPreconIts": 2,
        "NKJacobianLag": 3,
        "NKOuterPreconIts": 3,
        "NKPCILUFill": 2,
        "NKSubspaceSize": 100,
        "nSubiterTurb": 25,
        "useDissContinuation": True,
        "dissContMagnitude": 2.0,
        "dissContMidpoint": 20.0,
        "dissContSharpness": 3.0,
        "L2Convergence": request.l2conv,
        "L2ConvergenceCoarse": 1.0e-2,
        "nCycles": max(1, request.cycles),
        "monitorvariables": ["resrho", "cl", "cd", "cmz"],
        "surfacevariables": ["cp", "cf", "mach"],
        "writeSurfaceSolution": False,
        "writeVolumeSolution": False,
        "outputsurfacefamily": "wall",
        "smoother": "DADI",
        "MGStartLevel": -1,
        "printIterations": request.print_iterations,
        "printAllOptions": False,
        "printIntro": False,
        "printTiming": False,
        "setMonitor": True,
        "storeConvHist": True,
    }
    if request.options_version == 1:
        options.update(
            {
                "CFL": 1.5,
                "ANKNSubiterTurb": 3,
                "nSubiterTurb": 10,
                "useDissContinuation": True,
            }
        )
    elif request.options_version == 2:
        pass
    elif request.options_version == 3:
        options.update({"CFL": 1.5, "ANKNSubiterTurb": 3, "nSubiterTurb": 10})
    elif request.options_version == 4:
        options.update(
            {
                "CFL": 3.0,
                "CFLCoarse": 1.5,
                "MGCycle": "3w",
                "MGStartLevel": 1,
                "nSubiter": 3,
                "resAveraging": "always",
            }
        )
    elif request.options_version == 5:
        options.update(
            {
                "CFL": 1.0,
                "CFLCoarse": 0.75,
                "MGCycle": "3w",
                "MGStartLevel": 1,
                "smoother": "Runge-Kutta",
                "nSubiter": 1,
                "resAveraging": "always",
            }
        )
    elif request.options_version == 6:
        options.update(
            {
                "CFL": 1.5,
                "CFLCoarse": 1.0,
                "MGCycle": "3w",
                "MGStartLevel": 1,
                "smoother": "Runge-Kutta",
                "nSubiter": 1,
                "nSubiterTurb": 8,
                "resAveraging": "alternate",
            }
        )
    else:
        raise ContractError(f"Unsupported ADflow options_version: {request.options_version}")
    return options


def _apply_solver_preset(options: dict[str, Any], preset: SolverPreset) -> dict[str, Any]:
    updated = dict(options)
    if preset == SolverPreset.PROD:
        return updated
    if preset == SolverPreset.ANK:
        updated["useANKSolver"] = True
        updated["useNKSolver"] = False
        updated["ANKSwitchTol"] = 1.0e30
        updated["ANKCFLReset"] = False
        return updated
    if preset == SolverPreset.PSEUDO:
        updated["useANKSolver"] = False
        updated["useNKSolver"] = False
        return updated
    if preset == SolverPreset.NK:
        updated["useANKSolver"] = False
        updated["useNKSolver"] = True
        updated["NKSwitchTol"] = 1.0e30
        return updated
    if preset == SolverPreset.NONE:
        updated["useANKSolver"] = False
        updated["useNKSolver"] = False
        updated["nCycles"] = 0
        return updated
    raise ContractError(f"Unsupported ADflow solver preset: {preset!r}")


def build_adflow_options(request: ADflowOptionRequest) -> dict[str, Any]:
    """Build backend-specific ADflow options from clean solver semantics."""

    return _apply_solver_preset(
        _base_adflow_options(request),
        request.solver_preset,
    )


def build_adflow_options_for_stage(
    stage: StagePlan,
    *,
    cgns_path: str | Path,
    output_dir: str | Path,
    options_version: int,
    l2conv: float,
    cycles: int,
    print_iterations: bool = False,
    turbulence_model: str = "SA",
) -> dict[str, Any]:
    """Build ADflow options for one NK_resume stage."""

    return build_adflow_options(
        ADflowOptionRequest(
            cgns_path=cgns_path,
            output_dir=output_dir,
            options_version=options_version,
            l2conv=l2conv,
            cycles=cycles,
            solver_preset=stage.work.solver_preset,
            print_iterations=print_iterations,
            turbulence_model=turbulence_model,
            metadata={
                "stage": stage.name,
                "source_state": stage.source_state,
                "cycle_policy": stage.work.cycle_policy.value,
            },
        )
    )
