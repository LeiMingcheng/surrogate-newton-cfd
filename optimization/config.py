"""Typed configuration for the single optimization driver."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from data.common.flow_conditions import coupled_reynolds_from_mach


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tuple_float(value: Any, default: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(float(item) for item in (default if value is None else value))


@dataclass(frozen=True)
class TaskConfig:
    mach: tuple[float, ...] = (0.71, 0.72, 0.73)
    target_cl: float = 0.8
    reynolds: float = 20.0e6
    reynolds_mode: str = "fixed"
    aoa_bounds: tuple[float, float] = (1.0, 5.0)

    def __post_init__(self) -> None:
        if not self.mach:
            raise ValueError("task.mach must not be empty")
        if self.reynolds_mode not in {"fixed", "coupled"}:
            raise ValueError("task.reynolds_mode must be 'fixed' or 'coupled'")
        if len(self.aoa_bounds) != 2 or self.aoa_bounds[0] >= self.aoa_bounds[1]:
            raise ValueError("task.aoa_bounds must be increasing")

    def reynolds_for(self, mach: float) -> float:
        if self.reynolds_mode == "coupled":
            return float(coupled_reynolds_from_mach(float(mach)))
        return float(self.reynolds)


@dataclass(frozen=True)
class GeometryConfig:
    tail: float = 0.002
    points: int = 1001
    preserve_baseline_area: bool = True
    write_candidate_foil: bool = True
    max_physical_t_max: float | None = None

    def __post_init__(self) -> None:
        if self.points < 3:
            raise ValueError("geometry.points must be at least 3")
        if (
            self.max_physical_t_max is not None
            and self.max_physical_t_max <= 0.0
        ):
            raise ValueError("geometry.max_physical_t_max must be positive")


@dataclass(frozen=True)
class ServingConfig:
    host: str = "127.0.0.1"
    port: int = 65432
    timeout_s: float = 120.0
    model_key: str = ""

    def __post_init__(self) -> None:
        if not 0 < self.port <= 65535:
            raise ValueError("serving.port must be between 1 and 65535")
        if self.timeout_s <= 0.0:
            raise ValueError("serving.timeout_s must be positive")


@dataclass(frozen=True)
class CFDConfig:
    ranks_per_case: int = 8
    pool_count: int = 3
    mpi_launcher: str = "auto"
    python: str = sys.executable
    max_iterations: int = 6000
    max_aoa_iterations: int = 15
    cl_tolerance: float = 1.0e-3
    l2_convergence: float = 1.0e-6
    options_version: int = 2
    reference_state_mode: str = "dataset_unified"
    timeout_s: float = 7200.0

    def __post_init__(self) -> None:
        if self.ranks_per_case <= 0 or self.pool_count <= 0:
            raise ValueError("cfd ranks_per_case and pool_count must be positive")
        if self.max_iterations <= 0 or self.max_aoa_iterations <= 0:
            raise ValueError("cfd iteration limits must be positive")
        if self.cl_tolerance <= 0.0 or self.l2_convergence <= 0.0:
            raise ValueError("cfd tolerances must be positive")
        if self.timeout_s <= 0.0:
            raise ValueError("cfd.timeout_s must be positive")


@dataclass(frozen=True)
class NKConfig:
    selection: str = "all"
    top_k: int = 1
    coupling: str = "sync"
    ranks_per_case: int = 8
    pool_count: int = 3
    resume_mode: str = "ank_nk"
    max_work_per_flow_solve: int = 1000
    max_aoa_solves: int = 5
    total_time_limit_s: float = 30.0
    repeated_nk_cycles: tuple[int, ...] = (6, 8, 10)
    residual_tolerance: float = 1.0e-8
    nk_switch_tolerance: float = 1.0e-4
    cl_tolerance: float = 1.0e-2
    max_corrections: int = 3
    initial_cl_alpha: float = 0.1
    max_aoa_step: float = 2.0
    late_max_aoa_step: float = 0.3
    initial_aoa_damping: float = 0.7
    secant_aoa_damping: float = 0.85
    minimum_aoa_step: float = 0.05
    aoa_epsilon: float = 0.01
    repeated_aoa_tolerance: float = 0.02
    options_version: int = 2
    mpi_launcher: str = "auto"
    mpi_omp_threads: int = 1
    timeout_s: float = 7200.0

    def __post_init__(self) -> None:
        if self.selection not in {"all", "topk"}:
            raise ValueError("nk.selection must be 'all' or 'topk'")
        if self.coupling not in {"sync", "async"}:
            raise ValueError("nk.coupling must be 'sync' or 'async'")
        if self.resume_mode not in {"ank_nk", "repeated_nk"}:
            raise ValueError("nk.resume_mode must be 'ank_nk' or 'repeated_nk'")
        if self.top_k <= 0:
            raise ValueError("nk.top_k must be positive")
        if not self.repeated_nk_cycles:
            raise ValueError("nk.repeated_nk_cycles must not be empty")
        if (
            self.ranks_per_case <= 0
            or self.pool_count <= 0
            or self.max_work_per_flow_solve <= 0
            or self.max_aoa_solves <= 0
        ):
            raise ValueError(
                "nk ranks_per_case, pool_count, max_work_per_flow_solve, and "
                "max_aoa_solves must be positive"
            )
        if min(self.repeated_nk_cycles) <= 0:
            raise ValueError("nk.repeated_nk_cycles values must be positive")
        if (
            tuple(sorted(self.repeated_nk_cycles)) != self.repeated_nk_cycles
            or len(set(self.repeated_nk_cycles)) != len(self.repeated_nk_cycles)
        ):
            raise ValueError("nk.repeated_nk_cycles must be strictly increasing")
        if (
            self.residual_tolerance <= 0.0
            or self.nk_switch_tolerance <= 0.0
            or self.cl_tolerance <= 0.0
        ):
            raise ValueError("nk tolerances must be positive")
        if self.total_time_limit_s <= 0.0:
            raise ValueError("nk.total_time_limit_s must be positive")
        if (
            self.max_corrections < 0
            or self.max_aoa_step <= 0.0
            or self.late_max_aoa_step <= 0.0
            or self.minimum_aoa_step <= 0.0
            or self.aoa_epsilon <= 0.0
            or self.repeated_aoa_tolerance <= 0.0
        ):
            raise ValueError("nk correction limits are invalid")
        if not (
            0.0 < self.initial_aoa_damping <= 1.0
            and 0.0 < self.secant_aoa_damping <= 1.0
        ):
            raise ValueError("nk AoA damping values must be in (0, 1]")
        if self.timeout_s <= 0.0:
            raise ValueError("nk.timeout_s must be positive")


@dataclass(frozen=True)
class ObjectiveConfig:
    residual_weight: float = 0.0
    failure_penalty: float = 1.0e7
    penalize_nonpositive_drag: bool = False
    curvature_hf_weight: float = 0.0
    curvature_hf_points: int = 2001
    curvature_hf_region: tuple[float, float] = (0.15, 0.80)
    curvature_hf_filter_region: tuple[float, float] = (0.10, 0.85)
    curvature_hf_filter_sigma: float = 0.04
    curvature_hf_upper_weight: float = 0.80
    curvature_hf_lower_weight: float = 0.20

    def __post_init__(self) -> None:
        if self.curvature_hf_weight < 0.0:
            raise ValueError("objective.curvature_hf_weight must be non-negative")
        if self.curvature_hf_points < 3:
            raise ValueError("objective.curvature_hf_points must be at least 3")
        if not (
            0.0 <= self.curvature_hf_filter_region[0]
            <= self.curvature_hf_region[0]
            < self.curvature_hf_region[1]
            <= self.curvature_hf_filter_region[1]
            <= 1.0
        ):
            raise ValueError(
                "objective curvature regions must be nested and increasing"
            )
        if self.curvature_hf_filter_sigma <= 0.0:
            raise ValueError(
                "objective.curvature_hf_filter_sigma must be positive"
            )
        if min(
            self.curvature_hf_upper_weight,
            self.curvature_hf_lower_weight,
        ) < 0.0 or abs(
            self.curvature_hf_upper_weight
            + self.curvature_hf_lower_weight
            - 1.0
        ) > 1.0e-12:
            raise ValueError("objective curvature surface weights must sum to one")


@dataclass(frozen=True)
class OptimizerConfig:
    seed: int = 0
    n_pop: int = 32
    n_gen: int = 50
    n_proc: int = 32
    initial_population_size: int = 64
    initial_population: str = ""

    def __post_init__(self) -> None:
        if min(self.n_pop, self.n_proc, self.initial_population_size) <= 0 or self.n_gen < 0:
            raise ValueError("optimizer sizes/process count must be positive and n_gen non-negative")
        if self.n_gen > 0 and self.n_pop < 4:
            raise ValueError("optimizer.n_pop must be at least 4 when n_gen is positive")


@dataclass(frozen=True)
class OptimizationConfig:
    mode: str
    output_dir: str
    baseline_dir: str
    use_enhanced_cst: bool = True
    task: TaskConfig = field(default_factory=TaskConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    cfd: CFDConfig = field(default_factory=CFDConfig)
    nk: NKConfig = field(default_factory=NKConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    def __post_init__(self) -> None:
        if self.mode not in {"surrogate", "cfd", "surrogate_nk"}:
            raise ValueError("mode must be surrogate, cfd, or surrogate_nk")
        baseline = Path(self.baseline_dir)
        missing = [
            name
            for name in ("cst_u0.txt", "cst_l0.txt", "t0.txt")
            if not (baseline / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Incomplete baseline directory {baseline}: {missing}")
        if self.mode == "cfd" and self.optimizer.n_proc != 1:
            raise ValueError("Pure CFD optimization requires optimizer.n_proc=1")
        if (
            self.mode == "surrogate_nk"
            and self.nk.selection == "all"
            and self.optimizer.n_proc != 1
        ):
            raise ValueError(
                "All-candidate surrogate-NK uses resident MPI pools and "
                "requires optimizer.n_proc=1"
            )
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _section(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    return dict(value)


def load_optimization_config(path: str | Path) -> OptimizationConfig:
    config_path = Path(path).expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Optimization config must be a JSON object")

    baseline_value = payload.get("baseline_dir")
    if baseline_value is None:
        baseline_name = str(payload.get("baseline", "rae2822"))
        baseline = PROJECT_ROOT / "optimization" / "baselines" / baseline_name
    else:
        baseline = Path(str(baseline_value)).expanduser()
        if not baseline.is_absolute():
            baseline = config_path.parent / baseline

    output = Path(str(payload.get("output_dir", "run"))).expanduser()
    if not output.is_absolute():
        output = config_path.parent / output

    task = _section(payload, "task")
    geometry = _section(payload, "geometry")
    serving = _section(payload, "serving")
    cfd = _section(payload, "cfd")
    nk = _section(payload, "nk")
    legacy_repeated_nk_cycles = nk.pop("adaptive_cycles", None)
    if legacy_repeated_nk_cycles is not None and "resume_mode" not in nk:
        nk["resume_mode"] = "repeated_nk"
    repeated_nk_cycles = nk.pop(
        "repeated_nk_cycles",
        legacy_repeated_nk_cycles or (6, 8, 10),
    )
    objective = _section(payload, "objective")
    optimizer = _section(payload, "optimizer")
    initial_population = str(optimizer.get("initial_population", ""))
    if initial_population:
        initial_path = Path(initial_population).expanduser()
        if not initial_path.is_absolute():
            initial_path = config_path.parent / initial_path
        optimizer["initial_population"] = str(initial_path.resolve())

    return OptimizationConfig(
        mode=str(payload.get("mode", "")).strip().lower(),
        output_dir=str(output.resolve()),
        baseline_dir=str(baseline.resolve()),
        use_enhanced_cst=bool(payload.get("use_enhanced_cst", True)),
        task=TaskConfig(
            mach=_tuple_float(task.pop("mach", None), (0.71, 0.72, 0.73)),
            aoa_bounds=tuple(
                _tuple_float(task.pop("aoa_bounds", None), (1.0, 5.0))
            ),
            **task,
        ),
        geometry=GeometryConfig(**geometry),
        serving=ServingConfig(**serving),
        cfd=CFDConfig(**cfd),
        nk=NKConfig(
            repeated_nk_cycles=tuple(
                int(value) for value in repeated_nk_cycles
            ),
            **nk,
        ),
        objective=ObjectiveConfig(
            curvature_hf_region=tuple(
                _tuple_float(
                    objective.pop("curvature_hf_region", None),
                    (0.15, 0.80),
                )
            ),
            curvature_hf_filter_region=tuple(
                _tuple_float(
                    objective.pop("curvature_hf_filter_region", None),
                    (0.10, 0.85),
                )
            ),
            **objective,
        ),
        optimizer=OptimizerConfig(**optimizer),
    )


def write_optimization_config(config: OptimizationConfig, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


__all__ = [
    "CFDConfig",
    "GeometryConfig",
    "NKConfig",
    "ObjectiveConfig",
    "OptimizationConfig",
    "OptimizerConfig",
    "ServingConfig",
    "TaskConfig",
    "load_optimization_config",
    "write_optimization_config",
]
