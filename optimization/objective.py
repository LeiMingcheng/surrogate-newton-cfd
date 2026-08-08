"""Common candidate preparation, objective, and artifact writing."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from optimization.config import ObjectiveConfig, OptimizationConfig
from optimization.contracts import CandidateEvaluation
from optimization.evaluators import CandidateEvaluator, create_evaluator
from optimization.geometry_parameterization import (
    PreparedCandidateGeometry,
    build_airfoil_from_design,
    compose_design_vector,
    design_variable_count,
    prepare_candidate_geometry,
)


OUTPUT_NAMES = (
    "obj",
    "CdAvg",
    "w",
    "conf",
    "AoAmax",
    "AoAmin",
    "Cm",
    "t15",
    "le_cst_viol",
    "le_cst_pen",
    "area",
    "da",
    "loss_mass",
    "loss_momx",
    "loss_momy",
)


def _read_design(path: Path, count: int) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < count:
        raise ValueError(f"{path} has {len(lines)} lines; expected at least {count}")
    return np.asarray([float(lines[index].split()[-1]) for index in range(count)])


def _load_geometry(workdir: Path, config: OptimizationConfig) -> PreparedCandidateGeometry:
    baseline = Path(config.baseline_dir)
    design = _read_design(
        workdir / "input.txt",
        design_variable_count(config.use_enhanced_cst),
    )
    return prepare_candidate_geometry(
        design,
        baseline_cst_u=np.loadtxt(baseline / "cst_u0.txt"),
        baseline_cst_l=np.loadtxt(baseline / "cst_l0.txt"),
        baseline_t_max=float(np.loadtxt(baseline / "t0.txt")),
        nn=config.geometry.points,
        tail=config.geometry.tail,
        preserve_baseline_area=config.geometry.preserve_baseline_area,
        use_enhanced=config.use_enhanced_cst,
    )


def _curvature_hf_energy(
    x: np.ndarray,
    surface: np.ndarray,
    objective: ObjectiveConfig,
) -> float:
    first = np.gradient(surface, x, edge_order=2)
    second = np.gradient(first, x, edge_order=2)
    curvature = second / np.power(1.0 + first * first, 1.5)
    d_kappa_dx = np.gradient(curvature, x, edge_order=2)
    ds_dx = np.sqrt(1.0 + first * first)
    d_kappa_ds = d_kappa_dx / ds_dx

    filter_keep = (
        (x >= objective.curvature_hf_filter_region[0])
        & (x <= objective.curvature_hf_filter_region[1])
    )
    filtered_x = x[filter_keep]
    filtered_d_kappa_ds = d_kappa_ds[filter_keep]
    smooth_trend = gaussian_filter1d(
        filtered_d_kappa_ds,
        sigma=objective.curvature_hf_filter_sigma / float(x[1] - x[0]),
        mode="reflect",
        truncate=4.0,
    )
    high_frequency_residual = filtered_d_kappa_ds - smooth_trend
    score_keep = (
        (filtered_x >= objective.curvature_hf_region[0])
        & (filtered_x <= objective.curvature_hf_region[1])
    )
    return float(
        np.trapz(
            high_frequency_residual[score_keep] ** 2
            * ds_dx[filter_keep][score_keep],
            filtered_x[score_keep],
        )
    )


@lru_cache(maxsize=None)
def _baseline_curvature_hf_energies(
    baseline_dir: str,
    use_enhanced_cst: bool,
    tail: float,
    objective: ObjectiveConfig,
) -> tuple[float, float]:
    baseline = Path(baseline_dir)
    baseline_design = compose_design_vector(
        np.loadtxt(baseline / "cst_u0.txt"),
        np.loadtxt(baseline / "cst_l0.txt"),
        use_enhanced=use_enhanced_cst,
    )
    x = np.linspace(
        0.0,
        1.0,
        objective.curvature_hf_points,
        dtype=np.float64,
    )
    _, y_upper, y_lower, _, _ = build_airfoil_from_design(
        objective.curvature_hf_points,
        baseline_design,
        tail=tail,
        t=float(np.loadtxt(baseline / "t0.txt")),
        x=x,
        use_enhanced=use_enhanced_cst,
    )
    return (
        _curvature_hf_energy(x, y_upper, objective),
        _curvature_hf_energy(x, y_lower, objective),
    )


def curvature_hf_score(
    geometry: PreparedCandidateGeometry,
    config: OptimizationConfig,
) -> float:
    objective = config.objective
    x = np.linspace(
        0.0,
        1.0,
        objective.curvature_hf_points,
        dtype=np.float64,
    )
    _, y_upper, y_lower, _, _ = build_airfoil_from_design(
        objective.curvature_hf_points,
        geometry.design_vector,
        tail=config.geometry.tail,
        t=geometry.t_max,
        x=x,
        use_enhanced=config.use_enhanced_cst,
    )
    baseline_upper, baseline_lower = _baseline_curvature_hf_energies(
        config.baseline_dir,
        config.use_enhanced_cst,
        config.geometry.tail,
        objective,
    )
    upper_ratio = _curvature_hf_energy(x, y_upper, objective) / baseline_upper
    lower_ratio = _curvature_hf_energy(x, y_lower, objective) / baseline_lower
    return float(
        objective.curvature_hf_upper_weight * upper_ratio
        + objective.curvature_hf_lower_weight * lower_ratio
    )


def _outputs_from_evaluation(
    evaluation: CandidateEvaluation,
    geometry: PreparedCandidateGeometry,
    config: OptimizationConfig,
) -> tuple[dict[str, float], dict[str, float | bool | None]]:
    residual_values = [
        abs(float(point.residual))
        for point in evaluation.points
        if point.residual is not None
    ]
    residual = float(np.mean(residual_values)) if residual_values else 0.0
    curvature_score = (
        curvature_hf_score(geometry, config)
        if config.objective.curvature_hf_weight > 0.0
        else None
    )
    curvature_penalty = (
        0.0
        if curvature_score is None
        else config.objective.curvature_hf_weight * curvature_score
    )
    residual_penalty = config.objective.residual_weight * residual
    raw_objective = evaluation.cd_average + residual_penalty + curvature_penalty
    nonpositive_drag = any(float(point.cd) <= 0.0 for point in evaluation.points)
    nonphysical_drag_penalty_applied = bool(
        config.objective.penalize_nonpositive_drag and nonpositive_drag
    )
    objective = raw_objective
    failure_penalty_applied = bool(
        not evaluation.converged or nonphysical_drag_penalty_applied
    )
    if failure_penalty_applied:
        objective += config.objective.failure_penalty
    outputs = {
        "obj": float(objective),
        "CdAvg": float(evaluation.cd_average),
        "w": float(config.objective.residual_weight),
        "conf": residual,
        "AoAmax": max(point.aoa for point in evaluation.points),
        "AoAmin": min(point.aoa for point in evaluation.points),
        "Cm": min(point.cm for point in evaluation.points),
        "t15": geometry.t15_margin,
        "le_cst_viol": geometry.le_cst_viol,
        "le_cst_pen": geometry.le_cst_pen,
        "area": geometry.area,
        "da": geometry.shape_distance,
        "loss_mass": residual,
        "loss_momx": residual,
        "loss_momy": residual,
    }
    terms: dict[str, Any] = {
        "cd_average": float(evaluation.cd_average),
        "raw_objective": float(raw_objective),
        "residual": residual,
        "residual_weight": float(config.objective.residual_weight),
        "residual_penalty": float(residual_penalty),
        "curvature_hf_score": curvature_score,
        "curvature_hf_weight": float(config.objective.curvature_hf_weight),
        "curvature_hf_penalty": float(curvature_penalty),
        "failure_penalty_applied": failure_penalty_applied,
        "nonpositive_drag": nonpositive_drag,
        "nonphysical_drag_penalty_applied": (
            nonphysical_drag_penalty_applied
        ),
        "fitness_rejection_reason": (
            "nonpositive_drag"
            if nonphysical_drag_penalty_applied
            else None
        ),
        "failure_penalty": (
            float(config.objective.failure_penalty)
            if failure_penalty_applied
            else 0.0
        ),
        "objective": float(objective),
    }
    return outputs, terms


def _penalty_outputs(
    geometry: PreparedCandidateGeometry,
    config: OptimizationConfig,
) -> dict[str, float]:
    return {
        "obj": config.objective.failure_penalty * (1.0 + geometry.le_cst_pen),
        "CdAvg": 1.0,
        "w": config.objective.residual_weight,
        "conf": 0.0,
        "AoAmax": float(np.mean(config.task.aoa_bounds)),
        "AoAmin": float(np.mean(config.task.aoa_bounds)),
        "Cm": 0.0,
        "t15": geometry.t15_margin,
        "le_cst_viol": max(geometry.le_cst_viol, 1.0 if not geometry.valid else 0.0),
        "le_cst_pen": geometry.le_cst_pen,
        "area": geometry.area,
        "da": geometry.shape_distance,
        "loss_mass": 0.0,
        "loss_momx": 0.0,
        "loss_momy": 0.0,
    }


def _write_output(path: Path, outputs: dict[str, float]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for name in OUTPUT_NAMES:
            handle.write(f"  {name}  {outputs[name]:.12e}\n")


def _write_cd_series(
    path: Path,
    config: OptimizationConfig,
    evaluation: CandidateEvaluation | None,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("VARIABLES= Ma CLTARG AoA CL CD Cm converged\n")
        handle.write(f'ZONE T="{config.mode}" I={len(config.task.mach)}\n')
        if evaluation is None:
            for mach in config.task.mach:
                handle.write(
                    f" {mach:.9f} {config.task.target_cl:.9f} "
                    f"{np.mean(config.task.aoa_bounds):.9f} 0.0 1.0 0.0 0\n"
                )
            return
        for point in evaluation.points:
            handle.write(
                f" {point.mach:.9f} {point.target_cl:.9f} {point.aoa:.9f} "
                f"{point.cl:.9f} {point.cd:.9f} {point.cm:.9f} "
                f"{int(point.converged)}\n"
            )


def _geometry_summary(
    geometry: PreparedCandidateGeometry,
    *,
    max_physical_t_max: float | None,
) -> dict[str, Any]:
    physical_valid = bool(
        max_physical_t_max is None
        or geometry.t_max <= float(max_physical_t_max)
    )
    return {
        "geometry27": geometry.geometry27.tolist(),
        "t_max": geometry.t_max,
        "leading_edge_radius": geometry.leading_edge_radius,
        "area": geometry.area,
        "baseline_area": geometry.baseline_area,
        "t15_margin": geometry.t15_margin,
        "shape_distance": geometry.shape_distance,
        "le_cst_viol": geometry.le_cst_viol,
        "le_cst_pen": geometry.le_cst_pen,
        "parameterization_valid": geometry.valid,
        "physical_valid": physical_valid,
        "physical_rejection_reason": (
            None if physical_valid else "t_max_exceeds_physical_limit"
        ),
        "max_physical_t_max": max_physical_t_max,
        "valid": bool(geometry.valid and physical_valid),
    }


def evaluate_workdir(
    workdir: str | Path,
    config: OptimizationConfig,
    *,
    refinement: bool = False,
    evaluator: CandidateEvaluator | None = None,
) -> dict[str, float]:
    """Evaluate one AeroOpt calculation directory through the selected evaluator."""

    directory = Path(workdir).resolve()
    geometry = _load_geometry(directory, config)
    if config.geometry.write_candidate_foil:
        geometry.write_foil(directory / "foil.dat")
    evaluation: CandidateEvaluation | None = None
    physical_valid = bool(
        config.geometry.max_physical_t_max is None
        or geometry.t_max <= float(config.geometry.max_physical_t_max)
    )
    if geometry.valid and physical_valid:
        selected_evaluator = (
            evaluator
            if evaluator is not None
            else create_evaluator(config, refinement=refinement)
        )
        evaluation = selected_evaluator.evaluate(
            geometry,
            directory,
        )
        outputs, objective_terms = _outputs_from_evaluation(
            evaluation,
            geometry,
            config,
        )
    else:
        outputs = _penalty_outputs(geometry, config)
        rejection_reason = (
            "parameterization_invalid"
            if not geometry.valid
            else "t_max_exceeds_physical_limit"
        )
        objective_terms = {
            "cd_average": None,
            "raw_objective": None,
            "residual": None,
            "residual_weight": float(config.objective.residual_weight),
            "residual_penalty": 0.0,
            "curvature_hf_score": None,
            "curvature_hf_weight": float(config.objective.curvature_hf_weight),
            "curvature_hf_penalty": 0.0,
            "failure_penalty_applied": True,
            "nonpositive_drag": False,
            "nonphysical_drag_penalty_applied": False,
            "fitness_rejection_reason": rejection_reason,
            "failure_penalty": float(outputs["obj"]),
            "objective": float(outputs["obj"]),
        }
    _write_output(directory / "output.txt", outputs)
    _write_cd_series(directory / "cd_series.dat", config, evaluation)
    artifact = {
        "schema_version": "optimization_candidate_v1",
        "mode": config.mode,
        "refinement": bool(refinement),
        "geometry": _geometry_summary(
            geometry,
            max_physical_t_max=config.geometry.max_physical_t_max,
        ),
        "evaluation": None if evaluation is None else evaluation.to_dict(),
        "objective_terms": objective_terms,
        "outputs": outputs,
    }
    (directory / "evaluation.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return outputs


def output_vector(
    outputs: dict[str, float],
    *,
    names: tuple[str, ...] | list[str] = OUTPUT_NAMES,
) -> np.ndarray:
    return np.asarray([outputs[name] for name in names], dtype=np.float64)


__all__ = [
    "OUTPUT_NAMES",
    "curvature_hf_score",
    "evaluate_workdir",
    "output_vector",
]
