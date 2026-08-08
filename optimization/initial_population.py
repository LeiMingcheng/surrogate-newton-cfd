"""Single initial-population generator used by the optimization driver."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydoe

from optimization.geometry_parameterization import (
    ENHANCED_L_LE_INIT_ABS_LIMITS,
    ENHANCED_U_LE_INIT_ABS_LIMITS,
    N_ENHANCED_LE,
    build_airfoil_from_design,
    compose_design_vector,
    design_variable_count,
)


from cst_modeling.section import foil_bump_modify


def _lhs(dimensions: int, sample_count: int, seed: int) -> np.ndarray:
    if sample_count == 0:
        return np.zeros((0, dimensions))
    criterion = None if sample_count == 1 else "m"
    return pydoe.lhs(
        dimensions,
        samples=sample_count,
        criterion=criterion,
        seed=seed,
    )


def generate_initial_population(
    count: int,
    *,
    baseline_dir: str | Path,
    use_enhanced: bool = True,
    mode: str = "bump",
    nn: int = 1001,
    tail: float = 0.002,
) -> np.ndarray:
    """Generate a deterministic-with-NumPy-seed population around one baseline."""

    baseline = Path(baseline_dir)
    cst_u = np.loadtxt(baseline / "cst_u0.txt")
    cst_l = np.loadtxt(baseline / "cst_l0.txt")
    t_max = float(np.loadtxt(baseline / "t0.txt"))
    baseline_design = compose_design_vector(cst_u, cst_l, use_enhanced=use_enhanced)
    x, yu, yl, _, _ = build_airfoil_from_design(
        nn,
        baseline_design,
        t=t_max,
        tail=tail,
        use_enhanced=use_enhanced,
    )
    samples = np.zeros((int(count), design_variable_count(use_enhanced)))
    samples[0] = baseline_design
    sample_count = max(int(count) - 1, 0)
    seed = int(np.random.randint(0, 2**31 - 1))

    if mode == "bump":
        dimensions = 4 + (2 * N_ENHANCED_LE if use_enhanced else 0)
        lhs = _lhs(dimensions, sample_count, seed)
        for index, values in enumerate(lhs, start=1):
            side_mode = int(values[0] * 4)
            height = 2.0 * (values[2] - 0.25) * 0.08
            side = 1 if side_mode < 2 else -1
            width = 0.3 if side_mode % 2 == 0 else 0.8
            _, _, upper, lower = foil_bump_modify(
                x,
                yu,
                yl,
                xc=values[1],
                h=height,
                s=width,
                side=side,
                n_cst=10,
                return_cst=True,
                keep_tmax=False,
            )
            upper_le = None
            lower_le = None
            if use_enhanced:
                upper_le = (
                    2.0 * values[4 : 4 + N_ENHANCED_LE] - 1.0
                ) * ENHANCED_U_LE_INIT_ABS_LIMITS
                lower_le = (
                    2.0 * values[4 + N_ENHANCED_LE : 4 + 2 * N_ENHANCED_LE] - 1.0
                ) * ENHANCED_L_LE_INIT_ABS_LIMITS
            samples[index] = compose_design_vector(
                upper,
                lower,
                cst_u_le=upper_le,
                cst_l_le=lower_le,
                use_enhanced=use_enhanced,
            )
    elif mode == "lhd":
        dimensions = design_variable_count(use_enhanced) + 1
        lhs = _lhs(dimensions, sample_count, seed)
        for index, values in enumerate(lhs, start=1):
            upper = cst_u * (1.0 + 0.3 * (values[:10] - 0.5))
            lower = cst_l * (1.0 + 0.1 * (values[10:20] - 0.5))
            upper_le = None
            lower_le = None
            if use_enhanced:
                upper_le = (
                    2.0 * values[20 : 20 + N_ENHANCED_LE] - 1.0
                ) * ENHANCED_U_LE_INIT_ABS_LIMITS
                lower_le = (
                    2.0 * values[20 + N_ENHANCED_LE : 20 + 2 * N_ENHANCED_LE] - 1.0
                ) * ENHANCED_L_LE_INIT_ABS_LIMITS
            samples[index] = compose_design_vector(
                upper,
                lower,
                cst_u_le=upper_le,
                cst_l_le=lower_le,
                use_enhanced=use_enhanced,
            )
    else:
        raise ValueError("initial population mode must be 'bump' or 'lhd'")
    return samples


__all__ = ["generate_initial_population"]
