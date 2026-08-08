from dataclasses import dataclass
import os
from pathlib import Path
import numpy as np

from surrogate.utils.cst import enhanced_cst_foil


N_BASE_CST = 10
N_ENHANCED_LE = 3
N_ENHANCED_TOTAL = N_BASE_CST * 2 + N_ENHANCED_LE * 2

# Supercritical dataset-derived enhanced LE-CST scales.
# Constraint limits use combined_supercritical abs(p99); initialization uses abs(p90).
ENHANCED_U_LE_ABS_LIMITS = np.array(
    [0.0014870738005265593, 0.00428439749404788, 0.005393652189522982],
    dtype=np.float64,
)
ENHANCED_L_LE_ABS_LIMITS = np.array(
    [0.0016277555841952562, 0.0046032872051000595, 0.005626779980957508],
    dtype=np.float64,
)
ENHANCED_U_LE_INIT_ABS_LIMITS = np.array(
    [0.001382918516173959, 0.004052681848406792, 0.0052255685441195965],
    dtype=np.float64,
)
ENHANCED_L_LE_INIT_ABS_LIMITS = np.array(
    [0.0015979800373315811, 0.004538261331617832, 0.0055783591233193874],
    dtype=np.float64,
)
ENHANCED_LE_ABS_LIMITS = np.concatenate([ENHANCED_U_LE_ABS_LIMITS, ENHANCED_L_LE_ABS_LIMITS])
ENHANCED_LE_INIT_ABS_LIMITS = np.concatenate(
    [ENHANCED_U_LE_INIT_ABS_LIMITS, ENHANCED_L_LE_INIT_ABS_LIMITS]
)


def use_enhanced_cst(default: bool = True) -> bool:
    value = os.environ.get('OPT_USE_ENHANCED_CST')
    if value is None:
        return default
    return value.strip().lower() not in {'0', 'false', 'no', 'off'}


def design_variable_count(use_enhanced: bool) -> int:
    return N_ENHANCED_TOTAL if use_enhanced else N_BASE_CST * 2


def settings_filename(use_enhanced: bool) -> str:
    return 'Settings_enhanced.txt' if use_enhanced else 'Settings.txt'


def compute_le_cst_constraint_metrics(
    design_vector: np.ndarray,
    use_enhanced: bool | None = None,
) -> dict[str, np.ndarray | float]:
    parts = split_design_vector(design_vector, use_enhanced=use_enhanced)

    le_cst = np.concatenate([parts['cst_u_le'], parts['cst_l_le']]).astype(np.float64, copy=False)
    limits = ENHANCED_LE_ABS_LIMITS.astype(np.float64, copy=False)
    ratio = np.divide(np.abs(le_cst), limits, out=np.zeros_like(le_cst), where=limits > 0.0)
    excess = np.maximum(ratio - 1.0, 0.0)

    return {
        'le_cst': le_cst,
        'le_cst_limits': limits,
        'le_cst_ratio': ratio,
        'le_cst_excess': excess,
        'le_cst_viol': float(np.max(excess)),
        'le_cst_pen': float(np.mean(excess ** 2)),
    }


def compose_design_vector(
    cst_u: np.ndarray,
    cst_l: np.ndarray,
    cst_u_le: np.ndarray | None = None,
    cst_l_le: np.ndarray | None = None,
    use_enhanced: bool = True,
) -> np.ndarray:
    cst_u = np.asarray(cst_u, dtype=np.float64).reshape(N_BASE_CST)
    cst_l = np.asarray(cst_l, dtype=np.float64).reshape(N_BASE_CST)

    if not use_enhanced:
        return np.concatenate([cst_u, cst_l])

    if cst_u_le is None:
        cst_u_le = np.zeros(N_ENHANCED_LE, dtype=np.float64)
    if cst_l_le is None:
        cst_l_le = np.zeros(N_ENHANCED_LE, dtype=np.float64)

    return np.concatenate([
        cst_u,
        np.asarray(cst_u_le, dtype=np.float64).reshape(N_ENHANCED_LE),
        cst_l,
        np.asarray(cst_l_le, dtype=np.float64).reshape(N_ENHANCED_LE),
    ])


def split_design_vector(design_vector: np.ndarray, use_enhanced: bool | None = None) -> dict:
    vector = np.asarray(design_vector, dtype=np.float64).reshape(-1)

    if use_enhanced is None:
        if vector.size == N_ENHANCED_TOTAL:
            use_enhanced = True
        elif vector.size == N_BASE_CST * 2:
            use_enhanced = False
        else:
            raise ValueError(f'Unsupported design vector size: {vector.size}')

    if use_enhanced:
        if vector.size != N_ENHANCED_TOTAL:
            raise ValueError(f'Expected {N_ENHANCED_TOTAL} vars, got {vector.size}')
        cst_u = vector[0:10]
        cst_u_le = vector[10:13]
        cst_l = vector[13:23]
        cst_l_le = vector[23:26]
    else:
        if vector.size != N_BASE_CST * 2:
            raise ValueError(f'Expected 20 vars, got {vector.size}')
        cst_u = vector[0:10]
        cst_u_le = np.zeros(N_ENHANCED_LE, dtype=np.float64)
        cst_l = vector[10:20]
        cst_l_le = np.zeros(N_ENHANCED_LE, dtype=np.float64)

    geometry26 = compose_design_vector(
        cst_u,
        cst_l,
        cst_u_le=cst_u_le,
        cst_l_le=cst_l_le,
        use_enhanced=True,
    )

    return {
        'use_enhanced': bool(use_enhanced),
        'cst_u': cst_u,
        'cst_u_le': cst_u_le,
        'cst_l': cst_l,
        'cst_l_le': cst_l_le,
        'geometry26': geometry26,
    }


def build_airfoil_from_design(
    nn: int,
    design_vector: np.ndarray,
    tail: float,
    t: float | None = None,
    x: np.ndarray | None = None,
    use_enhanced: bool | None = None,
):
    parts = split_design_vector(design_vector, use_enhanced=use_enhanced)

    if parts['use_enhanced']:
        return enhanced_cst_foil(
            nn,
            parts['cst_u'],
            parts['cst_l'],
            cst_le_u=parts['cst_u_le'],
            cst_le_l=parts['cst_l_le'],
            x=x,
            t=t,
            tail=tail,
        )

    return enhanced_cst_foil(
        nn,
        parts['cst_u'],
        parts['cst_l'],
        x=x,
        t=t,
        tail=tail,
    )


def airfoil_area(x: np.ndarray, thickness: np.ndarray) -> float:
    """Match the historical cell-centered area integration."""

    x_array = np.asarray(x, dtype=np.float64)
    dy = np.asarray(thickness, dtype=np.float64)
    centers = 0.5 * (x_array[1:] + x_array[:-1])
    return float(np.sum((centers[1:] - centers[:-1]) * dy[1:-1]))


@dataclass(frozen=True)
class PreparedCandidateGeometry:
    design_vector: np.ndarray
    geometry27: np.ndarray
    x: np.ndarray
    y_upper: np.ndarray
    y_lower: np.ndarray
    t_max: float
    leading_edge_radius: float
    area: float
    baseline_area: float
    t15_margin: float
    shape_distance: float
    le_cst_viol: float
    le_cst_pen: float

    @property
    def valid(self) -> bool:
        return bool(
            self.le_cst_viol <= 0.0
            and np.all(self.y_upper - self.y_lower >= -1.0e-4)
        )

    def write_foil(self, path: str | Path) -> Path:
        target = Path(path)
        with target.open("w", encoding="utf-8") as handle:
            handle.write("Variables= X Y\n")
            handle.write(f"zone i= {len(self.x)}\n")
            for x_value, y_value in zip(self.x, self.y_upper):
                handle.write(f"   {x_value:.9f}  {y_value:.9f}\n")
            handle.write(f"\nzone i= {len(self.x)}\n")
            for x_value, y_value in zip(self.x, self.y_lower):
                handle.write(f"   {x_value:.9f}  {y_value:.9f}\n")
        return target


def prepare_candidate_geometry(
    design_vector: np.ndarray,
    *,
    baseline_cst_u: np.ndarray,
    baseline_cst_l: np.ndarray,
    baseline_t_max: float,
    nn: int = 1001,
    tail: float = 0.002,
    preserve_baseline_area: bool = True,
    use_enhanced: bool = True,
) -> PreparedCandidateGeometry:
    """Apply the one canonical optimization geometry transformation."""

    vector = np.asarray(design_vector, dtype=np.float64).reshape(-1)
    baseline_design = compose_design_vector(
        baseline_cst_u,
        baseline_cst_l,
        use_enhanced=use_enhanced,
    )
    x0, yu0, yl0, _, _ = build_airfoil_from_design(
        nn,
        baseline_design,
        t=float(baseline_t_max),
        tail=float(tail),
        use_enhanced=use_enhanced,
    )
    baseline_area = airfoil_area(x0, yu0 - yl0)
    t15_base = float(yu0[int(nn * 0.15)] - yl0[int(nn * 0.15)])

    x, yu, yl, raw_t_max, _ = build_airfoil_from_design(
        nn,
        vector,
        t=None,
        tail=float(tail),
        use_enhanced=use_enhanced,
    )
    raw_area = airfoil_area(x, yu - yl)
    target_t_max = float(raw_t_max)
    if preserve_baseline_area:
        target_t_max *= baseline_area / raw_area
    x, yu, yl, t_max, radius = build_airfoil_from_design(
        nn,
        vector,
        t=target_t_max,
        tail=float(tail),
        use_enhanced=use_enhanced,
    )
    area = airfoil_area(x, yu - yl)
    shape_distance = airfoil_area(x, np.abs(yu - yu0)) + airfoil_area(
        x, np.abs(yl - yl0)
    )
    t15 = float(yu[int(nn * 0.15)] - yl[int(nn * 0.15)])
    le_metrics = compute_le_cst_constraint_metrics(vector, use_enhanced=use_enhanced)
    parts = split_design_vector(vector, use_enhanced=use_enhanced)
    geometry27 = np.concatenate(
        [np.asarray(parts["geometry26"], dtype=np.float64), [float(tail)]]
    ).astype(np.float32)
    return PreparedCandidateGeometry(
        design_vector=vector,
        geometry27=geometry27,
        x=np.asarray(x, dtype=np.float64),
        y_upper=np.asarray(yu, dtype=np.float64),
        y_lower=np.asarray(yl, dtype=np.float64),
        t_max=float(t_max),
        leading_edge_radius=float(radius),
        area=float(area),
        baseline_area=float(baseline_area),
        t15_margin=float(t15 - 0.9 * t15_base),
        shape_distance=float(shape_distance),
        le_cst_viol=float(le_metrics["le_cst_viol"]),
        le_cst_pen=float(le_metrics["le_cst_pen"]),
    )
