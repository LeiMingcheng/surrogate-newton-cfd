"""Canonical CST and enhanced-CST geometry parameterization.

The repository uses one 27D enhanced-CST layout::

    [upper_base10, upper_le3, lower_base10, lower_le3, tail]

The six leading-edge coefficients are the first three coefficients of a
20-function CST basis on each surface.  The trailing-edge term is removed from
both surfaces before fitting and is added exactly once during reconstruction.
"""

from __future__ import annotations

from math import comb
from typing import Optional

import numpy as np


BASE_COEFFICIENTS_PER_SURFACE = 10
ENHANCED_COEFFICIENTS_PER_SURFACE = 3
ENHANCED_PARENT_COEFFICIENTS = 20
CST20_DIM = 20
CST26_DIM = 26
CST27_DIM = 27


def _geometry_deps():
    from cst_modeling.math import clustcos, find_circle_3p, interp_from_curve

    return clustcos, find_circle_3p, interp_from_curve


def _cst_basis(
    x: np.ndarray,
    n_coefficients: int,
    *,
    xn1: float = 0.5,
    xn2: float = 1.0,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    order = int(n_coefficients) - 1
    powers = np.arange(int(n_coefficients), dtype=np.int64)
    bernstein = (
        np.asarray(
            [comb(order, int(index)) for index in powers], dtype=np.float64
        )[None, :]
        * x_arr[:, None] ** powers[None, :]
        * (1.0 - x_arr[:, None]) ** (order - powers)[None, :]
    )
    class_function = x_arr**float(xn1) * (1.0 - x_arr) ** float(xn2)
    return class_function[:, None] * bernstein


def _as_surface(
    x: np.ndarray,
    y: np.ndarray,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"{label} x/y coordinate arrays must have matching shapes")
    minimum_points = (
        BASE_COEFFICIENTS_PER_SURFACE + ENHANCED_COEFFICIENTS_PER_SURFACE
    )
    if x_arr.shape[0] < minimum_points:
        raise ValueError(f"{label} surface has too few points for enhanced-CST fitting")
    if float(x_arr[-1]) <= float(x_arr[0]):
        raise ValueError(
            f"{label} surface must be ordered from leading edge to trailing edge"
        )
    return x_arr, y_arr


def _normalize_surfaces(
    xu: np.ndarray,
    yu: np.ndarray,
    xl: np.ndarray,
    yl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    xu_arr, yu_arr = _as_surface(xu, yu, "Upper")
    xl_arr, yl_arr = _as_surface(xl, yl, "Lower")

    upper_span = float(xu_arr[-1] - xu_arr[0])
    lower_span = float(xl_arr[-1] - xl_arr[0])
    chord = 0.5 * (upper_span + lower_span)

    xu_norm = (xu_arr - xu_arr[0]) / upper_span
    xl_norm = (xl_arr - xl_arr[0]) / lower_span
    yu_norm = (yu_arr - yu_arr[0]) / chord
    yl_norm = (yl_arr - yl_arr[0]) / chord
    tail = float(yu_norm[-1] - yl_norm[-1])
    return xu_norm, yu_norm, xl_norm, yl_norm, tail


def _remove_trailing_edge(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.asarray(y, dtype=np.float64) - float(y[-1]) * np.asarray(x, dtype=np.float64)


def _fit_surface(
    x: np.ndarray,
    y_shape: np.ndarray,
    *,
    xn1: float,
    xn2: float,
) -> tuple[np.ndarray, np.ndarray]:
    base_basis = _cst_basis(
        x,
        BASE_COEFFICIENTS_PER_SURFACE,
        xn1=xn1,
        xn2=xn2,
    )
    base, *_ = np.linalg.lstsq(base_basis, y_shape, rcond=None)
    residual = y_shape - base_basis @ base

    enhanced_basis = _cst_basis(
        x,
        ENHANCED_PARENT_COEFFICIENTS,
        xn1=xn1,
        xn2=xn2,
    )[:, :ENHANCED_COEFFICIENTS_PER_SURFACE]
    enhanced, *_ = np.linalg.lstsq(enhanced_basis, residual, rcond=None)
    return np.asarray(base, dtype=np.float64), np.asarray(enhanced, dtype=np.float64)


def cst_curve(
    nn: int,
    coefficients: np.ndarray,
    x: Optional[np.ndarray] = None,
    *,
    xn1: float = 0.5,
    xn2: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one standard CST surface without a trailing-edge term."""
    if x is None:
        clustcos, _, _ = _geometry_deps()
        x_arr = np.asarray([clustcos(index, int(nn)) for index in range(int(nn))])
    else:
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if x_arr.shape[0] != int(nn):
            raise ValueError("x must have length nn")
    coeffs = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    y = _cst_basis(x_arr, coeffs.shape[0], xn1=xn1, xn2=xn2) @ coeffs
    y[0] = 0.0
    y[-1] = 0.0
    return x_arr, y


def cst_foil_fit(
    xu: np.ndarray,
    yu: np.ndarray,
    xl: np.ndarray,
    yl: np.ndarray,
    *,
    n_cst: int = BASE_COEFFICIENTS_PER_SURFACE,
    xn1: float = 0.5,
    xn2: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit standard CST surfaces after removing their independent TE terms."""
    xu_norm, yu_norm, xl_norm, yl_norm, _ = _normalize_surfaces(xu, yu, xl, yl)
    upper_basis = _cst_basis(xu_norm, int(n_cst), xn1=xn1, xn2=xn2)
    lower_basis = _cst_basis(xl_norm, int(n_cst), xn1=xn1, xn2=xn2)
    upper, *_ = np.linalg.lstsq(
        upper_basis,
        _remove_trailing_edge(xu_norm, yu_norm),
        rcond=None,
    )
    lower, *_ = np.linalg.lstsq(
        lower_basis,
        _remove_trailing_edge(xl_norm, yl_norm),
        rcond=None,
    )
    return np.asarray(upper, dtype=np.float64), np.asarray(lower, dtype=np.float64)


def enhanced_le_cst_fit(
    xu: np.ndarray,
    yu: np.ndarray,
    xl: np.ndarray,
    yl: np.ndarray,
    *,
    xn1: float = 0.5,
    xn2: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit tail-independent base10 + LE3 coefficients for both surfaces."""
    xu_norm, yu_norm, xl_norm, yl_norm, _ = _normalize_surfaces(xu, yu, xl, yl)
    upper_base, upper_le = _fit_surface(
        xu_norm,
        _remove_trailing_edge(xu_norm, yu_norm),
        xn1=xn1,
        xn2=xn2,
    )
    lower_base, lower_le = _fit_surface(
        xl_norm,
        _remove_trailing_edge(xl_norm, yl_norm),
        xn1=xn1,
        xn2=xn2,
    )
    return upper_base, lower_base, upper_le, lower_le


def enhanced_cst_foil(
    nn: int,
    cst_base_u: np.ndarray,
    cst_base_l: np.ndarray,
    *,
    cst_le_u: Optional[np.ndarray] = None,
    cst_le_l: Optional[np.ndarray] = None,
    x: Optional[np.ndarray] = None,
    t: Optional[float] = None,
    tail: float = 0.0,
    xn1: float = 0.5,
    xn2: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Reconstruct base10 + LE3 with the same 20-function enhanced basis."""
    upper_base = np.asarray(cst_base_u, dtype=np.float64).reshape(
        BASE_COEFFICIENTS_PER_SURFACE
    )
    lower_base = np.asarray(cst_base_l, dtype=np.float64).reshape(
        BASE_COEFFICIENTS_PER_SURFACE
    )
    upper_le = (
        np.zeros(ENHANCED_COEFFICIENTS_PER_SURFACE, dtype=np.float64)
        if cst_le_u is None
        else np.asarray(cst_le_u, dtype=np.float64).reshape(
            ENHANCED_COEFFICIENTS_PER_SURFACE
        )
    )
    lower_le = (
        np.zeros(ENHANCED_COEFFICIENTS_PER_SURFACE, dtype=np.float64)
        if cst_le_l is None
        else np.asarray(cst_le_l, dtype=np.float64).reshape(
            ENHANCED_COEFFICIENTS_PER_SURFACE
        )
    )

    if x is None:
        clustcos, _, _ = _geometry_deps()
        x_arr = np.asarray(
            [clustcos(index, int(nn)) for index in range(int(nn))],
            dtype=np.float64,
        )
    else:
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if x_arr.shape[0] != int(nn):
            raise ValueError("x must have length nn")

    base_basis = _cst_basis(
        x_arr,
        BASE_COEFFICIENTS_PER_SURFACE,
        xn1=xn1,
        xn2=xn2,
    )
    enhanced_basis = _cst_basis(
        x_arr,
        ENHANCED_PARENT_COEFFICIENTS,
        xn1=xn1,
        xn2=xn2,
    )[:, :ENHANCED_COEFFICIENTS_PER_SURFACE]
    yu = base_basis @ upper_base + enhanced_basis @ upper_le
    yl = base_basis @ lower_base + enhanced_basis @ lower_le

    thickness = yu - yl
    thickness_index = int(np.argmax(thickness))
    t0 = float(thickness[thickness_index])
    if t is not None:
        ratio = (float(t) - float(tail) * x_arr[thickness_index]) / t0
        yu = yu * ratio
        yl = yl * ratio

    yu = yu + 0.5 * float(tail) * x_arr
    yl = yl - 0.5 * float(tail) * x_arr
    t0 = float(np.max(yu - yl))

    _, find_circle_3p, interp_from_curve = _geometry_deps()
    x_rle = 0.005
    yu_rle = float(interp_from_curve(x_rle, x_arr, yu))
    yl_rle = float(interp_from_curve(x_rle, x_arr, yl))
    r0, _ = find_circle_3p([0.0, 0.0], [x_rle, yu_rle], [x_rle, yl_rle])
    return x_arr, yu, yl, t0, float(r0)


def coords_to_cst27(
    xu: np.ndarray,
    yu: np.ndarray,
    xl: np.ndarray,
    yl: np.ndarray,
    *,
    tail: Optional[float] = None,
) -> np.ndarray:
    """Fit the canonical 27D vector from chord-aligned upper/lower coordinates."""
    xu_norm, yu_norm, xl_norm, yl_norm, fitted_tail = _normalize_surfaces(
        xu, yu, xl, yl
    )
    upper_base, upper_le = _fit_surface(
        xu_norm,
        _remove_trailing_edge(xu_norm, yu_norm),
        xn1=0.5,
        xn2=1.0,
    )
    lower_base, lower_le = _fit_surface(
        xl_norm,
        _remove_trailing_edge(xl_norm, yl_norm),
        xn1=0.5,
        xn2=1.0,
    )
    return np.concatenate(
        [
            upper_base,
            upper_le,
            lower_base,
            lower_le,
            np.asarray([fitted_tail if tail is None else float(tail)]),
        ]
    ).astype(np.float32)


def coords_to_cst27_batch(
    x: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    *,
    tail: Optional[np.ndarray | float] = None,
) -> np.ndarray:
    """Vectorized canonical fitting for surfaces sampled on one shared x grid."""
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    upper_arr = np.asarray(upper, dtype=np.float64)
    lower_arr = np.asarray(lower, dtype=np.float64)
    if upper_arr.ndim != 2 or lower_arr.shape != upper_arr.shape:
        raise ValueError("upper and lower must be matching arrays with shape (N, points)")
    if upper_arr.shape[1] != x_arr.shape[0]:
        raise ValueError("surface point count must match x")

    x_norm = (x_arr - x_arr[0]) / (x_arr[-1] - x_arr[0])
    chord = float(x_arr[-1] - x_arr[0])
    upper_norm = (upper_arr - upper_arr[:, :1]) / chord
    lower_norm = (lower_arr - lower_arr[:, :1]) / chord
    fitted_tail = upper_norm[:, -1] - lower_norm[:, -1]
    upper_shape = upper_norm - upper_norm[:, -1:] * x_norm[None, :]
    lower_shape = lower_norm - lower_norm[:, -1:] * x_norm[None, :]

    base_basis = _cst_basis(x_norm, BASE_COEFFICIENTS_PER_SURFACE)
    enhanced_basis = _cst_basis(x_norm, ENHANCED_PARENT_COEFFICIENTS)[
        :, :ENHANCED_COEFFICIENTS_PER_SURFACE
    ]
    base_pinv = np.linalg.pinv(base_basis)
    enhanced_pinv = np.linalg.pinv(enhanced_basis)
    upper_base = upper_shape @ base_pinv.T
    lower_base = lower_shape @ base_pinv.T
    upper_residual = upper_shape - upper_base @ base_basis.T
    lower_residual = lower_shape - lower_base @ base_basis.T
    upper_le = upper_residual @ enhanced_pinv.T
    lower_le = lower_residual @ enhanced_pinv.T

    if tail is None:
        tail_column = fitted_tail
    else:
        tail_arr = np.asarray(tail, dtype=np.float64)
        tail_column = (
            np.full(upper_arr.shape[0], float(tail_arr))
            if tail_arr.ndim == 0
            else tail_arr.reshape(-1)
        )
        if tail_column.shape[0] != upper_arr.shape[0]:
            raise ValueError("tail array must have one value per geometry")
    return np.column_stack(
        [upper_base, upper_le, lower_base, lower_le, tail_column]
    ).astype(np.float32)


def cst26_shape_metric_transform(x: np.ndarray) -> np.ndarray:
    """Return the per-surface transform that preserves sampled shape RMS.

    The base10 and parent20-LE3 functions are strongly non-orthogonal, so
    Euclidean distances between their raw coefficients are not geometric
    distances. QR orthogonalization of the shared evaluation matrix yields a
    13D coordinate system whose Euclidean norm equals the sampled surface norm.
    The returned matrix maps a row of 13 surface coefficients to that system.
    """
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    x_norm = (x_arr - x_arr[0]) / (x_arr[-1] - x_arr[0])
    basis = np.column_stack(
        [
            _cst_basis(x_norm, BASE_COEFFICIENTS_PER_SURFACE),
            _cst_basis(x_norm, ENHANCED_PARENT_COEFFICIENTS)[
                :, :ENHANCED_COEFFICIENTS_PER_SURFACE
            ],
        ]
    )
    _, upper_triangular = np.linalg.qr(basis, mode="reduced")
    return upper_triangular.T / np.sqrt(2.0 * x_arr.shape[0])


def cst26_to_shape_embedding(
    cst26: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Map CST26 to a 26D embedding with physical surface-RMS distance."""
    coefficients = np.asarray(cst26, dtype=np.float64)
    scalar_input = coefficients.ndim == 1
    if scalar_input:
        coefficients = coefficients.reshape(1, -1)
    if coefficients.ndim != 2 or coefficients.shape[1] != CST26_DIM:
        raise ValueError("cst26 must have shape (26,) or (N, 26)")
    transform = cst26_shape_metric_transform(x)
    embedding = np.column_stack(
        [
            coefficients[:, :13] @ transform,
            coefficients[:, 13:26] @ transform,
        ]
    )
    return embedding[0] if scalar_input else embedding


def enhanced_cst_foil_with_fit(
    xu: np.ndarray,
    yu: np.ndarray,
    xl: np.ndarray,
    yl: np.ndarray,
    *,
    nn: int = 101,
    t: Optional[float] = None,
    tail: Optional[float] = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Fit canonical enhanced CST and reconstruct it on a requested grid."""
    geometry = coords_to_cst27(xu, yu, xl, yl, tail=tail)
    x_out, yu_out, yl_out, t0, r0 = cst27_to_coords(
        geometry,
        nn=nn,
        t_max=t,
        return_metrics=True,
    )
    return (
        x_out,
        yu_out,
        yl_out,
        t0,
        r0,
        geometry[0:10].astype(np.float64),
        geometry[13:23].astype(np.float64),
        geometry[10:13].astype(np.float64),
        geometry[23:26].astype(np.float64),
    )


def cst20_to_coords(
    cst_u: np.ndarray,
    cst_l: np.ndarray,
    *,
    t_max: Optional[float] = None,
    tail: float = 0.0,
    nn: int = 1001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample coordinates from a base-only 10+10 CST pair."""
    x, yu, yl, _, _ = enhanced_cst_foil(
        int(nn),
        np.asarray(cst_u, dtype=np.float64).reshape(BASE_COEFFICIENTS_PER_SURFACE),
        np.asarray(cst_l, dtype=np.float64).reshape(BASE_COEFFICIENTS_PER_SURFACE),
        t=t_max,
        tail=float(tail),
    )
    return x, yu, yl


def scale_cst20_to_max_thickness(
    cst_u: np.ndarray,
    cst_l: np.ndarray,
    *,
    t_max: float,
    tail: float = 0.0,
    nn: int = 1001,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale a CST20 shape to a target thickness before format conversion."""
    upper = np.asarray(cst_u, dtype=np.float64).reshape(BASE_COEFFICIENTS_PER_SURFACE)
    lower = np.asarray(cst_l, dtype=np.float64).reshape(BASE_COEFFICIENTS_PER_SURFACE)
    x, yu, yl = cst20_to_coords(upper, lower, tail=0.0, nn=nn)
    thickness = yu - yl
    index = int(np.argmax(thickness))
    ratio = (float(t_max) - float(tail) * x[index]) / float(thickness[index])
    return upper * ratio, lower * ratio


def cst20_to_cst27(
    cst_u: np.ndarray,
    cst_l: np.ndarray,
    *,
    tail: float = 0.0,
) -> np.ndarray:
    """Directly pack CST20 into CST27 by inserting six zero LE values."""
    upper = np.asarray(cst_u, dtype=np.float64).reshape(
        BASE_COEFFICIENTS_PER_SURFACE
    )
    lower = np.asarray(cst_l, dtype=np.float64).reshape(
        BASE_COEFFICIENTS_PER_SURFACE
    )
    return np.concatenate(
        [
            upper,
            np.zeros(ENHANCED_COEFFICIENTS_PER_SURFACE),
            lower,
            np.zeros(ENHANCED_COEFFICIENTS_PER_SURFACE),
            np.asarray([float(tail)]),
        ]
    ).astype(np.float32)


def cst27_to_coords(
    cst27: np.ndarray,
    *,
    nn: int = 1001,
    x: Optional[np.ndarray] = None,
    t_max: Optional[float] = None,
    return_metrics: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[
    np.ndarray, np.ndarray, np.ndarray, float, float
]:
    """Sample coordinates from the canonical enhanced-CST27 vector."""
    coefficients = np.asarray(cst27, dtype=np.float64).reshape(CST27_DIM)
    point_count = int(nn) if x is None else int(np.asarray(x).size)
    result = enhanced_cst_foil(
        point_count,
        coefficients[0:10],
        coefficients[13:23],
        cst_le_u=coefficients[10:13],
        cst_le_l=coefficients[23:26],
        x=x,
        t=t_max,
        tail=float(coefficients[26]),
    )
    return result if return_metrics else result[:3]


def cst27_to_coords_batch(
    cst27_batch: np.ndarray,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct a batch of canonical CST27 vectors on one shared x grid."""
    coefficients = np.asarray(cst27_batch, dtype=np.float64)
    if coefficients.ndim != 2 or coefficients.shape[1] != CST27_DIM:
        raise ValueError("cst27_batch must have shape (N, 27)")
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    base_basis = _cst_basis(x_arr, BASE_COEFFICIENTS_PER_SURFACE)
    enhanced_basis = _cst_basis(x_arr, ENHANCED_PARENT_COEFFICIENTS)[
        :, :ENHANCED_COEFFICIENTS_PER_SURFACE
    ]
    tail_term = 0.5 * coefficients[:, 26:27] * x_arr[None, :]
    upper = (
        coefficients[:, 0:10] @ base_basis.T
        + coefficients[:, 10:13] @ enhanced_basis.T
        + tail_term
    )
    lower = (
        coefficients[:, 13:23] @ base_basis.T
        + coefficients[:, 23:26] @ enhanced_basis.T
        - tail_term
    )
    upper[:, 0] = 0.0
    lower[:, 0] = 0.0
    return upper, lower


def cst20_to_cst27_batch(
    cst_u_batch: np.ndarray,
    cst_l_batch: np.ndarray,
    *,
    tail: float = 0.0,
) -> np.ndarray:
    """Batch form of the direct CST20-to-CST27 packing conversion."""
    upper = np.asarray(cst_u_batch, dtype=np.float64)
    lower = np.asarray(cst_l_batch, dtype=np.float64)
    if upper.ndim != 2 or upper.shape != lower.shape or upper.shape[1] != 10:
        raise ValueError("cst_u_batch and cst_l_batch must both have shape (N, 10)")
    zeros = np.zeros(
        (upper.shape[0], ENHANCED_COEFFICIENTS_PER_SURFACE),
        dtype=np.float64,
    )
    tails = np.full((upper.shape[0], 1), float(tail), dtype=np.float64)
    return np.column_stack([upper, zeros, lower, zeros, tails]).astype(np.float32)


__all__ = [
    "BASE_COEFFICIENTS_PER_SURFACE",
    "CST20_DIM",
    "CST26_DIM",
    "CST27_DIM",
    "ENHANCED_COEFFICIENTS_PER_SURFACE",
    "ENHANCED_PARENT_COEFFICIENTS",
    "coords_to_cst27",
    "coords_to_cst27_batch",
    "cst20_to_coords",
    "cst20_to_cst27",
    "cst20_to_cst27_batch",
    "cst26_shape_metric_transform",
    "cst26_to_shape_embedding",
    "cst27_to_coords",
    "cst27_to_coords_batch",
    "cst_curve",
    "cst_foil_fit",
    "enhanced_cst_foil",
    "enhanced_cst_foil_with_fit",
    "enhanced_le_cst_fit",
    "scale_cst20_to_max_thickness",
]
