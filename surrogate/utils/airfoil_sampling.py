"""Internal airfoil surface sampling used before pyHyp mesh generation."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def conical_spacing(
    start: float,
    end: float,
    n: int,
    *,
    m: float = math.pi,
    coeff: float = 1.0,
) -> np.ndarray:
    """Return the conical spacing distribution used by the dataset mesh pipeline."""
    if int(n) <= 0:
        raise ValueError(f"n must be positive, got {n}")
    x = np.linspace(float(m), 0.0, int(n), dtype=np.float64)
    b = float(coeff)
    if b >= 1.0:
        denom = np.sqrt(np.cos(x) ** 2 + np.sin(x) ** 2 / (b * b))
        spacing = 0.5 * (1.0 + np.cos(x) / denom)
    else:
        cos_x = np.cos(x)
        spacing = (((cos_x + 1.0) * 0.5) - x[::-1] / math.pi) * b + x[::-1] / math.pi
    return spacing * (float(end) - float(start)) + float(start)


def joined_conical_spacing(
    n: int,
    *,
    s_le: float,
    upper_m: float = math.pi,
    upper_coeff: float = 1.0,
    lower_m: float = math.pi,
    lower_coeff: float = 1.0,
) -> np.ndarray:
    """Join upper and lower conical distributions at the leading edge."""
    s_le = float(np.clip(s_le, 1.0e-8, 1.0 - 1.0e-8))
    upper = conical_spacing(
        0.0,
        s_le,
        int(int(n) * s_le) + 1,
        m=float(upper_m),
        coeff=float(upper_coeff),
    )
    lower = conical_spacing(
        s_le,
        1.0,
        int(int(n) - int(n) * s_le) + 1,
        m=float(lower_m),
        coeff=float(lower_coeff),
    )
    return np.concatenate([upper, lower[1:]])


def build_openloop_airfoil_coords(xx: np.ndarray, yu: np.ndarray, yl: np.ndarray) -> np.ndarray:
    """Build TE-upper -> LE -> TE-lower open-loop surface coordinates."""
    return np.vstack(
        [
            np.column_stack([np.asarray(xx, dtype=np.float64)[::-1], np.asarray(yu, dtype=np.float64)[::-1]]),
            np.column_stack([np.asarray(xx, dtype=np.float64)[1:], np.asarray(yl, dtype=np.float64)[1:]]),
        ]
    ).astype(np.float64)


def resample_openloop_airfoil_coords(
    coords: np.ndarray,
    *,
    upper_count: int,
) -> np.ndarray:
    """Downsample dense open-loop coordinates before normalization and spline-free sampling."""
    coords_arr = np.asarray(coords, dtype=np.float64)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
        raise ValueError(f"Expected open-loop coordinates with shape (N,2), got {coords_arr.shape}")
    split_idx = int(np.argmin(coords_arr[:, 0]))
    upper = coords_arr[: split_idx + 1]
    lower = coords_arr[split_idx:]
    if upper.shape[0] < 2 or lower.shape[0] < 2:
        raise ValueError(f"Invalid open-loop coordinates for resampling: {coords_arr.shape}")
    upper_indices = np.linspace(0, upper.shape[0] - 1, int(upper_count), dtype=np.float64)
    lower_indices = np.linspace(0, lower.shape[0] - 1, int(upper_count), dtype=np.float64)
    upper_sampled = upper[np.unique(np.round(upper_indices).astype(np.int64))]
    lower_sampled = lower[np.unique(np.round(lower_indices).astype(np.int64))][1:]
    sampled = np.vstack([upper_sampled, lower_sampled]).astype(np.float64)
    if sampled.shape[0] < 4:
        raise ValueError(f"Resampled open-loop coordinates are too short: {sampled.shape}")
    return sampled


def normalize_openloop_airfoil(coords: np.ndarray) -> tuple[np.ndarray, float]:
    """Move LE to origin, rotate chord to +x, and normalize chord length to one."""
    coords_arr = np.asarray(coords, dtype=np.float64)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 2 or coords_arr.shape[0] < 4:
        raise ValueError(f"Invalid open-loop airfoil coordinates: {coords_arr.shape}")

    te = 0.5 * (coords_arr[0] + coords_arr[-1])
    le_index = int(np.argmax(np.linalg.norm(coords_arr - te[None, :], axis=1)))
    le = coords_arr[le_index]
    chord_vec = te - le
    chord = float(np.linalg.norm(chord_vec))
    if not np.isfinite(chord) or chord <= 0.0:
        raise ValueError("Airfoil chord must be finite and positive")

    translated = coords_arr - le[None, :]
    theta = -math.atan2(float(chord_vec[1]), float(chord_vec[0]))
    c = math.cos(theta)
    s = math.sin(theta)
    rotation = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    normalized = translated @ rotation.T / chord
    return normalized, float(le_index) / float(max(coords_arr.shape[0] - 1, 1))


def sample_pyhyp_surface_points(
    coords: np.ndarray,
    *,
    n_pts: int,
    n_te_pts: int,
    chord: float = 1.0,
    upper_sampling_m: Optional[float] = None,
    upper_sampling_coeff: Optional[float] = None,
    lower_sampling_m: Optional[float] = None,
    lower_sampling_coeff: Optional[float] = None,
) -> np.ndarray:
    """Sample normalized open-loop airfoil coordinates for pyHyp Plot3D input.

    This is the project-local equivalent of the former prefoil surface step used
    by the dataset pipeline. It covers the runtime path needed by surrogate
    inference: normalize to unit chord, apply conical upper/lower spacing, scale
    by chord, add blunt-TE points, and close the surface loop.
    """
    coords_arr = resample_openloop_airfoil_coords(np.asarray(coords, dtype=np.float64), upper_count=501)
    normalized, s_le = normalize_openloop_airfoil(coords_arr)

    deltas = np.linalg.norm(np.diff(normalized, axis=0), axis=1)
    path = np.concatenate([[0.0], np.cumsum(deltas)])
    total = float(path[-1])
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Airfoil surface path length must be finite and positive")
    path /= total
    s_le = float(path[int(np.argmax(np.linalg.norm(coords_arr - (0.5 * (coords_arr[0] + coords_arr[-1]))[None, :], axis=1)))])

    sampling_s = joined_conical_spacing(
        int(n_pts),
        s_le=s_le,
        upper_m=math.pi if upper_sampling_m is None else float(upper_sampling_m),
        upper_coeff=1.0 if upper_sampling_coeff is None else float(upper_sampling_coeff),
        lower_m=math.pi if lower_sampling_m is None else float(lower_sampling_m),
        lower_coeff=1.0 if lower_sampling_coeff is None else float(lower_sampling_coeff),
    )
    sampled = np.column_stack(
        [
            np.interp(sampling_s, path, normalized[:, 0]),
            np.interp(sampling_s, path, normalized[:, 1]),
        ]
    )

    sampled *= float(chord)
    if int(n_te_pts) > 0 and np.linalg.norm(sampled[0] - sampled[-1]) > 1.0e-14:
        te_points = np.linspace(sampled[-1], sampled[0], int(n_te_pts) + 2, dtype=np.float64)[1:-1]
        sampled = np.vstack([sampled, te_points])
    if np.linalg.norm(sampled[0] - sampled[-1]) > 1.0e-14:
        sampled = np.vstack([sampled, sampled[0]])

    if sampled.ndim != 2 or sampled.shape[1] != 2:
        raise ValueError(f"Invalid sampled surface coordinates: {sampled.shape}")
    return sampled.astype(np.float64, copy=False)


__all__ = [
    "build_openloop_airfoil_coords",
    "conical_spacing",
    "joined_conical_spacing",
    "normalize_openloop_airfoil",
    "resample_openloop_airfoil_coords",
    "sample_pyhyp_surface_points",
]
