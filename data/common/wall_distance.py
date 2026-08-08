from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def coerce_wall_distance_array(
    value: Any,
    *,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected wall_distance with shape (H, W), got {array.shape}")
    if expected_shape is not None and tuple(array.shape) != tuple(expected_shape):
        raise ValueError(
            f"Expected wall_distance shape {tuple(expected_shape)}, got {tuple(array.shape)}"
        )
    return array


def is_valid_wall_distance(
    value: Any,
    *,
    expected_shape: tuple[int, int] | None = None,
) -> bool:
    try:
        array = coerce_wall_distance_array(value, expected_shape=expected_shape)
    except Exception:
        return False
    if array is None:
        return False
    return bool(np.isfinite(array).all() and float(array.max()) > 0.0)


def _extract_blocks_recursive(obj: Any, blocks_list: list[Any]) -> None:
    import pyvista as pv

    if isinstance(obj, pv.MultiBlock):
        for sub in obj:
            _extract_blocks_recursive(sub, blocks_list)
        return
    if isinstance(obj, pv.DataSet):
        blocks_list.append(obj)


def extract_wall_distance_from_cgns(cgns_path: str | Path) -> np.ndarray | None:
    import pyvista as pv

    path = Path(cgns_path)
    if not path.exists():
        return None

    try:
        reader = pv.CGNSReader(str(path))
        reader.load_boundary_patch = False
        cgns_data = reader.read()

        all_blocks: list[Any] = []
        _extract_blocks_recursive(cgns_data, all_blocks)
        for block in all_blocks:
            if not hasattr(block, "cell_data") or "TurbulentDistance" not in block.cell_data:
                continue

            d2wall_1d = block.cell_data["TurbulentDistance"]
            ni, nj, nk = block.dimensions
            dims = (int(ni), int(nj), int(nk))
            thin_axis = dims.index(2) if 2 in dims else int(np.argmin(dims))
            cell_dims = tuple(max(d - 1, 1) for d in dims)
            d2wall_3d = d2wall_1d.reshape(cell_dims, order="F")
            d2wall_2d = np.take(d2wall_3d, indices=0, axis=thin_axis)
            if d2wall_2d.shape[0] > d2wall_2d.shape[1]:
                d2wall_2d = d2wall_2d.T
            return d2wall_2d.astype(np.float32, copy=False)
        return None
    except Exception:
        return None


__all__ = [
    "coerce_wall_distance_array",
    "extract_wall_distance_from_cgns",
    "is_valid_wall_distance",
]
