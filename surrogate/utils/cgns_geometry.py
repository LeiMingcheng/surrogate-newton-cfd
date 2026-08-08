"""Read dataset-layout 2D O-grid coordinates from CGNS."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _structured_grids(data: Any) -> list[Any]:
    import pyvista as pv

    if isinstance(data, pv.MultiBlock):
        grids: list[Any] = []
        for block in data:
            grids.extend(_structured_grids(block))
        return grids
    if isinstance(data, pv.StructuredGrid):
        return [data]
    return []


def load_cgns_geometry_2d(cgns_path: str | Path) -> dict[str, np.ndarray]:
    """Load a thin 3D CGNS O-grid as dataset-layout 2D coordinates."""

    import pyvista as pv

    path = Path(cgns_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CGNS file not found: {path}")

    reader = pv.CGNSReader(str(path))
    reader.load_boundary_patch = False
    grids = _structured_grids(reader.read())
    if not grids:
        raise ValueError(f"No StructuredGrid block found in CGNS file: {path}")

    block = max(grids, key=lambda item: int(item.n_points))
    node_dims = tuple(int(value) for value in block.dimensions)
    thin_axes = [axis for axis, size in enumerate(node_dims) if size == 2]
    if len(thin_axes) != 1:
        raise ValueError(
            f"Expected one two-node thin axis in 2D CGNS grid, got {node_dims}: {path}"
        )
    thin_axis = thin_axes[0]
    cell_dims = tuple(max(size - 1, 1) for size in node_dims)

    vertices = np.asarray(block.points, dtype=np.float64)
    centers = np.asarray(block.cell_centers().points, dtype=np.float64)
    if vertices.shape != (int(np.prod(node_dims)), 3):
        raise ValueError(
            f"CGNS vertex count does not match dimensions {node_dims}: {vertices.shape}"
        )
    if centers.shape != (int(np.prod(cell_dims)), 3):
        raise ValueError(
            f"CGNS cell-center count does not match dimensions {cell_dims}: {centers.shape}"
        )

    vertex_grid = vertices.reshape(node_dims + (3,), order="F")
    center_grid = centers.reshape(cell_dims + (3,), order="F")
    vertex_plane = np.take(vertex_grid, indices=0, axis=thin_axis)
    center_plane = np.take(center_grid, indices=0, axis=thin_axis)

    center_shape = tuple(int(value) for value in center_plane.shape[:2])
    radial_size = min(center_shape)
    radial_axis = center_shape.index(radial_size)
    if radial_axis != 0:
        center_plane = center_plane.swapaxes(0, radial_axis)

    vertex_shape = tuple(int(value) for value in vertex_plane.shape[:2])
    radial_vertex_size = radial_size + 1
    if radial_vertex_size not in vertex_shape:
        raise ValueError(
            f"CGNS vertex and center radial dimensions do not match: "
            f"vertex={vertex_shape}, center={center_shape}"
        )
    radial_vertex_axis = vertex_shape.index(radial_vertex_size)
    if radial_vertex_axis != 0:
        vertex_plane = vertex_plane.swapaxes(0, radial_vertex_axis)

    tangential_size = max(center_shape)
    center_plane = center_plane[:, :tangential_size]
    vertex_plane = vertex_plane[:, : tangential_size + 1]
    periodic_delta = np.abs(vertex_plane[:, -1, :2] - vertex_plane[:, 0, :2])
    periodic_max_delta = float(periodic_delta.max())
    if periodic_max_delta >= 1.0e-6:
        raise ValueError(
            f"CGNS O-grid is not periodically closed: max_delta={periodic_max_delta:.3e}, "
            f"path={path}"
        )
    vertex_plane[:, -1, :2] = vertex_plane[:, 0, :2]

    return {
        "coords_vertex": np.stack(
            [vertex_plane[:, :, 0], vertex_plane[:, :, 1]],
            axis=0,
        ),
        "coords_center": np.stack(
            [center_plane[:, :, 0], center_plane[:, :, 1]],
            axis=0,
        ),
        "node_dims": np.asarray(node_dims, dtype=np.int32),
        "thin_axis": np.asarray(thin_axis, dtype=np.int32),
    }


__all__ = ["load_cgns_geometry_2d"]
