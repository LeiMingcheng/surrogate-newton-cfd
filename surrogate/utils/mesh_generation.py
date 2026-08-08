"""Runtime mesh generation entrypoints for surrogate serving."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Optional

import numpy as np

from surrogate.utils.airfoil_sampling import (
    build_openloop_airfoil_coords,
    resample_openloop_airfoil_coords,
    sample_pyhyp_surface_points,
)
from surrogate.utils.cst import cst20_to_coords, cst27_to_coords
from surrogate.utils.cgns_geometry import load_cgns_geometry_2d
from surrogate.utils.runtime_paths import resolve_runtime_dir
from surrogate.utils.timing_profile import emit_profile_event


CGNSGeometryLoader = Callable[[str | Path], dict[str, np.ndarray]]
_PYHYP_LOCK = threading.Lock()
_PYHYP_WORKSPACE_LOCK = threading.Lock()


@dataclass
class _PersistentPyHypRuntime:
    topology_key: tuple[Any, ...]
    hyp: Any
    node_map: np.ndarray
    mapping_max_abs: float


_PYHYP_RUNTIME: _PersistentPyHypRuntime | None = None
_PYHYP_WORKSPACE: tempfile.TemporaryDirectory[str] | None = None


@contextmanager
def silence_native_output():
    """Discard native pyHyp/PETSc output inside a dedicated worker process."""

    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    sink = os.open(os.devnull, os.O_WRONLY)
    os.dup2(sink, 1)
    os.dup2(sink, 2)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(sink)


def _pyhyp_workspace() -> Path:
    """Return one process-local workspace reused across transient meshes."""

    global _PYHYP_WORKSPACE

    if _PYHYP_WORKSPACE is None:
        runtime_tmpdir = resolve_runtime_dir(
            None,
            env_var="CFD_RUNTIME_TMPDIR",
            default_subdir="mesh_generation_tmp",
        )
        _PYHYP_WORKSPACE = tempfile.TemporaryDirectory(
            prefix=f"pyhyp_worker_{os.getpid()}_",
            dir=str(runtime_tmpdir),
        )
    return Path(_PYHYP_WORKSPACE.name)


@contextmanager
def _persisted_mesh_lock(target: Path):
    lock_path = target.with_suffix(f"{target.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def model_coords_from_centers(coords_center: np.ndarray) -> np.ndarray:
    """Append normalized grid indices to physical cell-center coordinates."""

    center_xy = np.asarray(coords_center, dtype=np.float64)
    if center_xy.ndim != 3 or center_xy.shape[0] != 2:
        raise ValueError(f"coords_center must be (2,H,W), got {center_xy.shape}")
    _, height, width = center_xy.shape
    i_norm = np.broadcast_to(
        np.linspace(0.0, 1.0, width, dtype=np.float32)[np.newaxis, :],
        (height, width),
    )
    j_norm = np.broadcast_to(
        np.linspace(0.0, 1.0, height, dtype=np.float32)[:, np.newaxis],
        (height, width),
    )
    return np.concatenate(
        [center_xy.astype(np.float32, copy=False), i_norm[np.newaxis], j_norm[np.newaxis]],
        axis=0,
    )


def centers_from_vertices(coords_vertex: np.ndarray) -> np.ndarray:
    """Build model-layout cell centers ``[x, y, i_norm, j_norm]`` from vertices."""

    vertices = np.asarray(coords_vertex, dtype=np.float64)
    if vertices.ndim != 3 or vertices.shape[0] != 2:
        raise ValueError(f"coords_vertex must be (2,H,W), got {vertices.shape}")
    center_xy = 0.25 * (
        vertices[:, :-1, :-1]
        + vertices[:, 1:, :-1]
        + vertices[:, :-1, 1:]
        + vertices[:, 1:, 1:]
    )
    return model_coords_from_centers(center_xy)


@dataclass(frozen=True)
class PyHypMeshConfig:
    """Local pyHyp mesh defaults for surrogate runtime mesh generation."""

    chord: float = 1.0
    n_pts: int = 293
    n_radial: int = 85
    n_te_pts: int = 11
    s0: float = 5e-7
    march_distance: float = 100.0
    spacing_coeff: float = 1.0

    @property
    def scaled_s0(self) -> float:
        return float(self.s0) * float(self.chord)

    @property
    def scaled_march_distance(self) -> float:
        return float(self.march_distance) * float(self.chord)


def generate_mesh(
    cst_u: np.ndarray,
    cst_l: np.ndarray,
    *,
    t_max: Optional[float] = None,
    tail: float = 0.0,
    mode: str = "pyhyp",
    tag: str = "mesh",
    persist_cgns_path: str | Path | None = None,
    cgns_loader: Optional[CGNSGeometryLoader] = None,
    pyhyp_config: Optional[PyHypMeshConfig] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``coords_vertex`` and model-layout ``coords_center`` from 20D CST."""
    xx, yu, yl = cst20_to_coords(cst_u, cst_l, t_max=t_max, tail=tail, nn=501)
    return generate_mesh_from_coords(
        xx,
        yu,
        yl,
        mode=mode,
        tag=tag,
        persist_cgns_path=persist_cgns_path,
        cgns_loader=cgns_loader,
        pyhyp_config=pyhyp_config,
    )


def generate_mesh_from_cst27(
    cst27: np.ndarray,
    *,
    t_max: Optional[float] = None,
    mode: str = "pyhyp",
    tag: str = "mesh",
    persist_cgns_path: str | Path | None = None,
    cgns_loader: Optional[CGNSGeometryLoader] = None,
    pyhyp_config: Optional[PyHypMeshConfig] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate mesh arrays from a 27D enhanced-CST geometry vector."""
    xx, yu, yl = cst27_to_coords(cst27, nn=501, t_max=t_max)
    return generate_mesh_from_coords(
        xx,
        yu,
        yl,
        mode=mode,
        tag=tag,
        persist_cgns_path=persist_cgns_path,
        cgns_loader=cgns_loader,
        pyhyp_config=pyhyp_config,
    )


def generate_mesh_from_coords(
    xx: np.ndarray,
    yu: np.ndarray,
    yl: np.ndarray,
    *,
    mode: str = "pyhyp",
    tag: str = "mesh",
    persist_cgns_path: str | Path | None = None,
    cgns_loader: Optional[CGNSGeometryLoader] = None,
    pyhyp_config: Optional[PyHypMeshConfig] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate mesh arrays from LE-to-TE upper/lower coordinates."""
    if mode == "pyhyp":
        return _generate_pyhyp_from_coords(
            xx,
            yu,
            yl,
            tag=tag,
            persist_cgns_path=persist_cgns_path,
            cgns_loader=cgns_loader,
            pyhyp_config=pyhyp_config,
        )
    raise ValueError(f"Unknown mesh mode {mode!r}; expected 'pyhyp'")


def write_plot3d_surface_xyz(xyz_path: str | Path, sampled_pts: np.ndarray) -> Path:
    """Write a single-zone Plot3D surface file accepted by pyHyp scripts."""
    path = Path(xyz_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(sampled_pts, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise ValueError(f"Invalid sampled surface coordinates: {points.shape}")
    with path.open("w", encoding="utf-8") as handle:
        handle.write("1\n")
        handle.write(f"{len(points)} 2 1\n")
        for _ in range(2):
            for value in points[:, 0]:
                handle.write(f"{value:g}\n")
        for _ in range(2):
            for value in points[:, 1]:
                handle.write(f"{value:g}\n")
        for z_value in (0.0, 1.0):
            for _ in range(len(points)):
                handle.write(f"{z_value:g}\n")
    return path


def _read_plot3d_surface_nodes(
    xyz_path: str | Path,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Read the exact decimal surface representation consumed by pyHyp."""

    lines = Path(xyz_path).read_text(encoding="utf-8").splitlines()
    if int(lines[0]) != 1:
        raise ValueError("Persistent pyHyp requires one Plot3D surface zone")
    ni, nj, nk = (int(value) for value in lines[1].split())
    values = np.fromstring(" ".join(lines[2:]), sep=" ", dtype=np.float64)
    expected = 3 * ni * nj * nk
    if values.size != expected:
        raise ValueError(
            f"Plot3D surface contains {values.size} coordinates; expected {expected}"
        )
    coordinates = values.reshape(3, nk, nj, ni)
    nodes = coordinates[:, 0].transpose(1, 2, 0).reshape(-1, 3)
    return nodes, (ni, nj, nk)


def _load_cgns_geometry(
    cgns_path: Path,
    *,
    cgns_loader: Optional[CGNSGeometryLoader],
) -> tuple[np.ndarray, np.ndarray]:
    if cgns_loader is None:
        cgns_loader = load_cgns_geometry_2d
    geometry = cgns_loader(cgns_path)
    coords_vertex = np.asarray(geometry["coords_vertex"], dtype=np.float64)
    if "coords_center" in geometry:
        coords_center_xy = np.asarray(geometry["coords_center"], dtype=np.float64)
        if coords_center_xy.ndim == 3 and coords_center_xy.shape[0] == 4:
            coords_center = coords_center_xy.astype(np.float32, copy=False)
        elif coords_center_xy.ndim == 3 and coords_center_xy.shape[0] == 2:
            coords_center = model_coords_from_centers(coords_center_xy)
        else:
            raise ValueError(
                "coords_center from CGNS loader must have shape (2,H,W) or (4,H,W), "
                f"got {coords_center_xy.shape}"
            )
    else:
        coords_center = centers_from_vertices(coords_vertex)
    return coords_vertex, coords_center


def _run_pyhyp(
    xyz_path: Path,
    cgns_path: Path,
    mesh_cfg: PyHypMeshConfig,
    *,
    auto_connect: bool = False,
) -> dict[str, Any]:
    """Generate one mesh while reusing the process-local pyHyp runtime."""

    global _PYHYP_RUNTIME

    with _PYHYP_LOCK:
        import_start = time.perf_counter()
        from pyhyp import pyHyp

        import_s = time.perf_counter() - import_start
        surface_nodes, surface_shape = _read_plot3d_surface_nodes(xyz_path)
        topology_key = (
            surface_shape,
            int(mesh_cfg.n_radial),
            float(mesh_cfg.scaled_s0),
            float(mesh_cfg.scaled_march_distance),
            bool(auto_connect),
        )
        constructor_s = 0.0
        reset_s = 0.0
        reused = (
            _PYHYP_RUNTIME is not None
            and _PYHYP_RUNTIME.topology_key == topology_key
        )
        if reused:
            reset_start = time.perf_counter()
            _PYHYP_RUNTIME.hyp.resetForNewSurface(
                surface_nodes[_PYHYP_RUNTIME.node_map]
            )
            reset_s = time.perf_counter() - reset_start
        else:
            constructor_start = time.perf_counter()
            hyp = pyHyp(
                options={
                    "inputFile": str(xyz_path),
                    "unattachedEdgesAreSymmetry": False,
                    "outerFaceBC": "farfield",
                    "autoConnect": bool(auto_connect),
                    "BC": {1: {"jLow": "zSymm", "jHigh": "zSymm"}},
                    "families": "wall",
                    "N": int(mesh_cfg.n_radial),
                    "s0": mesh_cfg.scaled_s0,
                    "marchDist": mesh_cfg.scaled_march_distance,
                }
            )
            constructor_s = time.perf_counter() - constructor_start
            internal_surface = np.asarray(
                hyp.getSurfaceCoordinates(),
                dtype=np.float64,
            )
            node_map = np.argmin(
                np.linalg.norm(
                    internal_surface[:, None, :] - surface_nodes[None, :, :],
                    axis=2,
                ),
                axis=1,
            )
            mapping_max_abs = float(
                np.max(np.abs(internal_surface - surface_nodes[node_map]))
            )
            if mapping_max_abs != 0.0:
                raise RuntimeError(
                    "Persistent pyHyp surface mapping is not exact: "
                    f"max_abs={mapping_max_abs}"
                )
            _PYHYP_RUNTIME = _PersistentPyHypRuntime(
                topology_key=topology_key,
                hyp=hyp,
                node_map=node_map,
                mapping_max_abs=mapping_max_abs,
            )

        run_start = time.perf_counter()
        _PYHYP_RUNTIME.hyp.run()
        run_s = time.perf_counter() - run_start
        write_start = time.perf_counter()
        _PYHYP_RUNTIME.hyp.writeCGNS(str(cgns_path))
        write_s = time.perf_counter() - write_start
        return {
            "pyhyp_import_s": float(import_s),
            "pyhyp_constructor_s": float(constructor_s),
            "pyhyp_reset_s": float(reset_s),
            "pyhyp_march_s": float(run_s),
            "pyhyp_write_s": float(write_s),
            "pyhyp_run_s": float(constructor_s + reset_s + run_s + write_s),
            "pyhyp_reused": bool(reused),
            "pyhyp_auto_connect": bool(auto_connect),
            "surface_mapping_max_abs": float(
                _PYHYP_RUNTIME.mapping_max_abs
            ),
            "topology": {
                "surface_shape": list(surface_shape),
                "n_radial": int(mesh_cfg.n_radial),
            },
        }


def _generate_pyhyp_from_coords(
    xx: np.ndarray,
    yu: np.ndarray,
    yl: np.ndarray,
    *,
    tag: str,
    persist_cgns_path: str | Path | None,
    cgns_loader: Optional[CGNSGeometryLoader],
    pyhyp_config: Optional[PyHypMeshConfig],
) -> tuple[np.ndarray, np.ndarray]:
    """Generate an authoritative pyHyp O-grid when pyHyp and a CGNS loader exist."""
    total_start = time.perf_counter()
    persisted = None if persist_cgns_path is None else Path(persist_cgns_path)
    if persisted is not None and persisted.is_file():
        read_start = time.perf_counter()
        coords_vertex, coords_center = _load_cgns_geometry(
            persisted,
            cgns_loader=cgns_loader,
        )
        cgns_read_s = time.perf_counter() - read_start
        emit_profile_event(
            "mesh_generation",
            mode="pyhyp_persisted",
            geometry_hash=str(tag),
            surface_s=0.0,
            pyhyp_import_s=0.0,
            pyhyp_run_s=0.0,
            cgns_read_s=float(cgns_read_s),
            total_s=float(time.perf_counter() - total_start),
        )
        return coords_vertex, coords_center
    if persisted is not None:
        with _persisted_mesh_lock(persisted):
            if persisted.is_file():
                return _load_cgns_geometry(
                    persisted,
                    cgns_loader=cgns_loader,
                )
            return _generate_and_persist_pyhyp(
                xx,
                yu,
                yl,
                tag=tag,
                persisted=persisted,
                cgns_loader=cgns_loader,
                pyhyp_config=pyhyp_config,
                total_start=total_start,
            )
    return _generate_and_persist_pyhyp(
        xx,
        yu,
        yl,
        tag=tag,
        persisted=None,
        cgns_loader=cgns_loader,
        pyhyp_config=pyhyp_config,
        total_start=total_start,
    )


def _generate_and_persist_pyhyp(
    xx: np.ndarray,
    yu: np.ndarray,
    yl: np.ndarray,
    *,
    tag: str,
    persisted: Path | None,
    cgns_loader: Optional[CGNSGeometryLoader],
    pyhyp_config: Optional[PyHypMeshConfig],
    total_start: float,
) -> tuple[np.ndarray, np.ndarray]:
    mesh_cfg = pyhyp_config or PyHypMeshConfig()
    surface_start = time.perf_counter()
    sampled_pts = sample_pyhyp_surface_points(
        build_openloop_airfoil_coords(xx, yu, yl),
        n_pts=int(mesh_cfg.n_pts),
        n_te_pts=int(mesh_cfg.n_te_pts),
        chord=float(mesh_cfg.chord),
        upper_sampling_coeff=float(mesh_cfg.spacing_coeff),
        lower_sampling_coeff=float(mesh_cfg.spacing_coeff),
    )
    surface_s = time.perf_counter() - surface_start

    if persisted is None:
        with _PYHYP_WORKSPACE_LOCK:
            return _generate_pyhyp_in_directory(
                sampled_pts,
                tag=tag,
                tmpdir=_pyhyp_workspace(),
                persisted=None,
                cgns_loader=cgns_loader,
                mesh_cfg=mesh_cfg,
                surface_s=surface_s,
                total_start=total_start,
            )
    runtime_tmpdir = resolve_runtime_dir(
        None,
        env_var="CFD_RUNTIME_TMPDIR",
        default_subdir="mesh_generation_tmp",
    )
    with tempfile.TemporaryDirectory(prefix="pyhyp_", dir=str(runtime_tmpdir)) as tmp_text:
        return _generate_pyhyp_in_directory(
            sampled_pts,
            tag=tag,
            tmpdir=Path(tmp_text),
            persisted=persisted,
            cgns_loader=cgns_loader,
            mesh_cfg=mesh_cfg,
            surface_s=surface_s,
            total_start=total_start,
        )


def _generate_pyhyp_in_directory(
    sampled_pts: np.ndarray,
    *,
    tag: str,
    tmpdir: Path,
    persisted: Path | None,
    cgns_loader: Optional[CGNSGeometryLoader],
    mesh_cfg: PyHypMeshConfig,
    surface_s: float,
    total_start: float,
) -> tuple[np.ndarray, np.ndarray]:
    if persisted is None:
        cgns_path = tmpdir / "mesh.cgns"
        xyz_path = tmpdir / "surface.xyz"
        if cgns_path.exists():
            cgns_path.unlink()
    else:
        cgns_path = tmpdir / f"{tag}.cgns"
        xyz_path = tmpdir / f"{tag}_mesh.xyz"
    write_plot3d_surface_xyz(xyz_path, sampled_pts)
    pyhyp_timing = _run_pyhyp(
        xyz_path,
        cgns_path,
        mesh_cfg,
        auto_connect=persisted is not None,
    )

    if persisted is not None:
        persisted.parent.mkdir(parents=True, exist_ok=True)
        temporary = persisted.with_suffix(
            f"{persisted.suffix}.{os.getpid()}.writing"
        )
        shutil.copy2(cgns_path, temporary)
        temporary.replace(persisted)
    read_start = time.perf_counter()
    coords_vertex, coords_center = _load_cgns_geometry(
        cgns_path,
        cgns_loader=cgns_loader,
    )
    cgns_read_s = time.perf_counter() - read_start

    emit_profile_event(
        "mesh_generation",
        mode="pyhyp",
        geometry_hash=str(tag),
        surface_s=float(surface_s),
        cgns_read_s=float(cgns_read_s),
        total_s=float(time.perf_counter() - total_start),
        **pyhyp_timing,
    )
    return coords_vertex, coords_center


__all__ = [
    "CGNSGeometryLoader",
    "PyHypMeshConfig",
    "build_openloop_airfoil_coords",
    "generate_mesh",
    "generate_mesh_from_coords",
    "generate_mesh_from_cst27",
    "load_cgns_geometry_2d",
    "model_coords_from_centers",
    "resample_openloop_airfoil_coords",
    "sample_pyhyp_surface_points",
    "silence_native_output",
    "write_plot3d_surface_xyz",
]
