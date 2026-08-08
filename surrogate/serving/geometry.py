"""Geometry preparation and caching for serving requests."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from surrogate.serving.online import compute_geometry_id
from surrogate.utils.cst import cst20_to_cst27, scale_cst20_to_max_thickness
from surrogate.utils.mesh_generation import (
    CGNSGeometryLoader,
    PyHypMeshConfig,
    generate_mesh,
    generate_mesh_from_cst27,
)


@dataclass
class GeometryPreparationConfig:
    """Controls for serving-time geometry preparation."""

    mesh_mode: str = "pyhyp"
    cache_size: int = 4096
    cst_tail: float = 0.002
    t_max: Optional[float] = None
    pyhyp_config: Optional[PyHypMeshConfig] = None
    authority_cgns_dir: Optional[str | Path] = None


@dataclass
class PreparedGeometry:
    """Prepared model geometry and mesh arrays."""

    geometry: np.ndarray
    coords: np.ndarray
    coords_vertex: np.ndarray
    geometry_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _as_array(value: Any, *, dtype: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _hash_payload_geometry(
    payload: Mapping[str, Any],
    *,
    mode: str,
    tail: float,
    t_max: Optional[float],
) -> str:
    digest = hashlib.sha256()
    digest.update(str(mode).encode("utf-8"))
    digest.update(
        np.asarray(
            [tail, 0.0 if t_max is None else float(t_max)],
            dtype=np.float64,
        ).tobytes()
    )
    digest.update(
        b"persist_cgns="
        + str(payload.get("persist_cgns", True) is not False).encode("utf-8")
    )
    explicit_path = payload.get("persist_cgns_path")
    if explicit_path is not None:
        digest.update(str(explicit_path).encode("utf-8"))
    for key in ("geometry", "cst_u", "cst_l"):
        value = payload.get(key)
        if value is None:
            continue
        digest.update(str(key).encode("utf-8"))
        digest.update(_as_array(value, dtype=np.float64).reshape(-1).tobytes())
    return digest.hexdigest()[:16]


class GeometryPreparer:
    """Prepare and cache geometry vectors plus O-grid arrays for serving."""

    def __init__(
        self,
        *,
        config: Optional[GeometryPreparationConfig] = None,
        cgns_loader: Optional[CGNSGeometryLoader] = None,
    ) -> None:
        self.config = config or GeometryPreparationConfig()
        self.cgns_loader = cgns_loader
        self._cache: "OrderedDict[str, PreparedGeometry]" = OrderedDict()

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()

    def _cache_get(self, key: str) -> Optional[PreparedGeometry]:
        value = self._cache.get(key)
        if value is not None:
            self._cache.move_to_end(key, last=True)
        return value

    def _cache_put(self, key: str, value: PreparedGeometry) -> PreparedGeometry:
        self._cache[key] = value
        self._cache.move_to_end(key, last=True)
        while len(self._cache) > int(self.config.cache_size):
            self._cache.popitem(last=False)
        return value

    def prepare(self, payload: Mapping[str, Any]) -> PreparedGeometry:
        """Prepare a serving payload that may already contain mesh arrays."""
        coords_value = payload.get("coords")
        coords_vertex_value = payload.get("coords_vertex")
        geometry_value = payload.get("geometry")

        if (
            geometry_value is not None
            and coords_value is not None
            and coords_vertex_value is not None
        ):
            geometry = _as_array(geometry_value, dtype=np.float32)
            if geometry.ndim not in (1, 2) or int(geometry.shape[-1]) != 27:
                raise ValueError(
                    "Prepared geometry must have shape (27,) or (N,27); "
                    f"got {geometry.shape}"
                )
            authority_value = payload.get("authority_cgns_path")
            prepared = PreparedGeometry(
                geometry=geometry,
                coords=_as_array(coords_value, dtype=np.float32),
                coords_vertex=_as_array(coords_vertex_value, dtype=np.float64),
                geometry_id=compute_geometry_id(geometry),
                metadata={
                    "mesh_mode": "prepared",
                    "t_max": payload.get("t_max"),
                    "authority_cgns_path": (
                        None
                        if authority_value is None
                        else str(Path(authority_value))
                    ),
                    "authority_cgns_basename": (
                        None
                        if authority_value is None
                        else Path(authority_value).name
                    ),
                },
            )
            return prepared

        key = _hash_payload_geometry(
            payload,
            mode=self.config.mesh_mode,
            tail=float(payload.get("tail", self.config.cst_tail)),
            t_max=payload.get("t_max", self.config.t_max),
        )
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        tail = float(payload.get("tail", self.config.cst_tail))
        t_max = payload.get("t_max", self.config.t_max)
        tag = str(payload.get("tag") or key)

        if geometry_value is not None:
            geometry = _as_array(geometry_value, dtype=np.float32).reshape(-1)
            if geometry.shape != (27,):
                raise ValueError(f"Expected 27D geometry when mesh is absent, got {geometry.shape}")
            persist_cgns_path = self._persist_cgns_path(payload, geometry)
            coords_vertex, coords = generate_mesh_from_cst27(
                geometry,
                t_max=None if t_max is None else float(t_max),
                mode=self.config.mesh_mode,
                tag=tag,
                persist_cgns_path=persist_cgns_path,
                cgns_loader=self.cgns_loader,
                pyhyp_config=self.config.pyhyp_config,
            )
        else:
            cst_u = payload.get("cst_u")
            cst_l = payload.get("cst_l")
            if cst_u is None or cst_l is None:
                raise ValueError(
                    "Serving geometry preparation requires either geometry+coords, 27D geometry, "
                    "or cst_u/cst_l surface coefficients."
                )
            cst_u_array = _as_array(cst_u, dtype=np.float64)
            cst_l_array = _as_array(cst_l, dtype=np.float64)
            if t_max is not None:
                cst_u_array, cst_l_array = scale_cst20_to_max_thickness(
                    cst_u_array,
                    cst_l_array,
                    t_max=float(t_max),
                    tail=tail,
                )
            geometry = cst20_to_cst27(cst_u_array, cst_l_array, tail=tail)
            persist_cgns_path = self._persist_cgns_path(payload, geometry)
            coords_vertex, coords = generate_mesh(
                _as_array(cst_u, dtype=np.float64),
                _as_array(cst_l, dtype=np.float64),
                t_max=None if t_max is None else float(t_max),
                tail=tail,
                mode=self.config.mesh_mode,
                tag=tag,
                persist_cgns_path=persist_cgns_path,
                cgns_loader=self.cgns_loader,
                pyhyp_config=self.config.pyhyp_config,
            )

        prepared = PreparedGeometry(
            geometry=np.asarray(geometry, dtype=np.float32),
            coords=np.asarray(coords, dtype=np.float32),
            coords_vertex=np.asarray(coords_vertex, dtype=np.float64),
            geometry_id=compute_geometry_id(geometry),
            metadata={
                "mesh_mode": self.config.mesh_mode,
                "cache_key": key,
                "t_max": None if t_max is None else float(t_max),
                "authority_cgns_path": (
                    None if persist_cgns_path is None else str(persist_cgns_path)
                ),
                "authority_cgns_basename": (
                    None if persist_cgns_path is None else Path(persist_cgns_path).name
                ),
            },
        )
        return self._cache_put(key, prepared)

    def _persist_cgns_path(
        self,
        payload: Mapping[str, Any],
        geometry: np.ndarray,
    ) -> Optional[Path]:
        if payload.get("persist_cgns", True) is False:
            return None
        explicit = payload.get("persist_cgns_path")
        if explicit is not None:
            return Path(explicit)
        if self.config.authority_cgns_dir is None:
            return None
        root = Path(self.config.authority_cgns_dir)
        root.mkdir(parents=True, exist_ok=True)
        return root / f"{compute_geometry_id(geometry)}.cgns"


__all__ = [
    "GeometryPreparationConfig",
    "GeometryPreparer",
    "PreparedGeometry",
]
