"""Wall-distance artifact path contract for NK_resume."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..exceptions import ContractError
from ..schema import SolverContext


WALL_DISTANCE_REF_SCHEMA = "wall_distance_ref_v1"
_DEFAULT_SUFFIXES = (".wall_distance.npy", ".wall_distance.npz", ".wall_distance.h5", ".wall_distance.hdf5")


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return {str(k): v for k, v in dict(value or {}).items()}


def _clean_text(value: str | Path | None, *, name: str) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ContractError(f"{name} is required")
    return text


def _stem_from_cgns(cgns_basename: str | Path) -> str:
    name = Path(_clean_text(cgns_basename, name="cgns_basename")).name
    if name.lower().endswith(".cgns"):
        return name[:-5]
    return Path(name).stem


@dataclass(frozen=True)
class WallDistanceRef:
    """Resolved wall-distance artifact reference."""

    path: str
    cgns_basename: str
    layers: int | None = None
    exists: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        path = _clean_text(self.path, name="WallDistanceRef.path")
        cgns_basename = _clean_text(self.cgns_basename, name="WallDistanceRef.cgns_basename")
        layers = None if self.layers is None else int(self.layers)
        if layers is not None and layers <= 0:
            raise ContractError("WallDistanceRef.layers must be positive when set")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "cgns_basename", cgns_basename)
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "exists", bool(self.exists))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": WALL_DISTANCE_REF_SCHEMA,
            "path": self.path,
            "cgns_basename": self.cgns_basename,
            "layers": self.layers,
            "exists": self.exists,
            "metadata": dict(self.metadata),
        }


def resolve_wall_distance(
    *,
    cgns_basename: str | Path,
    wall_distance_path: str | Path | None = None,
    wall_distance_root: str | Path | None = None,
    layers: int | None = None,
    require_exists: bool = False,
) -> WallDistanceRef:
    """Resolve a wall-distance artifact path.

    Pass `wall_distance_path` for an explicit artifact, or
    `wall_distance_root` to derive `<cgns-stem>.wall_distance.*`.
    """

    if wall_distance_path is not None and str(wall_distance_path).strip():
        path = Path(wall_distance_path)
    else:
        root = Path(_clean_text(wall_distance_root, name="wall_distance_root"))
        stem = _stem_from_cgns(cgns_basename)
        candidates = [root / f"{stem}{suffix}" for suffix in _DEFAULT_SUFFIXES]
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    exists = path.exists()
    if require_exists and not exists:
        raise ContractError(f"Wall-distance path does not exist: {path}")
    return WallDistanceRef(
        path=str(path),
        cgns_basename=Path(str(cgns_basename)).name,
        layers=layers,
        exists=exists,
    )


def wall_distance_ref_from_solver_context(
    solver_context: SolverContext,
    *,
    wall_distance_path: str | Path | None = None,
    wall_distance_root: str | Path | None = None,
    require_exists: bool = False,
) -> WallDistanceRef | None:
    """Build an optional wall-distance reference from solver context."""

    if wall_distance_path is None and wall_distance_root is None:
        metadata_path = solver_context.metadata.get("wall_distance_path")
        metadata_root = solver_context.metadata.get("wall_distance_root")
        wall_distance_path = str(metadata_path) if metadata_path else None
        wall_distance_root = str(metadata_root) if metadata_root else None
    if wall_distance_path is None and wall_distance_root is None:
        return None
    return resolve_wall_distance(
        cgns_basename=solver_context.cgns_basename,
        wall_distance_path=wall_distance_path,
        wall_distance_root=wall_distance_root,
        layers=solver_context.wall_layers,
        require_exists=require_exists,
    )
