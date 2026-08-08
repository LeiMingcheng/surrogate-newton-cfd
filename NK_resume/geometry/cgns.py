"""CGNS path contract for NK_resume."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Mapping

from ..exceptions import ContractError
from ..schema import SolverContext


CGNS_REF_SCHEMA = "cgns_ref_v1"

_LONG_CASE_BASENAME = re.compile(
    r"^(?P<geometry>airfoil_\d+_G2_A_L0)_case_\d+_000_vol\.cgns$"
)
_SHORT_CASE_BASENAME = re.compile(r"^(?P<geometry>af\d+)_c\d+(?:_vol)?\.cgns$")


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return {str(k): v for k, v in dict(value or {}).items()}


def _clean_text(value: str | Path | None, *, name: str) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ContractError(f"{name} is required")
    return text


def _candidate_path(cgns_root: str | Path, cgns_basename: str | Path) -> Path:
    basename = Path(_clean_text(cgns_basename, name="cgns_basename"))
    if basename.is_absolute():
        return basename
    root = Path(_clean_text(cgns_root, name="cgns_root"))
    return root / basename


@dataclass(frozen=True)
class CGNSRef:
    """Resolved CGNS path reference.

    This is a path-level contract only.  It does not parse or validate CGNS
    contents.
    """

    path: str
    basename: str
    root: str = ""
    exists: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        path = _clean_text(self.path, name="CGNSRef.path")
        basename = _clean_text(self.basename, name="CGNSRef.basename")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "basename", basename)
        object.__setattr__(self, "root", "" if self.root is None else str(self.root))
        object.__setattr__(self, "exists", bool(self.exists))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CGNS_REF_SCHEMA,
            "path": self.path,
            "basename": self.basename,
            "root": self.root,
            "exists": self.exists,
            "metadata": dict(self.metadata),
        }


def resolve_cgns_path(
    cgns_root: str | Path,
    cgns_basename: str | Path,
    *,
    require_exists: bool = False,
) -> str:
    """Resolve a CGNS path from root and basename."""

    ref = resolve_cgns_ref(
        cgns_root=cgns_root,
        cgns_basename=cgns_basename,
        require_exists=require_exists,
    )
    return ref.path


def cgns_geometry_key(cgns_basename: str | Path) -> str:
    """Return the mesh identity shared by flow cases of one airfoil."""

    basename = Path(_clean_text(cgns_basename, name="cgns_basename")).name
    for pattern in (_LONG_CASE_BASENAME, _SHORT_CASE_BASENAME):
        match = pattern.match(basename)
        if match is not None:
            return str(match.group("geometry"))
    return basename


def resolve_cgns_ref(
    cgns_root: str | Path,
    cgns_basename: str | Path,
    *,
    require_exists: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> CGNSRef:
    """Resolve a CGNS path and optionally require it to exist."""

    path = _candidate_path(cgns_root, cgns_basename)
    exists = path.exists()
    if require_exists and not exists:
        raise ContractError(f"CGNS path does not exist: {path}")
    basename = Path(cgns_basename).name if str(cgns_basename).strip() else ""
    return CGNSRef(
        path=str(path),
        basename=basename,
        root=str(cgns_root),
        exists=exists,
        metadata=metadata,
    )


def cgns_ref_from_solver_context(
    solver_context: SolverContext,
    *,
    require_exists: bool = False,
) -> CGNSRef:
    """Build a CGNS reference from canonical solver context."""

    return resolve_cgns_ref(
        cgns_root=solver_context.cgns_root,
        cgns_basename=solver_context.cgns_basename,
        require_exists=require_exists,
        metadata={
            "options_version": solver_context.options_version,
            "source_info": solver_context.source_info,
        },
    )
