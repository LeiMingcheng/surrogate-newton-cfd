"""Manifest schema for clean replay planning."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..exceptions import ContractError
from ..plans import ResumePlan
from .bundle import PayloadRef


MANIFEST_SCHEMA = "resume_manifest_v1"


def _metadata(value: dict[str, Any] | None) -> dict[str, Any]:
    return {str(k): v for k, v in dict(value or {}).items()}


def _output_file(path_text: str, *, name: str) -> Path:
    path = Path(path_text)
    if not str(path).strip():
        raise ContractError(f"{name} is required")
    if path.exists() and path.is_dir():
        raise ContractError(f"{name} must be a file path, got directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


@dataclass(frozen=True)
class ManifestJob:
    payload: PayloadRef
    result_path: str
    output_dir: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.result_path).strip():
            raise ContractError("ManifestJob.result_path is required")
        if not str(self.output_dir).strip():
            raise ContractError("ManifestJob.output_dir is required")
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "payload": self.payload.to_dict(),
            "result_path": self.result_path,
            "output_dir": self.output_dir,
            "case_id": self.payload.case_id,
            "state_name": self.payload.state_name,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Manifest:
    plan: ResumePlan
    jobs: tuple[ManifestJob, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        jobs = tuple(self.jobs)
        if not jobs:
            raise ContractError("Manifest requires at least one job")
        object.__setattr__(self, "jobs", jobs)
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA,
            "plan": self.plan.to_dict(),
            "jobs": [job.to_dict() for job in self.jobs],
            "metadata": dict(self.metadata),
        }


def write_manifest(manifest: Manifest, output_path: str) -> str:
    """Write a clean replay manifest JSON file."""

    path = _output_file(output_path, name="output_path")
    _write_json_atomic(path, manifest.to_dict())
    return str(path)


def load_manifest_dict(path: str | Path) -> dict[str, Any]:
    """Load and validate a clean replay manifest JSON file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != MANIFEST_SCHEMA:
        raise ContractError(
            f"Expected {MANIFEST_SCHEMA}, got {payload.get('schema_version')!r}"
        )
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ContractError("Manifest JSON must contain at least one job")
    return payload
