"""Replay planning for clean payload manifests.

This module validates a manifest and builds replay job requests.  It does not
launch MPI, ADflow, or any historical runtime code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..exceptions import ContractError
from .manifest import MANIFEST_SCHEMA, Manifest, load_manifest_dict


REPLAY_PLAN_SCHEMA = "replay_plan_v1"
REPLAY_JOB_SCHEMA = "replay_job_v1"


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return {str(k): v for k, v in dict(value or {}).items()}


def _manifest_dict(manifest: Manifest | Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(manifest, Manifest):
        return manifest.to_dict()
    if isinstance(manifest, (str, Path)):
        return load_manifest_dict(manifest)
    if isinstance(manifest, Mapping):
        payload = {str(k): v for k, v in manifest.items()}
        if payload.get("schema_version") != MANIFEST_SCHEMA:
            raise ContractError(
                f"Expected {MANIFEST_SCHEMA}, got {payload.get('schema_version')!r}"
            )
        return payload
    raise ContractError(f"Unsupported manifest type: {type(manifest).__name__}")


def _payload_path(job: Mapping[str, Any]) -> str:
    payload = job.get("payload")
    if isinstance(payload, Mapping):
        path = str(payload.get("path") or "").strip()
    else:
        path = str(payload or "").strip()
    if not path:
        raise ContractError("Replay job payload path is required")
    return path


def _geometry_bundle_path(job: Mapping[str, Any]) -> str:
    payload = job.get("payload")
    path = ""
    if isinstance(payload, Mapping):
        bundle = payload.get("geometry_bundle")
        if isinstance(bundle, Mapping):
            path = str(bundle.get("path") or "").strip()
    if not path:
        path = str(job.get("geometry_bundle_path") or "").strip()
    if not path:
        raise ContractError("Replay job geometry bundle path is required")
    return path


def _job_metadata(job: Mapping[str, Any]) -> dict[str, Any]:
    metadata = job.get("metadata")
    return _metadata(metadata if isinstance(metadata, Mapping) else None)


@dataclass(frozen=True)
class ReplayJobRequest:
    """One planned replay job from a clean manifest."""

    case_id: str
    state_name: str
    payload_path: str
    geometry_bundle_path: str
    result_path: str
    output_dir: str
    solver_options: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        case_id = str(self.case_id).strip()
        state_name = str(self.state_name).strip().lower()
        payload_path = str(self.payload_path).strip()
        geometry_bundle_path = str(self.geometry_bundle_path).strip()
        result_path = str(self.result_path).strip()
        output_dir = str(self.output_dir).strip()
        if not case_id:
            raise ContractError("ReplayJobRequest.case_id is required")
        if not state_name:
            raise ContractError("ReplayJobRequest.state_name is required")
        if not payload_path:
            raise ContractError("ReplayJobRequest.payload_path is required")
        if not geometry_bundle_path:
            raise ContractError("ReplayJobRequest.geometry_bundle_path is required")
        if not result_path:
            raise ContractError("ReplayJobRequest.result_path is required")
        if not output_dir:
            raise ContractError("ReplayJobRequest.output_dir is required")
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "state_name", state_name)
        object.__setattr__(self, "payload_path", payload_path)
        object.__setattr__(self, "geometry_bundle_path", geometry_bundle_path)
        object.__setattr__(self, "result_path", result_path)
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "solver_options", _metadata(self.solver_options))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REPLAY_JOB_SCHEMA,
            "case_id": self.case_id,
            "state_name": self.state_name,
            "payload_path": self.payload_path,
            "geometry_bundle_path": self.geometry_bundle_path,
            "result_path": self.result_path,
            "output_dir": self.output_dir,
            "solver_options": dict(self.solver_options),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ReplayPlan:
    """Validated non-executing replay plan."""

    jobs: tuple[ReplayJobRequest, ...]
    mode: str = "dry_run"
    plan: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        jobs = tuple(self.jobs)
        if not jobs:
            raise ContractError("ReplayPlan requires at least one job")
        mode = str(self.mode).strip().lower()
        if mode != "dry_run":
            raise ContractError("ReplayPlan.mode must be dry_run")
        seen: set[tuple[str, str]] = set()
        for job in jobs:
            key = (job.case_id, job.state_name)
            if key in seen:
                raise ContractError(f"Duplicate replay job case/state pair: {key}")
            seen.add(key)
        object.__setattr__(self, "jobs", jobs)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "plan", _metadata(self.plan))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REPLAY_PLAN_SCHEMA,
            "mode": self.mode,
            "job_count": len(self.jobs),
            "plan": dict(self.plan),
            "jobs": [job.to_dict() for job in self.jobs],
            "metadata": dict(self.metadata),
        }


def plan_replay_manifest(
    manifest: Manifest | Mapping[str, Any] | str | Path,
    *,
    mode: str = "dry_run",
) -> ReplayPlan:
    """Validate a clean manifest and build replay job requests."""

    payload = _manifest_dict(manifest)
    jobs: list[ReplayJobRequest] = []
    for job in payload.get("jobs", ()):
        if not isinstance(job, Mapping):
            raise ContractError("Manifest jobs must be objects")
        metadata = _job_metadata(job)
        jobs.append(
            ReplayJobRequest(
                case_id=str(job.get("case_id") or metadata.get("case_id") or ""),
                state_name=str(job.get("state_name") or metadata.get("state_name") or ""),
                payload_path=_payload_path(job),
                geometry_bundle_path=_geometry_bundle_path(job),
                result_path=str(job.get("result_path") or job.get("result") or ""),
                output_dir=str(job.get("output_dir") or ""),
                solver_options=dict(metadata.get("solver_options") or {}),
                metadata=metadata,
            )
        )
    return ReplayPlan(
        jobs=tuple(jobs),
        mode=mode,
        plan=dict(payload.get("plan") or {}),
        metadata={
            "manifest_schema": payload.get("schema_version"),
            "manifest_metadata": dict(payload.get("metadata") or {}),
        },
    )


def replay_manifest(manifest: Manifest | Mapping[str, Any] | str | Path) -> ReplayPlan:
    """Build a dry-run replay plan from a manifest.

    The name is intentionally retained as the payload-layer entrypoint, but the
    return value is a plan.  Execution belongs to solver service/backends.
    """

    return plan_replay_manifest(manifest, mode="dry_run")
