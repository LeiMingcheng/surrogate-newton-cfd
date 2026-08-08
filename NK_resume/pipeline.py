"""High-level orchestration boundary for NK_resume."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

from .exceptions import ContractError
from .metrics import aggregate_results as aggregate_metric_results
from .payload import (
    Manifest,
    ManifestJob,
    load_manifest_dict,
    write_case_payload,
    write_geometry_bundle,
    write_manifest,
    resume_case_from_payload,
)
from .plans import ResumePlan, StagePlan, resume_plan_from_dict
from .schema import ResumeCase
from .solver import (
    ADflowBackend,
    MPIPoolManifestProjectionResult,
    ProjectionRequest,
    ProjectionResult,
    SolverBackend,
    WarmPoolManifestProjectionResult,
    project_manifest_pools,
    project_manifest_warm_pools,
)
from .solver.service import ReplayService
from .solver.options import solver_options_for_stage


@dataclass(frozen=True)
class PipelineResult:
    mode: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"mode": self.mode, "detail": self.detail}


@dataclass(frozen=True)
class ExportResult:
    """Artifacts produced by a clean non-executing export."""

    manifest_path: str
    job_count: int
    geometry_bundle_paths: tuple[str, ...]
    payload_paths: tuple[str, ...]
    result_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_path": self.manifest_path,
            "job_count": self.job_count,
            "geometry_bundle_paths": list(self.geometry_bundle_paths),
            "payload_paths": list(self.payload_paths),
            "result_paths": list(self.result_paths),
        }


@dataclass(frozen=True)
class ManifestProjectionResult:
    """Projection artifacts produced by executing a clean manifest locally."""

    manifest_path: str
    job_count: int
    result_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_path": self.manifest_path,
            "job_count": self.job_count,
            "result_paths": list(self.result_paths),
        }


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    slug = slug.strip("._")
    return slug or "case"


def _output_root(output_dir: str) -> Path:
    if not str(output_dir).strip():
        raise ContractError("output_dir is required for export_cases")
    root = Path(output_dir)
    if root.exists() and not root.is_dir():
        raise ContractError(f"output_dir must be a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _validate_case_plan_compatibility(cases: tuple[ResumeCase, ...], plan: ResumePlan) -> None:
    plan_dict = plan.to_dict()
    allowed_states = {stage["name"] for stage in plan_dict["stages"]}
    plan_predictor = str(plan_dict["predictor_kind"]).lower()
    seen: set[tuple[str, str]] = set()
    for case in cases:
        case_predictor = str(case.model_inputs.predictor_kind).lower()
        if case_predictor != plan_predictor:
            raise ContractError(
                f"Case {case.case_id} predictor_kind={case_predictor!r} "
                f"does not match plan predictor_kind={plan_predictor!r}"
            )
        state_name = case.prediction.state_name
        if state_name not in allowed_states:
            raise ContractError(
                f"Case {case.case_id} state_name={state_name!r} is not in plan stages "
                f"{sorted(allowed_states)}"
            )
        key = (case.case_id, state_name)
        if key in seen:
            raise ContractError(f"Duplicate export case/state pair: {key}")
        seen.add(key)


def _stage_by_name(plan: ResumePlan) -> dict[str, StagePlan]:
    return {stage.name: stage for stage in plan.stages}


class NKResumePipeline:
    """Clean facade for future export/replay/aggregate flows."""

    def export_cases(
        self,
        cases: Iterable[ResumeCase],
        plan: ResumePlan,
        *,
        output_dir: str,
        manifest_name: str = "manifest.json",
    ) -> ExportResult:
        case_tuple = tuple(cases)
        if not case_tuple:
            raise ContractError("export_cases requires at least one ResumeCase")
        _validate_case_plan_compatibility(case_tuple, plan)

        root = _output_root(output_dir)
        geometry_dir = root / "geometry"
        payload_dir = root / "payloads"
        result_dir = root / "results"
        geometry_dir.mkdir(exist_ok=True)
        payload_dir.mkdir(exist_ok=True)
        result_dir.mkdir(exist_ok=True)

        jobs: list[ManifestJob] = []
        geometry_paths: list[str] = []
        payload_paths: list[str] = []
        result_paths: list[str] = []
        stages = _stage_by_name(plan)
        for case in case_tuple:
            stem = _slug(f"{case.case_id}.{case.prediction.state_name}")
            stage = stages[case.prediction.state_name]
            bundle_ref = write_geometry_bundle(case, str(geometry_dir / f"{stem}.geometry.json"))
            payload_ref = write_case_payload(
                case,
                str(payload_dir / f"{stem}.payload.npz"),
                geometry_bundle=bundle_ref,
            )
            case_result_dir = result_dir / stem
            case_result_dir.mkdir(exist_ok=True)
            result_path = str(case_result_dir / f"{stem}.result.json")
            jobs.append(
                ManifestJob(
                    payload=payload_ref,
                    result_path=result_path,
                    output_dir=str(case_result_dir),
                    metadata={
                        "case_id": case.case_id,
                        "state_name": case.prediction.state_name,
                        "solver_options": solver_options_for_stage(
                            stage,
                            options_version=case.solver_context.options_version,
                            l2conv=case.solver_context.l2conv,
                        ),
                    },
                )
            )
            geometry_paths.append(bundle_ref.path)
            payload_paths.append(payload_ref.path)
            result_paths.append(result_path)

        manifest = Manifest(
            plan=plan,
            jobs=tuple(jobs),
            metadata={
                "export_root": str(root),
                "package": "NK_resume",
                "mode": "clean_payload_export",
            },
        )
        manifest_path = write_manifest(manifest, str(root / manifest_name))
        return ExportResult(
            manifest_path=manifest_path,
            job_count=len(jobs),
            geometry_bundle_paths=tuple(geometry_paths),
            payload_paths=tuple(payload_paths),
            result_paths=tuple(result_paths),
        )

    def replay_manifest(self, manifest_path: str) -> PipelineResult:
        if not str(manifest_path).strip():
            raise ValueError("manifest_path is required")
        replay_plan = ReplayService(mode="dry_run").submit(manifest_path)
        return PipelineResult(
            mode="replay_plan",
            detail=f"{replay_plan.mode}:{len(replay_plan.jobs)} jobs",
        )

    def project_case(
        self,
        case: ResumeCase,
        plan: ResumePlan,
        *,
        output_dir: str,
        backend: SolverBackend | None = None,
    ) -> ProjectionResult:
        """Execute one canonical case through a solver backend."""

        request = ProjectionRequest(case=case, plan=plan, output_dir=output_dir)
        solver_backend = backend if backend is not None else ADflowBackend()
        return solver_backend.project(request)

    def project_manifest(
        self,
        manifest_path: str,
        *,
        backend: SolverBackend | None = None,
    ) -> ManifestProjectionResult:
        """Execute all jobs in one clean manifest through a solver backend."""

        manifest = load_manifest_dict(manifest_path)
        plan = resume_plan_from_dict(dict(manifest.get("plan") or {}))
        replay_plan = ReplayService(mode="dry_run").plan(manifest)
        solver_backend = backend if backend is not None else ADflowBackend()
        result_paths: list[str] = []
        for job in replay_plan.jobs:
            case = resume_case_from_payload(job.payload_path)
            result = solver_backend.project(
                ProjectionRequest(
                    case=case,
                    plan=plan,
                    output_dir=job.output_dir,
                    metadata={
                        "manifest_path": str(manifest_path),
                        "manifest_job": job.to_dict(),
                        "result_path": job.result_path,
                    },
                )
            )
            result_paths.append(result.result_path)
        return ManifestProjectionResult(
            manifest_path=str(manifest_path),
            job_count=len(result_paths),
            result_paths=tuple(result_paths),
        )

    def project_manifest_mpi_pools(
        self,
        manifest_path: str,
        *,
        ranks_per_case: int = 8,
    ) -> MPIPoolManifestProjectionResult:
        """Execute all manifest jobs through static MPI solver pools."""

        return project_manifest_pools(
            manifest_path,
            ranks_per_case=int(ranks_per_case),
        )

    def project_manifest_warm_pools(
        self,
        manifest_path: str,
        *,
        ranks_per_case: int = 8,
        pool_count: int | None = None,
        mpi_launcher: str = "auto",
        mpi_omp_threads: int = 1,
        output_dir: str | Path | None = None,
        ready_timeout_sec: float = 30.0,
        submit_timeout_sec: float = 60.0,
        wait_for_manifest_sec: float = 60.0,
        injection_strategy: str = "restart_info",
    ) -> WarmPoolManifestProjectionResult:
        """Execute all manifest jobs through warm independent MPI pools."""

        return project_manifest_warm_pools(
            manifest_path,
            ranks_per_case=int(ranks_per_case),
            pool_count=pool_count,
            mpi_launcher=str(mpi_launcher),
            mpi_omp_threads=int(mpi_omp_threads),
            output_dir=output_dir,
            ready_timeout_sec=float(ready_timeout_sec),
            submit_timeout_sec=float(submit_timeout_sec),
            wait_for_manifest_sec=float(wait_for_manifest_sec),
            injection_strategy=str(injection_strategy),
        )

    def aggregate_results(self, summary_path: str) -> PipelineResult:
        if not str(summary_path).strip():
            raise ValueError("summary_path is required")
        path = Path(summary_path)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            results = payload
        elif isinstance(payload, dict) and isinstance(payload.get("results"), list):
            results = payload["results"]
        elif isinstance(payload, dict) and isinstance(payload.get("stages"), list):
            results = [payload]
        else:
            raise ContractError(
                "aggregate_results expects a JSON list, a dict with results, "
                "or one result dict with stages"
            )
        aggregate = aggregate_metric_results(results)
        output_path = path.with_name(f"{path.stem}.aggregate.json")
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(aggregate, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return PipelineResult(mode="aggregate", detail=str(output_path))


def create_pipeline() -> NKResumePipeline:
    return NKResumePipeline()
