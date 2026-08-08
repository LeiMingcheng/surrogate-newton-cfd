"""Experiment-level final-only entrypoints for clean NK_resume runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from NK_resume import (
    ContractError,
    ExportResult,
    ManifestRunResult,
    NKWorkPlan,
    ResumeCase,
    SolverPreset,
    create_pipeline,
    finalonly_plan,
    run_manifest,
)
from surrogate.nk_resume.finalonly import collect_finalonly_cases_from_config


FINALONLY_EXPERIMENT_SUMMARY_SCHEMA = "finalonly_experiment_summary_v1"


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return {str(key): item for key, item in dict(value or {}).items()}


def _int_tuple(values: Iterable[int] | None) -> tuple[int, ...]:
    if values is None:
        return ()
    return tuple(int(value) for value in values)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def _parse_ordinals(text: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in str(text).split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ContractError(f"ordinal range must be increasing: {item}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(item))
    if not values:
        raise ContractError("ordinals must not be empty")
    if any(value < 0 for value in values):
        raise ContractError("ordinals must be non-negative")
    return tuple(values)


def _predictor_kind(cases: tuple[ResumeCase, ...], explicit: str | None) -> str:
    if explicit:
        return str(explicit).strip().lower()
    kinds = {case.model_inputs.predictor_kind for case in cases}
    if len(kinds) != 1:
        raise ContractError(f"final-only experiment cases must share predictor_kind: {sorted(kinds)}")
    return next(iter(kinds))


@dataclass(frozen=True)
class FinalOnlyExperimentRequest:
    """Request for collecting, exporting, and optionally running final-only cases."""

    config_path: str | Path
    ordinals: tuple[int, ...]
    output_dir: str | Path
    index_path: str | Path | None = None
    stats_path: str | Path | None = None
    checkpoint_path: str | Path | None = None
    predictor_kind: str | None = None
    device: str = "cuda"
    use_ema: bool = True
    n_inference_steps: int | None = None
    custom_timesteps: tuple[int, ...] = ()
    eta: float = 0.0
    noise_mode: str = "zeros"
    cgns_root: str | Path = ""
    ranks_per_case: int = 8
    mpi_launcher: str = "auto"
    mpi_omp_threads: int = 1
    solver_preset: str = "nk"
    fixed_cycles: int = 5
    l2conv: float = 1.0e-8
    executor: str = "export"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ordinals = _int_tuple(self.ordinals)
        if not ordinals:
            raise ContractError("FinalOnlyExperimentRequest.ordinals must not be empty")
        if any(value < 0 for value in ordinals):
            raise ContractError("FinalOnlyExperimentRequest.ordinals must be non-negative")
        executor = str(self.executor).strip().lower()
        if executor not in {"export", "sequential", "pools"}:
            raise ContractError("executor must be one of: export, sequential, pools")
        if int(self.fixed_cycles) <= 0:
            raise ContractError("fixed_cycles must be positive")
        object.__setattr__(self, "ordinals", ordinals)
        object.__setattr__(self, "custom_timesteps", _int_tuple(self.custom_timesteps))
        object.__setattr__(self, "executor", executor)
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True)
class FinalOnlyExperimentResult:
    """Artifacts produced by one final-only experiment entrypoint."""

    output_dir: str
    predictor_kind: str
    ordinals: tuple[int, ...]
    manifest_path: str
    summary_path: str
    executor: str = "export"
    export_result: ExportResult | None = None
    run_result: ManifestRunResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "predictor_kind": self.predictor_kind,
            "ordinals": list(self.ordinals),
            "manifest_path": self.manifest_path,
            "summary_path": self.summary_path,
            "executor": self.executor,
            "export": None if self.export_result is None else self.export_result.to_dict(),
            "run": None if self.run_result is None else self.run_result.to_dict(),
            "metadata": dict(self.metadata),
        }


def run_finalonly_experiment(request: FinalOnlyExperimentRequest) -> FinalOnlyExperimentResult:
    """Collect final-only cases, export a clean manifest, and optionally execute it."""

    output_dir = Path(request.output_dir)
    export_dir = output_dir / "export"
    cases = tuple(
        collect_finalonly_cases_from_config(
            config_path=request.config_path,
            ordinals=request.ordinals,
            index_path=request.index_path,
            stats_path=request.stats_path,
            checkpoint_path=request.checkpoint_path,
            predictor_kind=request.predictor_kind,
            device=request.device,
            use_ema=request.use_ema,
            n_inference_steps=request.n_inference_steps,
            custom_timesteps=request.custom_timesteps,
            eta=request.eta,
            noise_mode=request.noise_mode,
            cgns_root=request.cgns_root,
            ranks_per_case=request.ranks_per_case,
            mpi_launcher=request.mpi_launcher,
            mpi_omp_threads=request.mpi_omp_threads,
            l2conv=request.l2conv,
            output_dir=output_dir / "cases",
            metadata={
                **request.metadata,
                "entrypoint": "run_finalonly_experiment",
            },
        )
    )
    if not cases:
        raise ContractError("final-only experiment collected no cases")
    predictor_kind = _predictor_kind(cases, request.predictor_kind)
    plan = finalonly_plan(
        predictor_kind,
        work=NKWorkPlan.fixed(
            int(request.fixed_cycles),
            solver_preset=SolverPreset(request.solver_preset),
        ),
    )
    export_result = create_pipeline().export_cases(
        cases,
        plan,
        output_dir=str(export_dir),
    )
    run_result = None
    if request.executor != "export":
        run_result = run_manifest(
            export_result.manifest_path,
            executor=request.executor,
            ranks_per_case=int(request.ranks_per_case),
            summary_path=output_dir / f"{request.executor}.run_summary.json",
        )

    summary_path = output_dir / "finalonly_experiment_summary.json"
    result = FinalOnlyExperimentResult(
        output_dir=str(output_dir),
        predictor_kind=predictor_kind,
        ordinals=request.ordinals,
        manifest_path=export_result.manifest_path,
        summary_path=str(summary_path),
        executor=request.executor,
        export_result=export_result,
        run_result=run_result,
        metadata={
            "schema_version": FINALONLY_EXPERIMENT_SUMMARY_SCHEMA,
            "config_path": str(request.config_path),
            "checkpoint_path": "" if request.checkpoint_path is None else str(request.checkpoint_path),
            "device": request.device,
            "solver_preset": str(request.solver_preset),
            "fixed_cycles": int(request.fixed_cycles),
        },
    )
    payload = {
        "schema_version": FINALONLY_EXPERIMENT_SUMMARY_SCHEMA,
        **result.to_dict(),
    }
    _write_json_atomic(summary_path, payload)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run clean final-only NK_resume experiments.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ordinals", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--index")
    parser.add_argument("--stats")
    parser.add_argument("--checkpoint")
    parser.add_argument("--predictor-kind", choices=("direct", "fsb"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--n-inference-steps", type=int)
    parser.add_argument("--custom-timesteps", default="")
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--noise-mode", default="zeros")
    parser.add_argument("--cgns-root", default="")
    parser.add_argument("--ranks-per-case", type=int, default=8)
    parser.add_argument("--solver-preset", choices=("none", "nk", "prod", "pseudo"), default="nk")
    parser.add_argument("--fixed-cycles", type=int, default=5)
    parser.add_argument("--executor", choices=("export", "sequential", "pools"), default="export")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    custom_timesteps = (
        ()
        if not str(args.custom_timesteps).strip()
        else tuple(int(item.strip()) for item in str(args.custom_timesteps).split(",") if item.strip())
    )
    result = run_finalonly_experiment(
        FinalOnlyExperimentRequest(
            config_path=args.config,
            ordinals=_parse_ordinals(args.ordinals),
            output_dir=args.output_dir,
            index_path=args.index,
            stats_path=args.stats,
            checkpoint_path=args.checkpoint,
            predictor_kind=args.predictor_kind,
            device=args.device,
            use_ema=not bool(args.no_ema),
            n_inference_steps=args.n_inference_steps,
            custom_timesteps=custom_timesteps,
            eta=float(args.eta),
            noise_mode=args.noise_mode,
            cgns_root=args.cgns_root,
            ranks_per_case=int(args.ranks_per_case),
            solver_preset=args.solver_preset,
            fixed_cycles=int(args.fixed_cycles),
            executor=args.executor,
        )
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FINALONLY_EXPERIMENT_SUMMARY_SCHEMA",
    "FinalOnlyExperimentRequest",
    "FinalOnlyExperimentResult",
    "run_finalonly_experiment",
]
