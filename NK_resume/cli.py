"""Command line entrypoint for the clean NK_resume boundary."""

from __future__ import annotations

import argparse
import importlib
import json
from typing import Any

from .exceptions import NotMigratedError
from .orchestration import run_manifest
from .plans import NKWorkPlan, ResumeMode, alternating_plan, finalonly_plan
from .pipeline import create_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean NK_resume runtime facade.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("describe", help="Print package boundary information.")

    plan_parser = sub.add_parser("validate-plan", help="Construct and validate a canonical plan.")
    plan_parser.add_argument("--kind", choices=("finalonly", "alternating"), required=True)
    plan_parser.add_argument("--predictor-kind", choices=("direct", "fsb"), default="fsb")
    plan_parser.add_argument(
        "--resume-mode",
        choices=tuple(mode.value for mode in ResumeMode),
        default="",
        help="Default: ank_nk for finalonly, repeated_nk for alternating.",
    )
    plan_parser.add_argument("--max-work", type=int, default=2000)
    plan_parser.add_argument("--time-limit-s", type=float, default=10.0)
    plan_parser.add_argument("--l2conv", type=float, default=1.0e-8)
    plan_parser.add_argument("--nk-switch-tol", type=float, default=1.0e-4)
    plan_parser.add_argument("--repeated-nk-cycles", default="6,8,10")

    replay_parser = sub.add_parser(
        "replay",
        help="Validate and plan manifest replay without solver execution.",
    )
    replay_parser.add_argument("--manifest", required=True)

    aggregate_parser = sub.add_parser("aggregate", help="Aggregate clean projection results.")
    aggregate_parser.add_argument("--summary", required=True)

    project_parser = sub.add_parser(
        "project-manifest",
        help="Execute a clean manifest with the single-case ADflow backend.",
    )
    project_parser.add_argument("--manifest", required=True)

    pool_parser = sub.add_parser(
        "project-manifest-pools",
        help="Execute a clean manifest with static MPI solver pools.",
    )
    pool_parser.add_argument("--manifest", required=True)
    pool_parser.add_argument("--ranks-per-case", type=int, default=8)

    warm_pool_parser = sub.add_parser(
        "project-manifest-warm-pools",
        help="Execute a clean manifest with warm independent MPI solver pools.",
    )
    warm_pool_parser.add_argument("--manifest", required=True)
    warm_pool_parser.add_argument("--ranks-per-case", type=int, default=8)
    warm_pool_parser.add_argument("--pool-count", type=int, default=0)
    warm_pool_parser.add_argument("--mpi-launcher", default="auto")
    warm_pool_parser.add_argument("--mpi-omp-threads", type=int, default=1)
    warm_pool_parser.add_argument("--runtime-output-dir", default="")
    warm_pool_parser.add_argument("--ready-timeout-sec", type=float, default=30.0)
    warm_pool_parser.add_argument("--submit-timeout-sec", type=float, default=60.0)
    warm_pool_parser.add_argument("--wait-for-manifest-sec", type=float, default=60.0)
    warm_pool_parser.add_argument(
        "--injection-strategy",
        choices=("restart_info", "states"),
        default="restart_info",
    )

    run_parser = sub.add_parser(
        "run-manifest",
        help="Execute a clean manifest and write a run summary.",
    )
    run_parser.add_argument("--manifest", required=True)
    run_parser.add_argument(
        "--executor",
        choices=("sequential", "pools", "warm_pools"),
        default="sequential",
    )
    run_parser.add_argument("--ranks-per-case", type=int, default=8)
    run_parser.add_argument("--pool-count", type=int, default=0)
    run_parser.add_argument("--mpi-launcher", default="auto")
    run_parser.add_argument("--mpi-omp-threads", type=int, default=1)
    run_parser.add_argument("--runtime-output-dir", default="")
    run_parser.add_argument("--ready-timeout-sec", type=float, default=30.0)
    run_parser.add_argument("--submit-timeout-sec", type=float, default=60.0)
    run_parser.add_argument("--wait-for-manifest-sec", type=float, default=60.0)
    run_parser.add_argument(
        "--injection-strategy",
        choices=("restart_info", "states"),
        default="restart_info",
    )
    run_parser.add_argument("--summary-path", default="")
    return parser


def _json_print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _is_primary_process() -> bool:
    try:
        module = importlib.import_module("mpi4py.MPI")
    except Exception:
        return True
    comm = getattr(module, "COMM_WORLD", None)
    getter = getattr(comm, "Get_rank", None)
    if not callable(getter):
        return True
    return int(getter()) == 0


def _primary_json_print(payload: dict[str, Any]) -> None:
    if _is_primary_process():
        _json_print(payload)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "describe":
        _primary_json_print(
            {
                "package": "NK_resume",
                "status": "clean boundary with ADflow projection adapters",
                "legacy_runtime_imports": False,
                "interaction_plan_kinds": ["finalonly", "alternating"],
                "terminal_resume_modes": [mode.value for mode in ResumeMode],
                "default_terminal_resume_mode": ResumeMode.ANK_NK.value,
                "stage_work_policies": ["adaptive", "fixed"],
                "solver_backends": [
                    "adflow_single_case",
                    "adflow_mpi_pools",
                    "adflow_warm_mpi_pools",
                ],
            }
        )
        return 0

    if args.command == "validate-plan":
        default_mode = (
            ResumeMode.ANK_NK
            if args.kind == "finalonly"
            else ResumeMode.REPEATED_NK
        )
        resume_mode = ResumeMode(args.resume_mode) if args.resume_mode else default_mode
        if resume_mode == ResumeMode.ANK_NK:
            work = NKWorkPlan.ank_nk(
                max_work=int(args.max_work),
                time_limit_s=float(args.time_limit_s),
                nk_switch_tolerance=float(args.nk_switch_tol),
            )
        else:
            repeated_cycles = tuple(
                int(value.strip())
                for value in str(args.repeated_nk_cycles).split(",")
                if value.strip()
            )
            work = NKWorkPlan.repeated_nk(
                repeated_cycles,
                threshold=float(args.l2conv),
            )
        if args.kind == "finalonly":
            plan = finalonly_plan(args.predictor_kind, work=work)
        else:
            plan = alternating_plan(transition_work=work, final_work=work)
        _primary_json_print(plan.to_dict())
        return 0

    pipeline = create_pipeline()
    try:
        if args.command == "replay":
            result = pipeline.replay_manifest(args.manifest)
        elif args.command == "aggregate":
            result = pipeline.aggregate_results(args.summary)
        elif args.command == "project-manifest":
            result = pipeline.project_manifest(args.manifest)
        elif args.command == "project-manifest-pools":
            result = pipeline.project_manifest_mpi_pools(
                args.manifest,
                ranks_per_case=int(args.ranks_per_case),
            )
        elif args.command == "project-manifest-warm-pools":
            result = pipeline.project_manifest_warm_pools(
                args.manifest,
                ranks_per_case=int(args.ranks_per_case),
                pool_count=int(args.pool_count) if int(args.pool_count) > 0 else None,
                mpi_launcher=args.mpi_launcher,
                mpi_omp_threads=int(args.mpi_omp_threads),
                output_dir=args.runtime_output_dir or None,
                ready_timeout_sec=float(args.ready_timeout_sec),
                submit_timeout_sec=float(args.submit_timeout_sec),
                wait_for_manifest_sec=float(args.wait_for_manifest_sec),
                injection_strategy=args.injection_strategy,
            )
        elif args.command == "run-manifest":
            result = run_manifest(
                args.manifest,
                executor=args.executor,
                ranks_per_case=int(args.ranks_per_case),
                pool_count=int(args.pool_count) if int(args.pool_count) > 0 else None,
                mpi_launcher=args.mpi_launcher,
                mpi_omp_threads=int(args.mpi_omp_threads),
                runtime_output_dir=args.runtime_output_dir or None,
                ready_timeout_sec=float(args.ready_timeout_sec),
                submit_timeout_sec=float(args.submit_timeout_sec),
                wait_for_manifest_sec=float(args.wait_for_manifest_sec),
                injection_strategy=args.injection_strategy,
                summary_path=args.summary_path or None,
            )
        else:
            result = None
    except NotMigratedError as exc:
        _primary_json_print({"status": "not_migrated", "error": str(exc)})
        return 2
    if result is not None:
        _primary_json_print({"status": "ok", **result.to_dict()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
