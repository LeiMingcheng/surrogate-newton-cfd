"""CLI dispatcher for clean surrogate workflows."""

from __future__ import annotations

import argparse
from typing import Optional

from surrogate.entrypoints import run_experiment, workflow_result_to_json


def _parse_timesteps(value: Optional[str]) -> Optional[list[int]]:
    if value is None or not str(value).strip():
        return None
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _parse_ordinals(value: Optional[str]) -> Optional[tuple[int, ...]]:
    if value is None or not str(value).strip():
        return None
    ordinals: list[int] = []
    for chunk in str(value).split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left)
            stop = int(right)
            if stop < start:
                raise argparse.ArgumentTypeError(f"ordinal range must be increasing: {token}")
            ordinals.extend(range(start, stop + 1))
        else:
            ordinals.append(int(token))
    if any(value < 0 for value in ordinals):
        raise argparse.ArgumentTypeError("ordinals must be non-negative")
    return tuple(ordinals)


def _add_optional_bool(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    dest: str,
    help_text: str,
) -> None:
    parser.add_argument(name, dest=dest, action="store_true", default=None, help=help_text)
    parser.add_argument(
        f"--no-{name[2:]}",
        dest=dest,
        action="store_false",
        help=f"Disable {help_text[:1].lower()}{help_text[1:]}",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Surrogate-Newton CFD surrogate workflow runner")
    parser.add_argument("--config", required=True, help="Clean surrogate config path")
    parser.add_argument(
        "--task",
        choices=["train", "validate", "infer", "nk_resume"],
        help="Override task.kind from the config",
    )
    parser.add_argument(
        "--checkpoint",
        help="Override training.checkpoint_path for train, or runtime.checkpoint for validate/infer/nk_resume",
    )
    parser.add_argument("--device", help="Override runtime.device")
    parser.add_argument("--output-dir", help="Override workflow output directory")

    parser.add_argument("--ddp", action="store_true", help="Force DDP mode")
    parser.add_argument("--dist-backend", default="nccl", help="Distributed backend")
    parser.add_argument("--sync-bn", action="store_true", help="Use SyncBatchNorm with DDP")

    parser.add_argument("--no-ema", action="store_true", help="Disable EMA checkpoint weights for runtime loading")
    parser.add_argument("--max-batches", type=int, help="Limit validation/inference batches")
    _add_optional_bool(
        parser,
        "--record-samples",
        dest="record_samples",
        help_text="write per-sample evaluation records",
    )
    _add_optional_bool(
        parser,
        "--compute-physical-field-metrics",
        dest="compute_physical_field_metrics",
        help_text="compute physical-space field metrics during evaluation",
    )
    _add_optional_bool(
        parser,
        "--compute-forces",
        dest="compute_forces",
        help_text="compute force metrics during evaluation",
    )
    _add_optional_bool(
        parser,
        "--compute-residuals",
        dest="compute_residuals",
        help_text="compute PDE residual metrics during evaluation",
    )

    parser.add_argument("--ordinals", help="Comma-separated ordinals/ranges for nk_resume export, e.g. 0-19")
    parser.add_argument("--max-cases", type=int, help="Export ordinals 0..N-1 for nk_resume")
    parser.add_argument("--index", help="Override the dataset index for nk_resume export")
    parser.add_argument("--stats", help="Override normalization stats for nk_resume export")
    parser.add_argument("--payload-only", action="store_true", help="Only write nk_resume model-side artifacts")
    parser.add_argument("--execute-backend", action="store_true", help="Deprecated; execute the exported manifest with NK_resume.cli")
    parser.add_argument("--backend-command", nargs=argparse.REMAINDER, help="Deprecated external nk_resume backend command")
    parser.add_argument("--cgns-root", default="", help="CGNS root passed to nk_resume payloads")
    parser.add_argument("--ranks-per-case", type=int, default=1, help="MPI ranks per nk_resume case")
    parser.add_argument("--mpi-launcher", default="auto", help="MPI launcher for external nk_resume backend")
    parser.add_argument("--mpi-omp-threads", type=int, default=1, help="OMP threads per MPI rank")
    parser.add_argument("--command-timeout-s", type=float, help="External backend command timeout")
    parser.add_argument("--resume-plan", default="finalonly", help="nk_resume plan preset (model-side CLI supports finalonly)")
    parser.add_argument(
        "--solver-preset",
        choices=("none", "nk", "prod", "pseudo"),
        default="nk",
        help="Solver preset recorded in the exported final-only plan",
    )
    parser.add_argument("--fixed-cycles", type=int, default=5, help="Fixed NK cycles recorded in the export plan")

    parser.add_argument("--n-inference-steps", type=int, help="Override FSB inference step count")
    parser.add_argument("--custom-timesteps", help="Comma-separated FSB custom timesteps")
    parser.add_argument("--eta", type=float, default=0.0, help="FSB inference eta")
    parser.add_argument("--noise-mode", default="zeros", help="FSB inference noise mode")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload_only = bool(args.payload_only or not args.execute_backend)
    result = run_experiment(
        args.config,
        task_kind=args.task,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        force_ddp=args.ddp,
        dist_backend=args.dist_backend,
        sync_bn=args.sync_bn,
        use_ema=not args.no_ema,
        max_batches=args.max_batches,
        record_samples=args.record_samples,
        compute_physical_field_metrics=args.compute_physical_field_metrics,
        compute_forces=args.compute_forces,
        compute_residuals=args.compute_residuals,
        max_cases=args.max_cases,
        payload_only=payload_only,
        backend_command=tuple(args.backend_command or ()),
        cgns_root=args.cgns_root,
        ranks_per_case=args.ranks_per_case,
        mpi_launcher=args.mpi_launcher,
        mpi_omp_threads=args.mpi_omp_threads,
        command_timeout_s=args.command_timeout_s,
        plan_preset=args.resume_plan,
        n_inference_steps=args.n_inference_steps,
        custom_timesteps=_parse_timesteps(args.custom_timesteps),
        eta=args.eta,
        noise_mode=args.noise_mode,
        ordinals=_parse_ordinals(args.ordinals),
        index_path=args.index,
        stats_path=args.stats,
        solver_preset=args.solver_preset,
        fixed_cycles=args.fixed_cycles,
    )
    print(workflow_result_to_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
