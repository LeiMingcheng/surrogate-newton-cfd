"""Command-line entrypoint for all optimization modes."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from optimization.config import load_optimization_config
from optimization.objective import evaluate_workdir
from surrogate.serving.client import SurrogateClient, SurrogateClientConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run one AeroOpt experiment")
    run.add_argument("--config", required=True)
    run.add_argument("--output-dir")

    resume = subparsers.add_parser("resume", help="resume one interrupted unified experiment")
    resume.add_argument("--run-dir", required=True)
    resume.add_argument("--target-n-gen", type=int)

    evaluate = subparsers.add_parser("evaluate", help="evaluate one calculation directory")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--workdir", required=True)
    evaluate.add_argument("--refinement", action="store_true")

    ping = subparsers.add_parser("ping", help="check the configured surrogate service")
    ping.add_argument("--config", required=True)
    return parser


def _client(config: object) -> SurrogateClient:
    serving = config.serving
    return SurrogateClient(
        SurrogateClientConfig(
            host=serving.host,
            port=serving.port,
            timeout_s=serving.timeout_s,
            model_key=serving.model_key,
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "resume":
        from optimization.aeroopt import resume_optimization

        run_dir = resume_optimization(args.run_dir, target_n_gen=args.target_n_gen)
        print(run_dir)
        return 0
    config = load_optimization_config(args.config)
    if args.command == "ping":
        print(json.dumps(_client(config).ping(), indent=2, sort_keys=True))
        return 0
    if args.command == "evaluate":
        outputs = evaluate_workdir(
            args.workdir,
            config,
            refinement=bool(args.refinement),
        )
        print(json.dumps(outputs, indent=2, sort_keys=True))
        return 0
    if args.output_dir:
        config = replace(config, output_dir=str(Path(args.output_dir).resolve()))
    if config.mode in {"surrogate", "surrogate_nk"}:
        _client(config).ping()
    from optimization.aeroopt import run_optimization

    run_dir = run_optimization(config)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
