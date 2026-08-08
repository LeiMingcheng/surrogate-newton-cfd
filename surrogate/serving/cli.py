"""Command-line entrypoint for the native surrogate socket service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import torch

from surrogate.configs import load_config
from surrogate.inference import (
    DirectPredictorBackend,
    DirectPredictorConfig,
    FSBPredictorBackend,
    FSBPredictorConfig,
)
from surrogate.physics.residual import get_residual_calculator
from surrogate.serving.aoa import AoASolverConfig, SurrogateAoASolver
from surrogate.serving.geometry import GeometryPreparationConfig, GeometryPreparer
from surrogate.serving.online import AsyncOnlineSampleWriter, SQLiteOnlineBuffer
from surrogate.serving.predictors import DirectServingPredictor, FSBServingPredictor
from surrogate.serving.server import SocketServingConfig, SurrogateServingApp, SurrogateSocketServer


def _parse_timesteps(value: Optional[str]) -> Optional[list[int]]:
    if value is None or not str(value).strip():
        return None
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _resolve_checkpoint(config: Any, override: Optional[str]) -> Path:
    value = override or config.runtime.checkpoint
    if value is None or not str(value).strip():
        raise ValueError(
            "Serving requires --checkpoint or runtime.checkpoint in the surrogate config"
        )
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Serving checkpoint not found: {path}")
    return path


def _model_version(checkpoint_path: Path) -> str:
    stat = checkpoint_path.stat()
    return f"{checkpoint_path.name}:{stat.st_size}:{stat.st_mtime_ns}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Native Surrogate-Newton CFD surrogate socket service")
    parser.add_argument("--config", required=True, help="Clean surrogate experiment config")
    parser.add_argument("--checkpoint", help="Override runtime.checkpoint from the config")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=65432)
    parser.add_argument("--device", help="Override runtime.device")
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Load model weights instead of EMA weights",
    )

    parser.add_argument("--n-inference-steps", type=int, help="Override FSB inference step count")
    parser.add_argument("--custom-timesteps", help="Comma-separated FSB inference timesteps")
    parser.add_argument("--eta", type=float, help="Override FSB inference eta")
    parser.add_argument("--noise-mode", default="zeros", help="FSB initial-noise mode")

    parser.add_argument("--mesh-mode", choices=["pyhyp"], default="pyhyp")
    parser.add_argument("--mesh-cache-size", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--batch-timeout-s", type=float, default=0.01)
    parser.add_argument("--request-timeout-s", type=float)
    parser.add_argument("--online-buffer-dir", help="Enable persistent online sample recording")
    parser.add_argument("--online-queue-size", type=int, default=4096)
    parser.add_argument(
        "--authority-cgns-dir",
        help="Persist authoritative pyHyp meshes here; defaults to ONLINE_BUFFER/meshes",
    )
    parser.add_argument("--compute-residuals", action="store_true")
    parser.add_argument("--residual-only-momentum", action="store_true")

    parser.add_argument("--aoa-min", type=float, default=-5.0)
    parser.add_argument("--aoa-max", type=float, default=10.0)
    parser.add_argument("--aoa-max-iter", type=int, default=15)
    parser.add_argument("--aoa-tol", type=float, default=1.0e-2)
    parser.add_argument("--aoa-fd-step", type=float, default=0.1)
    return parser


def create_serving_app(args: argparse.Namespace) -> SurrogateServingApp:
    """Build the native serving application from parsed CLI arguments."""

    torch.set_float32_matmul_precision("high")
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    device = str(args.device or config.runtime.device)
    checkpoint_path = _resolve_checkpoint(config, args.checkpoint)
    use_ema = not bool(args.no_ema)
    model_key = config.model.get_public_model_key()

    runtime_metadata: dict[str, Any] = {
        "model_key": model_key,
        "model_version": _model_version(checkpoint_path),
        "experiment_name": config.experiment.name,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "use_ema": use_ema,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }

    if config.model.family == "direct":
        backend = DirectPredictorBackend(
            DirectPredictorConfig(
                config_path=str(config_path),
                checkpoint_path=str(checkpoint_path),
                device=device,
                use_ema=use_ema,
            )
        )
        predictor = DirectServingPredictor(backend)
        runtime_metadata.update(
            {
                "n_inference_steps": None,
                "custom_timesteps": None,
                "eta": None,
                "noise_mode": None,
            }
        )
    elif config.model.family == "fsb":
        custom_timesteps = _parse_timesteps(args.custom_timesteps)
        eta = float(config.fsb.inference.eta if args.eta is None else args.eta)
        backend = FSBPredictorBackend(
            FSBPredictorConfig(
                config_path=str(config_path),
                checkpoint_path=str(checkpoint_path),
                device=device,
                use_ema=use_ema,
                n_inference_steps=args.n_inference_steps,
                custom_timesteps=custom_timesteps,
                eta=eta,
                noise_mode=str(args.noise_mode),
            )
        )
        predictor = FSBServingPredictor(backend)
        runtime_metadata.update(
            {
                "n_inference_steps": int(backend.engine.n_inference_steps),
                "custom_timesteps": backend.engine.runtime_config.get("custom_timesteps"),
                "eta": float(backend.engine.eta),
                "noise_mode": str(backend.engine.noise_mode),
            }
        )
    else:
        raise ValueError(f"Unsupported serving model family: {config.model.family}")

    socket_config = SocketServingConfig(
        host=str(args.host),
        port=int(args.port),
        max_batch_size=int(args.max_batch_size),
        batch_timeout_s=float(args.batch_timeout_s),
        mesh_mode=str(args.mesh_mode),
        mesh_cache_size=int(args.mesh_cache_size),
        device=device,
        request_timeout_s=args.request_timeout_s,
    )
    online_buffer_dir = None
    if args.online_buffer_dir is not None:
        online_buffer_dir = Path(args.online_buffer_dir).expanduser().resolve()
    authority_cgns_dir = args.authority_cgns_dir
    if authority_cgns_dir is None and online_buffer_dir is not None:
        authority_cgns_dir = str(online_buffer_dir / "meshes")
    geometry_preparer = GeometryPreparer(
        config=GeometryPreparationConfig(
            mesh_mode=socket_config.mesh_mode,
            cache_size=socket_config.mesh_cache_size,
            authority_cgns_dir=authority_cgns_dir,
        )
    )
    aoa_solver = SurrogateAoASolver(
        predictor,
        config=AoASolverConfig(
            aoa_range=(float(args.aoa_min), float(args.aoa_max)),
            max_iter=int(args.aoa_max_iter),
            tol=float(args.aoa_tol),
            fd_step=float(args.aoa_fd_step),
            device=device,
        ),
    )
    residual_calculator = None
    if bool(args.compute_residuals):
        residual_calculator = get_residual_calculator(
            device=device,
            compute_only_momentum=bool(args.residual_only_momentum),
        )
    online_writer = None
    if online_buffer_dir is not None:
        online_writer = AsyncOnlineSampleWriter(
            SQLiteOnlineBuffer(online_buffer_dir),
            max_queue_size=int(args.online_queue_size),
        )
    runtime_metadata.update(
        {
            "online_buffer_dir": None if online_buffer_dir is None else str(online_buffer_dir),
            "authority_cgns_dir": None if authority_cgns_dir is None else str(authority_cgns_dir),
            "residual_only_momentum": bool(args.residual_only_momentum),
        }
    )
    return SurrogateServingApp(
        predictor,
        geometry_preparer=geometry_preparer,
        aoa_solver=aoa_solver,
        config=socket_config,
        runtime_metadata=runtime_metadata,
        online_writer=online_writer,
        residual_calculator=residual_calculator,
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    app = create_serving_app(args)
    server = SurrogateSocketServer(app)
    startup = {
        "event": "surrogate_server_starting",
        **app.runtime_metadata,
        "serving": app.serving_metadata(),
    }
    print(json.dumps(startup, sort_keys=True), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_parser",
    "create_serving_app",
    "main",
]
