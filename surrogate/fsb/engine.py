"""FSB runtime construction entry points."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from surrogate.common.checkpointing import load_model_checkpoint
from surrogate.configs import ExperimentConfig, load_config
from surrogate.inference.loading import create_normalizer_from_config
from surrogate.fsb.bridge import create_i2sb_bridge
from surrogate.fsb.runtime import FSBEngine
from surrogate.models import create_model


def resolve_inference_wall_layers(inference_engine: Any, full_height: int) -> int:
    """Resolve retained near-wall rows for FSB inference."""

    candidates = [
        getattr(getattr(inference_engine, "model", None), "wall_layers", None),
        getattr(inference_engine, "wall_layers", None),
    ]
    config = getattr(inference_engine, "config", None)
    if isinstance(config, dict):
        candidates.append(config.get("wall_layers"))

    for value in candidates:
        if value is None:
            continue
        try:
            return int(min(max(1, int(value)), int(full_height)))
        except (TypeError, ValueError):
            continue
    return int(full_height)


_resolve_inference_wall_layers = resolve_inference_wall_layers


def get_i2sb_config(config: Any) -> Dict[str, Any]:
    """Return the clean FSB bridge config as a plain dictionary."""
    if hasattr(config, "fsb"):
        return get_i2sb_config(getattr(config, "fsb"))
    if hasattr(config, "bridge"):
        bridge = getattr(config, "bridge")
        if hasattr(bridge, "__dict__"):
            return dict(bridge.__dict__)
        if isinstance(bridge, dict):
            return dict(bridge)
    if isinstance(config, dict):
        if "fsb" in config:
            return get_i2sb_config(config["fsb"])
        if "bridge" in config:
            return dict(config["bridge"] or {})
        if "i2sb" in config:
            return dict(config["i2sb"] or {})
    return {
        "num_timesteps": 1000,
        "beta_max": 0.3,
        "beta_schedule": "symmetric_sine",
        "timestep_spacing": "quadratic",
    }


def _runtime_config_from(
    config: ExperimentConfig,
    inference_params: dict[str, Any] | None,
) -> dict[str, Any]:
    params = dict(inference_params or {})
    inference_cfg = config.fsb.inference
    runtime_config = {
        "n_inference_steps": params.pop("n_inference_steps", inference_cfg.n_steps or len(inference_cfg.custom_timesteps)),
        "custom_timesteps": params.pop("custom_timesteps", inference_cfg.custom_timesteps),
        "eta": params.pop("eta", inference_cfg.eta),
        "i2sb": get_i2sb_config(config),
    }
    runtime_config.update(params)
    return runtime_config


def create_fsb_engine(
    config_path: str,
    checkpoint_path: str | None,
    inference_params: dict[str, Any] | None,
    device: str,
) -> Tuple[FSBEngine, ExperimentConfig, dict[str, Any], Any, dict[str, Any]]:
    """Create a loaded FSB runtime engine from clean config and checkpoint."""

    config = load_config(config_path)
    if config.model.family != "fsb":
        raise ValueError("create_fsb_engine requires model.family='fsb'")

    torch_device = torch.device(device)
    model = create_model(config.model).to(torch_device)

    checkpoint_payload: dict[str, Any] = {}
    resolved_checkpoint = checkpoint_path or config.runtime.checkpoint
    if resolved_checkpoint is not None:
        checkpoint_payload = load_model_checkpoint(
            model,
            resolved_checkpoint,
            torch_device,
            use_ema=bool((inference_params or {}).get("use_ema", True)),
            context=f"FSB checkpoint {resolved_checkpoint}",
        )

    model.eval()
    bridge = create_i2sb_bridge(config, torch_device)
    runtime_config = _runtime_config_from(config, inference_params)
    normalizer = create_normalizer_from_config(config)
    engine = FSBEngine(
        model=model,
        config=runtime_config,
        device=torch_device,
        bridge=bridge,
        normalizer=normalizer,
    )
    metadata = {
        "config_path": str(config_path),
        "checkpoint_path": None if resolved_checkpoint is None else str(resolved_checkpoint),
        "i2sb": get_i2sb_config(config),
    }
    return engine, config, metadata, normalizer, checkpoint_payload


__all__ = [
    "FSBEngine",
    "create_fsb_engine",
    "get_i2sb_config",
    "resolve_inference_wall_layers",
    "_resolve_inference_wall_layers",
]
