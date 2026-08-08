"""FSB scheduler state persisted between alternating NK corrections."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from NK_resume import ContractError


def _batched_field_array(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4:
        raise ContractError(
            f"{name} must have shape (B,C,H,W) or (C,H,W), got {tuple(array.shape)}"
        )
    if int(array.shape[1]) not in {4, 5}:
        raise ContractError(f"{name} must have 4 or 5 channels, got {tuple(array.shape)}")
    array = np.asarray(array, dtype=np.float32)
    if array.size == 0:
        raise ContractError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ContractError(f"{name} must contain only finite values")
    return array


def _timesteps_tuple(value: Iterable[int] | torch.Tensor | np.ndarray) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(tuple(value) if not isinstance(value, np.ndarray) else value)
    array = np.asarray(array, dtype=np.int64).reshape(-1)
    if array.size < 2:
        raise ContractError("alternating scheduler state requires at least two timesteps")
    return tuple(int(item) for item in array.tolist())


@dataclass(frozen=True)
class FSBAlternatingSchedulerState:
    """Scheduler state needed after one NK-corrected FSB transition."""

    resolved_timesteps: Iterable[int] | torch.Tensor | np.ndarray
    target_step: int
    x_t_before_step: Any
    x1_norm: Any
    t_current: int | None = None
    t_next: int | None = None
    eta: float = 0.0
    noise_mode: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        timesteps = _timesteps_tuple(self.resolved_timesteps)
        target_step = int(self.target_step)
        if target_step < 0 or target_step >= len(timesteps) - 1:
            raise ContractError("FSBAlternatingSchedulerState.target_step is out of range")
        x_t = _batched_field_array(self.x_t_before_step, name="x_t_before_step")
        x1 = _batched_field_array(self.x1_norm, name="x1_norm")
        if x_t.shape != x1.shape:
            raise ContractError(
                "FSBAlternatingSchedulerState x_t_before_step and x1_norm shapes must match"
            )
        t_current = int(timesteps[target_step]) if self.t_current is None else int(self.t_current)
        t_next = int(timesteps[target_step + 1]) if self.t_next is None else int(self.t_next)
        if t_current != int(timesteps[target_step]):
            raise ContractError(
                "FSBAlternatingSchedulerState.t_current does not match timesteps"
            )
        if t_next != int(timesteps[target_step + 1]):
            raise ContractError("FSBAlternatingSchedulerState.t_next does not match timesteps")
        object.__setattr__(self, "resolved_timesteps", timesteps)
        object.__setattr__(self, "target_step", target_step)
        object.__setattr__(self, "x_t_before_step", x_t)
        object.__setattr__(self, "x1_norm", x1)
        object.__setattr__(self, "t_current", t_current)
        object.__setattr__(self, "t_next", t_next)
        object.__setattr__(self, "eta", float(self.eta))
        object.__setattr__(self, "noise_mode", str(self.noise_mode or ""))
        object.__setattr__(
            self,
            "metadata",
            {str(key): value for key, value in dict(self.metadata).items()},
        )

    @classmethod
    def from_engine_state(
        cls,
        state: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "FSBAlternatingSchedulerState":
        return cls(
            resolved_timesteps=state.get("resolved_timesteps", ()),
            target_step=int(state.get("target_step")),
            x_t_before_step=state.get("x_t_before_step"),
            x1_norm=state.get("x1_norm"),
            t_current=state.get("t_current"),
            t_next=state.get("t_next"),
            eta=float(state.get("eta", 0.0)),
            noise_mode=str(state.get("noise_mode", "")),
            metadata={**dict(state.get("metadata") or {}), **dict(metadata or {})},
        )

    def to_engine_state(self) -> dict[str, Any]:
        return {
            "resolved_timesteps": np.asarray(self.resolved_timesteps, dtype=np.int64),
            "target_step": int(self.target_step),
            "t_current": int(self.t_current),
            "t_next": int(self.t_next),
            "x_t_before_step": np.asarray(self.x_t_before_step, dtype=np.float32),
            "x1_norm": np.asarray(self.x1_norm, dtype=np.float32),
            "eta": float(self.eta),
            "noise_mode": self.noise_mode,
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolved_timesteps": list(self.resolved_timesteps),
            "target_step": self.target_step,
            "t_current": self.t_current,
            "t_next": self.t_next,
            "x_t_before_step_shape": list(np.shape(self.x_t_before_step)),
            "x1_norm_shape": list(np.shape(self.x1_norm)),
            "eta": self.eta,
            "noise_mode": self.noise_mode,
            "metadata": dict(self.metadata),
        }


def write_alternating_scheduler_state(
    state: FSBAlternatingSchedulerState,
    path: str | Path,
) -> str:
    """Persist alternating FSB scheduler state as a compact NPZ artifact."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        resolved_timesteps=np.asarray(state.resolved_timesteps, dtype=np.int64),
        target_step=np.asarray(state.target_step, dtype=np.int64),
        t_current=np.asarray(state.t_current, dtype=np.int64),
        t_next=np.asarray(state.t_next, dtype=np.int64),
        x_t_before_step=np.asarray(state.x_t_before_step, dtype=np.float32),
        x1_norm=np.asarray(state.x1_norm, dtype=np.float32),
        eta=np.asarray(state.eta, dtype=np.float64),
        noise_mode=np.asarray(state.noise_mode),
        metadata_json=np.asarray(json.dumps(state.metadata, sort_keys=True)),
    )
    return str(output_path)


def load_alternating_scheduler_state(path: str | Path) -> FSBAlternatingSchedulerState:
    """Load a persisted alternating FSB scheduler state artifact."""

    input_path = Path(path)
    if not input_path.is_file():
        raise ContractError(f"alternating scheduler state does not exist: {input_path}")
    with np.load(input_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item())) if "metadata_json" in data else {}
        return FSBAlternatingSchedulerState(
            resolved_timesteps=data["resolved_timesteps"],
            target_step=int(np.asarray(data["target_step"]).item()),
            x_t_before_step=data["x_t_before_step"],
            x1_norm=data["x1_norm"],
            t_current=int(np.asarray(data["t_current"]).item()),
            t_next=int(np.asarray(data["t_next"]).item()),
            eta=float(np.asarray(data["eta"]).item()),
            noise_mode=str(data["noise_mode"].item()) if "noise_mode" in data else "",
            metadata=metadata,
        )


__all__ = [
    "FSBAlternatingSchedulerState",
    "load_alternating_scheduler_state",
    "write_alternating_scheduler_state",
]
