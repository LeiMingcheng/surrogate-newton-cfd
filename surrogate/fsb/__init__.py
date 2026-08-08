"""Flow-state-bridge multi-step models and runtime components."""

from __future__ import annotations

from surrogate.fsb.dit import FSBDiT
from surrogate.fsb.fno import FSBFNO

__all__ = [
    "FSBFNO",
    "FSBDiT",
    "FSBEngine",
    "FSBTrainer",
    "FSBTrainerConfig",
    "create_fsb_engine",
]


def __getattr__(name: str):
    if name in {"FSBEngine", "create_fsb_engine"}:
        from surrogate.fsb import engine

        return getattr(engine, name)
    if name == "FSBTrainer":
        from surrogate.fsb.training import FSBTrainer

        return FSBTrainer
    if name == "FSBTrainerConfig":
        from surrogate.fsb.training import FSBTrainerConfig

        return FSBTrainerConfig
    raise AttributeError(name)
