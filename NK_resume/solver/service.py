"""Replay planning service for clean manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..exceptions import ContractError
from ..payload import Manifest, ReplayPlan, plan_replay_manifest


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return {str(k): v for k, v in dict(value or {}).items()}


@dataclass(frozen=True)
class ReplayService:
    """Validate manifests and produce replay plans for solver executors."""

    mode: str = "dry_run"
    ranks_per_case: int = 8
    pool_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode != "dry_run":
            raise ContractError("ReplayService.mode must be dry_run")
        ranks_per_case = int(self.ranks_per_case)
        pool_count = int(self.pool_count)
        if ranks_per_case <= 0:
            raise ContractError("ReplayService.ranks_per_case must be positive")
        if pool_count <= 0:
            raise ContractError("ReplayService.pool_count must be positive")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "ranks_per_case", ranks_per_case)
        object.__setattr__(self, "pool_count", pool_count)
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def plan(self, manifest: Manifest | Mapping[str, Any] | str) -> ReplayPlan:
        replay_plan = plan_replay_manifest(manifest, mode=self.mode)
        return ReplayPlan(
            jobs=replay_plan.jobs,
            mode=replay_plan.mode,
            plan=replay_plan.plan,
            metadata={
                **replay_plan.metadata,
                "service": self.to_dict(),
            },
        )

    def submit(self, manifest: Manifest | Mapping[str, Any] | str) -> ReplayPlan:
        return self.plan(manifest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "ranks_per_case": self.ranks_per_case,
            "pool_count": self.pool_count,
            "metadata": dict(self.metadata),
        }
