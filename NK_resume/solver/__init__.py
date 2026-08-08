"""Solver backend boundary for NK_resume."""

from __future__ import annotations

from .backend import (
    PROJECTION_RESULT_SCHEMA,
    PROJECTION_STAGE_RESULT_SCHEMA,
    ProjectionRequest,
    ProjectionResult,
    ProjectionStageResult,
    SolverBackend,
    load_projection_result_dict,
    write_projection_result,
)
from .adflow import ADflowBackend
from .adflow_options import (
    ADflowOptionRequest,
    build_adflow_options,
    build_adflow_options_for_stage,
)
from .adflow_runtime import ensure_adflow_runtime_on_path, select_adflow_runtime_path
from .mpi_pool import MPIPoolManifestProjectionResult, project_manifest_pools
from .warm_pool import (
    ResidentWarmPoolController,
    WarmPoolManifestProjectionResult,
    project_manifest_warm_pools,
)
from .options import (
    SOLVER_OPTIONS_SCHEMA,
    SolverOptions,
    build_solver_options,
    solver_options_for_plan,
    solver_options_for_stage,
)
from .service import ReplayService
from .state import ADflowStateAdapter

__all__ = [
    "ADflowBackend",
    "ADflowOptionRequest",
    "ADflowStateAdapter",
    "MPIPoolManifestProjectionResult",
    "PROJECTION_RESULT_SCHEMA",
    "PROJECTION_STAGE_RESULT_SCHEMA",
    "ProjectionRequest",
    "ProjectionResult",
    "ProjectionStageResult",
    "ReplayService",
    "ResidentWarmPoolController",
    "SOLVER_OPTIONS_SCHEMA",
    "SolverBackend",
    "SolverOptions",
    "WarmPoolManifestProjectionResult",
    "build_solver_options",
    "build_adflow_options",
    "build_adflow_options_for_stage",
    "load_projection_result_dict",
    "ensure_adflow_runtime_on_path",
    "project_manifest_pools",
    "project_manifest_warm_pools",
    "select_adflow_runtime_path",
    "solver_options_for_plan",
    "solver_options_for_stage",
    "write_projection_result",
]
