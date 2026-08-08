"""MPI worker for the unified pure-CFD evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from mpi4py import MPI

from optimization.adflow_case import run_adflow_cfd


def _run_job(job: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(job["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    target_cl = job.get("target_cl")
    result = run_adflow_cfd(
        cgns_file=str(Path(job["mesh_path"]).resolve()),
        output_dir=str(output_dir),
        mach=float(job["mach"]),
        aoa=float(job["aoa_init"]),
        reynolds=float(job["reynolds"]),
        max_iterations=int(job["max_iterations"]),
        l2_convergence=float(job["l2_convergence"]),
        options_version=int(job["options_version"]),
        reference_state_mode=str(job["reference_state_mode"]),
        cl_target=None if target_cl is None else float(target_cl),
        cl_tolerance=float(job["cl_tolerance"]),
        cl_solve_max_iter=int(job["max_aoa_iterations"]),
    )
    forces = dict(result.get("force_coefficients") or {})
    flow = dict(result.get("flow_conditions") or {})
    return {
        "mach": float(job["mach"]),
        "target_cl": None if target_cl is None else float(target_cl),
        "reynolds": float(flow.get("Reynolds", job["reynolds"])),
        "aoa": float(flow.get("AoA", job["aoa_init"])),
        "cl": float(forces.get("cl", 0.0)),
        "cd": float(forces.get("cd", 0.0)),
        "cm": float(forces.get("cmz", 0.0)),
        "converged": bool(result.get("converged", False)),
        "n_iter": int(result.get("iterations", 0)),
        "residual": result.get("l2_ratio"),
        "wall_time_s": float(time.perf_counter() - start),
        "provenance": {
            "solver": "ADFLOW.solveCL",
            "stop_reason": result.get("stop_reason"),
            "force_stability": result.get("force_stability"),
            "output_files": result.get("output_files", {}),
            "cdp": forces.get("cdp"),
            "cdv": forces.get("cdv"),
            "cl_error": result.get("cl_error"),
            "target_cl_converged": result.get("target_cl_converged"),
            "cl_solve": result.get("cl_solve"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    manifest = None
    if comm.rank == 0:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    manifest = comm.bcast(manifest, root=0)
    for job in manifest["jobs"]:
        result = _run_job(job)
        if comm.rank == 0:
            path = Path(job["result_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        comm.Barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
