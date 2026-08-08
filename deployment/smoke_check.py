#!/usr/bin/env python
"""Verify the public repository, exact solver forks, and optional runtime assets."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _solver_root(value: str | None, sibling: str) -> Path:
    text = str(value or "").strip()
    return Path(text).expanduser().resolve() if text else (REPO_ROOT.parent / sibling).resolve()


def _contains(path: Path, marker: str) -> bool:
    return path.is_file() and marker in path.read_text(encoding="utf-8", errors="ignore")


def _git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    revision_file = path / ".surrogate-newton-revision"
    return revision_file.read_text(encoding="utf-8").strip() if revision_file.is_file() else ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compiled_solver_probe(pyhyp_root: Path, adflow_root: Path) -> dict[str, object]:
    probe = """
import json
from pathlib import Path
import adflow
from adflow import libadflow
import pyhyp
from pyhyp import pyHyp

payload = {
    "pyhyp_import_root": str(Path(pyhyp.__file__).resolve()),
    "adflow_import_root": str(Path(adflow.__file__).resolve()),
    "pyhyp_reset": hasattr(pyHyp, "resetForNewSurface"),
    "adflow_injection_reinit": hasattr(libadflow.initializeflow, "reinitafterinjection"),
    "adflow_restart_rebuild": hasattr(
        libadflow.initializeflow,
        "rebuildrestartderivedstateaftersetinfo",
    ),
}
print(json.dumps(payload, sort_keys=True))
"""
    environment = os.environ.copy()
    python_path = [str(pyhyp_root), str(adflow_root), str(REPO_ROOT)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {
            "ok": False,
            "stderr": result.stderr.strip(),
        }
    payload = json.loads(result.stdout)
    payload["ok"] = bool(
        Path(str(payload["pyhyp_import_root"])).is_relative_to(pyhyp_root)
        and Path(str(payload["adflow_import_root"])).is_relative_to(adflow_root)
        and payload["pyhyp_reset"]
        and payload["adflow_injection_reinit"]
        and payload["adflow_restart_rebuild"]
    )
    return payload


def _result_checks(
    result_dir: Path,
    *,
    residual_threshold: float,
) -> tuple[dict[str, bool], dict[str, object]]:
    summary_path = result_dir / "summary.json"
    state_path = result_dir / "surrogate_state.npz"
    result_paths = sorted(result_dir.glob("newton_export/results/*/*.result.json"))
    checks = {
        "deployment_summary": summary_path.is_file(),
        "surrogate_state": state_path.is_file(),
        "newton_result": len(result_paths) == 1,
    }
    if not all(checks.values()):
        return checks, {}

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = json.loads(result_paths[0].read_text(encoding="utf-8"))
    stage = result["stages"][-1]
    contract = stage["metrics"]["nk_residual_contract"]
    forces = stage["metrics"]["force_coefficients"]
    pre_residual = float(contract["pre_nk_totalr"])
    post_residual = float(contract["post_nk_totalr"])
    final_ratio = float(contract["post_over_reference_totalr0"])
    with np.load(state_path) as state:
        field_shape = list(state["fields"].shape)
        center_shape = list(state["coords"].shape)
        vertex_shape = list(state["coords_vertex"].shape)

    checks.update(
        {
            "summary_schema": set(summary) == {
                "authority_cgns_path",
                "flow_conditions",
                "geometry",
                "geometry_id",
                "newton",
                "surrogate",
            },
            "projection_schema": result["schema_version"] == "projection_result_v1",
            "projection_status": result["status"] == "ok" and stage["status"] == "ok",
            "field_shape": field_shape == [5, 84, 304]
            and summary["surrogate"]["field_shape"] == field_shape,
            "mesh_shape": center_shape == [4, 84, 304]
            and vertex_shape == [2, 85, 305],
            "finite_residual": all(
                math.isfinite(value) for value in (pre_residual, post_residual, final_ratio)
            ),
            "newton_reduces_residual": 0.0 < post_residual < pre_residual,
            "newton_meets_threshold": final_ratio <= residual_threshold,
            "finite_force_coefficients": all(
                math.isfinite(float(forces[name])) for name in ("cl", "cd", "cm")
            ),
        }
    )
    details = {
        "field_shape": field_shape,
        "center_coordinate_shape": center_shape,
        "vertex_coordinate_shape": vertex_shape,
        "pre_nk_totalr": pre_residual,
        "post_nk_totalr": post_residual,
        "post_over_reference_totalr0": final_ratio,
        "force_coefficients": {name: float(forces[name]) for name in ("cl", "cd", "cm")},
    }
    return checks, details


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=("source", "runtime", "result"), default="source")
    parser.add_argument("--pyhyp-root")
    parser.add_argument("--adflow-root", default=os.environ.get("SURROGATE_NEWTON_ADFLOW_ROOT"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--stats")
    parser.add_argument("--result-dir")
    args = parser.parse_args()
    if args.level in {"runtime", "result"} and (args.checkpoint is None or args.stats is None):
        parser.error("--level runtime/result requires --checkpoint and --stats")
    if args.level == "result" and args.result_dir is None:
        parser.error("--level result requires --result-dir")

    pyhyp_root = _solver_root(args.pyhyp_root, "pyhyp")
    adflow_root = _solver_root(args.adflow_root, "adflow")
    deployment = yaml.safe_load((REPO_ROOT / "deployment/config.yaml").read_text(encoding="utf-8"))
    solver_lock = yaml.safe_load((REPO_ROOT / "solver-stack.lock.yaml").read_text(encoding="utf-8"))
    model_manifest = json.loads((REPO_ROOT / "model-manifest.json").read_text(encoding="utf-8"))
    model_config = REPO_ROOT / deployment["model"]["config"]
    baseline = REPO_ROOT / deployment["geometry"]["baseline_dir"]

    checks: dict[str, bool] = {
        "license": (REPO_ROOT / "LICENSE").is_file(),
        "source_provenance": (REPO_ROOT / "SOURCE_PROVENANCE.md").is_file(),
        "model_config": model_config.is_file(),
        "model_config_sha256": _sha256(model_config) == model_manifest["model"]["config_sha256"],
        "baseline_cst_u": (baseline / "cst_u0.txt").is_file(),
        "baseline_cst_l": (baseline / "cst_l0.txt").is_file(),
        "baseline_thickness": (baseline / "t0.txt").is_file(),
        "pyhyp_exact_commit": _git_head(pyhyp_root) == solver_lock["pyhyp"]["fork_commit"],
        "adflow_exact_commit": _git_head(adflow_root) == solver_lock["adflow"]["fork_commit"],
        "pyhyp_surface_reset": _contains(pyhyp_root / "pyhyp/pyHyp.py", "resetForNewSurface"),
        "adflow_injection_reinit": _contains(
            adflow_root / "src/f2py/adflow.pyf", "reinitafterinjection"
        ),
        "adflow_restart_rebuild": _contains(
            adflow_root / "src/f2py/adflow.pyf", "rebuildrestartderivedstateaftersetinfo"
        ),
    }
    modules = {
        name: importlib.util.find_spec(name) is not None
        for name in ("numpy", "torch", "yaml", "mpi4py", "cst_modeling")
    }

    runtime: dict[str, object] = {}
    if args.level in {"runtime", "result"}:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        stats = Path(args.stats).expanduser().resolve()
        runtime = _compiled_solver_probe(pyhyp_root, adflow_root)
        checks.update(
            {
                "checkpoint_sha256": checkpoint.is_file()
                and _sha256(checkpoint) == model_manifest["model"]["sha256"],
                "stats_sha256": stats.is_file()
                and _sha256(stats) == model_manifest["normalization_statistics"]["sha256"],
                "compiled_solver_imports": bool(runtime["ok"]),
            }
        )
    if args.level == "result":
        result_checks, result_details = _result_checks(
            Path(args.result_dir).expanduser().resolve(),
            residual_threshold=float(deployment["newton"]["residual_ratio"]),
        )
        checks.update(result_checks)
        runtime["deployment_result"] = result_details

    payload = {
        "status": "ok" if all(checks.values()) and all(modules.values()) else "incomplete",
        "level": args.level,
        "repo_root": str(REPO_ROOT),
        "pyhyp_root": str(pyhyp_root),
        "adflow_root": str(adflow_root),
        "checks": checks,
        "modules": modules,
        "runtime": runtime,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
