#!/usr/bin/env python
"""Compare a completed RAE2822 run with the sanitized golden baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _close(actual: float, expected: float, *, absolute: float, relative: float) -> bool:
    return math.isclose(actual, expected, abs_tol=absolute, rel_tol=relative)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--baseline", required=True)
    args = parser.parse_args()

    result_dir = Path(args.result_dir).expanduser().resolve()
    baseline = json.loads(Path(args.baseline).expanduser().resolve().read_text(encoding="utf-8"))
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    result_paths = sorted(result_dir.glob("newton_export/results/*/*.result.json"))
    if len(result_paths) != 1:
        raise SystemExit(f"expected one projection result, found {len(result_paths)}")
    projection = json.loads(result_paths[0].read_text(encoding="utf-8"))
    stage = projection["stages"][-1]
    solver_work = stage["metrics"]["solver_work"]
    actual_mode = str(solver_work["resume_mode"])
    expected_mode = str(baseline["case"]["resume_mode"])
    if actual_mode != expected_mode:
        payload = {
            "status": "failed",
            "baseline": str(Path(args.baseline).expanduser().resolve()),
            "result_dir": str(result_dir),
            "checks": {"resume_mode": False},
            "observed": {"resume_mode": actual_mode},
            "expected": {"resume_mode": expected_mode},
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    corrected = stage["metrics"]["force_coefficients"]
    residual = stage["metrics"]["nk_residual_contract"]
    expected = baseline["expected"]
    tolerance = baseline["tolerances"]

    with np.load(result_dir / "surrogate_state.npz") as state:
        shapes = {
            "center_coordinate_shape": list(state["coords"].shape),
            "vertex_coordinate_shape": list(state["coords_vertex"].shape),
            "field_shape": list(state["fields"].shape),
        }

    checks: dict[str, bool] = {
        name: value == expected[name] for name, value in shapes.items()
    }
    checks["resume_mode"] = actual_mode == expected_mode
    checks["geometry"] = summary["geometry"] == baseline["case"]["geometry"]
    checks["inference_steps"] = (
        int(summary["surrogate"]["inference_steps"])
        == int(baseline["case"]["inference_steps"])
    )
    checks["mpi_ranks"] = int(stage["metadata"]["mpi"]["size"]) == int(
        baseline["case"]["mpi_ranks"]
    )
    checks["requested_newton_cycles"] = int(residual["requested_cycles"]) == int(
        baseline["case"]["requested_newton_cycles"]
    )
    for name, expected_value in baseline["case"]["flow_conditions"].items():
        checks[f"flow_{name}"] = _close(
            float(summary["flow_conditions"][name]),
            float(expected_value),
            absolute=float(tolerance["flow_condition_absolute"]),
            relative=0.0,
        )
    for name, expected_value in expected["surrogate_force_coefficients"].items():
        checks[f"surrogate_{name}"] = _close(
            float(summary["surrogate"][name]),
            float(expected_value),
            absolute=float(tolerance["surrogate_force_absolute"]),
            relative=float(tolerance["surrogate_force_relative"]),
        )
    for name, expected_value in expected["corrected_force_coefficients"].items():
        checks[f"corrected_{name}"] = _close(
            float(corrected[name]),
            float(expected_value),
            absolute=float(tolerance["corrected_force_absolute"]),
            relative=float(tolerance["corrected_force_relative"]),
        )

    pre_nk_totalr = float(residual["pre_nk_totalr"])
    post_nk_totalr = float(residual["post_nk_totalr"])
    final_ratio = float(residual["post_over_reference_totalr0"])
    checks["pre_nk_totalr"] = _close(
        pre_nk_totalr,
        float(expected["pre_nk_totalr"]),
        absolute=0.0,
        relative=float(tolerance["pre_nk_totalr_relative"]),
    )
    checks["newton_reduces_residual"] = 0.0 < post_nk_totalr < pre_nk_totalr
    checks["newton_threshold"] = (
        0.0 < final_ratio <= float(tolerance["maximum_post_over_reference_totalr0"])
    )
    checks["newton_cycle_budget"] = (
        1 <= int(residual["executed_cycles"]) <= int(tolerance["maximum_newton_cycles"])
    )

    payload = {
        "status": "ok" if all(checks.values()) else "failed",
        "baseline": str(Path(args.baseline).expanduser().resolve()),
        "result_dir": str(result_dir),
        "checks": checks,
        "observed": {
            **shapes,
            "resume_mode": actual_mode,
            "surrogate_force_coefficients": {
                name: float(summary["surrogate"][name]) for name in ("cl", "cd", "cm")
            },
            "corrected_force_coefficients": {
                name: float(corrected[name]) for name in ("cl", "cd", "cm")
            },
            "pre_nk_totalr": pre_nk_totalr,
            "post_nk_totalr": post_nk_totalr,
            "post_over_reference_totalr0": final_ratio,
            "executed_newton_cycles": int(residual["executed_cycles"]),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
