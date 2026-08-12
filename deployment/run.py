#!/usr/bin/env python
"""Run the maintained RAE2822 surrogate-to-Newton deployment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import yaml

from data.common.flow_conditions import (
    REFERENCE_CHORD,
    REFERENCE_P_INF,
    REFERENCE_T_INF,
    coupled_reynolds_from_mach,
)
from NK_resume import (
    NKWorkPlan,
    build_fsb_case,
    create_pipeline,
    finalonly_plan,
    run_manifest,
)
from surrogate.serving.client import SurrogateClient, SurrogateClientConfig
from surrogate.serving.geometry import GeometryPreparationConfig, GeometryPreparer
from surrogate.utils.cst import cst20_to_cst27, scale_cst20_to_max_thickness


REPO_ROOT = Path(__file__).resolve().parents[1]


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a YAML mapping: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _scalar(value: Any) -> float:
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


def _flow_mapping(mach: float, aoa: float, reynolds: float) -> dict[str, float | str]:
    return {
        "mach": float(mach),
        "alpha": float(aoa),
        "reynolds": float(reynolds),
        "temperature": float(REFERENCE_T_INF),
        "pressure": float(REFERENCE_P_INF),
        "area_ref": 1.0,
        "chord_ref": float(REFERENCE_CHORD),
        "reference_state_mode": "dataset_unified",
    }


def _runtime_model_config(
    source: Path,
    *,
    checkpoint: Path,
    stats: Path,
    output_dir: Path,
) -> Path:
    config = _load_yaml(source)
    config["data"]["stats_path"] = str(stats)
    config["runtime"]["checkpoint"] = str(checkpoint)
    config["evaluation"]["output_dir"] = str(output_dir / "model_evaluation")
    target = output_dir / "runtime_model_config.yaml"
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return target


def _start_service(
    config: dict[str, Any],
    *,
    model_config: Path,
    checkpoint: Path,
    mesh_dir: Path,
    output_dir: Path,
) -> tuple[subprocess.Popen[str], Any, SurrogateClient]:
    model = config["model"]
    command = [
        sys.executable,
        "-m",
        "surrogate.serving.cli",
        "--config",
        str(model_config),
        "--checkpoint",
        str(checkpoint),
        "--device",
        str(model["device"]),
        "--host",
        str(model["host"]),
        "--port",
        str(model["port"]),
        "--authority-cgns-dir",
        str(mesh_dir),
    ]
    log = (output_dir / "surrogate_service.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    client = SurrogateClient(
        SurrogateClientConfig(
            host=str(model["host"]),
            port=int(model["port"]),
            timeout_s=float(model["request_timeout_sec"]),
        )
    )
    try:
        deadline = time.monotonic() + float(model["startup_timeout_sec"])
        while time.monotonic() < deadline:
            if process.poll() is not None:
                log.flush()
                raise RuntimeError(
                    f"Surrogate service exited with code {process.returncode}; see {log.name}"
                )
            try:
                client.ping()
                return process, log, client
            except (ConnectionError, OSError, RuntimeError):
                time.sleep(0.5)
        raise TimeoutError(f"Surrogate service did not become ready; see {log.name}")
    except Exception:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        log.close()
        raise


def _prepare_geometry(config: dict[str, Any], mesh_dir: Path) -> Any:
    geometry_config = config["geometry"]
    baseline = _path(geometry_config["baseline_dir"])
    cst_u = np.loadtxt(baseline / "cst_u0.txt", dtype=np.float64)
    cst_l = np.loadtxt(baseline / "cst_l0.txt", dtype=np.float64)
    t_max = float(np.loadtxt(baseline / "t0.txt", dtype=np.float64))
    tail = float(geometry_config["trailing_edge_thickness"])
    cst_u, cst_l = scale_cst20_to_max_thickness(
        cst_u,
        cst_l,
        t_max=t_max,
        tail=tail,
    )
    geometry = cst20_to_cst27(cst_u, cst_l, tail=tail).astype(np.float32)
    preparer = GeometryPreparer(
        config=GeometryPreparationConfig(
            mesh_mode="pyhyp",
            cache_size=4,
            authority_cgns_dir=mesh_dir,
        )
    )
    return preparer.prepare(
        {
            "geometry": geometry,
            "tag": str(geometry_config["name"]).lower(),
            "persist_cgns": True,
        }
    )


def _surrogate_prediction(
    config: dict[str, Any],
    *,
    client: SurrogateClient,
    prepared: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    flow = config["flow"]
    mach = float(flow["mach"])
    aoa = float(flow["aoa_degrees"])
    if str(flow["reynolds_mode"]) != "coupled":
        raise ValueError("The maintained deployment requires flow.reynolds_mode=coupled")
    reynolds = coupled_reynolds_from_mach(mach)
    flow_conditions = np.asarray([mach, aoa, reynolds], dtype=np.float64)
    response = client.request(
        {
            "geometry": prepared.geometry,
            "coords": prepared.coords,
            "coords_vertex": prepared.coords_vertex,
            "mach": mach,
            "aoa": aoa,
            "reynolds": reynolds,
            "n_inference_steps": int(config["model"]["inference_steps"]),
            "return_fields": True,
            "record_sample": False,
            "metadata": {"source": "rae2822_deployment"},
        }
    )
    fields = np.asarray(response["fields"], dtype=np.float32)
    if fields.ndim == 4:
        fields = fields[0]
    surrogate_summary = {
        "cl": _scalar(response["cl"]),
        "cd": _scalar(response["cd"]),
        "cm": _scalar(response["cm"]),
        "field_shape": list(fields.shape),
        "inference_steps": int(config["model"]["inference_steps"]),
    }
    residual = response.get("residual_components") or {}
    if residual.get("l2_ratio") is not None:
        surrogate_summary["residual_l2_ratio"] = _scalar(residual["l2_ratio"])
    return flow_conditions, fields, surrogate_summary


def _resume_work_plan(
    newton: dict[str, Any],
    resume_mode: str | None = None,
) -> NKWorkPlan:
    selected_mode = str(resume_mode or newton["resume_mode"])
    if selected_mode == "ank_nk":
        return NKWorkPlan.ank_nk(
            max_work=int(newton["max_work"]),
            time_limit_s=float(newton["time_limit_s"]),
            nk_switch_tolerance=float(newton["nk_switch_tolerance"]),
        )
    if selected_mode == "repeated_nk":
        return NKWorkPlan.repeated_nk(
            newton["repeated_nk_cycles"],
            threshold=float(newton["residual_ratio"]),
            name="terminal_recovery",
        )
    raise ValueError(f"Unsupported newton.resume_mode: {selected_mode}")


def _correct_field(
    config: dict[str, Any],
    *,
    prepared: Any,
    flow_conditions: np.ndarray,
    fields: np.ndarray,
    model_config: Path,
    checkpoint: Path,
    stats: Path,
    output_dir: Path,
    resume_mode: str | None = None,
) -> dict[str, Any]:
    newton = config["newton"]
    plan = finalonly_plan(
        "fsb",
        work=_resume_work_plan(newton, resume_mode),
    )
    cgns_path = Path(str(prepared.metadata["authority_cgns_path"]))
    case = build_fsb_case(
        case_id="rae2822_deployment",
        cgns_basename=cgns_path.name,
        cgns_root=cgns_path.parent,
        prediction_field=fields,
        state_name="final",
        flow_conditions=flow_conditions.tolist(),
        flow_conditions_dict=_flow_mapping(*flow_conditions),
        coords_center=prepared.coords,
        coords_vertex=prepared.coords_vertex,
        ranks_per_case=int(newton["ranks_per_case"]),
        mpi_launcher=str(newton["mpi_launcher"]),
        mpi_omp_threads=int(newton["mpi_omp_threads"]),
        options_version=2,
        l2conv=float(newton["residual_ratio"]),
        output_dir=output_dir / "newton_case",
        created_by="surrogate_newton_deployment",
        config_path=model_config,
        checkpoint_path=checkpoint,
        stats_path=stats,
        device=str(config["model"]["device"]),
        inference_steps=int(config["model"]["inference_steps"]),
    )
    export = create_pipeline().export_cases(
        (case,),
        plan,
        output_dir=str(output_dir / "newton_export"),
    )
    result = run_manifest(
        export.manifest_path,
        executor="warm_pools",
        ranks_per_case=int(newton["ranks_per_case"]),
        pool_count=int(newton["pool_count"]),
        mpi_launcher=str(newton["mpi_launcher"]),
        mpi_omp_threads=int(newton["mpi_omp_threads"]),
        runtime_output_dir=output_dir / "newton_runtime",
        ready_timeout_sec=float(newton["ready_timeout_sec"]),
        submit_timeout_sec=float(newton["submit_timeout_sec"]),
        summary_path=output_dir / "newton_summary.json",
    )
    return result.to_dict()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="deployment/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adflow-root")
    parser.add_argument("--resume-mode", choices=("ank_nk", "repeated_nk"))
    parser.add_argument("--surrogate-only", action="store_true")
    args = parser.parse_args()

    config = _load_yaml(_path(args.config))
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    stats = Path(args.stats).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not stats.is_file():
        raise FileNotFoundError(f"Normalization statistics not found: {stats}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    mesh_dir = output_dir / "meshes"
    mesh_dir.mkdir()
    if args.adflow_root:
        os.environ["SURROGATE_NEWTON_ADFLOW_ROOT"] = str(
            Path(args.adflow_root).expanduser().resolve()
        )

    model_config_source = _path(config["model"]["config"])
    runtime_model_config = _runtime_model_config(
        model_config_source,
        checkpoint=checkpoint,
        stats=stats,
        output_dir=output_dir,
    )
    process = None
    log = None
    try:
        process, log, client = _start_service(
            config,
            model_config=runtime_model_config,
            checkpoint=checkpoint,
            mesh_dir=mesh_dir,
            output_dir=output_dir,
        )
        prepared = _prepare_geometry(config, mesh_dir)
        flow_conditions, fields, surrogate_summary = _surrogate_prediction(
            config,
            client=client,
            prepared=prepared,
        )
        np.savez_compressed(
            output_dir / "surrogate_state.npz",
            geometry=prepared.geometry,
            coords=prepared.coords,
            coords_vertex=prepared.coords_vertex,
            fields=fields,
            flow_conditions=flow_conditions,
        )
        summary: dict[str, Any] = {
            "geometry": str(config["geometry"]["name"]),
            "geometry_id": prepared.geometry_id,
            "authority_cgns_path": str(prepared.metadata["authority_cgns_path"]),
            "flow_conditions": {
                "mach": float(flow_conditions[0]),
                "aoa_degrees": float(flow_conditions[1]),
                "reynolds": float(flow_conditions[2]),
            },
            "surrogate": surrogate_summary,
            "newton": None,
        }
        if bool(config["newton"]["enabled"]) and not args.surrogate_only:
            summary["newton"] = _correct_field(
                config,
                prepared=prepared,
                flow_conditions=flow_conditions,
                fields=fields,
                model_config=runtime_model_config,
                checkpoint=checkpoint,
                stats=stats,
                output_dir=output_dir,
                resume_mode=args.resume_mode,
            )
        _write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        if log is not None:
            log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
