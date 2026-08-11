"""Geometry, surrogate, and ADflow adapters for the local interactive demo."""

from __future__ import annotations

import hashlib
import html
import importlib.util
import json
import os
import re
import shlex
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data.common.flow_conditions import (
    REFERENCE_CHORD,
    REFERENCE_GAMMA,
    REFERENCE_GAS_CONSTANT,
    REFERENCE_P_INF,
    REFERENCE_T_INF,
    coupled_reynolds_from_mach,
    sutherland_viscosity,
)
from demo.ood import OodGeometryIndex
from NK_resume import (
    NKWorkPlan,
    ResidentWarmPoolController,
    SolverPreset,
    build_fsb_case,
    create_pipeline,
    finalonly_plan,
)
from surrogate.data.uniform_flow_initializer import UniformFlowInitializer
from surrogate.serving.client import SurrogateClient, SurrogateClientConfig
from surrogate.serving.geometry import (
    GeometryPreparationConfig,
    GeometryPreparer,
)
from surrogate.utils.cst import (
    coords_to_cst27,
    cst20_to_cst27,
    cst27_to_coords,
    scale_cst20_to_max_thickness,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PRESET_ROOT = REPO_ROOT / "optimization" / "baselines"
MODEL_MANIFEST = REPO_ROOT / "model-manifest.json"
FSB_CONFIG = REPO_ROOT / "surrogate" / "configs" / "training" / "fsb_dit.yaml"
DEFAULT_AIRFOIL_LIBRARY_ROOT = Path(__file__).resolve().parent / "airfoils" / "uiuc"
GAMMA = 1.4
EDITOR_POINTS = 241
MAX_UPLOAD_BYTES = 2_000_000
MAX_COORDINATE_POINTS = 20_000
FIXED_TRAILING_EDGE_THICKNESS = 0.002
CP_PLOT_X_MAX = 0.999
UIUC_CATALOG_URL = "https://m-selig.ae.illinois.edu/ads/coord_database.html"
UIUC_COORDINATE_ROOT = "https://m-selig.ae.illinois.edu/ads/coord"
UIUC_SITE_URL = "https://m-selig.ae.illinois.edu/ads.html"
UIUC_CST_FIT_MSE_LIMIT = 1.0e-5
FIELD_CHANNELS = (
    ("density", "Density"),
    ("velocity_x", "Velocity X"),
    ("velocity_y", "Velocity Y"),
    ("pressure", "Pressure"),
    ("sa_nu_tilde", "SA working variable"),
)


def _default_runtime_root() -> Path:
    configured = os.environ.get("SURROGATE_NEWTON_RUNTIME_DIR") or os.environ.get(
        "SURROGATE_NEWTON_RUNTIME_ROOT"
    )
    if configured:
        return Path(configured).expanduser() / "demo"
    return Path(tempfile.gettempdir()) / "surrogate-newton-cfd-demo"


def _model_artifacts(model_dir: Path) -> tuple[Path, Path]:
    manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    checkpoint = model_dir / str(manifest["model"]["filename"])
    statistics = model_dir / str(manifest["normalization_statistics"]["filename"])
    return checkpoint, statistics


def _launcher_available(launcher: str) -> bool:
    command = shlex.split(str(launcher))
    if not command or command[0] == "auto":
        return shutil.which("mpiexec") is not None or shutil.which("mpirun") is not None
    executable = Path(command[0]).expanduser()
    if executable.is_absolute():
        return executable.is_file()
    return shutil.which(command[0]) is not None


def _parse_uiuc_catalog(content: str) -> list[dict[str, str]]:
    """Extract the coordinate-file catalog from the official UIUC A--Z page."""

    pattern = re.compile(
        r'<a\s+href="coord/([^"?#]+\.dat)"[^>]*>.*?</a>(.*?)<br\s*/?>',
        flags=re.IGNORECASE | re.DOTALL,
    )
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for filename, trailing_html in pattern.findall(content):
        key = filename.lower()
        if key in seen:
            continue
        seen.add(key)
        fields = [
            re.sub(r"\s+", " ", part).strip()
            for part in html.unescape(re.sub(r"<[^>]+>", " ", trailing_html)).split("\\")
        ]
        description = next((field for field in fields if field), Path(filename).stem)
        entries.append(
            {
                "key": f"uiuc:{key}",
                "name": Path(filename).stem.upper(),
                "filename": filename,
                "description": description,
                "coordinate_url": f"{UIUC_COORDINATE_ROOT}/{filename}",
            }
        )
    if not entries:
        raise ValueError("The official UIUC catalog did not contain coordinate entries.")
    return entries


def reference_state_for_mach(mach: float) -> dict[str, float]:
    """Return the thermodynamic state used to generate the training data."""

    speed_of_sound = float(
        np.sqrt(REFERENCE_GAMMA * REFERENCE_GAS_CONSTANT * REFERENCE_T_INF)
    )
    return {
        "mach": float(mach),
        "temperature_k": float(REFERENCE_T_INF),
        "pressure_pa": float(REFERENCE_P_INF),
        "gamma": float(REFERENCE_GAMMA),
        "gas_constant_j_kg_k": float(REFERENCE_GAS_CONSTANT),
        "chord_m": float(REFERENCE_CHORD),
        "density_kg_m3": float(
            REFERENCE_P_INF / (REFERENCE_GAS_CONSTANT * REFERENCE_T_INF)
        ),
        "dynamic_viscosity_pa_s": float(sutherland_viscosity(REFERENCE_T_INF)),
        "speed_of_sound_m_s": speed_of_sound,
        "velocity_m_s": float(mach) * speed_of_sound,
        "reynolds": float(coupled_reynolds_from_mach(mach)),
    }


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _first_scalar(value: Any) -> float:
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


def _force_mae_against_reference(
    forces: dict[str, Any],
    reference_forces: dict[str, Any],
) -> float:
    return 10.0 * abs(float(forces["cd"]) - float(reference_forces["cd"])) + abs(
        float(forces["cl"]) - float(reference_forces["cl"])
    )


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


_PRIVATE_RUNTIME_KEYS = {
    "authority_cgns_path",
    "field_path",
    "result_path",
    "solver_runtime",
}


def _public_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _public_payload(item)
            for key, item in value.items()
            if key not in _PRIVATE_RUNTIME_KEYS
        }
    if isinstance(value, list):
        return [_public_payload(item) for item in value]
    return value


def _deduplicate_surface(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(points[:, 0], kind="stable")
    ordered = points[order]
    x_values, inverse = np.unique(ordered[:, 0], return_inverse=True)
    y_sum = np.zeros_like(x_values, dtype=np.float64)
    counts = np.zeros_like(x_values, dtype=np.int64)
    np.add.at(y_sum, inverse, ordered[:, 1])
    np.add.at(counts, inverse, 1)
    return x_values, y_sum / counts


def _numeric_rows(lines: list[str]) -> np.ndarray:
    rows: list[tuple[float, float]] = []
    numeric_started = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "!", "%")):
            continue
        parts = stripped.replace(",", " ").split()
        if len(parts) < 2:
            if numeric_started:
                raise ValueError("Coordinate data contains a malformed row.")
            continue
        try:
            row = (float(parts[0]), float(parts[1]))
        except ValueError as exc:
            if numeric_started:
                raise ValueError("Coordinate data contains a non-numeric row.") from exc
            continue
        if not np.all(np.isfinite(row)):
            raise ValueError("Coordinate values must be finite numbers.")
        rows.append(row)
        numeric_started = True
    if len(rows) < 26:
        raise ValueError("Coordinate input must contain at least 26 numeric x-y rows.")
    if len(rows) > MAX_COORDINATE_POINTS:
        raise ValueError(f"Coordinate input exceeds the {MAX_COORDINATE_POINTS}-point limit.")
    return np.asarray(rows, dtype=np.float64)


def _validate_surface_order(points: np.ndarray) -> None:
    delta_x = np.diff(np.asarray(points, dtype=np.float64)[:, 0])
    resolved = delta_x[np.abs(delta_x) > 1.0e-12]
    if resolved.size == 0 or (np.any(resolved > 0.0) and np.any(resolved < 0.0)):
        raise ValueError(
            "Each surface must follow a monotonic leading-edge-to-trailing-edge sequence."
        )


def _split_coordinate_text(content: str) -> tuple[np.ndarray, np.ndarray]:
    if len(content.encode("utf-8")) > MAX_UPLOAD_BYTES:
        raise ValueError("Coordinate file exceeds the 2 MB local-demo limit.")
    lines = content.splitlines()
    zone_starts = [
        index for index, line in enumerate(lines) if line.strip().lower().startswith("zone")
    ]
    if len(zone_starts) >= 2:
        boundaries = zone_starts + [len(lines)]
        zones = [
            _numeric_rows(lines[boundaries[index] + 1 : boundaries[index + 1]])
            for index in range(len(zone_starts))
        ]
        surfaces = sorted(zones, key=len, reverse=True)[:2]
    else:
        points = _numeric_rows(lines)
        keep = np.ones(len(points), dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1.0e-12
        points = points[keep]
        declared_counts = np.rint(points[0]).astype(np.int64)
        lednicer_format = bool(
            np.all(declared_counts >= 13)
            and np.allclose(points[0], declared_counts)
            and 1 + int(np.sum(declared_counts)) <= len(points)
        )
        if lednicer_format:
            first_count, second_count = (int(value) for value in declared_counts)
            surfaces = [
                points[1 : 1 + first_count],
                points[1 + first_count : 1 + first_count + second_count],
            ]
        else:
            candidates = sorted(
                (int(np.argmin(points[:, 0])), int(np.argmax(points[:, 0]))),
                key=lambda index: min(index + 1, len(points) - index),
                reverse=True,
            )
            split_index = candidates[0]
            if split_index < 12 or len(points) - split_index < 13:
                raise ValueError(
                    "Closed-contour input must traverse both surfaces and turn at the leading "
                    "or trailing edge."
                )
            surfaces = [points[: split_index + 1], points[split_index:]]

    _validate_surface_order(surfaces[0])
    _validate_surface_order(surfaces[1])
    first_x, first_y = _deduplicate_surface(surfaces[0])
    second_x, second_y = _deduplicate_surface(surfaces[1])
    if len(first_x) < 13 or len(second_x) < 13:
        raise ValueError("Each surface must contain at least 13 distinct x locations.")

    chord_min = min(float(first_x.min()), float(second_x.min()))
    chord_max = max(float(first_x.max()), float(second_x.max()))
    chord = chord_max - chord_min
    if chord <= 0.0:
        raise ValueError("Coordinate input has zero chord.")
    first_x = (first_x - chord_min) / chord
    second_x = (second_x - chord_min) / chord
    first_y = first_y / chord
    second_y = second_y / chord
    common = np.linspace(0.05, 0.95, 101)
    first_mean = float(np.mean(np.interp(common, first_x, first_y)))
    second_mean = float(np.mean(np.interp(common, second_x, second_y)))
    first = np.column_stack([first_x, first_y])
    second = np.column_stack([second_x, second_y])
    upper, lower = (first, second) if first_mean >= second_mean else (second, first)
    leading_edge_y = 0.5 * (
        float(np.interp(0.0, upper[:, 0], upper[:, 1]))
        + float(np.interp(0.0, lower[:, 0], lower[:, 1]))
    )
    upper[:, 1] -= leading_edge_y
    lower[:, 1] -= leading_edge_y
    return upper, lower


def _validate_geometry(x: np.ndarray, upper: np.ndarray, lower: np.ndarray) -> None:
    if x.ndim != 1 or upper.shape != x.shape or lower.shape != x.shape:
        raise ValueError("Geometry arrays must be matching one-dimensional x/upper/lower data.")
    if not np.all(np.isfinite(np.concatenate([x, upper, lower]))):
        raise ValueError("Geometry arrays must contain only finite numbers.")
    if len(x) < 26 or not np.all(np.diff(x) > 0.0):
        raise ValueError("Geometry x coordinates must be strictly increasing.")
    if not np.isclose(x[0], 0.0, atol=1.0e-6) or not np.isclose(
        x[-1], 1.0, atol=1.0e-6
    ):
        raise ValueError("Geometry must span the normalized chord from x=0 to x=1.")
    thickness = upper - lower
    resolved_interior = (x > 1.0e-3) & (x < 0.995)
    if float(np.min(thickness[resolved_interior])) <= 2.0e-4:
        raise ValueError("Upper and lower surfaces cross or become too thin.")
    if float(np.max(thickness)) > 0.35:
        raise ValueError("Maximum thickness exceeds the demo mesh envelope (35% chord).")
    if float(np.max(np.abs(0.5 * (upper + lower)))) > 0.25:
        raise ValueError("Airfoil camber exceeds the demo mesh envelope (25% chord).")


def _fit_coordinate_text_to_cst(content: str) -> tuple[np.ndarray, float]:
    upper, lower = _split_coordinate_text(content)
    geometry27 = coords_to_cst27(
        upper[:, 0],
        upper[:, 1],
        lower[:, 0],
        lower[:, 1],
        tail=FIXED_TRAILING_EDGE_THICKNESS,
    )
    x, canonical_upper, canonical_lower = cst27_to_coords(
        geometry27,
        nn=EDITOR_POINTS,
    )
    source_upper = np.interp(x, upper[:, 0], upper[:, 1])
    source_lower = np.interp(x, lower[:, 0], lower[:, 1])
    fit_mse = float(
        np.mean(
            np.concatenate(
                [
                    canonical_upper - source_upper,
                    canonical_lower - source_lower,
                ]
            )
            ** 2
        )
    )
    return geometry27, fit_mse


def _geometry_payload(
    name: str,
    geometry27: np.ndarray,
    *,
    fit_mse: float | None = None,
    ood: dict[str, Any] | None = None,
) -> dict[str, Any]:
    x, upper, lower, thickness, leading_edge_radius = cst27_to_coords(
        geometry27,
        nn=EDITOR_POINTS,
        return_metrics=True,
    )
    _validate_geometry(x, upper, lower)
    return {
        "name": str(name),
        "parameterization": "CST",
        "geometry27": np.asarray(geometry27, dtype=np.float32).tolist(),
        "x": x.tolist(),
        "upper": upper.tolist(),
        "lower": lower.tolist(),
        "metrics": {
            "max_thickness": float(thickness),
            "leading_edge_radius": float(leading_edge_radius),
            "fit_mse": fit_mse,
        },
        "ood": ood,
    }


def _pressure_view(
    field: np.ndarray,
    coords_center: np.ndarray,
    coords_vertex: np.ndarray,
    *,
    mach: float,
) -> dict[str, Any]:
    state = np.asarray(field, dtype=np.float64)
    if state.ndim == 4 and state.shape[0] == 1:
        state = state[0]
    if state.ndim != 3 or state.shape[0] < len(FIELD_CHANNELS):
        raise ValueError(f"Expected a five-channel field, got {state.shape}.")
    coords = np.asarray(coords_center, dtype=np.float64)
    if coords.ndim == 4 and coords.shape[0] == 1:
        coords = coords[0]
    pressure = state[3]
    vertices = np.asarray(coords_vertex, dtype=np.float64)
    if vertices.ndim == 4 and vertices.shape[0] == 1:
        vertices = vertices[0]
    if vertices.shape != (2, pressure.shape[0] + 1, pressure.shape[1] + 1):
        raise ValueError(
            "Vertex coordinates must have shape (2,H+1,W+1) for an O-grid field; "
            f"got {vertices.shape} for {pressure.shape}."
        )

    wall_x = coords[0, 0]
    wall_y = coords[1, 0]
    wall_cp = (pressure[0] - 1.0) / (0.5 * GAMMA * float(mach) ** 2)
    leading_edge = int(np.argmin(wall_x))
    first = np.column_stack(
        [wall_x[: leading_edge + 1], wall_y[: leading_edge + 1], wall_cp[: leading_edge + 1]]
    )
    second = np.column_stack(
        [wall_x[leading_edge:], wall_y[leading_edge:], wall_cp[leading_edge:]]
    )
    branches = []
    for branch in (first, second):
        branch = branch[np.argsort(branch[:, 0], kind="stable")]
        branches.append(branch)
    upper, lower = (
        (branches[0], branches[1])
        if float(np.mean(branches[0][:, 1])) >= float(np.mean(branches[1][:, 1]))
        else (branches[1], branches[0])
    )
    upper = upper[upper[:, 0] < CP_PLOT_X_MAX]
    lower = lower[lower[:, 0] < CP_PLOT_X_MAX]
    if len(upper) < 2 or len(lower) < 2:
        raise ValueError("Cp surface branches contain too few points after trailing-edge trim.")
    cp_trailing_edge = 0.5 * (float(upper[-1, 2]) + float(lower[-1, 2]))
    upper = np.vstack([upper, [1.0, 0.0, cp_trailing_edge]])
    lower = np.vstack([lower, [1.0, 0.0, cp_trailing_edge]])
    channels = {}
    for index, (key, label) in enumerate(FIELD_CHANNELS):
        values = state[index]
        channels[key] = {
            "label": label,
            "values": values.reshape(-1).tolist(),
            "range": [
                float(np.percentile(values, 2.0)),
                float(np.percentile(values, 98.0)),
            ],
        }
    return {
        "field": {
            "height": int(pressure.shape[0]),
            "width": int(pressure.shape[1]),
            "node_height": int(vertices.shape[1]),
            "node_width": int(vertices.shape[2]),
            "x": vertices[0].reshape(-1).tolist(),
            "y": vertices[1].reshape(-1).tolist(),
            "channels": channels,
        },
        "cp": {
            "upper": {"x": upper[:, 0].tolist(), "cp": upper[:, 2].tolist()},
            "lower": {"x": lower[:, 0].tolist(), "cp": lower[:, 2].tolist()},
        },
    }


class DemoEngine:
    """Own local demo state and bridge the browser to the research runtimes."""

    def __init__(
        self,
        *,
        runtime_root: Path | None = None,
        surrogate_host: str = "127.0.0.1",
        surrogate_port: int = 65432,
        airfoil_library_root: Path | None = None,
        ood_asset_root: Path | None = None,
        mpi_launcher: str = "auto",
        mpi_ranks: int = 8,
        model_dir: Path | None = None,
        model_config: Path = FSB_CONFIG,
        checkpoint: Path | None = None,
        statistics: Path | None = None,
        device: str = "cuda:0",
        resident_pool_root: Path | None = None,
    ) -> None:
        self.runtime_root = Path(runtime_root or _default_runtime_root()).expanduser()
        if not self.runtime_root.is_absolute():
            raise ValueError("The demo runtime directory must be an absolute path.")
        self.airfoil_library_root = Path(
            airfoil_library_root
            or os.environ.get("DEMO_AIRFOIL_LIBRARY_ROOT")
            or DEFAULT_AIRFOIL_LIBRARY_ROOT
        ).expanduser()
        configured_ood_root = ood_asset_root or os.environ.get("DEMO_OOD_ASSET_ROOT")
        self.ood_asset_root = (
            None if configured_ood_root is None else Path(configured_ood_root).expanduser()
        )
        selected_model_dir = Path(
            model_dir
            or os.environ.get("SURROGATE_NEWTON_MODEL_DIR")
            or (REPO_ROOT / "artifacts")
        ).expanduser()
        default_checkpoint, default_statistics = _model_artifacts(selected_model_dir)
        self.model_config = Path(model_config).expanduser()
        self.checkpoint = Path(checkpoint or default_checkpoint).expanduser()
        self.statistics = Path(statistics or default_statistics).expanduser()
        self.device = str(device)
        self.mpi_launcher = str(mpi_launcher)
        self.mpi_ranks = int(mpi_ranks)
        if self.mpi_ranks < 1:
            raise ValueError("DEMO_MPI_RANKS must be positive.")
        self.case_root = self.runtime_root / "cases"
        self.mesh_root = self.runtime_root / "meshes"
        self.case_root.mkdir(parents=True, exist_ok=True)
        self.mesh_root.mkdir(parents=True, exist_ok=True)
        self.resident_pool_root = Path(
            resident_pool_root or (self.runtime_root / "resident_adflow")
        ).expanduser()
        if not self.resident_pool_root.is_absolute():
            raise ValueError("The resident solver pool directory must be absolute.")
        self.client = SurrogateClient(
            SurrogateClientConfig(
                host=surrogate_host,
                port=int(surrogate_port),
                timeout_s=240.0,
            )
        )
        self.geometry_preparer = GeometryPreparer(
            config=GeometryPreparationConfig(
                mesh_mode="pyhyp",
                cache_size=128,
                authority_cgns_dir=self.mesh_root,
            )
        )
        self._ood_index: OodGeometryIndex | None = None
        self._ood_available: bool | None = None
        self._meshes: dict[str, dict[str, Any]] = {}
        self._uiuc_entries: list[dict[str, Any]] | None = None
        self._uiuc_lookup: dict[str, dict[str, Any]] = {}
        self._uiuc_geometries: dict[str, dict[str, Any]] = {}
        self._uiuc_source: dict[str, Any] = {}
        self._uiuc_filter: dict[str, Any] = {}
        self._solver_pool = ResidentWarmPoolController(
            ranks_per_case=self.mpi_ranks,
            pool_count=1,
            mpi_launcher=self.mpi_launcher,
            mpi_omp_threads=1,
            output_dir=self.resident_pool_root,
            ready_timeout_sec=120.0,
            submit_timeout_sec=7200.0,
            request_wait_timeout_sec=7200.0,
        )
        self._prewarm_summary: dict[str, Any] | None = None

    def status(self) -> dict[str, Any]:
        model: dict[str, Any] | None = None
        surrogate_online = False
        try:
            ping = self.client.ping()
            surrogate_online = True
            model = {
                "model_key": ping.get("model_key"),
                "experiment": ping.get("experiment_name"),
                "device": (ping.get("serving") or {}).get("device"),
                "checkpoint": Path(str(ping.get("checkpoint", ""))).name,
                "pde_residual_enabled": (ping.get("serving") or {}).get(
                    "pde_residual_enabled", False
                ),
            }
        except (ConnectionError, OSError, RuntimeError):
            pass
        solver_modules = all(
            importlib.util.find_spec(name) is not None
            for name in ("adflow", "cgnsutilities", "pyhyp")
        )
        solver_ready = solver_modules and _launcher_available(self.mpi_launcher)
        return {
            "mode": "local-only",
            "surrogate_online": surrogate_online,
            "solver_ready": solver_ready,
            "airfoil_library_available": (
                self.airfoil_library_root / "catalog.json"
            ).is_file(),
            "ood_assets_available": self._ood_assets_available(),
            "model": model,
            "resources": {
                "gpu": 1,
                "cpu_ranks_per_case": self.mpi_ranks,
                "reference_state": reference_state_for_mach(1.0),
            },
            "prewarm": self._prewarm_summary,
        }

    def prewarm(self) -> dict[str, Any]:
        """Warm the MPI runtime, pyHyp process, model, forces, and PDE residual."""
        started = time.perf_counter()
        presets = self.presets()["presets"]
        warm_geometry = np.asarray(presets[0]["geometry27"], dtype=np.float32).copy()
        warm_geometry[0] += 1.0e-5
        prewarm_root = self.runtime_root / "prewarm"
        prewarm_root.mkdir(parents=True, exist_ok=True)

        pool_started = time.perf_counter()
        pool_summary = self._solver_pool.start()
        pool_wall = time.perf_counter() - pool_started

        pyhyp_started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="pyhyp_", dir=prewarm_root) as directory:
            prepared = self.geometry_preparer.prepare(
                {
                    "geometry": warm_geometry,
                    "tag": "demo_pyhyp_runtime_warmup",
                    "persist_cgns_path": Path(directory) / "warmup.cgns",
                }
            )
        pyhyp_wall = time.perf_counter() - pyhyp_started

        inference_started = time.perf_counter()
        self.client.request(
            {
                "geometry": prepared.geometry,
                "coords": prepared.coords,
                "coords_vertex": prepared.coords_vertex,
                "mach": 0.74,
                "aoa": 1.0,
                "reynolds": coupled_reynolds_from_mach(0.74),
                "n_inference_steps": 5,
                "return_fields": True,
                "record_sample": False,
                "metadata": {"source": "local_demo_prewarm"},
            }
        )
        inference_wall = time.perf_counter() - inference_started
        self._prewarm_summary = {
            "resident_pool_wall_sec": float(pool_wall),
            "resident_pool_ready": bool(pool_summary),
            "pyhyp_wall_sec": float(pyhyp_wall),
            "inference_wall_sec": float(inference_wall),
            "total_wall_sec": float(time.perf_counter() - started),
        }
        return self._prewarm_summary

    def presets(self) -> dict[str, Any]:
        return {
            "presets": [
                self._preset("rae2822", "RAE2822"),
                self._preset("oat15a", "OAT15A"),
            ]
        }

    def uiuc_catalog(self) -> dict[str, Any]:
        if self._uiuc_entries is None:
            catalog_path = self.airfoil_library_root / "catalog.json"
            if not catalog_path.is_file():
                raise FileNotFoundError("The external UIUC airfoil library is not mounted.")
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            self._uiuc_entries = catalog["airfoils"]
            self._uiuc_lookup = {
                entry["filename"].lower(): entry for entry in self._uiuc_entries
            }
            self._uiuc_source = catalog["source"]
            self._uiuc_filter = catalog["filter"]
        return {
            "source": self._uiuc_source,
            "filter": self._uiuc_filter,
            "count": len(self._uiuc_entries),
            "airfoils": self._uiuc_entries,
        }

    def uiuc_airfoil(self, filename: str) -> dict[str, Any]:
        self.uiuc_catalog()
        if not re.fullmatch(r"[A-Za-z0-9_.+()-]+\.dat", filename, flags=re.IGNORECASE):
            raise ValueError("Invalid UIUC coordinate filename.")
        key = filename.lower()
        if key not in self._uiuc_lookup:
            raise KeyError(f"Unknown UIUC coordinate file: {filename}")
        if key not in self._uiuc_geometries:
            entry = self._uiuc_lookup[key]
            relative_path = Path(str(entry["coordinate_path"]))
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError("Invalid UIUC asset manifest path.")
            coordinate_path = (self.airfoil_library_root / relative_path).resolve()
            if not coordinate_path.is_relative_to(self.airfoil_library_root.resolve()):
                raise ValueError("Invalid UIUC asset manifest path.")
            if not coordinate_path.is_file():
                raise FileNotFoundError("The selected UIUC coordinate asset is unavailable.")
            content = coordinate_path.read_text(encoding="iso-8859-1")
            geometry = self.import_coordinates(content, entry["filename"])
            geometry["name"] = entry["name"]
            geometry["source"] = {
                "name": "UIUC Airfoil Data Site",
                "filename": entry["filename"],
                "description": entry["description"],
                "url": entry["coordinate_url"],
            }
            self._uiuc_geometries[key] = geometry
        return {"geometry": self._uiuc_geometries[key]}

    def _preset(self, folder_name: str, display_name: str) -> dict[str, Any]:
        root = PRESET_ROOT / folder_name
        upper = np.loadtxt(root / "cst_u0.txt", dtype=np.float64)
        lower = np.loadtxt(root / "cst_l0.txt", dtype=np.float64)
        thickness = float(np.loadtxt(root / "t0.txt", dtype=np.float64))
        upper, lower = scale_cst20_to_max_thickness(
            upper,
            lower,
            t_max=thickness,
            tail=FIXED_TRAILING_EDGE_THICKNESS,
        )
        geometry27 = cst20_to_cst27(
            upper,
            lower,
            tail=FIXED_TRAILING_EDGE_THICKNESS,
        )
        return _geometry_payload(
            display_name,
            geometry27,
            ood=self.ood_score(geometry27),
        )

    def import_coordinates(self, content: str, filename: str) -> dict[str, Any]:
        candidate = Path(str(filename))
        if (
            candidate.name != str(filename)
            or candidate.suffix.lower() not in {".dat", ".txt", ".csv"}
        ):
            raise ValueError("Coordinate filename must end in .dat, .txt, or .csv.")
        geometry27, mse = _fit_coordinate_text_to_cst(content)
        return _geometry_payload(
            Path(filename).stem or "Uploaded airfoil",
            geometry27,
            fit_mse=mse,
            ood=self.ood_score(geometry27),
        )

    def project_geometry(
        self,
        x: list[float],
        upper: list[float],
        lower: list[float],
        name: str,
    ) -> dict[str, Any]:
        x_array = np.asarray(x, dtype=np.float64)
        upper_array = np.asarray(upper, dtype=np.float64)
        lower_array = np.asarray(lower, dtype=np.float64)
        _validate_geometry(x_array, upper_array, lower_array)
        geometry27 = coords_to_cst27(
            x_array,
            upper_array,
            x_array,
            lower_array,
            tail=FIXED_TRAILING_EDGE_THICKNESS,
        )
        canonical = _geometry_payload(
            name,
            geometry27,
            ood=self.ood_score(geometry27),
        )
        reconstructed_upper = np.asarray(canonical["upper"])
        reconstructed_lower = np.asarray(canonical["lower"])
        source_upper = np.interp(canonical["x"], x_array, upper_array)
        source_lower = np.interp(canonical["x"], x_array, lower_array)
        canonical["metrics"]["projection_mse"] = float(
            np.mean(
                np.concatenate(
                    [
                        reconstructed_upper - source_upper,
                        reconstructed_lower - source_lower,
                    ]
                )
                ** 2
            )
        )
        return canonical

    def _ood_assets_available(self) -> bool:
        return self._load_ood()

    def _load_ood(self) -> bool:
        if self._ood_available is not None:
            return self._ood_available
        if self.ood_asset_root is None:
            self._ood_available = False
            return False
        try:
            self._ood_index = OodGeometryIndex.from_asset_root(self.ood_asset_root)
        except (FileNotFoundError, OSError, ValueError):
            self._ood_available = False
            return False
        self._ood_available = True
        return True

    def ood_score(self, geometry27: np.ndarray) -> dict[str, Any] | None:
        if not self._load_ood():
            return None
        assert self._ood_index is not None
        return self._ood_index.score(geometry27)

    def predict(
        self,
        *,
        geometry27: list[float],
        mach: float,
        aoa: float,
        n_inference_steps: int,
        name: str,
    ) -> dict[str, Any]:
        if not 0.2 <= float(mach) <= 0.9:
            raise ValueError("Mach must be between 0.2 and 0.9 for this model.")
        if not -5.0 <= float(aoa) <= 10.0:
            raise ValueError("Angle of attack must be between -5 and 10 degrees.")
        reference_state = reference_state_for_mach(mach)
        reynolds = reference_state["reynolds"]
        geometry = np.asarray(geometry27, dtype=np.float32).reshape(27)
        if int(n_inference_steps) < 1 or int(n_inference_steps) > 20:
            raise ValueError("Surrogate inference steps must be between 1 and 20.")
        geometry_key = hashlib.sha256(geometry.tobytes()).hexdigest()[:12]
        if geometry_key not in self._meshes:
            persisted_mesh = self.mesh_root / f"{geometry_key}.demo.json"
            if not persisted_mesh.is_file():
                raise ValueError("Generate the geometry mesh before running the surrogate.")
            self._meshes[geometry_key] = json.loads(
                persisted_mesh.read_text(encoding="utf-8")
            )
        mesh_lookup_start = time.perf_counter()
        prepared = self.geometry_preparer.prepare(
            {
                "geometry": geometry,
                "tag": geometry_key,
                "persist_cgns": True,
            }
        )
        mesh_lookup_wall = time.perf_counter() - mesh_lookup_start
        inference_start = time.perf_counter()
        response = self.client.request(
            {
                "geometry": prepared.geometry,
                "coords": prepared.coords,
                "coords_vertex": prepared.coords_vertex,
                "mach": float(mach),
                "aoa": float(aoa),
                "reynolds": float(reynolds),
                "n_inference_steps": int(n_inference_steps),
                "return_fields": True,
                "record_sample": False,
                "metadata": {"source": "local_demo"},
            }
        )
        inference_wall = time.perf_counter() - inference_start
        fields = np.asarray(response["fields"], dtype=np.float32)
        if fields.ndim == 4 and fields.shape[0] == 1:
            fields = fields[0]
        digest = hashlib.sha256(
            geometry.tobytes()
            + np.asarray([mach, aoa, reynolds], dtype=np.float64).tobytes()
        ).hexdigest()[:10]
        case_id = f"case_{_utc_stamp()}_{digest}"
        case_dir = self.case_root / case_id
        case_dir.mkdir(parents=True, exist_ok=False)
        authority_path = Path(str(prepared.metadata["authority_cgns_path"]))
        np.savez_compressed(
            case_dir / "surrogate_state.npz",
            geometry=prepared.geometry,
            coords=prepared.coords,
            coords_vertex=prepared.coords_vertex,
            fields=fields,
            flow_conditions=np.asarray([mach, aoa, reynolds], dtype=np.float64),
        )
        view = _pressure_view(
            fields,
            prepared.coords,
            prepared.coords_vertex,
            mach=float(mach),
        )
        residual_components = response.get("residual_components") or {}
        residual_ratio = residual_components.get("l2_ratio")
        summary = {
            "case_id": case_id,
            "name": str(name),
            "flow": {
                "mach": float(mach),
                "aoa": float(aoa),
                "reynolds": float(reynolds),
                "reference_state": reference_state,
            },
            "geometry_id": prepared.geometry_id,
            "authority_cgns_path": str(authority_path),
            "mesh": self._meshes[geometry_key],
            "stage": {
                "key": "surrogate",
                "label": "Neural estimate",
                "status": "complete",
                "forces": {
                    "cl": _first_scalar(response["cl"]),
                    "cd": _first_scalar(response["cd"]),
                    "cm": _first_scalar(response["cm"]),
                },
                "residual": {
                    "kind": "rans_sa_l2_ratio",
                    "final": None
                    if residual_ratio is None
                    else _first_scalar(residual_ratio),
                    "values": []
                    if residual_ratio is None
                    else [_first_scalar(residual_ratio)],
                    "budgets": [int(n_inference_steps)] if residual_ratio is not None else [],
                },
                "timing": {
                    "mesh_lookup_sec": float(mesh_lookup_wall),
                    "inference_wall_sec": float(inference_wall),
                },
                "n_inference_steps": int(n_inference_steps),
                **view,
            },
        }
        _json_dump(case_dir / "summary.json", summary)
        return _public_payload(summary)

    def prepare_mesh(
        self,
        *,
        geometry27: list[float],
        name: str,
    ) -> dict[str, Any]:
        geometry = np.asarray(geometry27, dtype=np.float32).reshape(27)
        geometry_key = hashlib.sha256(geometry.tobytes()).hexdigest()[:12]
        mesh_started = time.perf_counter()
        prepared = self.geometry_preparer.prepare(
            {
                "geometry": geometry,
                "tag": geometry_key,
                "persist_cgns": True,
            }
        )
        mesh_wall = time.perf_counter() - mesh_started
        flow = np.asarray(
            [0.74, 1.0, coupled_reynolds_from_mach(0.74)],
            dtype=np.float64,
        )
        initial_field = UniformFlowInitializer(
            normalizer=None,
            device="cpu",
        ).generate_uniform_field(
            torch.as_tensor(flow[None, :], dtype=torch.float32),
            spatial_shape=tuple(int(value) for value in prepared.coords.shape[-2:]),
        )[0].numpy()
        plan = finalonly_plan(
            "fsb",
            work=NKWorkPlan.fixed(
                1,
                solver_preset=SolverPreset.NK,
            ),
        )
        authority_path = Path(str(prepared.metadata["authority_cgns_path"]))
        warm_root = self.runtime_root / "solver_prepare" / geometry_key
        case = build_fsb_case(
            case_id=f"demo_prepare_{geometry_key}",
            cgns_basename=authority_path.name,
            cgns_root=authority_path.parent,
            prediction_field=initial_field,
            state_name="final",
            flow_conditions=flow.tolist(),
            flow_conditions_dict=self._flow_mapping(flow),
            coords_center=prepared.coords,
            coords_vertex=prepared.coords_vertex,
            ranks_per_case=self.mpi_ranks,
            mpi_launcher=self.mpi_launcher,
            mpi_omp_threads=1,
            options_version=2,
            l2conv=1.0e-8,
            output_dir=warm_root / "case",
            created_by="surrogate_newton_demo_solver_prepare",
            config_path=self.model_config,
            checkpoint_path=self.checkpoint,
            stats_path=self.statistics,
            device=self.device,
        )
        export = create_pipeline().export_cases(
            (case,),
            plan,
            output_dir=str(warm_root / "export"),
        )
        solver_started = time.perf_counter()
        solver_prepare = self._solver_pool.prepare(
            export.manifest_path,
            output_dir=warm_root / "resident",
        )
        solver_wall = time.perf_counter() - solver_started
        result = {
            "geometry_id": prepared.geometry_id,
            "geometry_key": geometry_key,
            "name": str(name),
            "authority_cgns_path": str(authority_path),
            "mesh_wall_sec": float(mesh_wall),
            "adflow_prepare_wall_sec": float(solver_wall),
            "adflow_solver_reused": bool(
                (solver_prepare.get("solver_warmup") or {}).get("solver_reused", False)
            ),
            "mpi_ranks": self.mpi_ranks,
        }
        _json_dump(self.mesh_root / f"{geometry_key}.demo.json", result)
        self._meshes[geometry_key] = result
        return _public_payload(result)

    def get_case(self, case_id: str) -> dict[str, Any]:
        return _public_payload(self._load_case_summary(case_id))

    def _load_case_summary(self, case_id: str) -> dict[str, Any]:
        return json.loads(
            (self._case_dir(case_id) / "summary.json").read_text(encoding="utf-8")
        )

    def _case_dir(self, case_id: str) -> Path:
        if re.fullmatch(r"case_[A-Za-z0-9_]+", case_id) is None:
            raise ValueError("Invalid demo case identifier.")
        path = (self.case_root / case_id).resolve()
        if not path.is_relative_to(self.case_root.resolve()):
            raise ValueError("Invalid demo case identifier.")
        if not path.is_dir():
            raise FileNotFoundError(f"Unknown demo case: {case_id}")
        return path

    @staticmethod
    def _flow_mapping(flow: np.ndarray) -> dict[str, Any]:
        return {
            "mach": float(flow[0]),
            "alpha": float(flow[1]),
            "reynolds": float(flow[2]),
            "temperature": float(REFERENCE_T_INF),
            "pressure": float(REFERENCE_P_INF),
            "area_ref": 1.0,
            "chord_ref": float(REFERENCE_CHORD),
            "reference_state_mode": "dataset_unified",
        }

    def recover(
        self,
        case_id: str,
        *,
        cycles: int = 6,
        residual_exponent: int = 8,
    ) -> dict[str, Any]:
        if cycles < 1 or cycles > 20:
            raise ValueError("Maximum terminal NK cycles must be between 1 and 20.")
        if residual_exponent < 2 or residual_exponent > 12:
            raise ValueError("NK stopping exponent must be between 2 and 12.")
        residual_threshold = 10.0 ** (-int(residual_exponent))
        case_dir = self._case_dir(case_id)
        with np.load(case_dir / "surrogate_state.npz", allow_pickle=False) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
        plan = finalonly_plan(
            "fsb",
            work=NKWorkPlan.adaptive(
                range(1, int(cycles) + 1),
                threshold=residual_threshold,
                name="demo_terminal_nk",
                solver_preset=SolverPreset.NK,
            ),
        )
        run_root = case_dir / "solver_runs" / f"terminal_nk_{_utc_stamp()}"
        result = self._run_solver(
            case_id=case_id,
            run_root=run_root,
            arrays=arrays,
            initial_field=arrays["fields"],
            plan=plan,
            created_by="surrogate_newton_demo_terminal_nk",
        )
        summary = self._load_case_summary(case_id)
        result["correction_mse"] = float(
            np.mean((result.pop("_field") - arrays["fields"]) ** 2)
        )
        result["stop_residual_l2_ratio"] = residual_threshold
        result["stop_residual_exponent"] = int(residual_exponent)
        summary["recovery"] = result
        if "reference" in summary:
            reference_field = np.load(Path(summary["reference"]["field_path"]))
            recovery_field = np.load(Path(result["field_path"]))
            summary["recovery"]["reference_mse"] = float(
                np.mean((recovery_field - reference_field) ** 2)
            )
            summary["recovery"]["pressure_reference_mse"] = float(
                np.mean((recovery_field[3] - reference_field[3]) ** 2)
            )
            summary["recovery"]["force_mae_reference"] = _force_mae_against_reference(
                summary["recovery"]["forces"],
                summary["reference"]["forces"],
            )
        _json_dump(case_dir / "summary.json", summary)
        return _public_payload(summary)

    def reference(self, case_id: str, *, max_cycles: int = 3000) -> dict[str, Any]:
        if max_cycles < 25 or max_cycles > 6000:
            raise ValueError("Cold-start ADflow budget must be between 25 and 6000 cycles.")
        case_dir = self._case_dir(case_id)
        with np.load(case_dir / "surrogate_state.npz", allow_pickle=False) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
        flow_tensor = torch.as_tensor(
            arrays["flow_conditions"][None, :],
            dtype=torch.float32,
        )
        uniform = UniformFlowInitializer(normalizer=None, device="cpu").generate_uniform_field(
            flow_tensor,
            spatial_shape=tuple(int(value) for value in arrays["fields"].shape[-2:]),
        )[0].numpy()
        plan = finalonly_plan(
            "fsb",
            work=NKWorkPlan.fixed(
                int(max_cycles),
                solver_preset=SolverPreset.PROD,
            ),
        )
        run_root = case_dir / "solver_runs" / f"cold_start_{_utc_stamp()}"
        result = self._run_solver(
            case_id=case_id,
            run_root=run_root,
            arrays=arrays,
            initial_field=uniform,
            plan=plan,
            created_by="surrogate_newton_demo_cold_start",
        )
        reference_field = result.pop("_field")
        summary = self._load_case_summary(case_id)
        summary["stage"]["reference_mse"] = float(
            np.mean((arrays["fields"] - reference_field) ** 2)
        )
        summary["stage"]["pressure_reference_mse"] = float(
            np.mean((arrays["fields"][3] - reference_field[3]) ** 2)
        )
        summary["stage"]["force_mae_reference"] = _force_mae_against_reference(
            summary["stage"]["forces"],
            result["forces"],
        )
        result["reference_mse"] = 0.0
        result["pressure_reference_mse"] = 0.0
        result["force_mae_reference"] = 0.0
        summary["reference"] = result
        if "recovery" in summary:
            recovery_field_path = Path(summary["recovery"]["field_path"])
            recovery_field = np.load(recovery_field_path)
            summary["recovery"]["reference_mse"] = float(
                np.mean((recovery_field - reference_field) ** 2)
            )
            summary["recovery"]["pressure_reference_mse"] = float(
                np.mean((recovery_field[3] - reference_field[3]) ** 2)
            )
            summary["recovery"]["force_mae_reference"] = _force_mae_against_reference(
                summary["recovery"]["forces"],
                result["forces"],
            )
        _json_dump(case_dir / "summary.json", summary)
        return _public_payload(summary)

    def _run_solver(
        self,
        *,
        case_id: str,
        run_root: Path,
        arrays: dict[str, np.ndarray],
        initial_field: np.ndarray,
        plan: Any,
        created_by: str,
    ) -> dict[str, Any]:
        summary = self._load_case_summary(case_id)
        cgns_path = Path(summary["authority_cgns_path"])
        flow = arrays["flow_conditions"]
        case = build_fsb_case(
            case_id=f"{case_id}_{plan.final_stage.work.solver_preset.value}",
            cgns_basename=cgns_path.name,
            cgns_root=cgns_path.parent,
            prediction_field=initial_field,
            state_name="final",
            flow_conditions=flow.tolist(),
            flow_conditions_dict=self._flow_mapping(flow),
            coords_center=arrays["coords"],
            coords_vertex=arrays["coords_vertex"],
            ranks_per_case=self.mpi_ranks,
            mpi_launcher=self.mpi_launcher,
            mpi_omp_threads=1,
            options_version=2,
            l2conv=1.0e-8,
            output_dir=run_root / "case",
            created_by=created_by,
            config_path=self.model_config,
            checkpoint_path=self.checkpoint,
            stats_path=self.statistics,
            device=self.device,
        )
        export = create_pipeline().export_cases(
            (case,),
            plan,
            output_dir=str(run_root / "export"),
        )
        run_root.mkdir(parents=True, exist_ok=True)
        request_started = time.perf_counter()
        pool_result = self._solver_pool.project(
            export.manifest_path,
            output_dir=run_root / "resident",
        )
        request_wall = time.perf_counter() - request_started
        result_payload = json.loads(Path(export.result_paths[0]).read_text(encoding="utf-8"))
        stage = result_payload["stages"][-1]
        field_path = Path(stage["output_paths"]["post_field"])
        field = np.load(field_path)
        metrics = stage["metrics"]
        residual = metrics.get("residual") or {}
        force = metrics.get("force_coefficients") or {}
        residual_kind = (residual.get("metadata") or {}).get(
            "value_kind", "raw_totalr"
        )
        residual_final = (residual.get("summary") or {}).get("final")
        residual_threshold = residual.get("threshold")
        converged = (
            residual_kind == "ratio_to_reference_totalr0"
            and residual_final is not None
            and residual_threshold is not None
            and float(residual_final) <= float(residual_threshold)
        )
        view = _pressure_view(
            field,
            arrays["coords"],
            arrays["coords_vertex"],
            mach=float(flow[0]),
        )
        work = plan.final_stage.work
        cycle_limit = (
            int(work.fixed_cycles)
            if int(work.fixed_cycles) > 0
            else int(work.adaptive_schedule.cumulative_cycles[-1])
        )
        residual_budgets = [int(value) for value in residual.get("budgets", [])]
        is_nk = plan.final_stage.work.solver_preset == SolverPreset.NK
        executed_cycles = residual_budgets[-1] if is_nk else None
        return {
            "key": "recovery"
            if is_nk
            else "reference",
            "label": f"Surrogate + NK (max {cycle_limit})"
            if is_nk
            else "Cold-start ADflow reference",
            "status": "complete",
            "converged": converged,
            "cycle_limit": cycle_limit,
            "executed_cycles": executed_cycles,
            "stopped_early": bool(
                is_nk and converged and executed_cycles < cycle_limit
            ),
            "forces": {
                key: float(force[key])
                for key in ("cl", "cd", "cm")
                if key in force
            },
            "residual": {
                "kind": residual_kind,
                "final": residual_final,
                "values": residual.get("values", []),
                "budgets": residual_budgets,
                "threshold": residual_threshold
                if residual_kind == "ratio_to_reference_totalr0"
                else None,
            },
            "timing": {
                "solver_wall_sec": float(stage["timing"]["solver_wall_sec"]),
                "total_wall_sec": float(stage["timing"]["total_wall_sec"]),
                "request_wall_sec": float(request_wall),
                "resident_pool_reused": bool(
                    pool_result.metadata.get("pool_controller_reused", False)
                ),
            },
            "field_path": str(field_path),
            "result_path": str(export.result_paths[0]),
            "solver_runtime": str(pool_result.status_paths[0]),
            **view,
            "_field": field,
        }

    def close(self) -> None:
        self._solver_pool.close()


__all__ = [
    "DemoEngine",
    "DEFAULT_AIRFOIL_LIBRARY_ROOT",
    "FIXED_TRAILING_EDGE_THICKNESS",
    "MAX_UPLOAD_BYTES",
    "UIUC_CATALOG_URL",
    "UIUC_COORDINATE_ROOT",
    "UIUC_CST_FIT_MSE_LIMIT",
    "UIUC_SITE_URL",
    "reference_state_for_mach",
    "_geometry_payload",
    "_pressure_view",
    "_split_coordinate_text",
]
