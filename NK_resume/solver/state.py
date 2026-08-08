"""Clean ADflow state injection and extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
import sys
from typing import Any, Mapping

import numpy as np

from ..exceptions import ContractError
from ..schema import ResumeCase
from .info_injection import build_restart_aligned_local_info


def _field_array(value: Any) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except Exception as exc:  # pragma: no cover - numpy owns exact exception type.
        raise ContractError("field cannot be converted to a numeric array") from exc
    if array.ndim != 3:
        raise ContractError(f"field must be 3D, got shape={array.shape}")
    if array.shape[0] in {4, 5}:
        return array
    if array.shape[-1] in {4, 5}:
        return np.moveaxis(array, -1, 0)
    raise ContractError(
        "field must be channel-first (4/5,H,W) or channel-last (H,W,4/5)"
    )


def _default_mpi_comm() -> Any:
    try:
        module = importlib.import_module("mpi4py.MPI")
    except Exception:
        return None
    return getattr(module, "COMM_WORLD", None)


def _comm_size(comm: Any) -> int:
    getter = getattr(comm, "Get_size", None)
    if not callable(getter):
        return 1
    return int(getter())


def _comm_rank(comm: Any) -> int:
    getter = getattr(comm, "Get_rank", None)
    if not callable(getter):
        return 0
    return int(getter())


def _trace(comm: Any, event: str, **payload: Any) -> None:
    if str(os.environ.get("NK_RESUME_MPI_TRACE", "")).strip() != "1":
        return
    parts = [f"rank={_comm_rank(comm)}", f"event={event}"]
    parts.extend(f"{key}={value}" for key, value in sorted(payload.items()))
    print("[NK_resume.state] " + " ".join(parts), file=sys.stderr, flush=True)


def _comm_barrier(comm: Any, event: str) -> None:
    barrier = getattr(comm, "Barrier", None)
    if not callable(barrier):
        return
    _trace(comm, f"before_{event}_barrier")
    barrier()
    _trace(comm, f"after_{event}_barrier")


def _state_vector(solver: Any, comm: Any = None) -> np.ndarray:
    getter = getattr(solver, "getStates", None)
    if not callable(getter):
        raise ContractError("ADflow state injection requires solver.getStates()")
    try:
        _trace(comm, "before_get_states")
        states = np.asarray(getter(), dtype=np.float64).reshape(-1)
        _trace(comm, "after_get_states", state_size=int(states.size))
    except Exception as exc:
        raise ContractError("solver.getStates() failed") from exc
    if states.size == 0:
        raise ContractError("solver.getStates() returned an empty state vector")
    return states


def _set_state_vector(solver: Any, states: np.ndarray, comm: Any = None) -> None:
    setter = getattr(solver, "setStates", None)
    if not callable(setter):
        raise ContractError("ADflow state injection requires solver.setStates()")
    try:
        _trace(comm, "before_set_states", state_size=int(np.asarray(states).size))
        setter(np.asarray(states, dtype=np.float64))
        _trace(comm, "after_set_states", state_size=int(np.asarray(states).size))
    except Exception as exc:
        raise ContractError("solver.setStates(...) failed") from exc


def _infer_n_vars(state_size: int, dataset_h: int, dataset_w: int) -> int:
    cell_count = int(dataset_h) * int(dataset_w)
    if cell_count <= 0:
        raise ContractError("field spatial shape must be positive")
    if state_size % cell_count != 0:
        raise ContractError(
            f"ADflow state size {state_size} is not divisible by field cell count {cell_count}"
        )
    n_vars = state_size // cell_count
    if n_vars < 5:
        raise ContractError(f"ADflow state vector needs at least 5 variables, got {n_vars}")
    return int(n_vars)


def _infer_n_vars_distributed(
    local_state_size: int,
    dataset_h: int,
    dataset_w: int,
    comm: Any,
) -> int:
    reducer = getattr(comm, "allreduce", None)
    if not callable(reducer):
        raise ContractError("distributed ADflow state injection requires MPI allreduce")
    global_size = int(reducer(int(local_state_size)))
    return _infer_n_vars(global_size, dataset_h, dataset_w)


def _local_cell_indices_from_rank(
    *,
    dataset_h: int,
    dataset_w: int,
    local_cell_count: int,
    comm: Any,
) -> np.ndarray:
    gather = getattr(comm, "allgather", None)
    if not callable(gather):
        raise ContractError("distributed ADflow state injection requires MPI allgather")
    rank = _comm_rank(comm)
    local_width, remainder = divmod(int(local_cell_count), int(dataset_h))
    if remainder:
        raise ContractError(
            f"local cell count {local_cell_count} is not divisible by dataset_h={dataset_h}"
        )
    widths = [int(value) for value in gather(local_width)]
    if sum(widths) != int(dataset_w):
        raise ContractError(
            "distributed ADflow local widths do not cover dataset width: "
            f"widths={widths}, dataset_w={dataset_w}"
        )
    start = sum(widths[:rank])
    cols = start + np.arange(local_width, dtype=np.int64)
    rows = np.arange(int(dataset_h), dtype=np.int64)[:, None]
    return (rows * int(dataset_w) + cols[None, :]).reshape(-1)


def _global_cell_centers_flat(
    *,
    coords_center: Any,
    dataset_h: int,
    dataset_w: int,
) -> np.ndarray | None:
    coords = np.asarray(coords_center, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[1:] != (int(dataset_h), int(dataset_w)):
        return None
    if coords.shape[0] < 2:
        return None
    return np.asarray(coords[:2], dtype=np.float64).transpose(1, 2, 0).reshape(-1, 2)


def _local_centers_xy(solver: Any, local_cell_count: int) -> np.ndarray | None:
    utils = getattr(getattr(solver, "adflow", None), "utils", None)
    getter = getattr(utils, "getcellcenters", None)
    if not callable(getter):
        return None
    try:
        local_centers = np.asarray(getter(1, int(local_cell_count)), dtype=np.float64)
    except Exception:
        return None
    if local_centers.ndim != 2:
        return None
    if local_centers.shape[0] in (2, 3) and local_centers.shape[1] == int(local_cell_count):
        local_centers = local_centers.T
    elif local_centers.shape[0] != int(local_cell_count):
        return None
    if local_centers.shape[1] < 2:
        return None
    return np.asarray(local_centers[:, :2], dtype=np.float64)


def _assign_local_cells_via_kdtree(
    *,
    local_xy: np.ndarray,
    global_xy: np.ndarray,
    max_dist2: float,
) -> np.ndarray | None:
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return None

    n_local = int(local_xy.shape[0])
    n_global = int(global_xy.shape[0])
    tree = cKDTree(global_xy)
    for k in (1, 4, 8, 16):
        k_eff = min(int(k), n_global)
        dist, idx = tree.query(local_xy, k=k_eff, workers=1)
        dist = np.asarray(dist, dtype=np.float64)
        idx = np.asarray(idx, dtype=np.int64)
        if dist.ndim == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        dist2 = dist * dist
        valid = np.isfinite(dist2) & (dist2 <= float(max_dist2))
        if k_eff == 1:
            matched = np.asarray(idx[:, 0], dtype=np.int64)
            if np.all(valid[:, 0]) and np.unique(matched).size == matched.size:
                return matched

        local_slots, candidate_slots = np.where(valid)
        if local_slots.size == 0:
            continue
        order = np.argsort(dist2[local_slots, candidate_slots], kind="mergesort")
        matched = np.full(n_local, -1, dtype=np.int64)
        used_global = np.zeros(n_global, dtype=bool)
        for order_pos in order.tolist():
            local_pos = int(local_slots[order_pos])
            candidate_pos = int(candidate_slots[order_pos])
            global_pos = int(idx[local_pos, candidate_pos])
            if matched[local_pos] >= 0 or used_global[global_pos]:
                continue
            matched[local_pos] = global_pos
            used_global[global_pos] = True
        if np.all(matched >= 0):
            return matched
    return None


def _assign_local_cells_via_lookup(
    *,
    local_xy: np.ndarray,
    global_xy: np.ndarray,
    max_dist2: float,
) -> np.ndarray | None:
    matched = np.full(int(local_xy.shape[0]), -1, dtype=np.int64)
    used: set[int] = set()
    for decimals in (12, 10, 8, 6):
        if np.all(matched >= 0):
            break
        scale = float(10**decimals)
        lookup: dict[tuple[int, int], list[int]] = {}
        global_keys = np.rint(global_xy * scale).astype(np.int64)
        for index, key_array in enumerate(global_keys):
            key = (int(key_array[0]), int(key_array[1]))
            lookup.setdefault(key, []).append(int(index))
        local_keys = np.rint(local_xy * scale).astype(np.int64)
        for local_pos in np.where(matched < 0)[0].tolist():
            key_array = local_keys[local_pos]
            key = (int(key_array[0]), int(key_array[1]))
            candidates = lookup.get(key) or []
            available = [candidate for candidate in candidates if candidate not in used]
            if len(available) != 1:
                continue
            chosen = int(available[0])
            dist2 = float(np.sum((global_xy[chosen] - local_xy[local_pos]) ** 2))
            if dist2 > float(max_dist2):
                continue
            matched[local_pos] = chosen
            used.add(chosen)
    if np.all(matched >= 0) and np.unique(matched).size == matched.size:
        return matched
    return None


def _local_cell_indices_from_centers(
    solver: Any,
    *,
    coords_center: Any,
    dataset_h: int,
    dataset_w: int,
    local_cell_count: int,
    comm: Any,
) -> tuple[np.ndarray, str] | None:
    global_xy = _global_cell_centers_flat(
        coords_center=coords_center,
        dataset_h=dataset_h,
        dataset_w=dataset_w,
    )
    if global_xy is None:
        return None
    local_xy = _local_centers_xy(solver, int(local_cell_count))
    if local_xy is None:
        return None
    max_dist2 = float(os.environ.get("NK_RESUME_CELL_MAP_MAX_DIST2", "1e-10") or "1e-10")
    local_width, remainder = divmod(int(local_cell_count), int(dataset_h))
    if remainder == 0:
        global_rows = np.asarray(global_xy, dtype=np.float64).reshape(
            int(dataset_h), int(dataset_w), 2
        )
        local_rows = np.asarray(local_xy, dtype=np.float64).reshape(
            int(dataset_h), local_width, 2
        )
        indices = np.full(int(local_cell_count), -1, dtype=np.int64)
        rowwise_ok = True
        for row in range(int(dataset_h)):
            local_row = local_rows[row]
            global_row = global_rows[row]
            first_dist2 = np.sum((global_row - local_row[0]) ** 2, axis=1)
            start_candidates = np.argsort(first_dist2, kind="mergesort")[:8]
            best_start: int | None = None
            best_dist2: float | None = None
            for start in start_candidates.tolist():
                cols = (int(start) + np.arange(local_width, dtype=np.int64)) % int(dataset_w)
                diff = global_row[cols] - local_row
                row_dist2 = float(np.max(np.sum(diff * diff, axis=1)))
                if best_dist2 is None or row_dist2 < best_dist2:
                    best_dist2 = row_dist2
                    best_start = int(start)
            if best_start is None or best_dist2 is None or best_dist2 > max_dist2:
                rowwise_ok = False
                break
            cols = (best_start + np.arange(local_width, dtype=np.int64)) % int(dataset_w)
            indices[row * local_width : (row + 1) * local_width] = (
                row * int(dataset_w) + cols
            )
        if rowwise_ok and np.unique(indices).size == indices.size:
            _trace(
                comm,
                "cell_mapping_rowwise",
                max_dist2=f"{max_dist2:.3e}",
                local_width=int(local_width),
            )
            return indices, "centers_rowwise"

    matched = _assign_local_cells_via_kdtree(
        local_xy=local_xy,
        global_xy=global_xy,
        max_dist2=max_dist2,
    )
    if matched is not None:
        _trace(comm, "cell_mapping_kdtree", max_dist2=f"{max_dist2:.3e}")
        return matched, "centers_kdtree"

    matched = _assign_local_cells_via_lookup(
        local_xy=local_xy,
        global_xy=global_xy,
        max_dist2=max_dist2,
    )
    if matched is not None:
        _trace(comm, "cell_mapping_lookup", max_dist2=f"{max_dist2:.3e}")
        return matched, "centers_lookup"

    return None


def _local_cell_indices(
    solver: Any,
    case: ResumeCase,
    *,
    dataset_h: int,
    dataset_w: int,
    local_cell_count: int,
    comm: Any,
) -> tuple[np.ndarray, str]:
    mode = str(os.environ.get("NK_RESUME_CELL_MAPPING", "centers")).strip().lower()
    if mode in {"", "center"}:
        mode = "centers"
    if mode not in {"centers", "rank"}:
        raise ContractError(
            "NK_RESUME_CELL_MAPPING must be one of: centers, rank"
        )
    if mode == "rank":
        rank_partition = _local_cell_indices_from_rank(
            dataset_h=dataset_h,
            dataset_w=dataset_w,
            local_cell_count=local_cell_count,
            comm=comm,
        )
        return rank_partition, "rank_contiguous"
    from_centers = _local_cell_indices_from_centers(
        solver,
        coords_center=case.geometry.coords_center,
        dataset_h=dataset_h,
        dataset_w=dataset_w,
        local_cell_count=local_cell_count,
        comm=comm,
    )
    if from_centers is not None:
        return from_centers
    raise ContractError(
        "Unable to map local ADflow cell centers to dataset cells. "
        "Check coords_center and ADflow grid compatibility; use "
        "NK_RESUME_CELL_MAPPING=rank only for diagnostics."
    )


def _primitive_to_state_vector(
    field: np.ndarray,
    prev_states: np.ndarray,
    *,
    gamma: float,
    turbulence_source: str,
    flatten_order: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    channels = _field_array(field)
    _, dataset_h, dataset_w = channels.shape
    n_vars = _infer_n_vars(prev_states.size, dataset_h, dataset_w)

    rho = channels[0].T
    u = channels[1].T
    v = channels[2].T
    p = channels[3].T
    rho_e = p / (float(gamma) - 1.0) + 0.5 * rho * (u * u + v * v)

    new_states = np.array(prev_states, dtype=np.float64, copy=True)
    try:
        new_states[0::n_vars] = rho.flatten(order=flatten_order)
        new_states[1::n_vars] = u.flatten(order=flatten_order)
        new_states[2::n_vars] = v.flatten(order=flatten_order)
        new_states[3::n_vars] = 0.0
        new_states[4::n_vars] = rho_e.flatten(order=flatten_order)
    except ValueError as exc:
        raise ContractError(
            "ADflow state injection shape mismatch: "
            f"field_shape={channels.shape}, state_size={prev_states.size}, n_vars={n_vars}"
        ) from exc

    if turbulence_source == "frozen":
        turbulence_mode = "preserved"
    elif turbulence_source == "model":
        if channels.shape[0] < 5:
            raise ContractError("turbulence_source='model' requires a fifth field channel")
        if n_vars <= 5:
            raise ContractError("turbulence_source='model' requires n_vars > 5")
        new_states[5::n_vars] = channels[4].T.flatten(order=flatten_order)
        turbulence_mode = "model_channel"
    else:
        raise ContractError(
            "ADflowStateAdapter.turbulence_source must be 'frozen' or 'model'"
        )

    return new_states, {
        "dataset_shape": [int(dataset_h), int(dataset_w)],
        "n_vars": int(n_vars),
        "state_size": int(new_states.size),
        "turbulence": turbulence_mode,
    }


def _primitive_to_local_state_vector(
    field: np.ndarray,
    prev_states: np.ndarray,
    *,
    local_cell_indices: np.ndarray,
    n_vars: int,
    gamma: float,
    turbulence_source: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    channels = _field_array(field)
    _, dataset_h, dataset_w = channels.shape
    indices = np.asarray(local_cell_indices, dtype=np.int64).reshape(-1)
    if prev_states.size != indices.size * int(n_vars):
        raise ContractError(
            "distributed ADflow state shape mismatch: "
            f"state_size={prev_states.size}, local_cells={indices.size}, n_vars={n_vars}"
        )
    flat = channels.reshape(channels.shape[0], -1)
    local = np.asarray(flat[:, indices], dtype=np.float64)

    rho = np.maximum(local[0], 1.0e-4)
    u = local[1]
    v = local[2]
    p = np.maximum(local[3], 1.0e-4)
    rho_e = p / (float(gamma) - 1.0) + 0.5 * rho * (u * u + v * v)

    new_states = np.array(prev_states, dtype=np.float64, copy=True)
    new_states[0::n_vars] = rho
    new_states[1::n_vars] = u
    new_states[2::n_vars] = v
    new_states[3::n_vars] = 0.0
    new_states[4::n_vars] = rho_e

    if turbulence_source == "frozen":
        turbulence_mode = "preserved"
    elif turbulence_source == "model":
        if channels.shape[0] < 5:
            raise ContractError("turbulence_source='model' requires a fifth field channel")
        if n_vars <= 5:
            raise ContractError("turbulence_source='model' requires n_vars > 5")
        new_states[5::n_vars] = np.maximum(local[4], 0.0)
        turbulence_mode = "model_channel"
    else:
        raise ContractError(
            "ADflowStateAdapter.turbulence_source must be 'frozen' or 'model'"
        )

    return new_states, {
        "dataset_shape": [int(dataset_h), int(dataset_w)],
        "n_vars": int(n_vars),
        "local_cell_count": int(indices.size),
        "local_state_size": int(new_states.size),
        "state_size": int(new_states.size),
        "turbulence": turbulence_mode,
    }


def _state_vector_to_field(
    states: np.ndarray,
    *,
    dataset_h: int,
    dataset_w: int,
    n_vars: int,
    gamma: float,
    flatten_order: str,
) -> np.ndarray:
    adflow_shape = (int(dataset_w), int(dataset_h))
    rho_af = states[0::n_vars].reshape(adflow_shape, order=flatten_order)
    u_af = states[1::n_vars].reshape(adflow_shape, order=flatten_order)
    v_af = states[2::n_vars].reshape(adflow_shape, order=flatten_order)
    rho_e_af = states[4::n_vars].reshape(adflow_shape, order=flatten_order)
    kinetic_af = 0.5 * rho_af * (u_af * u_af + v_af * v_af)
    p_af = (float(gamma) - 1.0) * (rho_e_af - kinetic_af)

    field = np.zeros((5, int(dataset_h), int(dataset_w)), dtype=np.float64)
    field[0] = rho_af.T
    field[1] = u_af.T
    field[2] = v_af.T
    field[3] = p_af.T
    if n_vars > 5:
        field[4] = states[5::n_vars].reshape(adflow_shape, order=flatten_order).T
    return field


def _local_state_vector_to_field(
    states: np.ndarray,
    *,
    n_vars: int,
    gamma: float,
) -> np.ndarray:
    rho = states[0::n_vars]
    u = states[1::n_vars]
    v = states[2::n_vars]
    rho_e = states[4::n_vars]
    kinetic = 0.5 * rho * (u * u + v * v)
    p = (float(gamma) - 1.0) * (rho_e - kinetic)
    field = np.zeros((5, rho.size), dtype=np.float64)
    field[0] = rho
    field[1] = u
    field[2] = v
    field[3] = p
    if n_vars > 5:
        field[4] = states[5::n_vars]
    return field


def _gather_global_field(
    *,
    local_field: np.ndarray,
    local_cell_indices: np.ndarray,
    dataset_h: int,
    dataset_w: int,
    comm: Any,
) -> np.ndarray:
    gather = getattr(comm, "allgather", None)
    if not callable(gather):
        raise ContractError("distributed ADflow field extraction requires MPI allgather")
    _trace(comm, "before_gather_indices", local_cells=int(np.asarray(local_cell_indices).size))
    gathered_indices = gather(np.asarray(local_cell_indices, dtype=np.int64))
    _trace(comm, "after_gather_indices", ranks=len(gathered_indices))
    _trace(comm, "before_gather_field", local_values=int(np.asarray(local_field).size))
    gathered_fields = gather(np.asarray(local_field, dtype=np.float64))
    _trace(comm, "after_gather_field", ranks=len(gathered_fields))
    global_flat = np.zeros((local_field.shape[0], int(dataset_h) * int(dataset_w)), dtype=np.float64)
    cover = np.zeros(int(dataset_h) * int(dataset_w), dtype=np.int32)
    for indices, field in zip(gathered_indices, gathered_fields):
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        values = np.asarray(field, dtype=np.float64)
        global_flat[:, idx] = values
        cover[idx] += 1
    if not np.all(cover == 1):
        raise ContractError(
            "distributed ADflow field gather coverage mismatch: "
            f"cover_values={sorted(set(int(v) for v in cover.tolist()))}"
        )
    return global_flat.reshape(local_field.shape[0], int(dataset_h), int(dataset_w))


def _attach_state_info(solver: Any, aero_problem: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    getter = getattr(solver, "_getInfo", None)
    if not callable(getter):
        return metadata
    try:
        info = getter()
    except Exception:
        return metadata
    try:
        aero_problem.adflowData.stateInfo = info
        metadata["state_info_attached"] = True
    except Exception:
        metadata["state_info_attached"] = False
    try:
        aero_problem.adflowData.oldWinf = solver.adflow.flowvarrefstate.winf[0:5].copy()
        aero_problem.adflowData.ank_cfl = solver.getOption("ANKCFL0")
        metadata["restart_continuation_attached"] = True
    except Exception:
        metadata["restart_continuation_attached"] = False
    return metadata


def _adflow_ref_state_flag(solver: Any, name: str) -> bool:
    adflow = getattr(solver, "adflow", None)
    ref_state = getattr(adflow, "flowvarrefstate", None)
    if ref_state is None or not hasattr(ref_state, name):
        raise ContractError(f"restart-info injection requires solver.adflow.flowvarrefstate.{name}")
    return bool(getattr(ref_state, name))


def _restart_info_hooks(solver: Any) -> tuple[Any, Any, Any]:
    getter = getattr(solver, "_getInfo", None)
    setter = getattr(solver, "_setInfo", None)
    initializer = getattr(getattr(solver, "adflow", None), "initializeflow", None)
    rebuilder = getattr(initializer, "rebuildrestartderivedstateaftersetinfo", None)
    if not callable(getter):
        raise ContractError("restart-info injection requires solver._getInfo()")
    if not callable(setter):
        raise ContractError("restart-info injection requires solver._setInfo(...)")
    if not callable(rebuilder):
        raise ContractError(
            "restart-info injection requires "
            "solver.adflow.initializeflow.rebuildrestartderivedstateaftersetinfo()"
        )
    return getter, setter, rebuilder


@dataclass(frozen=True)
class ADflowStateAdapter:
    """Stateless instance adapter for one ADflow solver state layout."""

    gamma: float = 1.4
    turbulence_source: str = "model"
    flatten_order: str = "F"
    verify_injection: bool = False
    injection_strategy: str = "restart_info"

    def __post_init__(self) -> None:
        if self.turbulence_source not in {"frozen", "model"}:
            raise ContractError("turbulence_source must be 'frozen' or 'model'")
        if self.flatten_order not in {"F", "C"}:
            raise ContractError("flatten_order must be 'F' or 'C'")
        if float(self.gamma) <= 1.0:
            raise ContractError("gamma must be greater than 1")
        strategy = str(self.injection_strategy).strip().lower()
        if strategy not in {"states", "restart_info"}:
            raise ContractError("injection_strategy must be 'states' or 'restart_info'")
        object.__setattr__(self, "injection_strategy", strategy)

    def _inject_restart_info(
        self,
        *,
        solver: Any,
        case: ResumeCase,
        aero_problem: Any,
        field: Any,
        comm: Any,
    ) -> dict[str, Any]:
        if case.geometry.coords_vertex is None:
            raise ContractError("restart-info injection requires GeometryContext.coords_vertex")
        prev_states = _state_vector(solver, comm)
        channels = _field_array(field)
        _, dataset_h, dataset_w = channels.shape
        if _comm_size(comm) > 1 and prev_states.size % (dataset_h * dataset_w) != 0:
            n_vars = _infer_n_vars_distributed(prev_states.size, dataset_h, dataset_w, comm)
        else:
            n_vars = _infer_n_vars(prev_states.size, dataset_h, dataset_w)
        local_cell_count = int(prev_states.size // int(n_vars))
        local_indices, cell_mapping = _local_cell_indices(
            solver,
            case,
            dataset_h=dataset_h,
            dataset_w=dataset_w,
            local_cell_count=local_cell_count,
            comm=comm,
        )
        getter, setter, rebuilder = _restart_info_hooks(solver)
        _trace(
            comm,
            "before_get_info",
            local_cells=int(local_cell_count),
            n_vars=int(n_vars),
            cell_mapping=cell_mapping,
        )
        info_template = np.asarray(getter(), dtype=np.float64).reshape(-1)
        _trace(comm, "after_get_info", info_size=int(info_template.size))
        info_payload = build_restart_aligned_local_info(
            info_template=info_template,
            field_phys=channels,
            flow_conditions=case.solver_context.flow_conditions,
            local_cell_indices=local_indices,
            dataset_h=int(dataset_h),
            dataset_w=int(dataset_w),
            n_vars=int(n_vars),
            has_viscous=_adflow_ref_state_flag(solver, "viscous"),
            has_eddy=_adflow_ref_state_flag(solver, "eddymodel"),
            coords_vertex=np.asarray(case.geometry.coords_vertex, dtype=np.float64),
            gamma=float(self.gamma),
        )
        _trace(comm, "before_set_info", info_size=int(info_payload.size))
        setter(info_payload)
        _trace(comm, "after_set_info", info_size=int(info_payload.size))
        rebuilder()
        _trace(comm, "after_rebuild_restart_derived_state")
        metadata: dict[str, Any] = {
            "strategy": "restart_info",
            "dataset_shape": [int(dataset_h), int(dataset_w)],
            "n_vars": int(n_vars),
            "local_cell_count": int(local_cell_count),
            "local_state_size": int(prev_states.size),
            "info_size": int(info_payload.size),
            "distribution": "rank_local_restart_info"
            if _comm_size(comm) > 1
            else "global_restart_info",
            "cell_mapping": cell_mapping,
            "rank": _comm_rank(comm),
            "mpi_size": _comm_size(comm),
            "turbulence": "model_channel",
        }
        if self.verify_injection:
            after_info = np.asarray(getter(), dtype=np.float64).reshape(-1)
            rms = float(np.sqrt(np.mean((after_info - info_payload) ** 2)))
            metadata["verification_info_rms"] = rms
            if rms > 1.0e-6:
                raise ContractError(f"ADflow restart-info injection verification failed: rms={rms}")
        metadata.update(_attach_state_info(solver, aero_problem))
        _trace(comm, "after_attach_state_info")
        _comm_barrier(comm, "inject")
        metadata["adapter"] = type(self).__name__
        metadata["case_id"] = case.case_id
        return metadata

    def inject(
        self,
        *,
        solver: Any,
        case: ResumeCase,
        aero_problem: Any,
        field: Any,
        comm: Any = None,
    ) -> dict[str, Any]:
        """Inject a canonical prediction field into an ADflow solver instance."""

        comm = comm if comm is not None else _default_mpi_comm()
        if self.injection_strategy == "restart_info":
            return self._inject_restart_info(
                solver=solver,
                case=case,
                aero_problem=aero_problem,
                field=field,
                comm=comm,
            )

        prev_states = _state_vector(solver, comm)
        channels = _field_array(field)
        _, dataset_h, dataset_w = channels.shape
        if _comm_size(comm) > 1 and prev_states.size % (dataset_h * dataset_w) != 0:
            n_vars = _infer_n_vars_distributed(prev_states.size, dataset_h, dataset_w, comm)
            local_cell_count = prev_states.size // int(n_vars)
            _trace(
                comm,
                "before_local_indices",
                dataset_h=int(dataset_h),
                dataset_w=int(dataset_w),
                local_cells=int(local_cell_count),
                n_vars=int(n_vars),
            )
            local_indices, cell_mapping = _local_cell_indices(
                solver,
                case,
                dataset_h=dataset_h,
                dataset_w=dataset_w,
                local_cell_count=local_cell_count,
                comm=comm,
            )
            _trace(
                comm,
                "after_local_indices",
                local_cells=int(local_indices.size),
                cell_mapping=cell_mapping,
            )
            next_states, metadata = _primitive_to_local_state_vector(
                channels,
                prev_states,
                local_cell_indices=local_indices,
                n_vars=n_vars,
                gamma=self.gamma,
                turbulence_source=self.turbulence_source,
            )
            metadata["distribution"] = "rank_local"
            metadata["cell_mapping"] = cell_mapping
            metadata["rank"] = _comm_rank(comm)
            metadata["mpi_size"] = _comm_size(comm)
        else:
            next_states, metadata = _primitive_to_state_vector(
                channels,
                prev_states,
                gamma=self.gamma,
                turbulence_source=self.turbulence_source,
                flatten_order=self.flatten_order,
            )
            metadata["distribution"] = "global"
        _set_state_vector(solver, next_states, comm)
        if self.verify_injection:
            after = _state_vector(solver, comm)
            rms = float(np.sqrt(np.mean((after - next_states) ** 2)))
            metadata["verification_rms"] = rms
            if rms > 1.0e-6:
                raise ContractError(f"ADflow state injection verification failed: rms={rms}")
        metadata.update(_attach_state_info(solver, aero_problem))
        _trace(comm, "after_attach_state_info")
        _comm_barrier(comm, "inject")
        metadata["adapter"] = type(self).__name__
        metadata["case_id"] = case.case_id
        metadata["strategy"] = "states"
        return metadata

    def extract(
        self,
        *,
        solver: Any,
        case: ResumeCase,
        aero_problem: Any,
        comm: Any = None,
    ) -> np.ndarray:
        """Extract the current ADflow state as a canonical channel-first field."""

        del aero_problem
        comm = comm if comm is not None else _default_mpi_comm()
        reference_field = _field_array(case.prediction.field)
        _, dataset_h, dataset_w = reference_field.shape
        states = _state_vector(solver, comm)
        if _comm_size(comm) > 1 and states.size % (dataset_h * dataset_w) != 0:
            n_vars = _infer_n_vars_distributed(states.size, dataset_h, dataset_w, comm)
            local_cell_count = states.size // int(n_vars)
            _trace(
                comm,
                "before_extract_local_indices",
                dataset_h=int(dataset_h),
                dataset_w=int(dataset_w),
                local_cells=int(local_cell_count),
                n_vars=int(n_vars),
            )
            local_indices, cell_mapping = _local_cell_indices(
                solver,
                case,
                dataset_h=dataset_h,
                dataset_w=dataset_w,
                local_cell_count=local_cell_count,
                comm=comm,
            )
            _trace(
                comm,
                "after_extract_local_indices",
                local_cells=int(local_indices.size),
                cell_mapping=cell_mapping,
            )
            local_field = _local_state_vector_to_field(
                states,
                n_vars=n_vars,
                gamma=self.gamma,
            )
            return _gather_global_field(
                local_field=local_field,
                local_cell_indices=local_indices,
                dataset_h=dataset_h,
                dataset_w=dataset_w,
                comm=comm,
            )
        n_vars = _infer_n_vars(states.size, dataset_h, dataset_w)
        return _state_vector_to_field(
            states,
            dataset_h=dataset_h,
            dataset_w=dataset_w,
            n_vars=n_vars,
            gamma=self.gamma,
            flatten_order=self.flatten_order,
        )
