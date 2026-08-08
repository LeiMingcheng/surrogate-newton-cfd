"""Restart-aligned ADflow info injection helpers.

This module is solver-side only.  It rebuilds the rank-local payload accepted by
ADflow ``_setInfo`` from a canonical 5-channel NK_resume field.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from ..exceptions import ContractError


def _build_primitive6_core(field_phys: np.ndarray, *, gamma: float) -> np.ndarray:
    field = np.asarray(field_phys, dtype=np.float64)
    if field.ndim != 3 or field.shape[0] < 5:
        raise ContractError(
            "restart-info injection requires a field shaped like "
            "(5,H,W) or larger with [rho,u,v,p,nuTilde]"
        )
    rho = np.maximum(field[0], 1.0e-4)
    u = field[1]
    v = field[2]
    p = np.maximum(field[3], 1.0e-4)
    nu_tilde = field[4]
    rho_e = p / (float(gamma) - 1.0) + 0.5 * rho * (u * u + v * v)
    return np.stack([rho, u, v, p, rho_e, nu_tilde], axis=0).astype(np.float64, copy=False)


def _cyclic_contiguous_order(row_i: np.ndarray, global_width: int) -> np.ndarray:
    values = np.asarray(row_i, dtype=np.int64).reshape(-1)
    if values.size == 0:
        raise ContractError("restart-info injection received an empty xi row")
    sorted_i = np.sort(values)
    diffs = np.diff(np.concatenate([sorted_i, sorted_i[:1] + int(global_width)]))
    start = int((np.argmax(diffs) + 1) % sorted_i.size)
    ordered = np.concatenate([sorted_i[start:], sorted_i[:start]])
    expected = (int(ordered[0]) + np.arange(ordered.size, dtype=np.int64)) % int(global_width)
    if not np.array_equal(ordered, expected):
        raise ContractError("local MPI partition is not a contiguous xi segment")
    return ordered


def _build_structured_local_core_gids(
    local_cell_indices: np.ndarray,
    *,
    global_height: int,
    global_width: int,
) -> np.ndarray:
    gids = np.asarray(local_cell_indices, dtype=np.int64).reshape(-1)
    if gids.size == 0:
        raise ContractError("local_cell_indices is empty")
    if gids.size % int(global_height) != 0:
        raise ContractError(
            f"local_cell_indices size {gids.size} is not divisible by global_height={global_height}"
        )
    local_width = int(gids.size // int(global_height))
    gk = gids // int(global_width)
    gi = gids % int(global_width)
    lattice = np.empty((int(global_height), local_width), dtype=np.int64)
    for row in range(int(global_height)):
        row_i = gi[gk == row]
        if row_i.size != local_width:
            raise ContractError(
                "local MPI partition does not cover all eta rows uniformly: "
                f"row={row}, count={row_i.size}, expected={local_width}"
            )
        lattice[row, :] = row * int(global_width) + _cyclic_contiguous_order(
            row_i, int(global_width)
        )
    return lattice


def _extend_periodic_xi_indices(
    core_row_gids: np.ndarray,
    *,
    global_width: int,
    halo_cols: int = 2,
) -> np.ndarray:
    core_row = np.asarray(core_row_gids, dtype=np.int64).reshape(-1)
    row_i = core_row % int(global_width)
    left = (int(row_i[0]) - np.arange(int(halo_cols), 0, -1, dtype=np.int64)) % int(global_width)
    right = (int(row_i[-1]) + np.arange(1, int(halo_cols) + 1, dtype=np.int64)) % int(global_width)
    return np.concatenate([left, row_i, right], axis=0)


def _rho_e_from_primitives(
    *,
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    gamma: float,
) -> np.ndarray:
    return p / (float(gamma) - 1.0) + 0.5 * rho * (u * u + v * v)


def _apply_extrapolated_second_halo(first_halo: np.ndarray, donor: np.ndarray, *, gamma: float) -> np.ndarray:
    first = np.asarray(first_halo, dtype=np.float64)
    donor = np.asarray(donor, dtype=np.float64)
    if first.shape != donor.shape or first.ndim != 2 or first.shape[0] != 6:
        raise ContractError("ADflow second halo construction received incompatible arrays")
    second = np.array(first, dtype=np.float64, copy=True)
    second[0] = np.maximum(0.5 * first[0], 2.0 * first[0] - donor[0])
    second[1] = 2.0 * first[1] - donor[1]
    second[2] = 2.0 * first[2] - donor[2]
    second[3] = np.maximum(0.5 * first[3], 2.0 * first[3] - donor[3])
    second[4] = _rho_e_from_primitives(
        rho=second[0],
        u=second[1],
        v=second[2],
        p=second[3],
        gamma=float(gamma),
    )
    second[5] = first[5]
    return second


def _compute_farfield_normals(coords_vertex: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords_vertex, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[0] < 2:
        raise ContractError(f"Expected coords_vertex shape (2,H+1,W+1), got {coords.shape}")
    x_top = coords[0, -1, :]
    y_top = coords[1, -1, :]
    dx = x_top[1:] - x_top[:-1]
    dy = y_top[1:] - y_top[:-1]
    mag = np.sqrt(dx * dx + dy * dy) + 1.0e-30
    return np.stack([dy / mag, -dx / mag], axis=0).astype(np.float64, copy=False)


def _sa_nutilde_inf(
    *,
    nu_lam: float,
    eddy_vis_inf_ratio: float = 0.009,
    cv1: float = 7.1,
    max_iter: int = 50,
    tol: float = 1.0e-10,
) -> float:
    if not np.isfinite(nu_lam) or nu_lam <= 0.0 or eddy_vis_inf_ratio <= 0.0:
        return 0.0
    cv13 = float(cv1) ** 3
    if eddy_vis_inf_ratio < 1.0e-4:
        chi = 0.5
    elif eddy_vis_inf_ratio < 1.0:
        chi = 5.0
    elif eddy_vis_inf_ratio < 10.0:
        chi = 10.0
    else:
        chi = float(eddy_vis_inf_ratio)
    for _ in range(int(max_iter)):
        chi2 = chi * chi
        chi3 = chi * chi2
        chi4 = chi * chi3
        f = chi4 - float(eddy_vis_inf_ratio) * (chi3 + cv13)
        df = 4.0 * chi3 - 3.0 * float(eddy_vis_inf_ratio) * chi2
        if abs(df) < 1.0e-30:
            break
        dchi = f / df
        chi -= dchi
        if abs(dchi / (chi + 1.0e-30)) <= tol:
            break
    return float(nu_lam) * float(chi)


def _apply_farfield_first_halo(
    farfield_donor: np.ndarray,
    *,
    flow_conditions: Tuple[float, float, float],
    coords_vertex: np.ndarray,
    gamma: float,
) -> np.ndarray:
    ma, aoa, reynolds = (float(x) for x in flow_conditions)
    normal = _compute_farfield_normals(coords_vertex)
    fields = np.asarray(farfield_donor, dtype=np.float64)
    if fields.ndim != 2 or fields.shape[0] < 4:
        raise ContractError(f"Expected farfield_donor shape (C,W), got {fields.shape}")
    n_x = normal[0]
    n_y = normal[1]
    gm1 = float(gamma) - 1.0
    ovgm1 = 1.0 / gm1
    aoa_rad = float(aoa) * (math.pi / 180.0)
    rho_inf = 1.0
    p_inf_corr = 1.0
    c0 = math.sqrt(float(gamma) * p_inf_corr / rho_inf)
    s0 = rho_inf ** float(gamma) / p_inf_corr
    u0 = float(ma) * c0 * math.cos(aoa_rad)
    v0 = float(ma) * c0 * math.sin(aoa_rad)
    qn0 = u0 * n_x + v0 * n_y
    vn0 = qn0
    rho_e = fields[0]
    u_e = fields[1]
    v_e = fields[2]
    p_e = fields[3]
    re = 1.0 / rho_e
    qne = u_e * n_x + v_e * n_y
    c_e = np.sqrt(float(gamma) * p_e * re)
    two_ovgm1 = 2.0 * ovgm1
    ac1 = np.where(vn0 > -c0, qne + two_ovgm1 * c_e, qn0 + two_ovgm1 * c0)
    ac2 = np.where(vn0 > c0, qne - two_ovgm1 * c_e, qn0 - two_ovgm1 * c0)
    qnf = 0.5 * (ac1 + ac2)
    cf = 0.25 * (ac1 - ac2) * gm1
    outflow = vn0 > 0.0
    uf_out = u_e + (qnf - qne) * n_x
    vf_out = v_e + (qnf - qne) * n_y
    sf_out = rho_e ** float(gamma) / p_e
    uf_in = u0 + (qnf - qn0) * n_x
    vf_in = v0 + (qnf - qn0) * n_y
    uf = np.where(outflow, uf_out, uf_in)
    vf = np.where(outflow, vf_out, vf_in)
    sf = np.where(outflow, sf_out, s0)
    cc = np.maximum((cf * cf) / float(gamma), 1.0e-30)
    rho_halo = np.maximum((sf * cc) ** ovgm1, 1.0e-30)
    p_halo = rho_halo * cc
    halo = np.array(fields, dtype=np.float64, copy=True)
    halo[0] = rho_halo
    halo[1] = uf
    halo[2] = vf
    halo[3] = p_halo
    if halo.shape[0] >= 5:
        halo[4] = _rho_e_from_primitives(
            rho=rho_halo,
            u=uf,
            v=vf,
            p=p_halo,
            gamma=float(gamma),
        )
    if halo.shape[0] >= 6:
        nu_lam = (float(ma) * math.sqrt(float(gamma))) / (float(reynolds) + 1.0e-30)
        halo[5] = np.where(outflow, fields[5], _sa_nutilde_inf(nu_lam=nu_lam))
    return halo.astype(np.float64, copy=False)


def _build_haloed_primitive6(
    primitive6_core: np.ndarray,
    *,
    flow_conditions: Optional[Tuple[float, float, float]],
    coords_vertex: Optional[np.ndarray],
    gamma: float,
) -> np.ndarray:
    core = np.asarray(primitive6_core, dtype=np.float64)
    if core.ndim != 3 or core.shape[0] != 6:
        raise ContractError(f"Expected primitive6_core shape (6,H,W), got {core.shape}")
    _, height, width = core.shape
    wall = core[:, 0, :]
    wall_first = np.array(wall, dtype=np.float64, copy=True)
    wall_first[1] = -wall_first[1]
    wall_first[2] = -wall_first[2]
    wall_first[4] = _rho_e_from_primitives(
        rho=wall_first[0],
        u=wall_first[1],
        v=wall_first[2],
        p=wall_first[3],
        gamma=float(gamma),
    )
    wall_first[5] = -wall[5]
    wall_second = _apply_extrapolated_second_halo(wall_first, core[:, 0, :], gamma=float(gamma))
    wall_second[5] = wall_first[5]
    include_farfield = flow_conditions is not None and coords_vertex is not None
    halo_rows = int(height) + (4 if include_farfield else 2)
    haloed = np.zeros((6, halo_rows, int(width) + 4), dtype=np.float64)
    haloed[:, 2 : int(height) + 2, 2 : int(width) + 2] = core
    haloed[:, 1, 2 : int(width) + 2] = wall_first
    haloed[:, 0, 2 : int(width) + 2] = wall_second
    if include_farfield:
        farfield_first = _apply_farfield_first_halo(
            core[:, -1, :],
            flow_conditions=flow_conditions,
            coords_vertex=np.asarray(coords_vertex, dtype=np.float64),
            gamma=float(gamma),
        )
        farfield_second = _apply_extrapolated_second_halo(
            farfield_first,
            core[:, -1, :],
            gamma=float(gamma),
        )
        farfield_second[5] = farfield_first[5]
        haloed[:, int(height) + 2, 2 : int(width) + 2] = farfield_first
        haloed[:, int(height) + 3, 2 : int(width) + 2] = farfield_second
    haloed[:, :, :2] = haloed[:, :, int(width) : int(width) + 2]
    haloed[:, :, int(width) + 2 : int(width) + 4] = haloed[:, :, 2:4]
    return haloed


def _compute_laminar_and_eddy_viscosity(
    primitive6: np.ndarray,
    *,
    flow_conditions: Tuple[float, float, float],
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    rho = np.asarray(primitive6[0], dtype=np.float64)
    p = np.asarray(primitive6[3], dtype=np.float64)
    nu_tilde = np.asarray(primitive6[5], dtype=np.float64)
    ma = float(flow_conditions[0])
    reynolds = float(flow_conditions[2])
    if not np.isfinite(reynolds) or reynolds <= 0.0:
        raise ContractError(f"invalid Reynolds number for restart-info injection: {reynolds}")
    ssuth = 110.55 / 300.0
    mu_inf = (ma * math.sqrt(float(gamma))) / reynolds
    t_ratio = np.clip(p / np.clip(rho, 1.0e-12, None), 1.0e-12, None)
    rlv = mu_inf * np.power(t_ratio, 1.5) * (1.0 + ssuth) / (t_ratio + ssuth)
    chi = (rho * nu_tilde) / np.clip(rlv, 1.0e-30, None)
    chi3 = chi**3
    fv1 = chi3 / (chi3 + 7.1**3)
    rev = fv1 * rho * nu_tilde
    return rlv.astype(np.float64, copy=False), rev.astype(np.float64, copy=False)


def _flow_conditions3(values: Sequence[float]) -> tuple[float, float, float]:
    if len(tuple(values)) < 3:
        raise ContractError("restart-info injection requires mach, alpha, reynolds")
    mach, alpha, reynolds = tuple(float(v) for v in tuple(values)[:3])
    return mach, alpha, reynolds


def build_restart_aligned_local_info(
    *,
    info_template: np.ndarray,
    field_phys: np.ndarray,
    flow_conditions: Sequence[float],
    local_cell_indices: np.ndarray,
    dataset_h: int,
    dataset_w: int,
    n_vars: int,
    has_viscous: bool,
    has_eddy: bool,
    coords_vertex: Optional[np.ndarray] = None,
    gamma: float = 1.4,
) -> np.ndarray:
    """Build one rank's ADflow ``_setInfo`` payload from a clean field."""

    conditions = _flow_conditions3(flow_conditions)
    primitive6_core = _build_primitive6_core(field_phys, gamma=float(gamma))
    haloed = _build_haloed_primitive6(
        primitive6_core,
        flow_conditions=conditions,
        coords_vertex=coords_vertex,
        gamma=float(gamma),
    )
    local_core_gids = _build_structured_local_core_gids(
        local_cell_indices,
        global_height=int(dataset_h),
        global_width=int(dataset_w),
    )
    local_width = int(local_core_gids.shape[1])
    local_cols = np.asarray(
        _extend_periodic_xi_indices(
            local_core_gids[0],
            global_width=int(dataset_w),
            halo_cols=2,
        )
        + 2,
        dtype=np.int64,
    )
    stride = int(n_vars) + 1 + int(bool(has_viscous)) + int(bool(has_eddy))
    expected_size = int(dataset_h + 4) * 5 * int(local_width + 4) * int(stride)
    info = np.asarray(info_template, dtype=np.float64).reshape(-1)
    if info.size != expected_size:
        raise ContractError(
            "restart-info template size mismatch: "
            f"got {info.size}, expected {expected_size} "
            f"(H={dataset_h}, local_width={local_width}, stride={stride})"
        )
    local_window = np.asarray(haloed[:, :, local_cols], dtype=np.float64)
    rlv_local, rev_local = _compute_laminar_and_eddy_viscosity(
        local_window,
        flow_conditions=conditions,
        gamma=float(gamma),
    )
    info_arr = info.reshape(int(dataset_h + 4), 5, int(local_width + 4), int(stride))
    rows = slice(0, int(local_window.shape[1]))
    info_arr[rows, :, :, 0] = local_window[0][:, None, :]
    info_arr[rows, :, :, 1] = local_window[1][:, None, :]
    info_arr[rows, :, :, 2] = local_window[2][:, None, :]
    if int(n_vars) > 3:
        info_arr[rows, :, :, 3] = 0.0
    if int(n_vars) > 4:
        info_arr[rows, :, :, 4] = local_window[4][:, None, :]
    if int(n_vars) > 5:
        info_arr[rows, :, :, 5] = local_window[5][:, None, :]
    info_arr[rows, :, :, int(n_vars)] = local_window[3][:, None, :]
    if has_viscous:
        info_arr[rows, :, :, int(n_vars) + 1] = rlv_local[:, None, :]
    if has_eddy:
        info_arr[rows, :, :, int(n_vars) + 1 + int(bool(has_viscous))] = rev_local[:, None, :]
    if int(local_window.shape[1]) >= int(dataset_h) + 4:
        info_arr[0, 0] = info_arr[1, 1]
        info_arr[0, -1] = info_arr[1, -2]
        info_arr[-1, 0] = info_arr[-2, 1]
        info_arr[-1, -1] = info_arr[-2, -2]
        if has_eddy:
            eddy_index = int(n_vars) + 1 + int(bool(has_viscous))
            info_arr[-1, 0, :, eddy_index] = 0.0
            info_arr[-1, -1, :, eddy_index] = 0.0
    return np.asarray(info_arr.reshape(-1), dtype=np.float64)
