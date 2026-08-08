"""
ADFLOW-like surface force helpers for field-based post-processing.

This module reuses the existing PDE geometry / halo / nodal-gradient /
viscous-flux stack to recover wall viscous forces from a predicted flow
field, instead of relying on the legacy one-cell shear proxy.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Union

import numpy as np
import torch

from .fluxes import viscous_flux
from .geometry import (
    compute_cell_centers_from_vertex,
    compute_cell_volume_adflow,
    compute_face_area_vectors_full,
    compute_halo_cell_volumes,
    compute_halo_eta_face_vectors,
    compute_halo_xi_face_vectors,
    extrapolate_halo_vertex_coords,
)
from .gradient import NodalGradientCalculator
from .halo import apply_farfield_bc, apply_wall_bc
from .sa_utils import compute_sa_eddy_viscosity, compute_sa_nuTilde_inf_tensor
from .thermodynamics import compute_rhoE_from_primitives


TensorLike = Union[np.ndarray, torch.Tensor]


def _to_tensor(
    value: TensorLike,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _build_halo_geometry_cache(
    *,
    coords_vertex: torch.Tensor,
    face_geom: Dict[str, torch.Tensor],
    periodic_xi: bool,
    sign: Union[float, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    coords_vertex_hat = extrapolate_halo_vertex_coords(coords_vertex, direction="eta")
    halo_vol_wall, halo_vol_ff = compute_halo_cell_volumes(coords_vertex_hat, periodic_xi=periodic_xi)
    si_x_halo_wall, si_y_halo_wall, si_x_halo_ff, si_y_halo_ff = compute_halo_xi_face_vectors(
        coords_vertex_hat,
        periodic_xi=periodic_xi,
        sign=sign,
    )
    sj_x_hat, sj_y_hat = compute_halo_eta_face_vectors(
        coords_vertex_hat,
        face_geom_eta=(face_geom["A_x_eta"], face_geom["A_y_eta"]),
        periodic_xi=periodic_xi,
        sign=sign,
    )
    return {
        "halo_vol_wall": halo_vol_wall,
        "halo_vol_ff": halo_vol_ff,
        "si_x_halo_wall": si_x_halo_wall,
        "si_y_halo_wall": si_y_halo_wall,
        "si_x_halo_ff": si_x_halo_ff,
        "si_y_halo_ff": si_y_halo_ff,
        "sj_x_hat": sj_x_hat,
        "sj_y_hat": sj_y_hat,
    }


def prepare_surface_force_geometry(
    coords_vertex: TensorLike,
    *,
    periodic_xi: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> Dict[str, torch.Tensor]:
    """
    Precompute geometry shared by repeated wall-force evaluations on one grid.

    Supports one structured O/C-grid sample with shape `(2, H+1, W+1)` or a
    batch with shape `(B, 2, H+1, W+1)`.
    """
    coords_vertex_t = _to_tensor(coords_vertex, device=device, dtype=dtype)
    if coords_vertex_t.ndim not in (3, 4):
        raise ValueError(
            "prepare_surface_force_geometry expects coords_vertex with shape "
            f"(2, H+1, W+1) or (B, 2, H+1, W+1), "
            f"got {tuple(coords_vertex_t.shape)}"
        )

    xv = coords_vertex_t[..., 0, :, :] if coords_vertex_t.ndim == 4 else coords_vertex_t[0]
    yv = coords_vertex_t[..., 1, :, :] if coords_vertex_t.ndim == 4 else coords_vertex_t[1]
    xc, yc = compute_cell_centers_from_vertex(xv, yv, periodic_xi=periodic_xi)
    coords_center = torch.stack([xc, yc], dim=0 if xc.ndim == 2 else 1)

    vol, sign = compute_cell_volume_adflow(coords_vertex_t, periodic_xi=periodic_xi)
    face_geom = compute_face_area_vectors_full(
        coords_center,
        coords_vertex_t,
        periodic_xi=periodic_xi,
        sign=sign,
    )
    halo_geom = _build_halo_geometry_cache(
        coords_vertex=coords_vertex_t,
        face_geom=face_geom,
        periodic_xi=periodic_xi,
        sign=sign,
    )

    A_x_top = face_geom["A_x_eta"][..., -1, :]
    A_y_top = face_geom["A_y_eta"][..., -1, :]
    A_mag_top = torch.sqrt(A_x_top * A_x_top + A_y_top * A_y_top) + 1e-30
    normal_farfield = torch.stack(
        [A_x_top / A_mag_top, A_y_top / A_mag_top],
        dim=0 if A_x_top.ndim == 1 else 1,
    )

    geometry_nodal = {
        "volumes": vol,
        "si_x": face_geom["A_x_xi"],
        "si_y": face_geom["A_y_xi"],
        "sj_x": face_geom["A_x_eta"],
        "sj_y": face_geom["A_y_eta"],
    }

    return {
        "coords_vertex": coords_vertex_t,
        "coords_center": coords_center,
        "periodic_xi": periodic_xi,
        "vol": vol,
        "sign": sign,
        "face_geom": face_geom,
        "halo_geom": halo_geom,
        "normal_farfield": normal_farfield,
        "geometry_nodal": geometry_nodal,
        "device": coords_vertex_t.device,
        "dtype": coords_vertex_t.dtype,
    }


def _extract_state_channels(
    fields: torch.Tensor,
    *,
    gamma: float,
) -> Dict[str, Optional[torch.Tensor]]:
    if fields.ndim not in (3, 4):
        raise ValueError(
            "Expected fields with shape (C, H, W) or (B, C, H, W), "
            f"got {tuple(fields.shape)}"
        )

    channel_dim = 0 if fields.ndim == 3 else 1
    n_channels = int(fields.shape[channel_dim])
    rho = fields[0] if fields.ndim == 3 else fields[:, 0]
    u = fields[1] if fields.ndim == 3 else fields[:, 1]
    v = fields[2] if fields.ndim == 3 else fields[:, 2]
    p = fields[3] if fields.ndim == 3 else fields[:, 3]
    rhoE = compute_rhoE_from_primitives(rho=rho, u=u, v=v, p=p, gamma=gamma)

    nu_tilde = None
    if n_channels >= 6:
        nu_tilde = fields[5] if fields.ndim == 3 else fields[:, 5]
    elif n_channels >= 5:
        nu_tilde = fields[4] if fields.ndim == 3 else fields[:, 4]

    if nu_tilde is not None:
        nu_tilde = torch.clamp(nu_tilde, min=0.0)

    return {
        "rho": rho,
        "u": u,
        "v": v,
        "p": p,
        "rhoE": rhoE,
        "nu_tilde": nu_tilde,
    }


def _stack_halo_state(
    *,
    rho: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    rhoE: torch.Tensor,
    nu_tilde: Optional[torch.Tensor],
    j_index: int,
) -> torch.Tensor:
    if rho.ndim == 2:
        channels = [
            rho[j_index, :],
            u[j_index, :],
            v[j_index, :],
            p[j_index, :],
            rhoE[j_index, :],
        ]
        if nu_tilde is not None:
            channels.append(nu_tilde[j_index, :])
        channel_dim = 0
    else:
        channels = [
            rho[:, j_index, :],
            u[:, j_index, :],
            v[:, j_index, :],
            p[:, j_index, :],
            rhoE[:, j_index, :],
        ]
        if nu_tilde is not None:
            channels.append(nu_tilde[:, j_index, :])
        channel_dim = 1
    return torch.stack(channels, dim=channel_dim)


def _build_mu_eff(
    *,
    rho: torch.Tensor,
    p: torch.Tensor,
    nu_tilde: Optional[torch.Tensor],
    mach: float,
    reynolds: float,
    gamma: float,
) -> Dict[str, torch.Tensor]:
    sqrt_gamma = math.sqrt(gamma)
    mu_inf = (mach * sqrt_gamma) / (reynolds + 1e-30)
    if rho.ndim == 3 and isinstance(mu_inf, torch.Tensor) and mu_inf.ndim == 1:
        mu_inf = mu_inf[:, None, None]

    ssuth = 110.55 / 300.0
    rho_safe = torch.clamp(rho, min=1e-12)
    t_ratio = torch.clamp(p / rho_safe, min=1e-12)
    suth_factor = torch.pow(t_ratio, 1.5) * (1.0 + ssuth) / (t_ratio + ssuth)
    mu_lam = mu_inf * suth_factor

    if nu_tilde is None:
        mu_turb = torch.zeros_like(mu_lam)
    else:
        mu_turb = compute_sa_eddy_viscosity(rho=rho, nuTilde=nu_tilde, mu_l=mu_lam)

    mu_inf_t = mu_inf if isinstance(mu_inf, torch.Tensor) else torch.tensor(mu_inf, device=rho.device, dtype=rho.dtype)
    mu_inf_t = mu_inf_t.to(device=rho.device, dtype=rho.dtype)
    return {
        "mu_inf": mu_inf_t,
        "mu_lam": mu_lam,
        "mu_turb": mu_turb,
        "mu_eff": mu_lam + mu_turb,
    }


def _resolve_flow_conditions_scalars(
    flow_conditions: Union[TensorLike, Dict[str, float]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(flow_conditions, dict):
        mach = torch.as_tensor(
            flow_conditions.get("Ma", flow_conditions.get("mach", 0.3)),
            device=device,
            dtype=dtype,
        )
        aoa = torch.as_tensor(
            flow_conditions.get("AOA", flow_conditions.get("aoa", flow_conditions.get("AoA", 0.0))),
            device=device,
            dtype=dtype,
        )
        reynolds = torch.as_tensor(
            flow_conditions.get("Re", flow_conditions.get("re", 1e6)),
            device=device,
            dtype=dtype,
        )
        def _scalar_or_vector(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(()) if value.numel() == 1 else value.reshape(-1)

        return _scalar_or_vector(mach), _scalar_or_vector(aoa), _scalar_or_vector(reynolds)

    flow_t = _to_tensor(flow_conditions, device=device, dtype=dtype)
    if flow_t.ndim == 1:
        mach = flow_t[0].reshape(())
        aoa = flow_t[1].reshape(())
        reynolds = flow_t[2].reshape(()) if flow_t.numel() > 2 else torch.tensor(1e6, device=device, dtype=dtype)
        return mach, aoa, reynolds
    if flow_t.ndim == 2:
        mach = flow_t[:, 0]
        aoa = flow_t[:, 1]
        reynolds = flow_t[:, 2] if flow_t.shape[1] > 2 else torch.full_like(mach, 1e6)
        return mach, aoa, reynolds
    raise ValueError(
        "flow_conditions must have shape (3,) or (B, 3), "
        f"got {tuple(flow_t.shape)}"
    )


def compute_viscous_wall_force_adflow_like_torch(
    fields: TensorLike,
    flow_conditions: Union[TensorLike, Dict[str, float]],
    prepared_geometry: Dict[str, torch.Tensor],
    *,
    gamma: float = 1.4,
    seg_mask: Optional[TensorLike] = None,
    eddy_vis_inf_ratio: float = 0.009,
    wall_force_sign: float = 1.0,
    return_details: bool = False,
) -> Dict[str, torch.Tensor]:
    device = prepared_geometry["device"]
    dtype = prepared_geometry["dtype"]
    fields_t = _to_tensor(fields, device=device, dtype=dtype)
    state = _extract_state_channels(fields_t, gamma=gamma)
    mach, aoa, reynolds = _resolve_flow_conditions_scalars(
        flow_conditions,
        device=device,
        dtype=dtype,
    )

    rho = state["rho"]
    u = state["u"]
    v = state["v"]
    p = state["p"]
    rhoE = state["rhoE"]
    nu_tilde = state["nu_tilde"]

    fields_wall = _stack_halo_state(
        rho=rho,
        u=u,
        v=v,
        p=p,
        rhoE=rhoE,
        nu_tilde=nu_tilde,
        j_index=0,
    )
    second_idx = 1 if int(rho.shape[-2]) > 1 else 0
    fields_second = _stack_halo_state(
        rho=rho,
        u=u,
        v=v,
        p=p,
        rhoE=rhoE,
        nu_tilde=nu_tilde,
        j_index=second_idx,
    )
    halo_wall = apply_wall_bc(
        fields_wall=fields_wall,
        fields_second_layer=fields_second,
        slip_velocity=None,
        wall_pressure_treatment="constant_pressure",
        gamma=gamma,
    )

    fields_farfield = _stack_halo_state(
        rho=rho,
        u=u,
        v=v,
        p=p,
        rhoE=rhoE,
        nu_tilde=nu_tilde,
        j_index=-1,
    )
    nu_tilde_inf = None
    if nu_tilde is not None:
        nu_lam_inf = (mach * math.sqrt(gamma)) / (reynolds + 1e-30)
        nu_tilde_inf = compute_sa_nuTilde_inf_tensor(
            nuLam=nu_lam_inf,
            eddyVisInfRatio=float(eddy_vis_inf_ratio),
            cv1=7.1,
        )
    halo_farfield = apply_farfield_bc(
        fields_farfield=fields_farfield,
        normal=prepared_geometry["normal_farfield"],
        Ma=mach,
        AoA=aoa,
        gamma=gamma,
        nuTilde_inf=nu_tilde_inf,
        Re=reynolds,
    )

    bc_common = dict(prepared_geometry["halo_geom"])
    wall_u = halo_wall[1, :] if halo_wall.ndim == 2 else halo_wall[:, 1, :]
    wall_v = halo_wall[2, :] if halo_wall.ndim == 2 else halo_wall[:, 2, :]
    farfield_u = halo_farfield[1, :] if halo_farfield.ndim == 2 else halo_farfield[:, 1, :]
    farfield_v = halo_farfield[2, :] if halo_farfield.ndim == 2 else halo_farfield[:, 2, :]
    bc_u = {
        "halo_eta_bottom": wall_u,
        "halo_eta_top": farfield_u,
        **bc_common,
    }
    bc_v = {
        "halo_eta_bottom": wall_v,
        "halo_eta_top": farfield_v,
        **bc_common,
    }

    gradient_calc_nodal = NodalGradientCalculator(
        periodic_xi=bool(prepared_geometry["periodic_xi"]),
        device=device,
        dtype=dtype,
    )
    du_dx_node, du_dy_node = gradient_calc_nodal.compute_gradient(
        u,
        prepared_geometry["geometry_nodal"],
        bc_u,
    )
    dv_dx_node, dv_dy_node = gradient_calc_nodal.compute_gradient(
        v,
        prepared_geometry["geometry_nodal"],
        bc_v,
    )

    mu_eff = _build_mu_eff(
        rho=rho,
        p=p,
        nu_tilde=nu_tilde,
        mach=mach,
        reynolds=reynolds,
        gamma=gamma,
    )

    zeros = torch.zeros_like(u)
    vfx, vfy = viscous_flux(
        u=u,
        v=v,
        du_dx=zeros,
        du_dy=zeros,
        dv_dx=zeros,
        dv_dy=zeros,
        mu_eff=mu_eff,
        face_geom=prepared_geometry["face_geom"],
        direction="eta",
        normal_correction="adflow",
        Ma=mach,
        AoA=aoa,
        gamma=gamma,
        du_dx_node=du_dx_node,
        du_dy_node=du_dy_node,
        dv_dx_node=dv_dx_node,
        dv_dy_node=dv_dy_node,
        halo_wall=halo_wall,
        halo_farfield=halo_farfield,
        use_nodal_gradients=True,
    )

    wall_vx = vfx[..., 0, :]
    wall_vy = vfy[..., 0, :]
    if seg_mask is None:
        mask_t = torch.ones_like(wall_vx)
    else:
        mask_t = _to_tensor(seg_mask, device=device, dtype=dtype).reshape_as(wall_vx)

    fx_v = wall_force_sign * torch.sum(wall_vx * mask_t, dim=-1)
    fy_v = wall_force_sign * torch.sum(wall_vy * mask_t, dim=-1)

    result: Dict[str, torch.Tensor] = {
        "Fx_v": fx_v,
        "Fy_v": fy_v,
    }
    if return_details:
        result.update(
            {
                "wall_vx": wall_vx,
                "wall_vy": wall_vy,
                "mask": mask_t,
            }
        )
    return result


def compute_viscous_wall_force_adflow_like(
    fields: TensorLike,
    flow_conditions: Union[TensorLike, Dict[str, float]],
    prepared_geometry: Dict[str, torch.Tensor],
    *,
    gamma: float = 1.4,
    seg_mask: Optional[TensorLike] = None,
    eddy_vis_inf_ratio: float = 0.009,
    wall_force_sign: float = 1.0,
    return_details: bool = False,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Recover wall viscous force from field state using the existing PDE stack.

    Returns body-force components in Cartesian axes. For the current face-orientation
    convention in the PDE stack, the wall-face viscous momentum flux already matches
    the body-force sign, so the default is `wall_force_sign=+1`.
    """
    result_t = compute_viscous_wall_force_adflow_like_torch(
        fields=fields,
        flow_conditions=flow_conditions,
        prepared_geometry=prepared_geometry,
        gamma=gamma,
        seg_mask=seg_mask,
        eddy_vis_inf_ratio=eddy_vis_inf_ratio,
        wall_force_sign=wall_force_sign,
        return_details=return_details,
    )
    result: Dict[str, Union[float, np.ndarray]] = {
        "Fx_v": float(result_t["Fx_v"].detach().cpu().item()),
        "Fy_v": float(result_t["Fy_v"].detach().cpu().item()),
    }
    if return_details:
        result.update(
            {
                "wall_vx": result_t["wall_vx"].detach().cpu().numpy(),
                "wall_vy": result_t["wall_vy"].detach().cpu().numpy(),
                "mask": result_t["mask"].detach().cpu().numpy(),
            }
        )
    return result
