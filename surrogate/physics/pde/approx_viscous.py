"""ADFLOW-style approximate viscous flux for residual-operator alignment."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


_SSUTH = 110.55 / 300.0
_PR_LAMINAR = 0.72
_PR_TURBULENT = 0.9
_TWO_THIRD = 2.0 / 3.0


def _ensure_batch_field(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if x.ndim == 2:
        return x.unsqueeze(0)
    return x


def _ensure_batch_halo(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if x.ndim == 2:
        return x.unsqueeze(0)
    return x


def _ensure_batch_line(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if x.ndim == 1:
        return x.unsqueeze(0)
    return x


def _match_field_shape(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if x.ndim == reference.ndim:
        return x.to(device=reference.device, dtype=reference.dtype)
    if x.ndim == 0:
        return x.to(device=reference.device, dtype=reference.dtype).view(*([1] * reference.ndim))
    if x.ndim == 1 and reference.ndim == 2 and x.shape[0] == reference.shape[0]:
        return x.to(device=reference.device, dtype=reference.dtype).view(-1, 1)
    if x.ndim == 1 and reference.ndim == 3 and x.shape[0] == reference.shape[0]:
        return x.to(device=reference.device, dtype=reference.dtype).view(-1, 1, 1)
    return x.to(device=reference.device, dtype=reference.dtype)


def _sutherland_mu(
    rho: torch.Tensor,
    p: torch.Tensor,
    mu_inf: torch.Tensor | float,
) -> torch.Tensor:
    rho_safe = torch.clamp(rho, min=1e-12)
    t_ratio = torch.clamp(p / rho_safe, min=1e-12)
    mu_inf_t = torch.as_tensor(mu_inf, device=rho.device, dtype=rho.dtype)
    mu_inf_t = _match_field_shape(mu_inf_t, rho)
    return mu_inf_t * torch.pow(t_ratio, 1.5) * (1.0 + _SSUTH) / (t_ratio + _SSUTH)


def _sa_mu_turb(
    rho: torch.Tensor,
    nu_tilde: torch.Tensor,
    mu_lam: torch.Tensor,
) -> torch.Tensor:
    chi = (rho * nu_tilde) / (mu_lam + 1e-30)
    chi3 = chi ** 3
    fv1 = chi3 / (chi3 + 7.1 ** 3)
    return fv1 * rho * nu_tilde


def _extract_halo_channel(halo: Optional[torch.Tensor], index: int) -> Optional[torch.Tensor]:
    halo_b = _ensure_batch_halo(halo)
    if halo_b is None:
        return None
    if halo_b.shape[1] <= index:
        return None
    return halo_b[:, index, :]


def _pair_xi(field: torch.Tensor, periodic: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    if periodic:
        return field, torch.roll(field, shifts=-1, dims=-1)
    return field[..., :, :-1], field[..., :, 1:]


def _pair_eta(
    field: torch.Tensor,
    *,
    halo_bottom: Optional[torch.Tensor],
    halo_top: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    bottom = halo_bottom if halo_bottom is not None else field[..., 0, :]
    top = halo_top if halo_top is not None else field[..., -1, :]
    lower = torch.cat([bottom.unsqueeze(-2), field], dim=-2)
    upper = torch.cat([field, top.unsqueeze(-2)], dim=-2)
    return lower, upper


def _prepare_mu_top_halo(
    mu_eff: dict[str, torch.Tensor],
    halo_farfield: Optional[torch.Tensor],
    rho: torch.Tensor,
) -> Optional[torch.Tensor]:
    mu_inf = mu_eff.get("mu_inf", None)
    if mu_inf is None or halo_farfield is None:
        return None

    halo_rho = _extract_halo_channel(halo_farfield, 0)
    halo_p = _extract_halo_channel(halo_farfield, 3)
    if halo_rho is None or halo_p is None:
        return None

    halo_rho = halo_rho.to(device=rho.device, dtype=rho.dtype)
    halo_p = halo_p.to(device=rho.device, dtype=rho.dtype)
    return _sutherland_mu(halo_rho, halo_p, mu_inf)


def _prepare_mu_wall_halo(
    mu_eff: dict[str, torch.Tensor],
    halo_wall: Optional[torch.Tensor],
    rho: torch.Tensor,
    fallback_mu_lam: torch.Tensor,
) -> torch.Tensor:
    mu_inf = mu_eff.get("mu_inf", None)
    if mu_inf is None or halo_wall is None:
        return fallback_mu_lam

    halo_rho = _extract_halo_channel(halo_wall, 0)
    halo_p = _extract_halo_channel(halo_wall, 3)
    if halo_rho is None or halo_p is None:
        return fallback_mu_lam

    halo_rho = halo_rho.to(device=rho.device, dtype=rho.dtype)
    halo_p = halo_p.to(device=rho.device, dtype=rho.dtype)
    return _sutherland_mu(halo_rho, halo_p, mu_inf)


def viscous_flux_approx(
    *,
    rho: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    rhoE: Optional[torch.Tensor],
    mu_eff: dict[str, torch.Tensor],
    face_geom: dict[str, torch.Tensor],
    direction: str,
    gamma: float,
    periodic_xi: bool = True,
    rfil: float = 1.0,
    halo_wall: Optional[torch.Tensor] = None,
    halo_farfield: Optional[torch.Tensor] = None,
    halo_nu_tilde_farfield: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Approximate viscous operator matching ADFLOW `viscousFluxApprox`."""

    rho_b = _ensure_batch_field(rho)
    u_b = _ensure_batch_field(u)
    v_b = _ensure_batch_field(v)
    p_b = _ensure_batch_field(p)
    rhoE_b = _ensure_batch_field(rhoE)
    squeeze_output = rho.ndim == 2

    mu_lam = _ensure_batch_field(mu_eff["mu_lam"]).to(device=rho_b.device, dtype=rho_b.dtype)
    mu_turb = _ensure_batch_field(mu_eff["mu_turb"]).to(device=rho_b.device, dtype=rho_b.dtype)
    aa = gamma * p_b / torch.clamp(rho_b, min=1e-12)

    if direction == "xi":
        A_x = face_geom["A_x_xi"]
        A_y = face_geom["A_y_xi"]
        ssx = face_geom["ssx_xi"]
        ssy = face_geom["ssy_xi"]
        inv_d = face_geom["inv_d_xi"]
        if A_x.ndim == 2:
            A_x = A_x.unsqueeze(0)
            A_y = A_y.unsqueeze(0)
            ssx = ssx.unsqueeze(0)
            ssy = ssy.unsqueeze(0)
            inv_d = inv_d.unsqueeze(0)

        u_l, u_r = _pair_xi(u_b, periodic_xi)
        v_l, v_r = _pair_xi(v_b, periodic_xi)
        aa_l, aa_r = _pair_xi(aa, periodic_xi)
        mu_l_l, mu_l_r = _pair_xi(mu_lam, periodic_xi)
        mu_t_l, mu_t_r = _pair_xi(mu_turb, periodic_xi)
        por = (0.5 * float(rfil)) * torch.ones_like(A_x)
    elif direction == "eta":
        A_x = face_geom["A_x_eta"]
        A_y = face_geom["A_y_eta"]
        ssx = face_geom["ssx_eta"]
        ssy = face_geom["ssy_eta"]
        inv_d = face_geom["inv_d_eta"]
        if A_x.ndim == 2:
            A_x = A_x.unsqueeze(0)
            A_y = A_y.unsqueeze(0)
            ssx = ssx.unsqueeze(0)
            ssy = ssy.unsqueeze(0)
            inv_d = inv_d.unsqueeze(0)

        halo_wall_u = _extract_halo_channel(halo_wall, 1)
        halo_wall_v = _extract_halo_channel(halo_wall, 2)
        if halo_wall_u is None:
            halo_wall_u = -u_b[:, 0, :]
        if halo_wall_v is None:
            halo_wall_v = -v_b[:, 0, :]
        halo_top_u = _extract_halo_channel(halo_farfield, 1)
        halo_top_v = _extract_halo_channel(halo_farfield, 2)

        u_l, u_r = _pair_eta(u_b, halo_bottom=halo_wall_u, halo_top=halo_top_u)
        v_l, v_r = _pair_eta(v_b, halo_bottom=halo_wall_v, halo_top=halo_top_v)

        halo_wall_aa = _extract_halo_channel(halo_wall, 3)
        halo_wall_rho = _extract_halo_channel(halo_wall, 0)
        if halo_wall_aa is not None and halo_wall_rho is not None:
            aa_wall = gamma * halo_wall_aa.to(device=rho_b.device, dtype=rho_b.dtype) / torch.clamp(
                halo_wall_rho.to(device=rho_b.device, dtype=rho_b.dtype), min=1e-12
            )
        else:
            aa_wall = aa[:, 0, :]
        halo_top_aa = _extract_halo_channel(halo_farfield, 3)
        halo_top_rho = _extract_halo_channel(halo_farfield, 0)
        if halo_top_aa is not None and halo_top_rho is not None:
            aa_top = gamma * halo_top_aa.to(device=rho_b.device, dtype=rho_b.dtype) / torch.clamp(
                halo_top_rho.to(device=rho_b.device, dtype=rho_b.dtype), min=1e-12
            )
        else:
            aa_top = aa[:, -1, :]
        aa_l, aa_r = _pair_eta(aa, halo_bottom=aa_wall, halo_top=aa_top)

        mu_lam_wall = _prepare_mu_wall_halo(
            mu_eff,
            halo_wall,
            rho_b,
            fallback_mu_lam=mu_lam[:, 0, :],
        )
        mu_turb_wall = torch.zeros_like(mu_turb[:, 0, :])
        # ADFLOW bcFarfield copies both laminar and eddy viscosity from the donor
        # cell into the halo. Do not recompute mu_lam from the farfield state here.
        mu_lam_top = mu_lam[:, -1, :]
        # ADFLOW bcEddyNoWall copies halo eddy viscosity from the interior cell.
        mu_turb_top = mu_turb[:, -1, :]
        mu_l_l, mu_l_r = _pair_eta(mu_lam, halo_bottom=mu_lam_wall, halo_top=mu_lam_top)
        mu_t_l, mu_t_r = _pair_eta(mu_turb, halo_bottom=mu_turb_wall, halo_top=mu_turb_top)

        por = (0.5 * float(rfil)) * torch.ones_like(A_x)
    else:
        raise ValueError(f"Invalid direction: {direction}")

    du = u_r - u_l
    dv = v_r - v_l
    daa = aa_r - aa_l

    grad_x = ssx * inv_d
    grad_y = ssy * inv_d

    u_x = du * grad_x
    u_y = du * grad_y
    v_x = dv * grad_x
    v_y = dv * grad_y

    mul = por * (mu_l_l + mu_l_r)
    mue = por * (mu_t_l + mu_t_r)
    mut = mul + mue

    gm1 = gamma - 1.0
    heat_coef = mul * (1.0 / (_PR_LAMINAR * gm1)) + mue * (1.0 / (_PR_TURBULENT * gm1))

    frac_div = _TWO_THIRD * (u_x + v_y)
    tau_xx = mut * (2.0 * u_x - frac_div)
    tau_yy = mut * (2.0 * v_y - frac_div)
    tau_xy = mut * (u_y + v_x)

    q_x = heat_coef * (-daa * grad_x)
    q_y = heat_coef * (-daa * grad_y)

    ubar = 0.5 * (u_l + u_r)
    vbar = 0.5 * (v_l + v_r)

    Vmx = tau_xx * A_x + tau_xy * A_y
    Vmy = tau_xy * A_x + tau_yy * A_y

    VE = None
    if rhoE_b is not None:
        VE = (
            (ubar * tau_xx + vbar * tau_xy) * A_x
            + (ubar * tau_xy + vbar * tau_yy) * A_y
            - q_x * A_x
            - q_y * A_y
        )

    import os
    if os.environ.get("SURROGATE_DEBUG_APPROX_VISCOUS", "") == "1" and direction == "eta":
        import numpy as np

        def _first_sample(x: torch.Tensor | None) -> np.ndarray | None:
            if x is None:
                return None
            xx = x.detach().cpu()
            if xx.ndim >= 3:
                xx = xx[0]
            return xx.numpy()

        debug_data = {
            "u_l": _first_sample(u_l),
            "u_r": _first_sample(u_r),
            "v_l": _first_sample(v_l),
            "v_r": _first_sample(v_r),
            "aa_l": _first_sample(aa_l),
            "aa_r": _first_sample(aa_r),
            "ssx": _first_sample(ssx),
            "ssy": _first_sample(ssy),
            "inv_d": _first_sample(inv_d),
            "grad_x": _first_sample(grad_x),
            "grad_y": _first_sample(grad_y),
            "A_x": _first_sample(A_x),
            "A_y": _first_sample(A_y),
            "por": _first_sample(por),
            "mu_l_l": _first_sample(mu_l_l),
            "mu_l_r": _first_sample(mu_l_r),
            "mu_t_l": _first_sample(mu_t_l),
            "mu_t_r": _first_sample(mu_t_r),
            "du": _first_sample(du),
            "dv": _first_sample(dv),
            "daa": _first_sample(daa),
            "u_x": _first_sample(u_x),
            "u_y": _first_sample(u_y),
            "v_x": _first_sample(v_x),
            "v_y": _first_sample(v_y),
            "mul": _first_sample(mul),
            "mue": _first_sample(mue),
            "mut": _first_sample(mut),
            "heat_coef": _first_sample(heat_coef),
            "tau_xx": _first_sample(tau_xx),
            "tau_xy": _first_sample(tau_xy),
            "tau_yy": _first_sample(tau_yy),
            "q_x": _first_sample(q_x),
            "q_y": _first_sample(q_y),
            "ubar": _first_sample(ubar),
            "vbar": _first_sample(vbar),
            "Vmx": _first_sample(Vmx),
            "Vmy": _first_sample(Vmy),
        }
        if VE is not None:
            debug_data["VE"] = _first_sample(VE)
        np.savez("pytorch_approx_viscous_debug.npz", **debug_data)

    if squeeze_output:
        Vmx = Vmx.squeeze(0)
        Vmy = Vmy.squeeze(0)
        if VE is not None:
            VE = VE.squeeze(0)

    return Vmx, Vmy, VE
