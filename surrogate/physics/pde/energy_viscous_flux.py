"""
能量方程粘性通量模块

实现ADFLOW对齐的能量粘性通量计算，包括：
1. 粘性功（viscous work）：速度 × 应力张量
2. 热传导（heat conduction）：-k_heat × q̂ · S，其中 q̂ = -∇(a²)（ADFLOW 的 qx/qy）

参考ADFLOW源码：
- fluxes.F90:3377-3388 (viscousFlux, energy part)
- fluxes.F90:3148 (heatCoef calculation)
- fluxes.F90:3157-3290 (面梯度插值 + 法向修正 + tau计算)

关键点：
- ADFLOW使用 grad(a²) 而不是 grad(T)，数值稳定性更好
- 双Prandtl数：层流Pr_l=0.72，湍流Pr_t=0.9
- **ADFLOW对齐修复**：tau必须在面上计算（使用面梯度+法向修正），而非cell-center tau平均
"""

import torch
from typing import Dict, Tuple, Optional
from .thermodynamics import compute_heat_conduction_coefficient


def compute_energy_viscous_flux(
    rho: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    rhoE: torch.Tensor,
    # 节点梯度（已由momentum viscous flux计算）
    du_dx_node: torch.Tensor,
    du_dy_node: torch.Tensor,
    dv_dx_node: torch.Tensor,
    dv_dy_node: torch.Tensor,
    daa_dx_node: torch.Tensor,
    daa_dy_node: torch.Tensor,  # q̂ = -grad(a²)（ADFLOW qx/qy），不是grad(T)！
    # 应力张量（cell-center，仅在use_face_tau=False时使用）
    tau_xx: torch.Tensor,
    tau_xy: torch.Tensor,
    tau_yy: torch.Tensor,
    # 粘度
    mu_l: torch.Tensor,
    mu_t: torch.Tensor,
    # 面几何
    face_geom: Dict[str, torch.Tensor],
    direction: str = 'xi',
    gamma: float = 1.4,
    Pr_laminar: float = 0.72,
    Pr_turbulent: float = 0.9,
    # ✅ ADFLOW对齐：halo参数（使壁面ubar=0, mu_t=0）
    halo_u_wall: Optional[torch.Tensor] = None,
    halo_v_wall: Optional[torch.Tensor] = None,
    halo_mu_t_wall: Optional[torch.Tensor] = None,
    halo_u_farfield: Optional[torch.Tensor] = None,
    halo_v_farfield: Optional[torch.Tensor] = None,
    halo_mu_t_farfield: Optional[torch.Tensor] = None,
    # ✅ ADFLOW对齐：热通量法向修正参数
    apply_heat_flux_normal_correction: bool = False,
    aa: Optional[torch.Tensor] = None,  # a² = gamma * p / rho
    ssx_sep: Optional[torch.Tensor] = None,  # 分隔法向x分量
    ssy_sep: Optional[torch.Tensor] = None,  # 分隔法向y分量
    inv_d: Optional[torch.Tensor] = None,  # 1/|d| (单元中心距离倒数)
    halo_aa_wall: Optional[torch.Tensor] = None,  # 壁面aa halo
    halo_aa_farfield: Optional[torch.Tensor] = None,  # 远场aa halo
    return_heat_parts: bool = False,
) -> torch.Tensor:
    """
    能量粘性通量（ADFLOW对齐）

    参考：fluxes.F90:3377-3388 (viscousFlux, energy part)

    公式：
        F_E_visc = (u_bar, v_bar) · (tau · n) - q · n

    其中：
        - 第一项：粘性功（速度×应力）
        - 第二项：热传导
        - ADFLOW中热通量相关的 qx/qy 为 q̂ = -∇(a²)（flowUtils.F90: allNodalGradients）
        - 物理热通量向量: q = heatCoef * q̂ = -k_heat * ∇(a²)
        - 能量粘性通量热项: -q·S = -heatCoef * q̂·S
        - k_heat = mu_l/(Pr_l*(gamma-1)) + mu_t/(Pr_t*(gamma-1))

    Args:
        rho, u, v, p, rhoE: 流场变量 (H, W) 或 (batch, H, W)
        du_dx_node, ..., daa_dy_node: 节点梯度 (H+1, W+1) 或 (batch, H+1, W+1)
        tau_xx, tau_xy, tau_yy: 应力张量分量（从momentum flux复用）
        mu_l, mu_t: 层流和湍流粘度
        face_geom: 面几何字典
        direction: 'xi' 或 'eta'
        gamma, Pr_laminar, Pr_turbulent: 热力学参数

    Returns:
        V_E: 能量粘性通量（与单元面相对应）

    Shape:
        - Input fields: (H, W) 或 (batch, H, W)
        - Input gradients: (H+1, W+1) 或 (batch, H+1, W+1)
        - Output: (H, W_faces) 或 (batch, H, W_faces)
            其中 W_faces = W+1 (xi方向) 或 W (eta方向)
    """
    # 提取面法向向量和面积
    if direction == 'xi':
        # ξ方向：(H, W+1)
        A_x = face_geom['A_x_xi']  # 面面积向量x分量
        A_y = face_geom['A_y_xi']  # 面面积向量y分量
    elif direction == 'eta':
        # η方向：(H+1, W)
        A_x = face_geom['A_x_eta']
        A_y = face_geom['A_y_eta']
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'xi' or 'eta'")

    # ========== 部分1：粘性功 u · (tau · n) ==========

    periodic_xi = bool(direction == 'xi' and face_geom.get('periodic_xi', False))
    xi_internal_faces_only = bool(direction == 'xi' and not periodic_xi)

    # ✅ ADFLOW对齐：速度平均到面（使用halo使壁面ubar=0）
    # 参考：fluxes.F90 viscousFlux中的ubar计算
    # 壁面：halo反射速度使 u_face = 0.5*(u + (-u)) = 0
    if direction == 'eta':
        u_face = _average_to_face(u, direction,
                                   halo_bottom=halo_u_wall,
                                   halo_top=halo_u_farfield)
        v_face = _average_to_face(v, direction,
                                   halo_bottom=halo_v_wall,
                                   halo_top=halo_v_farfield)
    else:
        u_face = _average_to_face(u, direction, periodic=periodic_xi)
        v_face = _average_to_face(v, direction, periodic=periodic_xi)
        if xi_internal_faces_only:
            u_face = u_face[..., 1:-1]
            v_face = v_face[..., 1:-1]

    # ========== ADFLOW对齐：面上tau计算（使用面梯度+法向修正）==========
    # 参考：fluxes.F90:3157-3290
    # 1. 节点梯度插值到面
    du_dx_face = _average_nodal_to_face(du_dx_node, direction, periodic=periodic_xi)
    du_dy_face = _average_nodal_to_face(du_dy_node, direction, periodic=periodic_xi)
    dv_dx_face = _average_nodal_to_face(dv_dx_node, direction, periodic=periodic_xi)
    dv_dy_face = _average_nodal_to_face(dv_dy_node, direction, periodic=periodic_xi)
    if xi_internal_faces_only:
        du_dx_face = du_dx_face[..., 1:-1]
        du_dy_face = du_dy_face[..., 1:-1]
        dv_dx_face = dv_dx_face[..., 1:-1]
        dv_dy_face = dv_dy_face[..., 1:-1]

    # 2. 法向修正（η方向和ξ方向都做，与动量粘性通量一致）
    # 参考：fluxes.F90:3227-3250 (η方向), fluxes.F90:4486 (ξ方向)
    if ssx_sep is not None and ssy_sep is not None and inv_d is not None:
        if direction == 'eta':
            # η方向：计算 Delta_u = u[k+1] - u[k]（包含边界处理）
            # 内部面
            Delta_u_interior = u[..., 1:, :] - u[..., :-1, :]
            Delta_v_interior = v[..., 1:, :] - v[..., :-1, :]
            # 壁面j=0：ghost反射，Delta = 2*u[0]
            Delta_u_bot = 2.0 * u[..., 0:1, :]
            Delta_v_bot = 2.0 * v[..., 0:1, :]
            # 远场j=H：外推，Delta ≈ 0
            Delta_u_top = torch.zeros_like(u[..., -1:, :])
            Delta_v_top = torch.zeros_like(v[..., -1:, :])
            # 拼接
            Delta_u = torch.cat([Delta_u_bot, Delta_u_interior, Delta_u_top], dim=-2)
            Delta_v = torch.cat([Delta_v_bot, Delta_v_interior, Delta_v_top], dim=-2)

        elif direction == 'xi':
            if periodic_xi:
                # ✅ ADFLOW对齐：ξ方向法向修正（周期边界）
                # 参考：fluxes.F90:4486
                Delta_u = torch.cat([
                    u[..., :, 1:] - u[..., :, :-1],
                    u[..., :, :1] - u[..., :, -1:]
                ], dim=-1)
                Delta_v = torch.cat([
                    v[..., :, 1:] - v[..., :, :-1],
                    v[..., :, :1] - v[..., :, -1:]
                ], dim=-1)
            else:
                Delta_u = u[..., :, 1:] - u[..., :, :-1]
                Delta_v = v[..., :, 1:] - v[..., :, :-1]

        # 法向修正：g ← g - [g·n - Δφ*inv_d]*n
        # 参考：fluxes.F90:3227-3250
        g_u = torch.stack([du_dx_face, du_dy_face], dim=0)
        g_v = torch.stack([dv_dx_face, dv_dy_face], dim=0)
        n_sep = torch.stack([ssx_sep, ssy_sep], dim=0)

        g_dot_n_u = (g_u * n_sep).sum(dim=0)
        corr_u = g_dot_n_u - Delta_u * inv_d
        du_dx_face = du_dx_face - corr_u * ssx_sep
        du_dy_face = du_dy_face - corr_u * ssy_sep

        g_dot_n_v = (g_v * n_sep).sum(dim=0)
        corr_v = g_dot_n_v - Delta_v * inv_d
        dv_dx_face = dv_dx_face - corr_v * ssx_sep
        dv_dy_face = dv_dy_face - corr_v * ssy_sep

    # 3. 粘度平均到面
    mu_l_face_tau = _average_to_face(mu_l, direction, periodic=periodic_xi)
    if direction == 'eta':
        mu_t_face_tau = _average_to_face(mu_t, direction,
                                          halo_bottom=halo_mu_t_wall,
                                          halo_top=halo_mu_t_farfield)
    else:
        mu_t_face_tau = _average_to_face(mu_t, direction, periodic=periodic_xi)
        if xi_internal_faces_only:
            mu_l_face_tau = mu_l_face_tau[..., 1:-1]
            mu_t_face_tau = mu_t_face_tau[..., 1:-1]
    mu_total_face = mu_l_face_tau + mu_t_face_tau

    # 4. 在面上计算应力张量（参考：fluxes.F90:3282-3290）
    lambda_visc_face = -2.0 / 3.0 * mu_total_face
    div_vel_face = du_dx_face + dv_dy_face
    tau_xx_face = 2.0 * mu_total_face * du_dx_face + lambda_visc_face * div_vel_face
    tau_yy_face = 2.0 * mu_total_face * dv_dy_face + lambda_visc_face * div_vel_face
    tau_xy_face = mu_total_face * (du_dy_face + dv_dx_face)

    # 应力张量·法向量：(tau · n) = (tau_xx*n_x + tau_xy*n_y, tau_xy*n_x + tau_yy*n_y)
    # 注意：ADFLOW中面积向量包含了单位法向量×面积，所以直接使用A_x, A_y
    tau_n_x = tau_xx_face * A_x + tau_xy_face * A_y
    tau_n_y = tau_xy_face * A_x + tau_yy_face * A_y

    # 粘性功：u · (tau · n)
    viscous_work = u_face * tau_n_x + v_face * tau_n_y

    # ========== 部分2：热传导 -k_heat * q̂ · S ==========

    # 热传导系数（复用tau计算时已得到的面粘度）
    k_heat_laminar = mu_l_face_tau / (Pr_laminar * (gamma - 1.0))
    k_heat_turbulent = mu_t_face_tau / (Pr_turbulent * (gamma - 1.0))
    k_heat = compute_heat_conduction_coefficient(
        mu_l_face_tau, mu_t_face_tau, gamma, Pr_laminar, Pr_turbulent
    )

    # q̂ = -∇(a²) 在面上（从节点量平均，xi方向使用周期边界）
    daa_dx_face = _average_nodal_to_face(daa_dx_node, direction, periodic=periodic_xi)
    daa_dy_face = _average_nodal_to_face(daa_dy_node, direction, periodic=periodic_xi)
    if xi_internal_faces_only:
        daa_dx_face = daa_dx_face[..., 1:-1]
        daa_dy_face = daa_dy_face[..., 1:-1]
    daa_dy_face_uncorrected = daa_dy_face

    heat_projection_uncorrected_n = daa_dx_face * A_x + daa_dy_face * A_y
    laminar_heat_factor = 1.0 / (Pr_laminar * (gamma - 1.0))
    heat_flux_uncorrected_laminar_unitcoef_n = (
        -laminar_heat_factor * heat_projection_uncorrected_n
    )
    heat_flux_uncorrected_laminar_unitcoef_x_n = (
        -laminar_heat_factor * daa_dx_face * A_x
    )
    heat_flux_uncorrected_laminar_unitcoef_y_n = (
        -laminar_heat_factor * daa_dy_face * A_y
    )
    heat_flux_uncorrected_laminar_unitcoef_qy_n = (
        -laminar_heat_factor * daa_dy_face
    )
    heat_flux_uncorrected_laminar_n = -k_heat_laminar * heat_projection_uncorrected_n
    heat_flux_uncorrected_turbulent_n = -k_heat_turbulent * heat_projection_uncorrected_n
    heat_flux_uncorrected_n = -k_heat * heat_projection_uncorrected_n

    # ✅ ADFLOW对齐：热通量法向修正
    # 参考：fluxes.F90:3264-3271
    # 公式：corr = g·n + (aa_R - aa_L)*ss （注意正号，与速度梯度不同！）
    # ✅ ADFLOW对齐：热通量法向修正（支持η方向和ξ方向）
    # 参考：fluxes.F90:3264 (η方向), fluxes.F90:4496 (ξ方向)
    g_dot_n_aa = torch.zeros_like(daa_dy_face)
    Delta_aa = torch.zeros_like(daa_dy_face)
    corr_aa = torch.zeros_like(daa_dy_face)
    if apply_heat_flux_normal_correction:
        if ssx_sep is None or ssy_sep is None or inv_d is None or aa is None:
            raise ValueError(
                "Heat flux normal correction requires: ssx_sep, ssy_sep, inv_d, aa"
            )

        if direction == 'eta':
            # 计算Delta_aa = aa_R - aa_L（各层面）
            # 内部面：aa[j+1] - aa[j]
            Delta_aa_interior = aa[..., 1:, :] - aa[..., :-1, :]

            # 壁面j=0：halo_aa_wall - aa[0]
            if halo_aa_wall is not None:
                # halo_aa_wall: (batch, W) or (W,)
                if halo_aa_wall.ndim == aa.ndim - 1:
                    halo_aa_wall_expanded = halo_aa_wall.unsqueeze(-2)
                else:
                    halo_aa_wall_expanded = halo_aa_wall
                Delta_aa_bot = aa[..., 0:1, :] - halo_aa_wall_expanded
            else:
                Delta_aa_bot = torch.zeros_like(aa[..., 0:1, :])

            # 远场j=H：halo_aa_farfield - aa[-1]
            if halo_aa_farfield is not None:
                if halo_aa_farfield.ndim == aa.ndim - 1:
                    halo_aa_farfield_expanded = halo_aa_farfield.unsqueeze(-2)
                else:
                    halo_aa_farfield_expanded = halo_aa_farfield
                Delta_aa_top = halo_aa_farfield_expanded - aa[..., -1:, :]
            else:
                Delta_aa_top = torch.zeros_like(aa[..., -1:, :])

            # 拼接：[j=0, j=1..H-1, j=H]
            Delta_aa = torch.cat([Delta_aa_bot, Delta_aa_interior, Delta_aa_top], dim=-2)

        elif direction == 'xi':
            if periodic_xi:
                # ✅ ADFLOW对齐：ξ方向热通量法向修正（周期边界）
                # 参考：fluxes.F90:4496
                Delta_aa = torch.cat([
                    aa[..., :, 1:] - aa[..., :, :-1],
                    aa[..., :, :1] - aa[..., :, -1:]
                ], dim=-1)
            else:
                Delta_aa = aa[..., :, 1:] - aa[..., :, :-1]

        # 法向修正：g ← g - [g·n + Δaa*inv_d]·n
        # ADFLOW热通量使用+号（与速度梯度的-号不同）
        # 参考：fluxes.F90:3264: corr = q_x*ssx + q_y*ssy + (aa(k+1)-aa(k))*ss
        g_aa = torch.stack([daa_dx_face, daa_dy_face], dim=0)
        n_sep = torch.stack([ssx_sep, ssy_sep], dim=0)

        g_dot_n_aa = (g_aa * n_sep).sum(dim=0)
        # 注意：ADFLOW用 + Delta*ss，我们的inv_d相当于ss
        corr_aa = g_dot_n_aa + Delta_aa * inv_d

        daa_dx_face = daa_dx_face - corr_aa * ssx_sep
        daa_dy_face = daa_dy_face - corr_aa * ssy_sep

    # 热通量项：-k_heat * (q̂ · S)
    # q̂ · S = daa_dx_face * A_x + daa_dy_face * A_y
    heat_projection_n = daa_dx_face * A_x + daa_dy_face * A_y
    heat_flux_laminar_unitcoef_n = -laminar_heat_factor * heat_projection_n
    heat_flux_laminar_unitcoef_x_n = -laminar_heat_factor * daa_dx_face * A_x
    heat_flux_laminar_unitcoef_y_n = -laminar_heat_factor * daa_dy_face * A_y
    heat_flux_laminar_n = -k_heat_laminar * heat_projection_n
    heat_flux_turbulent_n = -k_heat_turbulent * heat_projection_n
    heat_flux_n = -k_heat * heat_projection_n
    heat_flux_correction_n = heat_flux_n - heat_flux_uncorrected_n
    heat_flux_correction_laminar_n = (
        heat_flux_laminar_n - heat_flux_uncorrected_laminar_n
    )
    heat_flux_correction_turbulent_n = (
        heat_flux_turbulent_n - heat_flux_uncorrected_turbulent_n
    )
    heat_flux_correction_laminar_unitcoef_n = (
        heat_flux_laminar_unitcoef_n - heat_flux_uncorrected_laminar_unitcoef_n
    )
    heat_flux_correction_laminar_unitcoef_x_n = (
        heat_flux_laminar_unitcoef_x_n
        - heat_flux_uncorrected_laminar_unitcoef_x_n
    )
    heat_flux_correction_laminar_unitcoef_y_n = (
        heat_flux_laminar_unitcoef_y_n
        - heat_flux_uncorrected_laminar_unitcoef_y_n
    )
    heat_flux_correction_laminar_unitcoef_qy_n = (
        -laminar_heat_factor * (daa_dy_face - daa_dy_face_uncorrected)
    )

    # ========== 总能量粘性通量 ==========

    # F_E_visc = viscous_work + heat_flux_n
    # ADFLOW公式: frhoE = viscous_work + heat_flux
    # 其中 heat_flux = -q·S（已包含负号）
    # 这里 heat_flux_n = -k_heat * (q̂·S) 也已包含负号
    V_E = viscous_work + heat_flux_n

    # DEBUG: 打印能量粘性通量关键值
    import os
    if os.environ.get('SURROGATE_DEBUG_ENERGY_VISC'):
        print(f"\n[DEBUG energy_viscous_flux.py] {direction} direction:")
        print(f"  tau_xx_face: min={tau_xx_face.min():.6e}, max={tau_xx_face.max():.6e}")
        print(f"  tau_xy_face: min={tau_xy_face.min():.6e}, max={tau_xy_face.max():.6e}")
        print(f"  tau_yy_face: min={tau_yy_face.min():.6e}, max={tau_yy_face.max():.6e}")
        print(f"  u_face: min={u_face.min():.6e}, max={u_face.max():.6e}")
        print(f"  v_face: min={v_face.min():.6e}, max={v_face.max():.6e}")
        print(f"  viscous_work: min={viscous_work.min():.6e}, max={viscous_work.max():.6e}, rms={torch.sqrt((viscous_work**2).mean()):.6e}")
        print(f"  k_heat: min={k_heat.min():.6e}, max={k_heat.max():.6e}")
        print(f"  daa_dx_face: min={daa_dx_face.min():.6e}, max={daa_dx_face.max():.6e}")
        print(f"  daa_dy_face: min={daa_dy_face.min():.6e}, max={daa_dy_face.max():.6e}")
        print(f"  heat_flux_n: min={heat_flux_n.min():.6e}, max={heat_flux_n.max():.6e}, rms={torch.sqrt((heat_flux_n**2).mean()):.6e}")
        print(f"  V_E: min={V_E.min():.6e}, max={V_E.max():.6e}, rms={torch.sqrt((V_E**2).mean()):.6e}")
        if direction == 'eta':
            # 壁面层详细输出
            print(f"  [壁面层 j=0] V_E[0, :5]={V_E[0, :5]}")
            print(f"  [壁面层 j=0] viscous_work[0, :5]={viscous_work[0, :5]}")
            print(f"  [壁面层 j=0] heat_flux_n[0, :5]={heat_flux_n[0, :5]}")
            print(f"  [壁面层 j=0] u_face[0, :5]={u_face[0, :5]}")
            print(f"  [壁面层 j=0] mu_total_face[0, :5]={mu_total_face[0, :5]}")

    # 返回总通量和分解（用于调试）
    if return_heat_parts:
        return (
            V_E,
            viscous_work,
            heat_flux_n,
            heat_flux_uncorrected_n,
            heat_flux_correction_n,
            heat_flux_laminar_n,
            heat_flux_turbulent_n,
            heat_flux_uncorrected_laminar_n,
            heat_flux_correction_laminar_n,
            heat_flux_uncorrected_turbulent_n,
            heat_flux_correction_turbulent_n,
            heat_flux_uncorrected_laminar_unitcoef_n,
            heat_flux_correction_laminar_unitcoef_n,
            heat_flux_uncorrected_laminar_unitcoef_x_n,
            heat_flux_uncorrected_laminar_unitcoef_y_n,
            heat_flux_correction_laminar_unitcoef_x_n,
            heat_flux_correction_laminar_unitcoef_y_n,
            heat_flux_uncorrected_laminar_unitcoef_qy_n,
            heat_flux_correction_laminar_unitcoef_qy_n,
            g_dot_n_aa,
            Delta_aa,
            corr_aa,
        )
    return V_E, viscous_work, heat_flux_n


def _average_to_face(
    field: torch.Tensor,
    direction: str,
    halo_bottom: Optional[torch.Tensor] = None,
    halo_top: Optional[torch.Tensor] = None,
    periodic: bool = False
) -> torch.Tensor:
    """
    单元中心值平均到面（支持halo的边界处理）

    ADFLOW对齐：边界面使用 0.5 * (physical + halo) 平均，
    而不是简单复制内部单元。这对壁面速度特别重要：
    - 壁面halo反射速度使 u_wall_face = 0.5 * (u_physical + u_halo) = 0
    - 壁面halo反射后 mu_t = 0

    参考：fluxes.F90 viscousFlux中的ubar计算

    Args:
        field: 单元中心值 (H, W) 或 (batch, H, W)
        direction: 'xi' 或 'eta'
        halo_bottom: 底部/左侧边界halo值 (W,) 或 (batch, W) 用于eta方向
                     (H,) 或 (batch, H) 用于xi方向
        halo_top: 顶部/右侧边界halo值（同上）
        periodic: 是否使用周期边界条件（仅用于xi方向）

    Returns:
        field_face: 面上的值
            - xi方向（非周期）: (H, W+1) - 左右两个单元平均
            - xi方向（周期）: (H, W) - 使用周期条件
            - eta方向: (H+1, W) - 上下两个单元平均

    ✅ ADFLOW对齐：
        - 当提供halo_bottom时，底部边界使用 0.5*(physical + halo)
        - 当提供halo_top时，顶部边界使用 0.5*(physical + halo)
        - 未提供halo时回退到复制边界单元（非ADFLOW标准）
    """
    # 如果是标量或0维张量，直接返回（面上值与单元中心值相同）
    if field.ndim == 0:
        return field

    # 检查batch维度
    has_batch = field.ndim == 3

    if direction == 'xi':
        # ξ方向：左右两个单元平均
        # face[i,j] = 0.5 * (cell[i,j-1] + cell[i,j])

        if periodic:
            # ✅ ADFLOW对齐：周期边界条件 face[i] = 0.5 * (cell[i] + cell[i+1])
            # ADFLOW使用 (i, i+1) 平均，而非 (i-1, i)
            # 参考：fluxes.F90:4511
            if has_batch:
                # (batch, H, W) → (batch, H, W)
                field_right = torch.roll(field, shifts=-1, dims=-1)
                field_face = 0.5 * (field + field_right)
            else:
                # (H, W) → (H, W)
                field_right = torch.roll(field, shifts=-1, dims=-1)
                field_face = 0.5 * (field + field_right)
        else:
            # 非周期边界
            if has_batch:
                # (batch, H, W) → (batch, H, W+1)

                # 左边界
                if halo_bottom is not None:
                    # halo_bottom: (batch, H) → (batch, H, 1)
                    halo_left = halo_bottom.unsqueeze(-1) if halo_bottom.ndim == 2 else halo_bottom
                    left_boundary = 0.5 * (field[:, :, 0:1] + halo_left)
                else:
                    left_boundary = field[:, :, 0:1]

                # 内部面：平均
                interior = 0.5 * (field[:, :, :-1] + field[:, :, 1:])

                # 右边界
                if halo_top is not None:
                    # halo_top: (batch, H) → (batch, H, 1)
                    halo_right = halo_top.unsqueeze(-1) if halo_top.ndim == 2 else halo_top
                    right_boundary = 0.5 * (field[:, :, -1:] + halo_right)
                else:
                    right_boundary = field[:, :, -1:]

                field_face = torch.cat([left_boundary, interior, right_boundary], dim=2)
            else:
                # (H, W) → (H, W+1)

                # 左边界
                if halo_bottom is not None:
                    halo_left = halo_bottom.unsqueeze(-1) if halo_bottom.ndim == 1 else halo_bottom
                    left_boundary = 0.5 * (field[:, 0:1] + halo_left)
                else:
                    left_boundary = field[:, 0:1]

                interior = 0.5 * (field[:, :-1] + field[:, 1:])

                # 右边界
                if halo_top is not None:
                    halo_right = halo_top.unsqueeze(-1) if halo_top.ndim == 1 else halo_top
                    right_boundary = 0.5 * (field[:, -1:] + halo_right)
                else:
                    right_boundary = field[:, -1:]

                field_face = torch.cat([left_boundary, interior, right_boundary], dim=1)

    elif direction == 'eta':
        # η方向：上下两个单元平均
        # face[i,j] = 0.5 * (cell[i-1,j] + cell[i,j])

        if has_batch:
            # (batch, H, W) → (batch, H+1, W)

            # 底部边界（壁面）
            if halo_bottom is not None:
                # halo_bottom: (batch, W) → (batch, 1, W)
                halo_wall = halo_bottom.unsqueeze(1) if halo_bottom.ndim == 2 else halo_bottom
                # ✅ ADFLOW对齐：壁面使用 0.5*(physical + halo_reflected)
                # 对于速度，halo已反射，所以 u_face = 0.5*(u + (-u)) = 0
                bottom_boundary = 0.5 * (field[:, 0:1, :] + halo_wall)
            else:
                bottom_boundary = field[:, 0:1, :]

            # 内部面：平均
            interior = 0.5 * (field[:, :-1, :] + field[:, 1:, :])

            # 顶部边界（远场）
            if halo_top is not None:
                # halo_top: (batch, W) → (batch, 1, W)
                halo_far = halo_top.unsqueeze(1) if halo_top.ndim == 2 else halo_top
                top_boundary = 0.5 * (field[:, -1:, :] + halo_far)
            else:
                top_boundary = field[:, -1:, :]

            field_face = torch.cat([bottom_boundary, interior, top_boundary], dim=1)
        else:
            # (H, W) → (H+1, W)

            # 底部边界（壁面）
            if halo_bottom is not None:
                halo_wall = halo_bottom.unsqueeze(0) if halo_bottom.ndim == 1 else halo_bottom
                bottom_boundary = 0.5 * (field[0:1, :] + halo_wall)
            else:
                bottom_boundary = field[0:1, :]

            interior = 0.5 * (field[:-1, :] + field[1:, :])

            # 顶部边界（远场）
            if halo_top is not None:
                halo_far = halo_top.unsqueeze(0) if halo_top.ndim == 1 else halo_top
                top_boundary = 0.5 * (field[-1:, :] + halo_far)
            else:
                top_boundary = field[-1:, :]

            field_face = torch.cat([bottom_boundary, interior, top_boundary], dim=0)

    else:
        raise ValueError(f"Invalid direction: {direction}")

    return field_face


def _average_nodal_to_face(
    nodal_field: torch.Tensor,
    direction: str,
    periodic: bool = False
) -> torch.Tensor:
    """
    节点值平均到面（2节点平均）

    面上的值由两个相邻节点平均得到

    Args:
        nodal_field: 节点值 (H+1, W+1) 或 (batch, H+1, W+1)
        direction: 'xi' 或 'eta'
        periodic: 是否使用周期边界条件（仅用于xi方向）

    Returns:
        face_field: 面上的值
            - xi方向（非周期）: (H, W+1) - 上下两个节点平均
            - xi方向（周期）: (H, W) - 上下两个节点平均，排除周期重复点
            - eta方向: (H+1, W) - 左右两个节点平均
    """
    # 检查batch维度
    has_batch = nodal_field.ndim == 3

    if direction == 'xi':
        # ξ方向面：上下两个节点平均
        # face[i,j] = 0.5 * (node[i,j] + node[i+1,j])

        if periodic:
            # ✅ ADFLOW对齐：周期边界节点插值
            # 与 _interpolate_nodal_to_face_xi 一致，使用节点列 1:W+1
            # 参考：fluxes.py:2077 (seam在列0，实际面从列1开始)
            if has_batch:
                # (batch, H+1, W+1) → (batch, H, W)
                face_field = 0.5 * (nodal_field[:, :-1, 1:] + nodal_field[:, 1:, 1:])
            else:
                # (H+1, W+1) → (H, W)
                face_field = 0.5 * (nodal_field[:-1, 1:] + nodal_field[1:, 1:])
        else:
            if has_batch:
                # (batch, H+1, W+1) → (batch, H, W+1)
                face_field = 0.5 * (nodal_field[:, :-1, :] + nodal_field[:, 1:, :])
            else:
                # (H+1, W+1) → (H, W+1)
                face_field = 0.5 * (nodal_field[:-1, :] + nodal_field[1:, :])

    elif direction == 'eta':
        # η方向面：左右两个节点平均
        # face[i,j] = 0.5 * (node[i,j] + node[i,j+1])

        if has_batch:
            # (batch, H+1, W+1) → (batch, H+1, W)
            face_field = 0.5 * (nodal_field[:, :, :-1] + nodal_field[:, :, 1:])
        else:
            # (H+1, W+1) → (H+1, W)
            face_field = 0.5 * (nodal_field[:, :-1] + nodal_field[:, 1:])

    else:
        raise ValueError(f"Invalid direction: {direction}")

    return face_field


# ========== 辅助函数：grad(a²)计算 ==========

def compute_speed_of_sound_squared_gradient(
    rho: torch.Tensor,
    p: torch.Tensor,
    drho_dx: torch.Tensor,
    drho_dy: torch.Tensor,
    dp_dx: torch.Tensor,
    dp_dy: torch.Tensor,
    gamma: float = 1.4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算声速平方梯度：grad(a²)

    公式：
        a² = gamma * p / rho

        d(a²)/dx = gamma * [dp/dx / rho - p / rho² * drho/dx]
                 = gamma / rho * [dp/dx - a² * drho/dx / gamma]
                 = gamma / rho * [dp/dx - p/rho * drho/dx]

    ADFLOW使用此梯度计算热传导通量，而不是温度梯度。
    原因：数值稳定性更好，避免温度计算的中间步骤。

    Args:
        rho, p: 密度和压力 (H, W) 或 (batch, H, W)
        drho_dx, drho_dy, dp_dx, dp_dy: 梯度
        gamma: 比热比

    Returns:
        (daa_dx, daa_dy): 声速平方梯度
    """
    # 避免除零
    rho_safe = rho + 1e-14

    # 声速平方：a² = gamma * p / rho
    aa = gamma * p / rho_safe

    # 梯度：d(a²)/dx = gamma / rho * [dp/dx - p/rho * drho/dx]
    daa_dx = gamma / rho_safe * (dp_dx - p / rho_safe * drho_dx)
    daa_dy = gamma / rho_safe * (dp_dy - p / rho_safe * drho_dy)

    return daa_dx, daa_dy
