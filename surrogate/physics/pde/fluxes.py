"""
通量模块 - Phase 2 & 3

统一对流和黏性通量计算，全部使用geometry-based方法

关键改进：
1. 对流通量：ADflow中央散度格式（已由 residual backend 迁移到此）
2. 黏性通量：使用geometry-based面向量投影，替代metric-based
3. 连续性方程也使用中央散度（rqsp+rqsm），替代简单插值

所有通量使用统一的face_geom（来自geometry.py），彻底解决几何不一致问题。
"""

import os
import torch
from typing import Dict, Tuple, Optional, Union

from .approx_viscous import viscous_flux_approx
from . import geometry as geom_module


_ENABLE_FLUX_NAN_GUARD = os.environ.get('SURROGATE_VALIDATE_FLUX_INPUTS', '') == '1'


# ========== Phase 2: 对流通量（Inviscid Flux）==========

def _stack_face_state_fields(*fields: torch.Tensor) -> torch.Tensor:
    """Pack independent scalar fields so face-state slicing happens once."""
    return torch.stack(fields, dim=1)


def _extract_face_states_xi(
    stacked_fields: torch.Tensor,
    *,
    periodic: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if periodic:
        return stacked_fields, torch.roll(stacked_fields, shifts=-1, dims=-1)
    return stacked_fields[..., :-1], stacked_fields[..., 1:]


def _extract_face_states_eta(
    stacked_fields: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return stacked_fields[..., :-1, :], stacked_fields[..., 1:, :]


def _assemble_eta_face_stack(
    bottom: torch.Tensor,
    interior: torch.Tensor,
    top: torch.Tensor,
) -> torch.Tensor:
    total_faces = int(bottom.shape[-2] + interior.shape[-2] + top.shape[-2])
    assembled = bottom.new_empty(*bottom.shape[:-2], total_faces, bottom.shape[-1])
    bottom_faces = int(bottom.shape[-2])
    interior_faces = int(interior.shape[-2])
    assembled[..., :bottom_faces, :] = bottom
    assembled[..., bottom_faces:bottom_faces + interior_faces, :] = interior
    assembled[..., bottom_faces + interior_faces:, :] = top
    return assembled


def _pad_eta_face_stack_with_zeros(interior: torch.Tensor) -> torch.Tensor:
    padded = interior.new_zeros(*interior.shape[:-2], int(interior.shape[-2]) + 2, interior.shape[-1])
    padded[..., 1:-1, :] = interior
    return padded


def inviscid_central_flux(
    rho: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    face_geom: Dict[str, torch.Tensor],
    direction: str = 'xi',
    rhoE: Optional[torch.Tensor] = None,  # 新增：能量密度（可选，用于5方程）
    dissipation_mode: str = 'jameson',
    vis2: float = 0.25,
    vis4: float = 0.0156,
    gamma: float = 1.4,
    dss_max: float = 0.25,
    sslim: float = 1e-3,
    return_dissipation: bool = False,
    basis: str = 'entropy',
    vol: Optional[torch.Tensor] = None,
    adis: float = 0.67,
    acoustic_scale_factor: float = 1.0,
    lumped_dissipation: bool = False,
    lumped_sigma: float = 1.0,
    frozen_shock_sensor: Optional[torch.Tensor] = None,
    frozen_ss_halo: Optional[torch.Tensor] = None,
    use_dissipation_continuation: bool = False,
    diss_cont_magnitude: float = 0.0,
    diss_cont_midpoint: float = 20.0,
    diss_cont_sharpness: float = 3.0,
    diss_cont_total_r: Optional[float] = None,
    diss_cont_total_r0: Optional[float] = None,
    diss_cont_rfil: float = 1.0,
    precomputed_cell_radius: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    ADflow中央散度对流通量 + Jameson人工耗散（geometry-based，完全统一）

    基于ADflow的inviscidCentralFlux格式（fluxes.F90:1088附近）：
    - 质量流率（包含面面积向量A）：
      rqsp = 0.5·ρ_R·vnp, rqsm = 0.5·ρ_L·vnm
      其中 vnp = u_R·A_x + v_R·A_y, vnm = u_L·A_x + v_L·A_y（法向速度乘面积）
    - 连续性通量：Fc = rqsp + rqsm
    - x动量通量：Fmx = rqsp·u_R + rqsm·u_L + p_avg·A_x
    - y动量通量：Fmy = rqsp·v_R + rqsm·v_L + p_avg·A_y

    **Jameson耗散（fluxes.F90:1210-1268行）：**
    - 总通量：F_total = F_conv - F_diss

    Args:
        rho, u, v, p: 物理场 (batch, H, W) 或 (H, W)
        face_geom: 面几何字典（来自geometry.compute_face_geometry）
        direction: 'xi' 或 'eta'
        dissipation_mode: 'jameson'（Jameson耗散）或 'none'（无耗散）
        vis2: 2阶耗散系数（ADflow默认0.5）
        vis4: 4阶耗散系数（当前live ADFLOW默认0.0156）
        gamma: 比热比（默认1.4）
        dss_max: 传感器上限（ADflow默认0.25）
        sslim: 压力传感器下限（ADflow: 0.001*pInfCorr，无量纲约1e-3）

    Returns:
        (Fc, Fmx, Fmy): 连续性、x动量、y动量通量（包含耗散项）
            - ξ面: (batch, H, W) if periodic else (batch, H, W-1)
            - η面: (batch, H-1, W)
    """
    # 添加batch维度（如果没有）
    if rho.ndim == 2:
        rho = rho.unsqueeze(0)
        u = u.unsqueeze(0)
        v = v.unsqueeze(0)
        p = p.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    # NaN检测：确保输入流场数据有效
    #
    # NOTE: This is an eager diagnostics guard. Under FX proxy tracing
    # (e.g. torch.func.linearize), Python truthiness of 0-dim tensors is
    # disallowed, so skip the guard in that case.
    if _ENABLE_FLUX_NAN_GUARD:
        def _has_nan(x: torch.Tensor) -> bool:
            try:
                return bool(torch.isnan(x).any())
            except RuntimeError as e:
                msg = str(e)
                if "tracing tensor" in msg and "aten._local_scalar_dense" in msg:
                    return False
                raise

        if _has_nan(rho):
            raise ValueError(f"Input rho contains NaN values in {direction} direction flux")
        if _has_nan(u):
            raise ValueError(f"Input u contains NaN values in {direction} direction flux")
        if _has_nan(v):
            raise ValueError(f"Input v contains NaN values in {direction} direction flux")
        if _has_nan(p):
            raise ValueError(f"Input p contains NaN values in {direction} direction flux")

    # 提取面几何信息（P0修复：使用面面积向量直接投影）
    if direction == 'xi':
        # 面面积向量（metric-based，与体积同源）
        A_x = face_geom['A_x_xi']
        A_y = face_geom['A_y_xi']
        periodic = face_geom.get('periodic_xi', False)
    elif direction == 'eta':
        A_x = face_geom['A_x_eta']
        A_y = face_geom['A_y_eta']
        periodic = False
    else:
        raise ValueError(f"Invalid direction: {direction}")

    # 确保面几何有batch维度
    if A_x.ndim == 2:
        A_x = A_x.unsqueeze(0)
        A_y = A_y.unsqueeze(0)

    state_fields = [rho, u, v, p]
    if rhoE is not None:
        state_fields.append(rhoE)
    stacked_states = _stack_face_state_fields(*state_fields)

    # 1. 提取左右状态
    if direction == 'xi':
        state_L, state_R = _extract_face_states_xi(stacked_states, periodic=periodic)
    elif direction == 'eta':
        state_L, state_R = _extract_face_states_eta(stacked_states)
    else:
        raise ValueError(f"Invalid direction: {direction}")

    rho_L = state_L[:, 0, ...]
    u_L = state_L[:, 1, ...]
    v_L = state_L[:, 2, ...]
    p_L = state_L[:, 3, ...]

    rho_R = state_R[:, 0, ...]
    u_R = state_R[:, 1, ...]
    v_R = state_R[:, 2, ...]
    p_R = state_R[:, 3, ...]
    if rhoE is not None:
        rhoE_L = state_L[:, 4, ...]
        rhoE_R = state_R[:, 4, ...]

    # 2. 计算法向速度分量（P0修复：用面面积向量A直接投影）
    # vnp/vnm = u·A_x + v·A_y（法向速度乘以面积）
    vnp = u_R * A_x + v_R * A_y  # 右侧法向速度×面积
    vnm = u_L * A_x + v_L * A_y  # 左侧法向速度×面积

    # 3. 质量流率（ADflow中央散度系数0.5）
    # 注意：vnp/vnm已包含面积，无需再乘s
    rqsp = 0.5 * rho_R * vnp
    rqsm = 0.5 * rho_L * vnm

    # 4. 压力平均
    p_avg = 0.5 * (p_L + p_R)

    # 5. 物理空间对流通量（P0修复：用面面积向量A直接投影）
    # 连续性：质量通量
    Fc_conv = rqsp + rqsm

    # x动量 / y动量：
    # ADflow格式改用面面积向量A直接投影：
    Fmx_conv = rqsp * u_R + rqsm * u_L + p_avg * A_x
    Fmy_conv = rqsp * v_R + rqsm * v_L + p_avg * A_y

    # 6. 添加Jameson人工耗散（如果启用）
    D_rho, D_rhou, D_rhov = None, None, None  # 默认无耗散

    if dissipation_mode == 'jameson':
        from . import dissipation as diss_module

        # 确定周期性（仅ξ方向可能周期）
        periodic_xi = face_geom.get('periodic_xi', False) if direction == 'xi' else False

        # 计算耗散通量（ADflow标量方案）
        # ADflow blockette.F90: RANS用熵基底，Euler用压力基底
        # ✅ 新增：如果提供rhoE，则独立计算能量耗散（ADFLOW对齐）
        dissipation_result = diss_module.compute_jameson_dissipation(
            rho, u, v, p, face_geom,
            direction=direction,
            vis2=vis2,
            vis4=vis4,
            gamma=gamma,
            dss_max=dss_max,
            periodic_xi=periodic_xi,
            sslim=sslim,
            basis=basis,
            vol=vol,
            adis=adis,
            acoustic_scale_factor=acoustic_scale_factor,
            rhoE=rhoE,  # ✅ 传入rhoE参数
            lumped_dissipation=lumped_dissipation,
            lumped_sigma=lumped_sigma,
            frozen_shock_sensor=frozen_shock_sensor,
            frozen_ss_halo=frozen_ss_halo,
            use_dissipation_continuation=use_dissipation_continuation,
            diss_cont_magnitude=diss_cont_magnitude,
            diss_cont_midpoint=diss_cont_midpoint,
            diss_cont_sharpness=diss_cont_sharpness,
            diss_cont_total_r=diss_cont_total_r,
            diss_cont_total_r0=diss_cont_total_r0,
            rfil=diss_cont_rfil,
            precomputed_cell_radius=precomputed_cell_radius,
        )

        # ✅ 解包返回值（3或4个，取决于是否提供rhoE）
        if rhoE is not None:
            D_rho, D_rhou, D_rhov, D_rhoE = dissipation_result
        else:
            D_rho, D_rhou, D_rhov = dissipation_result
            D_rhoE = None

        # 总通量 = 对流通量 - 耗散通量
        Fc = Fc_conv - D_rho
        Fmx = Fmx_conv - D_rhou
        Fmy = Fmy_conv - D_rhov

    elif dissipation_mode == 'none':
        # 无耗散：纯中央差分
        Fc = Fc_conv
        Fmx = Fmx_conv
        Fmy = Fmy_conv

    else:
        raise ValueError(f"Unknown dissipation_mode: {dissipation_mode}. Choose 'jameson' or 'none'.")

    # 6.5 能量方程通量（如果提供了rhoE）
    FE = None
    FE_conv = None

    if rhoE is not None:
        # 复用已提取的面状态和法向投影，避免重复 stack / slice / project。
        qsp = 0.5 * vnp
        qsm = 0.5 * vnm
        FE_conv_raw = (
            qsp * rhoE_R +
            qsm * rhoE_L +
            0.5 * (vnp * p_R + vnm * p_L)
        )

        # ✅ 能量方程Jameson耗散（独立计算，ADFLOW对齐）
        # D_rhoE已在前面的compute_jameson_dissipation调用中获得
        if dissipation_mode == 'jameson' and D_rhoE is not None:
            FE = FE_conv_raw - D_rhoE
        else:
            # 无耗散
            FE = FE_conv_raw

        FE_conv = FE_conv_raw

    # 7. 移除batch维度（如果输入没有）
    if squeeze_output:
        Fc = Fc.squeeze(0)
        Fmx = Fmx.squeeze(0)
        Fmy = Fmy.squeeze(0)
        Fc_conv = Fc_conv.squeeze(0)
        Fmx_conv = Fmx_conv.squeeze(0)
        Fmy_conv = Fmy_conv.squeeze(0)
        if D_rho is not None:
            D_rho = D_rho.squeeze(0)
            D_rhou = D_rhou.squeeze(0)
            D_rhov = D_rhov.squeeze(0)
        if FE is not None:
            FE = FE.squeeze(0)
            FE_conv = FE_conv.squeeze(0)
            if D_rhoE is not None:
                D_rhoE = D_rhoE.squeeze(0)

    if return_dissipation:
        # 返回: 总通量, 耗散, 纯对流通量
        if rhoE is not None:
            return Fc, Fmx, Fmy, FE, D_rho, D_rhou, D_rhov, D_rhoE, Fc_conv, Fmx_conv, Fmy_conv, FE_conv
        else:
            return Fc, Fmx, Fmy, D_rho, D_rhou, D_rhov, Fc_conv, Fmx_conv, Fmy_conv

    # 标准返回
    if rhoE is not None:
        return Fc, Fmx, Fmy, FE
    else:
        return Fc, Fmx, Fmy


def inviscid_energy_flux_central(
    rho: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    rhoE: torch.Tensor,
    p: torch.Tensor,
    face_geom: Dict[str, torch.Tensor],
    direction: str = 'xi'
) -> torch.Tensor:
    """
    能量方程中央散度通量（ADFLOW对齐）

    参考：ADFLOW fluxes.F90:126-129 (inviscidCentralFlux, energy part)

    公式：
        F_E = qsp·rhoE_R + qsm·rhoE_L + porFlux·(vnp·p_R + vnm·p_L)

    其中：
        qsp = 0.5 · ρ_R · vnp  (质量流率，右侧)
        qsm = 0.5 · ρ_L · vnm  (质量流率，左侧)
        vnp = u_R·A_x + v_R·A_y  (法向速度 × 面积)
        vnm = u_L·A_x + v_L·A_y
        porFlux = 0.5  (内部面)

    Args:
        rho, u, v, rhoE, p: 物理场 (batch, H, W) 或 (H, W)
        face_geom: 面几何字典（来自geometry.compute_face_geometry）
        direction: 'xi' 或 'eta'

    Returns:
        F_E: 能量通量
            - ξ面: (batch, H, W) if periodic else (batch, H, W-1)
            - η面: (batch, H-1, W)
    """
    # 添加batch维度（如果没有）
    if rho.ndim == 2:
        rho = rho.unsqueeze(0)
        u = u.unsqueeze(0)
        v = v.unsqueeze(0)
        rhoE = rhoE.unsqueeze(0)
        p = p.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    # 提取面几何信息
    if direction == 'xi':
        A_x = face_geom['A_x_xi']
        A_y = face_geom['A_y_xi']
        periodic = face_geom.get('periodic_xi', False)
    elif direction == 'eta':
        A_x = face_geom['A_x_eta']
        A_y = face_geom['A_y_eta']
        periodic = False
    else:
        raise ValueError(f"Invalid direction: {direction}")

    # 确保面几何有batch维度
    if A_x.ndim == 2:
        A_x = A_x.unsqueeze(0)
        A_y = A_y.unsqueeze(0)

    stacked_states = _stack_face_state_fields(rho, u, v, p, rhoE)
    if direction == 'xi':
        state_L, state_R = _extract_face_states_xi(stacked_states, periodic=periodic)
    elif direction == 'eta':
        state_L, state_R = _extract_face_states_eta(stacked_states)
    else:
        raise ValueError(f"Invalid direction: {direction}")

    rho_L = state_L[:, 0, ...]
    u_L = state_L[:, 1, ...]
    v_L = state_L[:, 2, ...]
    p_L = state_L[:, 3, ...]
    rhoE_L = state_L[:, 4, ...]

    rho_R = state_R[:, 0, ...]
    u_R = state_R[:, 1, ...]
    v_R = state_R[:, 2, ...]
    p_R = state_R[:, 3, ...]
    rhoE_R = state_R[:, 4, ...]

    # 2. 计算法向速度分量（用面面积向量A直接投影）
    vnp = u_R * A_x + v_R * A_y  # 右侧法向速度×面积
    vnm = u_L * A_x + v_L * A_y  # 左侧法向速度×面积

    # 3. 质量流率（ADFLOW对齐：fluxes.F90:86, 126-129）
    # ADFLOW公式：qsp = (vnp - sFace) * porVel
    # ADFLOW: porVel = one * porFlux = 0.5 (fluxes.F90:81)
    sFace = 0.0  # 静态网格
    porVel = 0.5  # ADFLOW标准: porVel = 1.0 * porFlux = 0.5
    porFlux = 0.5  # 中心格式系数

    qsp = (vnp - sFace) * porVel  # ✅ 不含rho（ADFLOW标准）
    qsm = (vnm - sFace) * porVel

    # 4. 能量通量（ADFLOW公式：fluxes.F90:126-129）
    # F_E = qsp*rhoE_R + qsm*rhoE_L + porFlux*(vnp*p_R + vnm*p_L)
    F_E = qsp * rhoE_R + qsm * rhoE_L + porFlux * (vnp * p_R + vnm * p_L)

    # 5. 移除batch维度（如果输入没有）
    if squeeze_output:
        F_E = F_E.squeeze(0)

    return F_E


def compute_inviscid_eta_flux(
    rho: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    face_geom: Dict[str, torch.Tensor],
    Ma: Optional[Union[float, torch.Tensor]] = None,
    AoA: Optional[Union[float, torch.Tensor]] = None,
    gamma: float = 1.4,
    dissipation_mode: str = 'jameson',
    vis2: float = 0.25,
    vis4: float = 0.0156,
    dss_max: float = 0.25,
    sslim: float = 1e-3,
    debug: bool = False,
    halo_wall: Optional[torch.Tensor] = None,
    halo_farfield: Optional[torch.Tensor] = None,
    return_dissipation: bool = False,
    basis: str = 'entropy',
    ss_halo: Optional[torch.Tensor] = None,
    vol: Optional[torch.Tensor] = None,
    adis: float = 0.67,
    acoustic_scale_factor: float = 1.0,
    # 新增：能量方程支持
    rhoE: Optional[torch.Tensor] = None,
    lumped_dissipation: bool = False,
    lumped_sigma: float = 1.0,
    frozen_shock_sensor: Optional[torch.Tensor] = None,
    frozen_ss_halo: Optional[torch.Tensor] = None,
    use_dissipation_continuation: bool = False,
    diss_cont_magnitude: float = 0.0,
    diss_cont_midpoint: float = 20.0,
    diss_cont_sharpness: float = 3.0,
    diss_cont_total_r: Optional[float] = None,
    diss_cont_total_r0: Optional[float] = None,
    diss_cont_rfil: float = 1.0,
    precomputed_cell_radius: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    计算完整η面对流通量（H+1个面，包含边界j=0和j=H）

    **ADflow对齐模式**：如果提供halo_wall，使用ADflow风格的halo单元+物理单元
    计算边界面通量，实现精确对齐。

    实现策略：
    - 内部面 (j=1..H-1): 使用标准中央散度 + Jameson耗散
    - 边界面 j=0 (壁面):
        * 如果提供halo_wall: 使用halo单元+物理单元计算（ADflow对齐）
        * 否则: 简化无滑移BC（Fc=0, Fmx=p·A_x）
    - 边界面 j=H (远场):
        * 如果提供halo_farfield: 使用halo单元+物理单元计算
        * 否则: 使用自由流BC或简化外推

    ADflow壁面通量计算（k=1面，连接halo单元k=1和物理单元k=2）：
    - rqsp = 0.5 * rho_physical * (u_physical·A)
    - rqsm = 0.5 * rho_halo * (u_halo·A)
    - Fc = rqsp + rqsm
    - 由于u_halo = -u_physical（反射），壁面处：
      Fc = 0.5*rho*(u·A) + 0.5*rho*(-u·A) = 0（质量守恒）

    Args:
        rho, u, v, p: 物理场 (batch, H, W) 或 (H, W)
        face_geom: 面几何字典（必须包含'A_x_eta', 'A_y_eta'，形状为H+1×W）
        Ma: 自由流马赫数（用于远场BC）
        AoA: 攻角（度，用于远场BC）
        gamma: 比热比（默认1.4）
        halo_wall: 壁面halo单元物理量 (C, W) 或 (batch, C, W)，C=4 [rho,u,v,p]
        halo_farfield: 远场halo单元物理量（可选）

    Returns:
        (Fc_eta, Fmx_eta, Fmy_eta): (batch, H+1, W) 或 (H+1, W) 完整η面通量
    """
    # 1. 添加batch维度（如果没有）
    if rho.ndim == 2:
        rho = rho.unsqueeze(0)
        u = u.unsqueeze(0)
        v = v.unsqueeze(0)
        p = p.unsqueeze(0)
        if rhoE is not None:
            rhoE = rhoE.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch, H, W = rho.shape

    # 2. 提取完整η面几何（H+1个面）
    if 'A_x_eta' not in face_geom or 'A_y_eta' not in face_geom:
        raise ValueError(
            "compute_inviscid_eta_flux requires 'A_x_eta' and 'A_y_eta' "
            "in face_geom with shape (H+1, W). Please call compute_face_area_vectors first."
        )

    A_x_full = face_geom['A_x_eta']  # (H+1, W)
    A_y_full = face_geom['A_y_eta']  # (H+1, W)

    # 确保几何有batch维度
    if A_x_full.ndim == 2:
        A_x_full = A_x_full.unsqueeze(0)
        A_y_full = A_y_full.unsqueeze(0)

    # 3. 计算内部面通量 (j=1..H-1, 共H-1个面)
    # 复用标准中央散度逻辑，但左右状态只做一次切片
    interior_fields = [rho, u, v, p]
    if rhoE is not None:
        interior_fields.append(rhoE)
    interior_state_L, interior_state_R = _extract_face_states_eta(
        _stack_face_state_fields(*interior_fields)
    )

    rho_L_interior = interior_state_L[:, 0, ...]
    u_L_interior = interior_state_L[:, 1, ...]
    v_L_interior = interior_state_L[:, 2, ...]
    p_L_interior = interior_state_L[:, 3, ...]

    rho_R_interior = interior_state_R[:, 0, ...]
    u_R_interior = interior_state_R[:, 1, ...]
    v_R_interior = interior_state_R[:, 2, ...]
    p_R_interior = interior_state_R[:, 3, ...]

    # 能量方程：提取 rhoE 状态（如果提供）
    if rhoE is not None:
        rhoE_L_interior = interior_state_L[:, 4, ...]
        rhoE_R_interior = interior_state_R[:, 4, ...]

    # 内部面几何 (j=1..H-1)
    A_x_interior = A_x_full[:, 1:-1, :]  # (batch, H-1, W)
    A_y_interior = A_y_full[:, 1:-1, :]

    # 法向速度×面积
    vnp_interior = u_R_interior * A_x_interior + v_R_interior * A_y_interior
    vnm_interior = u_L_interior * A_x_interior + v_L_interior * A_y_interior

    # 质量流率
    rqsp_interior = 0.5 * rho_R_interior * vnp_interior
    rqsm_interior = 0.5 * rho_L_interior * vnm_interior

    # 压力平均
    p_avg_interior = 0.5 * (p_L_interior + p_R_interior)

    # 内部面对流通量
    Fc_interior_conv = rqsp_interior + rqsm_interior
    Fmx_interior_conv = rqsp_interior * u_R_interior + rqsm_interior * u_L_interior + p_avg_interior * A_x_interior
    Fmy_interior_conv = rqsp_interior * v_R_interior + rqsm_interior * v_L_interior + p_avg_interior * A_y_interior

    # 能量方程：内部面对流通量（如果提供rhoE）
    if rhoE is not None:
        # ✅ ADFLOW能量方程使用qsp（不含rho），与动量方程的rqsp不同
        # 参考 ADFLOW fluxes.F90:86, 126-129
        # qsp = (vnp - sFace) * porVel（不含rho！）
        # FE = qsp*rhoE_R + qsm*rhoE_L + porFlux*(vnp*p_R + vnm*p_L)
        # ADFLOW: porVel = one * porFlux = 0.5 (fluxes.F90:81)
        sFace = 0.0  # 静态网格
        porVel = 0.5  # ADFLOW标准: porVel = 1.0 * porFlux = 0.5
        porFlux = 0.5  # 中心格式系数

        # 能量方程专用qsp（不含rho）
        qsp_interior = (vnp_interior - sFace) * porVel
        qsm_interior = (vnm_interior - sFace) * porVel

        FE_interior_conv = (
            qsp_interior * rhoE_R_interior +
            qsm_interior * rhoE_L_interior +
            porFlux * (vnp_interior * p_R_interior + vnm_interior * p_L_interior)
        )

    # ===== DEBUG: 打印η方向内部面通量（与ADflow K面对比）=====
    # ADflow索引(i=152,k=2,10,40) -> PyTorch索引(i=150,j=0,8,38)  adflow 存在halo元
    # 注意：A_x_interior已经是[1:-1]切片，所以j=1对应interior_j=0
    if debug:
        debug_i = 150  # ADflow i=152
        debug_js_interior = [0, 8, 38]  # ADflow k=2,10,40 -> interior面索引
        print("===== DEBUG: PyTorch η-direction (eta) interior face flux =====")
        for interior_j in debug_js_interior:
            adflow_k = interior_j + 2  # 转回ADflow索引
            print(f"  i={debug_i+1}(ADflow), k={adflow_k}(ADflow), interior_j={interior_j}(PyTorch)")
            print(f"    A_x_eta={A_x_interior[0, interior_j, debug_i].item():.6e}, "
                  f"A_y_eta={A_y_interior[0, interior_j, debug_i].item():.6e}")
            print(f"    vnp={vnp_interior[0, interior_j, debug_i].item():.6e}, "
                  f"vnm={vnm_interior[0, interior_j, debug_i].item():.6e}")
            print(f"    rqsp={rqsp_interior[0, interior_j, debug_i].item():.6e}, "
                  f"rqsm={rqsm_interior[0, interior_j, debug_i].item():.6e}")
            print(f"    Fc_eta={Fc_interior_conv[0, interior_j, debug_i].item():.6e}")
        print("=" * 60)

    # 3.5. 组装纯对流 η 面通量 (H+1, W)
    # 边界面无耗散时：对流=总通量；但ADflow inviscidDissFluxScalar 会在 farfield 边界面也计算耗散，
    # 因此这里先构造纯对流通量，后续统一减去 D_full（包含 farfield 边界面）。
    # 注意：壁面 porK=0 -> 不计算耗散（D_full[0]=0）

    # 4. 计算边界面 j=0（壁面）
    A_x_bot = A_x_full[:, 0:1, :]  # (batch, 1, W)
    A_y_bot = A_y_full[:, 0:1, :]

    if halo_wall is not None:
        # **ADflow对齐模式**：壁面边界使用porK=boundFlux处理
        #
        # 关键：ADflow在壁面边界(porK=0)时，porVel=0，这意味着：
        # - 所有基于速度的对流项被置零
        # - 只保留压力项
        #
        # 这与使用halo反射计算中央通量不同！
        # ADflow日志显示：porK=0, porVel=0, vnp=0, vnm=0
        #
        # 因此壁面通量应该是：
        # - Fc_bot = 0（质量通量为0）
        # - Fmx_bot = p_wall * A_x（仅压力项）
        # - Fmy_bot = p_wall * A_y（仅压力项）

        # 确保halo_wall有正确的维度
        if halo_wall.ndim == 2:  # (C, W)
            halo_wall = halo_wall.unsqueeze(0)  # (1, C, W)

        # 提取物理单元压力
        p_wall = p[..., 0:1, :]  # (batch, 1, W)

        # 也可以使用halo和物理的压力平均（更接近中央差分）
        p_halo = halo_wall[:, 3:4, :]
        p_avg = 0.5 * (p_halo + p_wall)

        # 壁面通量（ADflow porK=0处理：porVel=0，只有压力项）
        Fc_bot = torch.zeros_like(A_x_bot)
        Fmx_bot = p_avg * A_x_bot  # 使用压力平均
        Fmy_bot = p_avg * A_y_bot

        # 能量方程：壁面通量（如果提供rhoE）
        if rhoE is not None:
            # 壁面速度为0，只有压力功项
            # FE_wall = 0 (质量通量为0) + 0 (速度为0) + p_avg * 0 (法向速度为0)
            # 实际上壁面能量通量为0（无滑移边界）
            FE_bot = torch.zeros_like(A_x_bot)

        # DEBUG: 验证壁面通量
        if debug:
            print(f"\n[DEBUG fluxes.py] 壁面通量（ADflow porK=0模式）:")
            print(f"  halo_wall shape: {halo_wall.shape}")
            print(f"  p_wall[0,0,0:5]: {p_wall[0,0,0:5].cpu().numpy()}")
            print(f"  p_halo[0,0,0:5]: {p_halo[0,0,0:5].cpu().numpy()}")
            print(f"  p_avg[0,0,0:5]: {p_avg[0,0,0:5].cpu().numpy()}")
            print(f"  Fc_bot[0,0,0:5]: {Fc_bot[0,0,0:5].cpu().numpy()}")
            print(f"  Fmx_bot[0,0,0:5]: {Fmx_bot[0,0,0:5].cpu().numpy()}")
    else:
        # 简化模式：无滑移BC (u_wall=0, v_wall=0)
        p_wall = p[..., 0:1, :]  # 使用第一层单元压力

        # 壁面通量：质量通量为0，动量通量仅压力项
        Fc_bot = torch.zeros_like(A_x_bot)
        Fmx_bot = p_wall * A_x_bot
        Fmy_bot = p_wall * A_y_bot

        # 能量方程：壁面通量（如果提供rhoE）
        if rhoE is not None:
            FE_bot = torch.zeros_like(A_x_bot)

    # 5. 计算边界面 j=H（远场）
    A_x_top = A_x_full[:, -1:, :]  # (batch, 1, W)
    A_y_top = A_y_full[:, -1:, :]

    if halo_farfield is not None:
        # **ADflow对齐模式**：使用halo单元+物理单元计算远场通量

        # 确保halo_farfield有正确的维度
        if halo_farfield.ndim == 2:  # (C, W)
            halo_farfield = halo_farfield.unsqueeze(0)  # (1, C, W)

        # 提取halo单元物理量（右侧状态，远场外侧）
        rho_R_far = halo_farfield[:, 0:1, :]
        u_R_far = halo_farfield[:, 1:2, :]
        v_R_far = halo_farfield[:, 2:3, :]
        p_R_far = halo_farfield[:, 3:4, :]

        # 提取物理单元物理量（左侧状态，最后一层物理单元j=H-1）
        rho_L_far = rho[..., -1:, :]
        u_L_far = u[..., -1:, :]
        v_L_far = v[..., -1:, :]
        p_L_far = p[..., -1:, :]

        # ADflow中央通量公式
        vnp_far = u_R_far * A_x_top + v_R_far * A_y_top
        vnm_far = u_L_far * A_x_top + v_L_far * A_y_top

        rqsp_far = 0.5 * rho_R_far * vnp_far
        rqsm_far = 0.5 * rho_L_far * vnm_far

        p_avg_far = 0.5 * (p_L_far + p_R_far)

        Fc_top = rqsp_far + rqsm_far
        Fmx_top = rqsp_far * u_R_far + rqsm_far * u_L_far + p_avg_far * A_x_top
        Fmy_top = rqsp_far * v_R_far + rqsm_far * v_L_far + p_avg_far * A_y_top

        # 能量方程：远场通量（如果提供rhoE）
        if rhoE is not None:
            # 提取远场rhoE状态
            rhoE_R_far = halo_farfield[:, 4:5, :] if halo_farfield.shape[1] > 4 else None
            rhoE_L_far = rhoE[..., -1:, :]

            # ✅ ADFLOW对齐：强制要求halo包含rhoE（5+通道）
            if rhoE_R_far is None:
                raise ValueError(
                    f"Farfield halo must include rhoE (5+ channels). "
                    f"Got halo_farfield.shape[1]={halo_farfield.shape[1]}, expected >= 5. "
                    f"Update torch residual backend to construct 5-channel halo."
                )

            # ✅ ADFLOW能量方程使用qsp（不含rho）
            # 参考 ADFLOW fluxes.F90:86, 126-129
            # ADFLOW: porVel = one * porFlux = 0.5 (fluxes.F90:81)
            sFace = 0.0
            porVel = 0.5  # ADFLOW标准: porVel = 1.0 * porFlux = 0.5
            porFlux_far = 0.5

            # 能量方程专用qsp（不含rho）
            qsp_far = (vnp_far - sFace) * porVel
            qsm_far = (vnm_far - sFace) * porVel

            FE_top = (
                qsp_far * rhoE_R_far +
                qsm_far * rhoE_L_far +
                porFlux_far * (vnp_far * p_R_far + vnm_far * p_L_far)
            )

    elif Ma is not None and AoA is not None:
        # 计算自由流状态（无量纲，ADflow convention）
        # 支持Ma/AoA为float或(B,) tensor

        # 转换Ma为tensor并广播为(batch, 1, 1)
        if isinstance(Ma, (int, float)):
            Ma_t = torch.full((batch, 1, 1), Ma, device=rho.device, dtype=rho.dtype)
        elif isinstance(Ma, torch.Tensor):
            if Ma.ndim == 0:  # 标量tensor
                Ma_t = Ma.view(1, 1, 1).expand(batch, 1, 1)
            elif Ma.shape == (batch,):  # (B,)
                Ma_t = Ma.view(batch, 1, 1)
            else:
                raise ValueError(f"Invalid Ma shape: {Ma.shape}, expected scalar or ({batch},)")
        else:
            Ma_t = torch.tensor(Ma, device=rho.device, dtype=rho.dtype).view(1, 1, 1).expand(batch, 1, 1)

        # 转换AoA为tensor并广播为(batch, 1, 1)
        if isinstance(AoA, (int, float)):
            AoA_rad = AoA * torch.pi / 180.0
            AoA_t = torch.full((batch, 1, 1), AoA_rad, device=rho.device, dtype=rho.dtype)
        elif isinstance(AoA, torch.Tensor):
            if AoA.ndim == 0:  # 标量tensor
                AoA_rad = AoA * torch.pi / 180.0
                AoA_t = AoA_rad.view(1, 1, 1).expand(batch, 1, 1)
            elif AoA.shape == (batch,):  # (B,)
                AoA_rad = AoA * torch.pi / 180.0
                AoA_t = AoA_rad.view(batch, 1, 1)
            else:
                raise ValueError(f"Invalid AoA shape: {AoA.shape}, expected scalar or ({batch},)")
        else:
            AoA_rad = AoA * torch.pi / 180.0
            AoA_t = torch.tensor(AoA_rad, device=rho.device, dtype=rho.dtype).view(1, 1, 1).expand(batch, 1, 1)

        # 自由流无量纲速度：u' = Ma * sqrt(gamma)
        gamma_t = torch.tensor(gamma, device=rho.device, dtype=rho.dtype)
        u_inf_nondim = Ma_t * torch.sqrt(gamma_t)  # (batch, 1, 1)

        # 速度分量 (batch, 1, 1)
        u_inf = u_inf_nondim * torch.cos(AoA_t)
        v_inf = u_inf_nondim * torch.sin(AoA_t)

        # 自由流状态（无量纲参考状态）
        rho_inf = torch.ones_like(p[..., -1:, :])  # ρ_inf = 1, (batch, 1, W)
        p_inf = torch.ones_like(p[..., -1:, :])    # p_inf = 1, (batch, 1, W)

        # 远场法向速度×面积（广播：(batch,1,1) × (batch,1,W) → (batch,1,W)）
        vnp_top = u_inf * A_x_top + v_inf * A_y_top

        # 远场通量（完整Riemann状态）
        Fc_top = rho_inf * vnp_top
        Fmx_top = rho_inf * u_inf * vnp_top + p_inf * A_x_top
        Fmy_top = rho_inf * v_inf * vnp_top + p_inf * A_y_top

        # 能量方程：远场通量（如果提供rhoE）
        if rhoE is not None:
            # 计算自由流rhoE_inf
            # rhoE_inf = p_inf/(gamma-1) + 0.5*rho_inf*(u_inf^2 + v_inf^2)
            kinetic_energy_inf = 0.5 * rho_inf * (u_inf**2 + v_inf**2)
            internal_energy_inf = p_inf / (gamma - 1.0)
            rhoE_inf = internal_energy_inf + kinetic_energy_inf

            # 能量通量（参考ADFLOW公式）
            FE_top = rhoE_inf * vnp_top + p_inf * vnp_top

    else:
        # 回退：如果未提供Ma/AoA，使用简化版本（仅压力，质量通量为0）
        # 注意：这会导致远场误差，应尽量提供Ma/AoA
        p_far = p[..., -1:, :]
        Fc_top = torch.zeros_like(A_x_top)
        Fmx_top = p_far * A_x_top
        Fmy_top = p_far * A_y_top

        # 能量方程：简化远场通量（如果提供rhoE）
        if rhoE is not None:
            # 简化：质量通量为0，只有压力项（但法向速度未知，近似为0）
            FE_top = torch.zeros_like(A_x_top)

    # 6. 拼接纯对流通量 (H+1, W)
    Fc_conv_full = _assemble_eta_face_stack(Fc_bot, Fc_interior_conv, Fc_top)  # (batch, H+1, W)
    Fmx_conv_full = _assemble_eta_face_stack(Fmx_bot, Fmx_interior_conv, Fmx_top)
    Fmy_conv_full = _assemble_eta_face_stack(Fmy_bot, Fmy_interior_conv, Fmy_top)

    # 能量方程：拼接能量对流通量（如果提供rhoE）
    if rhoE is not None:
        FE_conv_full = _assemble_eta_face_stack(FE_bot, FE_interior_conv, FE_top)  # (batch, H+1, W)

    # 6.1. 添加耗散（如果启用）：Fc_full = Fc_conv_full - D_full
    D_rho_full, D_rhou_full, D_rhov_full, D_rhoE_full = None, None, None, None
    if dissipation_mode == 'jameson':
        from . import dissipation as diss_module

        # ✅ 新增：如果提供rhoE，则独立计算能量耗散（ADFLOW对齐）
        dissipation_result = diss_module.compute_jameson_dissipation(
            rho, u, v, p, face_geom,
            direction='eta',
            vis2=vis2,
            vis4=vis4,
            gamma=gamma,
            dss_max=dss_max,
            periodic_xi=False,
            sslim=sslim,
            basis=basis,
            ss_halo=ss_halo,
            halo_wall=halo_wall,
            halo_farfield=halo_farfield,
            vol=vol,
            adis=adis,
            acoustic_scale_factor=acoustic_scale_factor,
            rhoE=rhoE,  # ✅ 传入rhoE参数
            lumped_dissipation=lumped_dissipation,
            lumped_sigma=lumped_sigma,
            frozen_shock_sensor=frozen_shock_sensor,
            frozen_ss_halo=frozen_ss_halo,
            use_dissipation_continuation=use_dissipation_continuation,
            diss_cont_magnitude=diss_cont_magnitude,
            diss_cont_midpoint=diss_cont_midpoint,
            diss_cont_sharpness=diss_cont_sharpness,
            diss_cont_total_r=diss_cont_total_r,
            diss_cont_total_r0=diss_cont_total_r0,
            rfil=diss_cont_rfil,
            precomputed_cell_radius=precomputed_cell_radius,
        )

        # ✅ 解包返回值（3或4个，取决于是否提供rhoE）
        if rhoE is not None:
            D_rho_tmp, D_rhou_tmp, D_rhov_tmp, D_rhoE_tmp = dissipation_result
        else:
            D_rho_tmp, D_rhou_tmp, D_rhov_tmp = dissipation_result
            D_rhoE_tmp = None

        # 兼容：如果返回的是内部面 (H-1, W)，则补零为 (H+1, W)
        if D_rho_tmp.shape[-2] == H - 1:
            D_rho_full = _pad_eta_face_stack_with_zeros(D_rho_tmp)
            D_rhou_full = _pad_eta_face_stack_with_zeros(D_rhou_tmp)
            D_rhov_full = _pad_eta_face_stack_with_zeros(D_rhov_tmp)
            # ✅ 能量耗散：直接补零（ADFLOW对齐，独立计算）
            if D_rhoE_tmp is not None:
                D_rhoE_full = _pad_eta_face_stack_with_zeros(D_rhoE_tmp)
        else:
            D_rho_full, D_rhou_full, D_rhov_full = D_rho_tmp, D_rhou_tmp, D_rhov_tmp
            # ✅ 能量耗散：直接赋值（ADFLOW对齐，独立计算）
            if D_rhoE_tmp is not None:
                D_rhoE_full = D_rhoE_tmp

        Fc_full = Fc_conv_full - D_rho_full
        Fmx_full = Fmx_conv_full - D_rhou_full
        Fmy_full = Fmy_conv_full - D_rhov_full
        if rhoE is not None:
            FE_full = FE_conv_full - D_rhoE_full

    elif dissipation_mode == 'none':
        Fc_full = Fc_conv_full
        Fmx_full = Fmx_conv_full
        Fmy_full = Fmy_conv_full
        if rhoE is not None:
            FE_full = FE_conv_full

    else:
        raise ValueError(f"Unknown dissipation_mode: {dissipation_mode}. Choose 'jameson' or 'none'.")

    # ========== DEBUG: 特定单元η面通量 ==========
    if debug:
        print(f"\n[DEBUG fluxes.py] η面通量特定单元（尾缘壁面，对应ADflow i=293..297）:")
        print(f"  PyTorch索引: 面j=0..2, i=291..295")
        print(f"  Fc_full shape: {Fc_full.shape}")
        # 面面积向量
        print(f"  A_x_eta[0:3, 291:296]:")
        print(f"    {A_x_full[0, 0:3, 291:296].cpu().numpy()}")
        print(f"  A_y_eta[0:3, 291:296]:")
        print(f"    {A_y_full[0, 0:3, 291:296].cpu().numpy()}")
        # 面通量
        print(f"  Fc_eta[0:3, 291:296] (质量通量):")
        print(f"    {Fc_full[0, 0:3, 291:296].cpu().numpy()}")
        print(f"  壁面面(j=0)应为0: Fc_bot[0, 0, 291:296]={Fc_bot[0, 0, 291:296].cpu().numpy()}")
        # 内部面面积模长
        A_eta_mag = torch.sqrt(A_x_full**2 + A_y_full**2)
        print(f"  |A_eta|[0:3, 291:296]:")
        print(f"    {A_eta_mag[0, 0:3, 291:296].cpu().numpy()}")
    # ========== END DEBUG ==========

    # ========== DEBUG: Save full eta flux to npz file ==========
    import os
    if os.environ.get('SURROGATE_DEBUG_FLUX', '') == '1':
        import numpy as np
        debug_data = {
            # 完整 eta 总通量 (H+1, W) - 包含壁面和远场
            'Fc_full': Fc_full[0].detach().cpu().numpy(),
            'Fmx_full': Fmx_full[0].detach().cpu().numpy(),
            'Fmy_full': Fmy_full[0].detach().cpu().numpy(),

            # 壁面通量 (1, W)
            'Fc_bot': Fc_bot[0].detach().cpu().numpy(),
            'Fmx_bot': Fmx_bot[0].detach().cpu().numpy(),
            'Fmy_bot': Fmy_bot[0].detach().cpu().numpy(),

            # 内部面通量 (H-1, W) - 从完整数组截取 (j=1..H-1)
            'Fc_interior': Fc_full[0, 1:-1, :].detach().cpu().numpy(),
            'Fmx_interior': Fmx_full[0, 1:-1, :].detach().cpu().numpy(),
            'Fmy_interior': Fmy_full[0, 1:-1, :].detach().cpu().numpy(),

            # 远场通量 (1, W)
            'Fc_top': Fc_top[0].detach().cpu().numpy(),
            'Fmx_top': Fmx_top[0].detach().cpu().numpy(),
            'Fmy_top': Fmy_top[0].detach().cpu().numpy(),

            # 面积向量
            'A_x_full': A_x_full[0].detach().cpu().numpy(),
            'A_y_full': A_y_full[0].detach().cpu().numpy(),
            'A_x_bot': A_x_bot[0].detach().cpu().numpy(),
            'A_y_bot': A_y_bot[0].detach().cpu().numpy(),

            # 壁面压力 (1, W) - p_wall 总是存在
            'p_wall': p_wall[0].detach().cpu().numpy(),

            # 维度信息
            'H': H,
            'W': W,
        }

        # 添加耗散通量和对流通量（如果启用）
        if dissipation_mode == 'jameson' and D_rho_full is not None:
            debug_data['D_rho_full'] = D_rho_full[0].detach().cpu().numpy()
            debug_data['D_rhou_full'] = D_rhou_full[0].detach().cpu().numpy()
            debug_data['D_rhov_full'] = D_rhov_full[0].detach().cpu().numpy()
            # 对流通量 = 总通量 + 耗散通量（因为 Fc = Fc_conv - D_rho）
            debug_data['Fc_conv_full'] = Fc_conv_full[0].detach().cpu().numpy()
            debug_data['Fmx_conv_full'] = Fmx_conv_full[0].detach().cpu().numpy()
            debug_data['Fmy_conv_full'] = Fmy_conv_full[0].detach().cpu().numpy()
        # 如果使用了 halo_wall，还保存 p_halo 和 p_avg
        if halo_wall is not None:
            debug_data['p_halo'] = p_halo[0].detach().cpu().numpy()
            debug_data['p_avg'] = p_avg[0].detach().cpu().numpy()

        # 添加能量通量（如果存在）
        if rhoE is not None and FE_full is not None:
            debug_data['FE_full'] = FE_full[0].detach().cpu().numpy()
            debug_data['FE_conv_full'] = FE_conv_full[0].detach().cpu().numpy()
            if D_rhoE_full is not None:
                debug_data['D_rhoE_full'] = D_rhoE_full[0].detach().cpu().numpy()
            print(f"        Energy flux (FE_full) included in debug data")

        np.savez('pytorch_eta_flux_debug.npz', **debug_data)
        print(f"[DEBUG] Saved full eta flux to: pytorch_eta_flux_debug.npz")
        print(f"        Full flux shape: ({H+1}, {W}), Wall: (1,{W}), Interior: ({H-1},{W})")
    # ========== END DEBUG ==========

    # 6.5. return_dissipation 时，D_full 已经构造好（可能包含 farfield 边界面耗散）

    # 7. 移除batch维度（如果输入没有）
    if squeeze_output:
        Fc_full = Fc_full.squeeze(0)
        Fmx_full = Fmx_full.squeeze(0)
        Fmy_full = Fmy_full.squeeze(0)
        Fc_conv_full = Fc_conv_full.squeeze(0)
        Fmx_conv_full = Fmx_conv_full.squeeze(0)
        Fmy_conv_full = Fmy_conv_full.squeeze(0)
        if rhoE is not None:
            FE_full = FE_full.squeeze(0)
            FE_conv_full = FE_conv_full.squeeze(0)
        if D_rho_full is not None:
            D_rho_full = D_rho_full.squeeze(0)
            D_rhou_full = D_rhou_full.squeeze(0)
            D_rhov_full = D_rhov_full.squeeze(0)
            if D_rhoE_full is not None:
                D_rhoE_full = D_rhoE_full.squeeze(0)

    # 8. 返回结果（支持3方程和4方程模式）
    if rhoE is not None:
        # 4方程模式：包含能量通量
        if return_dissipation:
            return (
                Fc_full, Fmx_full, Fmy_full, FE_full,
                D_rho_full, D_rhou_full, D_rhov_full, D_rhoE_full,
                Fc_conv_full, Fmx_conv_full, Fmy_conv_full, FE_conv_full,
            )
        return Fc_full, Fmx_full, Fmy_full, FE_full
    else:
        # 3方程模式：仅动量通量
        if return_dissipation:
            return (
                Fc_full, Fmx_full, Fmy_full,
                D_rho_full, D_rhou_full, D_rhov_full,
                Fc_conv_full, Fmx_conv_full, Fmy_conv_full,
            )
        return Fc_full, Fmx_full, Fmy_full


# ========== Phase 3: 黏性通量（Viscous Flux）==========

def viscous_flux(
    u: torch.Tensor,
    v: torch.Tensor,
    du_dx: torch.Tensor,
    du_dy: torch.Tensor,
    dv_dx: torch.Tensor,
    dv_dy: torch.Tensor,
    mu_eff: Dict[str, torch.Tensor],  # 方案B：字典{'mu_lam', 'mu_turb', 'mu_eff'}
    face_geom: Dict[str, torch.Tensor],
    direction: str = 'xi',
    normal_correction: str = 'adflow',
    Ma: Optional[float] = None,
    AoA: Optional[float] = None,
    gamma: float = 1.4,
    # 新增：节点梯度参数（用于ADFLOW对齐的四节点平均）
    du_dx_node: Optional[torch.Tensor] = None,
    du_dy_node: Optional[torch.Tensor] = None,
    dv_dx_node: Optional[torch.Tensor] = None,
    dv_dy_node: Optional[torch.Tensor] = None,
    halo_wall: Optional[torch.Tensor] = None,
    halo_farfield: Optional[torch.Tensor] = None,
    use_nodal_gradients: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    黏性通量（geometry-based，统一几何，ADFLOW完全对齐）

    关键特性：
    1. 使用face_geom的geometry-based面向量（s·n）
    2. 替代当前的metric-based投影（yη, -xη）
    3. 保持ADflow风格的法向修正（g ← g - [g·n - Δφ/|d|]·n）
    4. **Plan39修复**：η方向强制使用完整边界面（H+1, W）
    5. **Plan86对齐**：强制使用ADFLOW节点梯度四节点平均

    黏性应力投影：
    - x动量：Vmx = s·(τ_xx·n_x + τ_xy·n_y)
    - y动量：Vmy = s·(τ_xy·n_x + τ_yy·n_y)

    梯度插值方法（强制）：
    - 四节点平均（ADFLOW标准）：du_dx_node参数必须提供
    - 参考：ADFLOW blockette.F90:5724-5727

    Args:
        u, v: 速度场 (batch, H, W) 或 (H, W)
        du_dx, du_dy, dv_dx, dv_dy: 速度梯度（cell-center，保留用于其他用途）
        mu_eff: 有效粘度（标量或场）
        face_geom: 面几何字典（η方向必须包含完整面几何）
        direction: 'xi' 或 'eta'
        normal_correction: 'adflow'（法向修正）或 'none'
        Ma, AoA: 自由流马赫数和攻角（度），用于远场边界条件
        gamma: 比热比（默认1.4）
        du_dx_node, du_dy_node, dv_dx_node, dv_dy_node: 节点梯度 (batch, H+1, W+1) 或 (H+1, W+1)
            - **必须提供**（通过NodalGradientCalculator计算）
            - 参考：ADFLOW blockette.F90:5724-5727
        use_nodal_gradients: 是否使用节点梯度（必须为True）

    Returns:
        (Vmx, Vmy): x/y动量的黏性通量
            - direction='xi': (H, W) if periodic else (H, W-1)
            - direction='eta': (H+1, W) 完整边界面（强制）

    Raises:
        ValueError: 如果未提供节点梯度或use_nodal_gradients=False
    """
    # 添加batch维度（如果没有）
    if u.ndim == 2:
        u = u.unsqueeze(0)
        v = v.unsqueeze(0)
        du_dx = du_dx.unsqueeze(0)
        du_dy = du_dy.unsqueeze(0)
        dv_dx = dv_dx.unsqueeze(0)
        dv_dy = dv_dy.unsqueeze(0)
        # 节点梯度batch处理
        if du_dx_node is not None:
            du_dx_node = du_dx_node.unsqueeze(0)
            du_dy_node = du_dy_node.unsqueeze(0)
            dv_dx_node = dv_dx_node.unsqueeze(0)
            dv_dy_node = dv_dy_node.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    # P2修复：强制转换为float64以提高近壁数值精度
    # 近壁小体积 + 大量相减（g·n - Δ/|d|）在float32下容易导致局部粘性通量虚高
    original_dtype = u.dtype
    u = u.to(torch.float64)
    v = v.to(torch.float64)
    du_dx = du_dx.to(torch.float64)
    du_dy = du_dy.to(torch.float64)
    dv_dx = dv_dx.to(torch.float64)
    dv_dy = dv_dy.to(torch.float64)
    if du_dx_node is not None:
        du_dx_node = du_dx_node.to(torch.float64)
        du_dy_node = du_dy_node.to(torch.float64)
        dv_dx_node = dv_dx_node.to(torch.float64)
        dv_dy_node = dv_dy_node.to(torch.float64)

    # 提取面几何信息（P0修复：双通道法向 + 面面积向量）
    if direction == 'xi':
        # 分隔法向（用于法向修正Δφ/|d|）
        ssx_sep = face_geom['ssx_xi'].to(torch.float64)
        ssy_sep = face_geom['ssy_xi'].to(torch.float64)
        inv_d = face_geom['inv_d_xi'].to(torch.float64)
        # 面面积向量（metric-based，用于通量投影）
        A_x = face_geom['A_x_xi'].to(torch.float64)
        A_y = face_geom['A_y_xi'].to(torch.float64)
        periodic = face_geom.get('periodic_xi', False)
    elif direction == 'eta':
        # 强制使用完整η边界面（H+1, W）
        if 'A_x_eta' not in face_geom:
            raise ValueError(
                "viscous_flux with direction='eta' requires face_geom to contain "
                "'A_x_eta', 'ssx_eta', etc. with shape (H+1, W) for complete eta faces. "
                "Ensure compute_face_geometry includes full eta faces."
            )
        A_x = face_geom['A_x_eta'].to(torch.float64)
        A_y = face_geom['A_y_eta'].to(torch.float64)
        ssx_sep = face_geom['ssx_eta'].to(torch.float64)
        ssy_sep = face_geom['ssy_eta'].to(torch.float64)
        inv_d = face_geom['inv_d_eta'].to(torch.float64)
        periodic = False
    else:
        raise ValueError(f"Invalid direction: {direction}")

    # 确保面几何有batch维度
    if ssx_sep.ndim == 2:
        ssx_sep = ssx_sep.unsqueeze(0)
        ssy_sep = ssy_sep.unsqueeze(0)
        A_x = A_x.unsqueeze(0)
        A_y = A_y.unsqueeze(0)
        inv_d = inv_d.unsqueeze(0)

    # 1. 将梯度插值到面中心
    # **Plan86对齐**：强制使用节点梯度四节点平均（ADFLOW标准）
    if not use_nodal_gradients or du_dx_node is None:
        raise ValueError(
            "viscous_flux requires nodal gradients (du_dx_node, du_dy_node, dv_dx_node, dv_dy_node). "
            "Nodal gradients must be computed using NodalGradientCalculator (method='nodal'). "
            "Cell-center gradient fallback has been removed to ensure ADFLOW alignment."
        )

    # 节点梯度四节点平均（ADFLOW blockette.F90:5724-5727）
    if direction == 'xi':
        du_dx_face, du_dy_face = _interpolate_nodal_to_face_xi(
            du_dx_node, du_dy_node, periodic=periodic
        )
        dv_dx_face, dv_dy_face = _interpolate_nodal_to_face_xi(
            dv_dx_node, dv_dy_node, periodic=periodic
        )
    elif direction == 'eta':
        du_dx_face, du_dy_face = _interpolate_nodal_to_face_eta(
            du_dx_node, du_dy_node
        )
        dv_dx_face, dv_dy_face = _interpolate_nodal_to_face_eta(
            dv_dx_node, dv_dy_node
        )

    import os
    debug_gradient = os.environ.get('SURROGATE_DEBUG_GRADIENT', '') == '1'
    if debug_gradient and direction == 'eta':
        du_dx_face_before = du_dx_face.clone()
        du_dy_face_before = du_dy_face.clone()
        dv_dx_face_before = dv_dx_face.clone()
        dv_dy_face_before = dv_dy_face.clone()

    # 2. 法向修正（ADflow风格）
    if normal_correction == 'adflow':
        # 计算相邻单元中心的速度差分
        if direction == 'xi':
            if periodic:
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

        elif direction == 'eta':
            # **Plan39修复**：强制使用完整η边界面（H+1）
            # 内部面j=1..H-1
            Delta_u_interior = u[..., 1:, :] - u[..., :-1, :]  # (H-1, W)
            Delta_v_interior = v[..., 1:, :] - v[..., :-1, :]

            # 边界面j=0（壁面）：优先使用真实halo状态，
            # 对齐ADFLOW viscousFlux中的 w(j+1) - w(j)。
            if halo_wall is not None:
                if halo_wall.ndim == 2:
                    halo_u_wall = halo_wall[1:2, :]
                    halo_v_wall = halo_wall[2:3, :]
                else:
                    halo_u_wall = halo_wall[:, 1:2, :]
                    halo_v_wall = halo_wall[:, 2:3, :]
                Delta_u_bot = u[..., 0:1, :] - halo_u_wall
                Delta_v_bot = v[..., 0:1, :] - halo_v_wall
            else:
                # 回退到无滑移反射: ghost_u = -u[0], ghost_v = -v[0]
                Delta_u_bot = 2.0 * u[..., 0:1, :]
                Delta_v_bot = 2.0 * v[..., 0:1, :]

            # 边界面j=H（远场）：优先使用真实farfield halo。
            if halo_farfield is not None:
                if halo_farfield.ndim == 2:
                    halo_u_top = halo_farfield[1:2, :]
                    halo_v_top = halo_farfield[2:3, :]
                else:
                    halo_u_top = halo_farfield[:, 1:2, :]
                    halo_v_top = halo_farfield[:, 2:3, :]
                Delta_u_top = halo_u_top - u[..., -1:, :]
                Delta_v_top = halo_v_top - v[..., -1:, :]
            else:
                # 回退到自由流状态，保持旧路径可用。
                if Ma is None or AoA is None:
                    raise ValueError(
                        "viscous_flux with direction='eta' requires Ma/AoA or halo_farfield "
                        "for correct far-field normal correction."
                    )

                if isinstance(AoA, torch.Tensor):
                    if AoA.ndim == 0:
                        AoA_rad = AoA * torch.pi / 180.0
                    elif AoA.ndim == 1:
                        AoA_rad = (AoA * torch.pi / 180.0).view(-1, 1, 1)
                    else:
                        raise ValueError(f"Invalid AoA shape: {AoA.shape}, expected scalar or (B,)")
                else:
                    AoA_rad = torch.tensor(AoA * torch.pi / 180.0, device=u.device, dtype=u.dtype)

                if isinstance(Ma, torch.Tensor):
                    if Ma.ndim == 0:
                        Ma_t = Ma
                    elif Ma.ndim == 1:
                        Ma_t = Ma.view(-1, 1, 1)
                    else:
                        raise ValueError(f"Invalid Ma shape: {Ma.shape}, expected scalar or (B,)")
                else:
                    Ma_t = torch.tensor(Ma, device=u.device, dtype=u.dtype)

                sqrt_gamma = torch.sqrt(torch.tensor(gamma, device=u.device, dtype=u.dtype))
                u_inf = Ma_t * sqrt_gamma * torch.cos(AoA_rad)
                v_inf = Ma_t * sqrt_gamma * torch.sin(AoA_rad)
                Delta_u_top = u_inf - u[..., -1:, :]
                Delta_v_top = v_inf - v[..., -1:, :]

            # 拼接：[j=0, j=1..H-1, j=H]
            Delta_u = torch.cat([Delta_u_bot, Delta_u_interior, Delta_u_top], dim=-2)
            Delta_v = torch.cat([Delta_v_bot, Delta_v_interior, Delta_v_top], dim=-2)

        # 法向修正：g ← g - [g·n - Δφ/|d|]·n（使用分隔法向）
        # 对u的梯度
        g_u = torch.stack([du_dx_face, du_dy_face], dim=0)  # (2, batch, ...)
        n_sep = torch.stack([ssx_sep, ssy_sep], dim=0)  # (2, batch, ...) 分隔法向

        g_dot_n_u = (g_u * n_sep).sum(dim=0)  # (batch, ...)
        corr_u = g_dot_n_u - Delta_u * inv_d

        du_dx_face = du_dx_face - corr_u * ssx_sep
        du_dy_face = du_dy_face - corr_u * ssy_sep

        # 对v的梯度
        g_v = torch.stack([dv_dx_face, dv_dy_face], dim=0)
        g_dot_n_v = (g_v * n_sep).sum(dim=0)
        corr_v = g_dot_n_v - Delta_v * inv_d

        dv_dx_face = dv_dx_face - corr_v * ssx_sep
        dv_dy_face = dv_dy_face - corr_v * ssy_sep

        if debug_gradient and direction == 'eta':
            import numpy as np

            def _to_numpy(arr: torch.Tensor) -> np.ndarray:
                if arr.ndim == 3:
                    arr = arr[0]
                return arr.detach().cpu().numpy()

            np.savez(
                'pytorch_face_gradient_debug.npz',
                face_ux_before=_to_numpy(du_dx_face_before),
                face_uy_before=_to_numpy(du_dy_face_before),
                face_vx_before=_to_numpy(dv_dx_face_before),
                face_vy_before=_to_numpy(dv_dy_face_before),
                face_ux_after=_to_numpy(du_dx_face),
                face_uy_after=_to_numpy(du_dy_face),
                face_vx_after=_to_numpy(dv_dx_face),
                face_vy_after=_to_numpy(dv_dy_face),
            )
            np.savez(
                'pytorch_normal_correction_debug.npz',
                g_dot_n_u=_to_numpy(g_dot_n_u),
                delta_u=_to_numpy(Delta_u),
                corr_u=_to_numpy(corr_u),
                g_dot_n_v=_to_numpy(g_dot_n_v),
                delta_v=_to_numpy(Delta_v),
                corr_v=_to_numpy(corr_v),
            )
            print("[DEBUG] Saved face gradients to: pytorch_face_gradient_debug.npz")
            print("[DEBUG] Saved normal correction intermediates to: pytorch_normal_correction_debug.npz")

    # 3. 计算应力张量（Stokes假设）
    div_u = du_dx_face + dv_dy_face

    # **方案B修复**：字典模式（包含mu_lam和mu_turb分离组分）
    # 对齐ADflow BCRoutines.F90:544（壁面面上涡粘为0）
    mu_lam = mu_eff['mu_lam']
    mu_turb = mu_eff['mu_turb']
    mu_eff_total = mu_eff['mu_eff']
    if isinstance(mu_lam, torch.Tensor):
        mu_lam = mu_lam.to(torch.float64)
    if isinstance(mu_turb, torch.Tensor):
        mu_turb = mu_turb.to(torch.float64)
    if isinstance(mu_eff_total, torch.Tensor):
        mu_eff_total = mu_eff_total.to(torch.float64)

    # 确保有batch维度
    if isinstance(mu_lam, torch.Tensor) and mu_lam.ndim == 2:
        mu_lam = mu_lam.unsqueeze(0)
    if mu_eff_total.ndim == 2:
        mu_turb = mu_turb.unsqueeze(0)
        mu_eff_total = mu_eff_total.unsqueeze(0)

    if direction == 'xi':
        # ξ方向：使用完整粘度（层流+湍流）
        if periodic:
            mu_face = torch.cat([
                0.5 * (mu_eff_total[..., :, :-1] + mu_eff_total[..., :, 1:]),
                0.5 * (mu_eff_total[..., :, -1:] + mu_eff_total[..., :, :1])
            ], dim=-1)
        else:
            mu_face = 0.5 * (mu_eff_total[..., :, :-1] + mu_eff_total[..., :, 1:])
    elif direction == 'eta':
        # **方案B关键修复**：η方向壁面边界仅使用层流粘度
        # 内部面j=1..H-1：完整粘度（层流+湍流）
        mu_interior = 0.5 * (mu_eff_total[..., :-1, :] + mu_eff_total[..., 1:, :])  # (H-1, W)

        # 壁面边界j=0：仅层流粘度（对齐ADflow BCRoutines.F90:544：rev_halo=-rev_cell使面上涡粘为0）
        # mu_lam 既可能是标量/批量常值，也可能已经是 Sutherland 空间场。
        mu_bot_template = mu_eff_total[..., 0:1, :]  # (1, W) 或 (B, 1, W)
        if (isinstance(mu_lam, torch.Tensor)
                and mu_lam.ndim == mu_eff_total.ndim
                and mu_lam.shape[-2:] == mu_eff_total.shape[-2:]):
            mu_bot = mu_lam[..., 0:1, :]
        else:
            mu_bot = torch.ones_like(mu_bot_template) * mu_lam  # 广播处理标量/(B,)/(B,1,1)

        # 远场边界j=H：完整粘度
        mu_top = mu_eff_total[..., -1:, :]  # (1, W) 或 (B, 1, W)

        # 拼接
        mu_face = torch.cat([mu_bot, mu_interior, mu_top], dim=-2)

    tau_xx = mu_face * (2.0 * du_dx_face - (2.0 / 3.0) * div_u)
    tau_yy = mu_face * (2.0 * dv_dy_face - (2.0 / 3.0) * div_u)
    tau_xy = mu_face * (du_dy_face + dv_dx_face)

    # 4. **P0修复完成**：使用面面积向量A直接投影
    # x动量黏性通量：Vmx = τ_xx·A_x + τ_xy·A_y
    # 与原来的 s·(τ·n) 等价，但A = s·n来自metric-based，与体积同源
    Vmx = tau_xx * A_x + tau_xy * A_y

    # y动量黏性通量：Vmy = τ_xy·A_x + τ_yy·A_y
    Vmy = tau_xy * A_x + tau_yy * A_y

    # 移除batch维度（如果输入没有）
    if squeeze_output:
        Vmx = Vmx.squeeze(0)
        Vmy = Vmy.squeeze(0)

    # P2修复：转回原始dtype
    Vmx = Vmx.to(original_dtype)
    Vmy = Vmy.to(original_dtype)

    # ========== DEBUG: 详细中间量调试 (用于ADFLOW对齐) ==========
    import os
    if os.environ.get('SURROGATE_DEBUG_VISCOUS_DETAIL', '') == '1' and direction == 'eta':
        # 只对eta方向输出（近壁粘性通量主要问题在eta方向）
        import numpy as np

        # Top-10热点 (j, i) - 来自logs/log_mom_debug_adflow_gradient.txt
        hotspots = [
            (1, 82), (0, 66), (0, 65), (0, 64), (2, 68),
            (0, 82), (0, 68), (1, 81), (0, 86), (0, 103)
        ]

        debug_data = {}

        # 移除batch维度以便索引
        def _collapse_debug_plane(arr: torch.Tensor) -> torch.Tensor:
            out = arr
            while out.ndim > 2:
                out = out[0]
            return out

        du_dx_node_2d = _collapse_debug_plane(du_dx_node)
        du_dy_node_2d = _collapse_debug_plane(du_dy_node)
        dv_dx_node_2d = _collapse_debug_plane(dv_dx_node)
        dv_dy_node_2d = _collapse_debug_plane(dv_dy_node)

        # Face梯度（插值后，修正前）
        if du_dx_face.ndim == 3:
            du_dx_face_2d = du_dx_face.squeeze(0)
            du_dy_face_2d = du_dy_face.squeeze(0)
            dv_dx_face_2d = dv_dx_face.squeeze(0)
            dv_dy_face_2d = dv_dy_face.squeeze(0)
        else:
            du_dx_face_2d = du_dx_face
            du_dy_face_2d = du_dy_face
            dv_dx_face_2d = dv_dx_face
            dv_dy_face_2d = dv_dy_face

        # 其他中间量
        if g_dot_n_u.ndim == 3:
            g_dot_n_u_2d = g_dot_n_u.squeeze(0)
            g_dot_n_v_2d = g_dot_n_v.squeeze(0)
            corr_u_2d = corr_u.squeeze(0)
            corr_v_2d = corr_v.squeeze(0)
            Delta_u_2d = Delta_u.squeeze(0)
            Delta_v_2d = Delta_v.squeeze(0)
            inv_d_2d = inv_d.squeeze(0)
            ssx_sep_2d = ssx_sep.squeeze(0)
            ssy_sep_2d = ssy_sep.squeeze(0)
            mu_face_2d = mu_face.squeeze(0)
            tau_xx_2d = tau_xx.squeeze(0)
            tau_xy_2d = tau_xy.squeeze(0)
            tau_yy_2d = tau_yy.squeeze(0)
            A_x_2d = A_x.squeeze(0)
            A_y_2d = A_y.squeeze(0)
            Vmx_2d = Vmx.squeeze(0) if Vmx.ndim == 3 else Vmx
            Vmy_2d = Vmy.squeeze(0) if Vmy.ndim == 3 else Vmy
        else:
            g_dot_n_u_2d = g_dot_n_u
            g_dot_n_v_2d = g_dot_n_v
            corr_u_2d = corr_u
            corr_v_2d = corr_v
            Delta_u_2d = Delta_u
            Delta_v_2d = Delta_v
            inv_d_2d = inv_d
            ssx_sep_2d = ssx_sep
            ssy_sep_2d = ssy_sep
            mu_face_2d = mu_face
            tau_xx_2d = tau_xx
            tau_xy_2d = tau_xy
            tau_yy_2d = tau_yy
            A_x_2d = A_x
            A_y_2d = A_y
            Vmx_2d = Vmx
            Vmy_2d = Vmy

        for j, i in hotspots:
            # 对每个cell，输出其两个eta-face (j和j+1) 的详细中间量
            for face_offset in [0, 1]:
                j_face = j + face_offset
                key_prefix = f'cell_j{j}_i{i}_face_j{j_face}'

                # 1. 节点梯度 (4个节点) - 在face (j_face, i)
                # 节点索引: (j_face, i), (j_face, i+1), (j_face+1, i), (j_face+1, i+1)
                # 但注意：NodeGradientCalculator输出shape (H+1, W+1)
                debug_data[f'{key_prefix}_du_dx_node_0_i'] = du_dx_node_2d[j_face, i].item()
                debug_data[f'{key_prefix}_du_dx_node_0_ip1'] = du_dx_node_2d[j_face, i+1].item()
                debug_data[f'{key_prefix}_du_dx_node_1_i'] = du_dx_node_2d[j_face+1, i].item()
                debug_data[f'{key_prefix}_du_dx_node_1_ip1'] = du_dx_node_2d[j_face+1, i+1].item()

                debug_data[f'{key_prefix}_du_dy_node_0_i'] = du_dy_node_2d[j_face, i].item()
                debug_data[f'{key_prefix}_du_dy_node_0_ip1'] = du_dy_node_2d[j_face, i+1].item()
                debug_data[f'{key_prefix}_du_dy_node_1_i'] = du_dy_node_2d[j_face+1, i].item()
                debug_data[f'{key_prefix}_du_dy_node_1_ip1'] = du_dy_node_2d[j_face+1, i+1].item()

                debug_data[f'{key_prefix}_dv_dx_node_0_i'] = dv_dx_node_2d[j_face, i].item()
                debug_data[f'{key_prefix}_dv_dx_node_0_ip1'] = dv_dx_node_2d[j_face, i+1].item()
                debug_data[f'{key_prefix}_dv_dx_node_1_i'] = dv_dx_node_2d[j_face+1, i].item()
                debug_data[f'{key_prefix}_dv_dx_node_1_ip1'] = dv_dx_node_2d[j_face+1, i+1].item()

                debug_data[f'{key_prefix}_dv_dy_node_0_i'] = dv_dy_node_2d[j_face, i].item()
                debug_data[f'{key_prefix}_dv_dy_node_0_ip1'] = dv_dy_node_2d[j_face, i+1].item()
                debug_data[f'{key_prefix}_dv_dy_node_1_i'] = dv_dy_node_2d[j_face+1, i].item()
                debug_data[f'{key_prefix}_dv_dy_node_1_ip1'] = dv_dy_node_2d[j_face+1, i+1].item()

                # 2. Face平均后的梯度（四节点平均，修正前）
                # 注意：这些是修正前的值，需要从viscous_flux函数开始时保存
                # 由于我们在normal_correction之后，这里的du_dx_face已经是修正后的
                # 所以我们标记为"corrected"
                debug_data[f'{key_prefix}_du_dx_face_corrected'] = du_dx_face_2d[j_face, i].item()
                debug_data[f'{key_prefix}_du_dy_face_corrected'] = du_dy_face_2d[j_face, i].item()
                debug_data[f'{key_prefix}_dv_dx_face_corrected'] = dv_dx_face_2d[j_face, i].item()
                debug_data[f'{key_prefix}_dv_dy_face_corrected'] = dv_dy_face_2d[j_face, i].item()

                # 3. Normal correction中间量
                debug_data[f'{key_prefix}_g_dot_n_u'] = g_dot_n_u_2d[j_face, i].item()
                debug_data[f'{key_prefix}_g_dot_n_v'] = g_dot_n_v_2d[j_face, i].item()
                debug_data[f'{key_prefix}_Delta_u'] = Delta_u_2d[j_face, i].item()
                debug_data[f'{key_prefix}_Delta_v'] = Delta_v_2d[j_face, i].item()
                debug_data[f'{key_prefix}_inv_d'] = inv_d_2d[j_face, i].item()
                debug_data[f'{key_prefix}_corr_u'] = corr_u_2d[j_face, i].item()
                debug_data[f'{key_prefix}_corr_v'] = corr_v_2d[j_face, i].item()

                # 4. Separation stencil
                debug_data[f'{key_prefix}_ssx_sep'] = ssx_sep_2d[j_face, i].item()
                debug_data[f'{key_prefix}_ssy_sep'] = ssy_sep_2d[j_face, i].item()

                # 5. 粘度
                debug_data[f'{key_prefix}_mu_face'] = mu_face_2d[j_face, i].item()
                # 层流粘度（标量或空间场）
                if isinstance(mu_lam, torch.Tensor):
                    mu_lam_debug = mu_lam
                    while mu_lam_debug.ndim > 2:
                        mu_lam_debug = mu_lam_debug[0]
                    if mu_lam_debug.ndim == 0:
                        debug_data[f'{key_prefix}_mu_lam'] = mu_lam_debug.item()
                    elif mu_lam_debug.ndim == 1:
                        debug_data[f'{key_prefix}_mu_lam'] = mu_lam_debug[min(i, mu_lam_debug.shape[0] - 1)].item()
                    else:
                        debug_data[f'{key_prefix}_mu_lam'] = mu_lam_debug[
                            min(j_face, mu_lam_debug.shape[0] - 1),
                            min(i, mu_lam_debug.shape[1] - 1),
                        ].item()
                else:
                    debug_data[f'{key_prefix}_mu_lam'] = float(mu_lam)

                # 6. 应力分量
                debug_data[f'{key_prefix}_tau_xx'] = tau_xx_2d[j_face, i].item()
                debug_data[f'{key_prefix}_tau_xy'] = tau_xy_2d[j_face, i].item()
                debug_data[f'{key_prefix}_tau_yy'] = tau_yy_2d[j_face, i].item()

                # 7. 面向量
                debug_data[f'{key_prefix}_A_x'] = A_x_2d[j_face, i].item()
                debug_data[f'{key_prefix}_A_y'] = A_y_2d[j_face, i].item()
                debug_data[f'{key_prefix}_s_mag'] = (A_x_2d[j_face, i]**2 + A_y_2d[j_face, i]**2).sqrt().item()

                # 8. 最终flux
                debug_data[f'{key_prefix}_Vmx'] = Vmx_2d[j_face, i].item()
                debug_data[f'{key_prefix}_Vmy'] = Vmy_2d[j_face, i].item()

        # 保存到npz文件
        np.savez('pytorch_viscous_detail_debug.npz', **debug_data)
        print(f"\n[DEBUG] Saved detailed viscous flux debug data to pytorch_viscous_detail_debug.npz")
        print(f"        {len(hotspots)} hotspots × 2 faces/cell × ~30 quantities = {len(debug_data)} values")

    return Vmx, Vmy


# ========== 便捷函数：完整通量计算 ==========

def compute_all_fluxes(
    rho: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    du_dx: torch.Tensor,
    du_dy: torch.Tensor,
    dv_dx: torch.Tensor,
    dv_dy: torch.Tensor,
    face_geom: Dict[str, torch.Tensor],
    mu_eff: Optional[Union[float, torch.Tensor, Dict]] = None,
    include_viscous: bool = False,
    Ma: Optional[float] = None,
    AoA: Optional[float] = None,
    gamma: float = 1.4,
    dissipation_mode: str = 'jameson',
    vis2: float = 0.25,
    vis4: float = 0.0156,
    dss_max: float = 0.25,
    sslim: float = 1e-3,
    debug: bool = False,
    halo_wall: Optional[torch.Tensor] = None,
    halo_farfield: Optional[torch.Tensor] = None,
    return_dissipation: bool = False,
    basis: str = 'entropy',
    ss_halo: Optional[torch.Tensor] = None,
    gradient_method: str = 'green_gauss',
    volumes: Optional[torch.Tensor] = None,
    periodic_xi: bool = True,
    # 新增：节点梯度参数（Plan86对齐）
    du_dx_node: Optional[torch.Tensor] = None,
    du_dy_node: Optional[torch.Tensor] = None,
    dv_dx_node: Optional[torch.Tensor] = None,
    dv_dy_node: Optional[torch.Tensor] = None,
    use_nodal_gradients: bool = True,
    vol: Optional[torch.Tensor] = None,
    # 新增：Jameson耗散参数（与ADflow对齐）
    adis: float = 0.67,
    acoustic_scale_factor: float = 1.0,
    # 新增：能量方程支持（阶段3）
    rhoE: Optional[torch.Tensor] = None,
    daa_dx_node: Optional[torch.Tensor] = None,
    daa_dy_node: Optional[torch.Tensor] = None,
    dp_dx_node: Optional[torch.Tensor] = None,
    dp_dy_node: Optional[torch.Tensor] = None,
    drho_dx_node: Optional[torch.Tensor] = None,
    drho_dy_node: Optional[torch.Tensor] = None,
    Pr_laminar: float = 0.72,
    Pr_turbulent: float = 0.9,
    lumped_dissipation: bool = False,
    lumped_sigma: float = 1.0,
    frozen_shock_sensor: Optional[torch.Tensor] = None,
    frozen_ss_halo: Optional[torch.Tensor] = None,
    use_dissipation_continuation: bool = False,
    diss_cont_magnitude: float = 0.0,
    diss_cont_midpoint: float = 20.0,
    diss_cont_sharpness: float = 3.0,
    diss_cont_total_r: Optional[float] = None,
    diss_cont_total_r0: Optional[float] = None,
    diss_cont_rfil: float = 1.0,
    approximate_viscous_operator: bool = False,
    halo_nu_tilde_farfield: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    计算所有方向的对流+黏性通量（统一使用完整η边界面，ADFLOW完全对齐）

    **ADflow对齐**：
    - 如果提供halo_wall/halo_farfield，使用ADflow风格的halo单元计算边界面通量
    - **Plan86对齐**：粘性通量强制使用节点梯度四节点平均（ADFLOW标准）

    Args:
        rho, u, v, p: 物理场
        du_dx, du_dy, dv_dx, dv_dy: 速度梯度（cell-center，用于耗散等）
        face_geom: 面几何（来自geometry.compute_face_geometry）
        mu_eff: 有效粘度（如果include_viscous=True）
        include_viscous: 是否包含黏性通量
        Ma: 自由流马赫数（用于η边界条件）
        AoA: 攻角（度，用于η边界条件）
        gamma: 比热比（默认1.4）
        halo_wall: 壁面halo单元物理量 (C, W)，用于ADflow对齐模式
        halo_farfield: 远场halo单元物理量 (C, W)，用于ADflow对齐模式
        ss_halo: 预计算的熵场 (batch, H+2, W)，用于η方向边界dss修复
            - 由compute_ss_with_halo()生成
        du_dx_node, du_dy_node, dv_dx_node, dv_dy_node: 节点梯度 (batch, H+1, W+1)
            - **粘性通量必需**：如果include_viscous=True，必须提供
            - 通过NodalGradientCalculator (method='nodal') 计算
        use_nodal_gradients: 是否使用节点梯度（必须为True，如果include_viscous=True）

    Returns:
        fluxes: 字典包含
            - 'Fc_xi', 'Fmx_xi', 'Fmy_xi': ξ面对流通量 (H, W)
            - 'Fc_eta', 'Fmx_eta', 'Fmy_eta': η面对流通量 (H+1, W) **完整边界面**
            - 'Vmx_xi', 'Vmy_xi': ξ面黏性通量（如果include_viscous）
            - 'Vmx_eta', 'Vmy_eta': η面黏性通量 (H+1, W) **完整边界面**（如果include_viscous）

    Raises:
        ValueError: 如果include_viscous=True但未提供节点梯度
    """
    fluxes = {}
    precomputed_xi_cell_radius = None
    precomputed_eta_cell_radius = None

    if dissipation_mode == 'jameson' and vol is not None:
        from . import dissipation as diss_module

        precomputed_xi_cell_radius, precomputed_eta_cell_radius = (
            diss_module.compute_cell_centered_spectral_radius_pair(
                rho,
                u,
                v,
                p,
                face_geom,
                vol=vol,
                gamma=gamma,
                acoustic_scale_factor=acoustic_scale_factor,
                adis=adis,
                apply_anisotropic=True,
            )
        )

    # ξ方向对流通量（周期边界，含耗散）
    xi_result = inviscid_central_flux(
        rho, u, v, p, rhoE=rhoE, face_geom=face_geom, direction='xi',
        dissipation_mode=dissipation_mode,
        vis2=vis2, vis4=vis4, gamma=gamma,
        dss_max=dss_max, sslim=sslim,
        return_dissipation=return_dissipation,
        basis=basis,
        vol=vol,
        adis=adis,
        acoustic_scale_factor=acoustic_scale_factor,
        lumped_dissipation=lumped_dissipation,
        lumped_sigma=lumped_sigma,
        frozen_shock_sensor=frozen_shock_sensor,
        frozen_ss_halo=frozen_ss_halo,
        use_dissipation_continuation=use_dissipation_continuation,
        diss_cont_magnitude=diss_cont_magnitude,
        diss_cont_midpoint=diss_cont_midpoint,
        diss_cont_sharpness=diss_cont_sharpness,
        diss_cont_total_r=diss_cont_total_r,
        diss_cont_total_r0=diss_cont_total_r0,
        diss_cont_rfil=diss_cont_rfil,
        precomputed_cell_radius=precomputed_xi_cell_radius,
    )

    # 解析返回值（支持3方程和4方程模式）
    if rhoE is not None:
        # 4方程模式：包含能量通量
        if return_dissipation:
            # 返回: 4个总通量, 4个耗散, 4个纯对流通量
            Fc_xi, Fmx_xi, Fmy_xi, FE_xi, D_rho_xi, D_rhou_xi, D_rhov_xi, D_rhoE_xi, Fc_conv_xi, Fmx_conv_xi, Fmy_conv_xi, FE_conv_xi = xi_result
            fluxes['D_rho_xi'] = D_rho_xi
            fluxes['D_rhou_xi'] = D_rhou_xi
            fluxes['D_rhov_xi'] = D_rhov_xi
            fluxes['D_rhoE_xi'] = D_rhoE_xi
            # 纯对流通量
            fluxes['Fc_conv_xi'] = Fc_conv_xi
            fluxes['Fmx_conv_xi'] = Fmx_conv_xi
            fluxes['Fmy_conv_xi'] = Fmy_conv_xi
            fluxes['FE_conv_xi'] = FE_conv_xi
        else:
            Fc_xi, Fmx_xi, Fmy_xi, FE_xi = xi_result

        fluxes['FE_xi'] = FE_xi
    else:
        # 3方程模式：仅动量
        if return_dissipation:
            # 返回: 总通量, 耗散, 纯对流通量
            Fc_xi, Fmx_xi, Fmy_xi, D_rho_xi, D_rhou_xi, D_rhov_xi, Fc_conv_xi, Fmx_conv_xi, Fmy_conv_xi = xi_result
            fluxes['D_rho_xi'] = D_rho_xi
            fluxes['D_rhou_xi'] = D_rhou_xi
            fluxes['D_rhov_xi'] = D_rhov_xi
            # 纯对流通量
            fluxes['Fc_conv_xi'] = Fc_conv_xi
            fluxes['Fmx_conv_xi'] = Fmx_conv_xi
            fluxes['Fmy_conv_xi'] = Fmy_conv_xi
        else:
            Fc_xi, Fmx_xi, Fmy_xi = xi_result

    fluxes['Fc_xi'] = Fc_xi
    fluxes['Fmx_xi'] = Fmx_xi
    fluxes['Fmy_xi'] = Fmy_xi

    # η方向对流通量（完整边界面 H+1，含耗散）
    # 如果提供halo参数，使用ADflow对齐模式
    # **plan74.md修复**：传递ss_halo用于正确的边界dss计算
    eta_result = compute_inviscid_eta_flux(
        rho, u, v, p, face_geom,
        Ma=Ma, AoA=AoA, gamma=gamma,
        dissipation_mode=dissipation_mode,
        vis2=vis2, vis4=vis4,
        dss_max=dss_max, sslim=sslim,
        debug=debug,
        halo_wall=halo_wall,
        halo_farfield=halo_farfield,
        return_dissipation=return_dissipation,
        basis=basis,
        ss_halo=ss_halo,  # 边界dss修复
        vol=vol,
        adis=adis,
        acoustic_scale_factor=acoustic_scale_factor,
        rhoE=rhoE,  # 新增：能量方程支持
        lumped_dissipation=lumped_dissipation,
        lumped_sigma=lumped_sigma,
        frozen_shock_sensor=frozen_shock_sensor,
        frozen_ss_halo=frozen_ss_halo,
        use_dissipation_continuation=use_dissipation_continuation,
        diss_cont_magnitude=diss_cont_magnitude,
        diss_cont_midpoint=diss_cont_midpoint,
        diss_cont_sharpness=diss_cont_sharpness,
        diss_cont_total_r=diss_cont_total_r,
        diss_cont_total_r0=diss_cont_total_r0,
        diss_cont_rfil=diss_cont_rfil,
        precomputed_cell_radius=precomputed_eta_cell_radius,
    )

    # 解析η方向返回值（支持3方程和4方程模式）
    if rhoE is not None:
        # 4方程模式
        if return_dissipation:
            Fc_eta, Fmx_eta, Fmy_eta, FE_eta, D_rho_eta, D_rhou_eta, D_rhov_eta, D_rhoE_eta, Fc_conv_eta, Fmx_conv_eta, Fmy_conv_eta, FE_conv_eta = eta_result
            fluxes['D_rho_eta'] = D_rho_eta
            fluxes['D_rhou_eta'] = D_rhou_eta
            fluxes['D_rhov_eta'] = D_rhov_eta
            fluxes['D_rhoE_eta'] = D_rhoE_eta
            # 纯对流通量
            fluxes['Fc_conv_eta'] = Fc_conv_eta
            fluxes['Fmx_conv_eta'] = Fmx_conv_eta
            fluxes['Fmy_conv_eta'] = Fmy_conv_eta
            fluxes['FE_conv_eta'] = FE_conv_eta
        else:
            Fc_eta, Fmx_eta, Fmy_eta, FE_eta = eta_result

        fluxes['FE_eta'] = FE_eta
    else:
        # 3方程模式
        if return_dissipation:
            # 返回: 总通量, 耗散, 纯对流通量
            Fc_eta, Fmx_eta, Fmy_eta, D_rho_eta, D_rhou_eta, D_rhov_eta, Fc_conv_eta, Fmx_conv_eta, Fmy_conv_eta = eta_result
            fluxes['D_rho_eta'] = D_rho_eta
            fluxes['D_rhou_eta'] = D_rhou_eta
            fluxes['D_rhov_eta'] = D_rhov_eta
            # 纯对流通量
            fluxes['Fc_conv_eta'] = Fc_conv_eta
            fluxes['Fmx_conv_eta'] = Fmx_conv_eta
            fluxes['Fmy_conv_eta'] = Fmy_conv_eta
        else:
            Fc_eta, Fmx_eta, Fmy_eta = eta_result

    fluxes['Fc_eta'] = Fc_eta
    fluxes['Fmx_eta'] = Fmx_eta
    fluxes['Fmy_eta'] = Fmy_eta

    # 黏性通量
    if include_viscous and mu_eff is not None:
        if bool(approximate_viscous_operator):
            if not isinstance(mu_eff, dict):
                raise ValueError(
                    "approximate_viscous_operator requires viscosity dict with mu_lam/mu_turb."
                )

            Vmx_xi, Vmy_xi, VE_xi = viscous_flux_approx(
                rho=rho,
                u=u,
                v=v,
                p=p,
                rhoE=rhoE,
                mu_eff=mu_eff,
                face_geom=face_geom,
                direction='xi',
                gamma=gamma,
                periodic_xi=periodic_xi,
                rfil=diss_cont_rfil,
            )
            fluxes['Vmx_xi'] = Vmx_xi
            fluxes['Vmy_xi'] = Vmy_xi

            Vmx_eta, Vmy_eta, VE_eta = viscous_flux_approx(
                rho=rho,
                u=u,
                v=v,
                p=p,
                rhoE=rhoE,
                mu_eff=mu_eff,
                face_geom=face_geom,
                direction='eta',
                gamma=gamma,
                periodic_xi=periodic_xi,
                rfil=diss_cont_rfil,
                halo_wall=halo_wall,
                halo_farfield=halo_farfield,
                halo_nu_tilde_farfield=halo_nu_tilde_farfield,
            )
            fluxes['Vmx_eta'] = Vmx_eta
            fluxes['Vmy_eta'] = Vmy_eta

            if VE_xi is not None and VE_eta is not None:
                fluxes['VE_xi'] = VE_xi
                fluxes['VE_eta'] = VE_eta
        else:
            # ξ面黏性通量
            Vmx_xi, Vmy_xi = viscous_flux(
                u, v, du_dx, du_dy, dv_dx, dv_dy, mu_eff, face_geom,
                direction='xi',
                # Plan86对齐：传入节点梯度（ADFLOW标准）
                du_dx_node=du_dx_node,
                du_dy_node=du_dy_node,
                dv_dx_node=dv_dx_node,
                dv_dy_node=dv_dy_node,
                use_nodal_gradients=use_nodal_gradients
            )

            fluxes['Vmx_xi'] = Vmx_xi
            fluxes['Vmy_xi'] = Vmy_xi

            # η面黏性通量（完整边界面 H+1）
            Vmx_eta, Vmy_eta = viscous_flux(
                u, v, du_dx, du_dy, dv_dx, dv_dy, mu_eff, face_geom,
                direction='eta',
                Ma=Ma, AoA=AoA, gamma=gamma,
                # Plan86对齐：传入节点梯度（ADFLOW标准）
                du_dx_node=du_dx_node,
                du_dy_node=du_dy_node,
                dv_dx_node=dv_dx_node,
                dv_dy_node=dv_dy_node,
                halo_wall=halo_wall,
                halo_farfield=halo_farfield,
                use_nodal_gradients=use_nodal_gradients
            )

            fluxes['Vmx_eta'] = Vmx_eta
            fluxes['Vmy_eta'] = Vmy_eta

            # 能量粘性通量（如果提供rhoE和相关梯度）
            if (rhoE is not None and
                daa_dx_node is not None and daa_dy_node is not None and
                dp_dx_node is not None and dp_dy_node is not None and
                drho_dx_node is not None and drho_dy_node is not None):

                from .energy_viscous_flux import compute_energy_viscous_flux

                # 解包粘度（支持SA模式字典或单一值）
                if isinstance(mu_eff, dict):
                    mu_l_raw = mu_eff['mu_lam']  # 可能是标量或 (B, 1, 1)
                    mu_t = mu_eff['mu_turb']      # (H, W) 或 (B, H, W)
                    # 广播 mu_l 到与 mu_t 相同的形状（避免 _average_to_face 中维度不匹配）
                    if mu_l_raw.ndim == 0:
                        # 标量：先 reshape 再 expand
                        mu_l = mu_l_raw.reshape(1, 1, 1).expand(mu_t.shape)
                    elif mu_l_raw.shape != mu_t.shape:
                        # (B, 1, 1) 等形状：直接 expand
                        mu_l = mu_l_raw.expand(mu_t.shape)
                    else:
                        mu_l = mu_l_raw
                else:
                    mu_l = mu_eff
                    mu_t = torch.zeros_like(mu_eff)

                # ========== ADFLOW对齐：使用nodal gradients计算应力张量 ==========
                # 参考：fluxes.F90:3227-3290 (viscousFlux中的tau计算)
                # ADFLOW使用节点梯度插值到面上，然后计算tau
                # 这与动量粘性通量使用相同的路径，保证一致性

                # 将节点梯度插值到cell-center（用于tau计算）
                # 取4个角节点的平均值
                # 支持 batch 模式：(B, H+1, W+1) -> (B, H, W)
                if du_dx_node.ndim == 3:
                    # Batch mode: (B, H+1, W+1)
                    du_dx_cc = 0.25 * (du_dx_node[:, :-1, :-1] + du_dx_node[:, :-1, 1:] +
                                       du_dx_node[:, 1:, :-1] + du_dx_node[:, 1:, 1:])
                    du_dy_cc = 0.25 * (du_dy_node[:, :-1, :-1] + du_dy_node[:, :-1, 1:] +
                                       du_dy_node[:, 1:, :-1] + du_dy_node[:, 1:, 1:])
                    dv_dx_cc = 0.25 * (dv_dx_node[:, :-1, :-1] + dv_dx_node[:, :-1, 1:] +
                                       dv_dx_node[:, 1:, :-1] + dv_dx_node[:, 1:, 1:])
                    dv_dy_cc = 0.25 * (dv_dy_node[:, :-1, :-1] + dv_dy_node[:, :-1, 1:] +
                                       dv_dy_node[:, 1:, :-1] + dv_dy_node[:, 1:, 1:])
                else:
                    # Single sample: (H+1, W+1)
                    du_dx_cc = 0.25 * (du_dx_node[:-1, :-1] + du_dx_node[:-1, 1:] +
                                       du_dx_node[1:, :-1] + du_dx_node[1:, 1:])
                    du_dy_cc = 0.25 * (du_dy_node[:-1, :-1] + du_dy_node[:-1, 1:] +
                                       du_dy_node[1:, :-1] + du_dy_node[1:, 1:])
                    dv_dx_cc = 0.25 * (dv_dx_node[:-1, :-1] + dv_dx_node[:-1, 1:] +
                                       dv_dx_node[1:, :-1] + dv_dx_node[1:, 1:])
                    dv_dy_cc = 0.25 * (dv_dy_node[:-1, :-1] + dv_dy_node[:-1, 1:] +
                                       dv_dy_node[1:, :-1] + dv_dy_node[1:, 1:])

                # 2D应力张量计算（使用nodal gradients插值结果）
                mu_total = mu_l + mu_t
                lambda_visc = -2.0 / 3.0 * mu_total
                div_vel = du_dx_cc + dv_dy_cc
                tau_xx = 2.0 * mu_total * du_dx_cc + lambda_visc * div_vel
                tau_yy = 2.0 * mu_total * dv_dy_cc + lambda_visc * div_vel
                tau_xy = mu_total * (du_dy_cc + dv_dx_cc)

                # ========== ADFLOW对齐：壁面halo设置 ==========
                # 优先使用真实halo状态；无halo时回退到解析反射。
                if halo_wall is not None:
                    halo_u_wall = halo_wall[:, 1, :] if halo_wall.ndim == 3 else halo_wall[1, :]
                    halo_v_wall = halo_wall[:, 2, :] if halo_wall.ndim == 3 else halo_wall[2, :]
                    halo_rho_wall = halo_wall[:, 0, :] if halo_wall.ndim == 3 else halo_wall[0, :]
                    halo_p_wall = halo_wall[:, 3, :] if halo_wall.ndim == 3 else halo_wall[3, :]
                else:
                    halo_u_wall = -u[:, 0, :] if u.ndim == 3 else -u[0, :]
                    halo_v_wall = -v[:, 0, :] if v.ndim == 3 else -v[0, :]
                    halo_rho_wall = None
                    halo_p_wall = None

                halo_u_farfield = None
                halo_v_farfield = None
                halo_rho_farfield = None
                halo_p_farfield = None
                if halo_farfield is not None:
                    halo_u_farfield = (
                        halo_farfield[:, 1, :] if halo_farfield.ndim == 3 else halo_farfield[1, :]
                    )
                    halo_v_farfield = (
                        halo_farfield[:, 2, :] if halo_farfield.ndim == 3 else halo_farfield[2, :]
                    )
                    halo_rho_farfield = (
                        halo_farfield[:, 0, :] if halo_farfield.ndim == 3 else halo_farfield[0, :]
                    )
                    halo_p_farfield = (
                        halo_farfield[:, 3, :] if halo_farfield.ndim == 3 else halo_farfield[3, :]
                    )

                # ADFLOW对齐：halo_mu_t = -mu_t_cell，使 mu_t_face = 0.5*(halo + physical) = 0
                mu_t_wall_cell = mu_t[:, 0, :] if mu_t.ndim == 3 else mu_t[0, :]
                halo_mu_t_wall = -mu_t_wall_cell

                # 计算a² = gamma * p / rho（用于热通量法向修正）
                # ADFLOW对齐：不要在rho分母中加入epsilon（会在近壁inv_d放大，导致layer0热通量残差偏差）
                aa = gamma * p / rho
                if halo_rho_wall is not None and halo_p_wall is not None:
                    halo_aa_wall = gamma * halo_p_wall / halo_rho_wall
                else:
                    halo_aa_wall = aa[:, 0, :] if aa.ndim == 3 else aa[0, :]
                if halo_rho_farfield is not None and halo_p_farfield is not None:
                    halo_aa_farfield = gamma * halo_p_farfield / halo_rho_farfield
                else:
                    halo_aa_farfield = aa[:, -1, :] if aa.ndim == 3 else aa[-1, :]
                if aa.ndim == 3:
                    aa_left_eta = torch.cat(
                        [halo_aa_wall.unsqueeze(1), aa[:, :-1, :], aa[:, -1:, :]],
                        dim=1,
                    )
                    aa_right_eta = torch.cat(
                        [aa[:, 0:1, :], aa[:, 1:, :], halo_aa_farfield.unsqueeze(1)],
                        dim=1,
                    )
                else:
                    aa_left_eta = torch.cat(
                        [halo_aa_wall.unsqueeze(0), aa[:-1, :], aa[-1:, :]],
                        dim=0,
                    )
                    aa_right_eta = torch.cat(
                        [aa[0:1, :], aa[1:, :], halo_aa_farfield.unsqueeze(0)],
                        dim=0,
                    )

                # ξ方向能量粘性通量（周期边界，启用法向修正）
                # ✅ ADFLOW对齐：ξ方向也需要法向修正
                (
                    VE_xi,
                    viscWork_xi,
                    heatFlux_xi,
                    heatFlux_xi_uncorrected,
                    heatFlux_xi_correction,
                    heatFlux_xi_laminar,
                    heatFlux_xi_turbulent,
                    heatFlux_xi_laminar_uncorrected,
                    heatFlux_xi_laminar_correction,
                    heatFlux_xi_turbulent_uncorrected,
                    heatFlux_xi_turbulent_correction,
                    heatFlux_xi_laminar_uncorrected_unitcoef,
                    heatFlux_xi_laminar_correction_unitcoef,
                    heatFlux_xi_laminar_uncorrected_unitcoef_x,
                    heatFlux_xi_laminar_uncorrected_unitcoef_y,
                    heatFlux_xi_laminar_correction_unitcoef_x,
                    heatFlux_xi_laminar_correction_unitcoef_y,
                    heatFlux_xi_laminar_uncorrected_unitcoef_qy,
                    heatFlux_xi_laminar_correction_unitcoef_qy,
                    heatFlux_xi_heat_q_dot_n_uncorrected,
                    heatFlux_xi_heat_delta_aa,
                    heatFlux_xi_heat_corr,
                ) = compute_energy_viscous_flux(
                    rho, u, v, p, rhoE,
                    du_dx_node, du_dy_node, dv_dx_node, dv_dy_node,
                    daa_dx_node, daa_dy_node,
                    tau_xx, tau_xy, tau_yy,
                    mu_l, mu_t, face_geom, direction='xi',
                    gamma=gamma, Pr_laminar=Pr_laminar, Pr_turbulent=Pr_turbulent,
                    # ✅ ADFLOW对齐：启用ξ方向法向修正
                    apply_heat_flux_normal_correction=True,
                    aa=aa,
                    ssx_sep=face_geom.get('ssx_xi'),
                    ssy_sep=face_geom.get('ssy_xi'),
                    inv_d=face_geom.get('inv_d_xi'),
                    return_heat_parts=True,
                )

                # η方向能量粘性通量（需要halo和热通量法向修正）
                (
                    VE_eta,
                    viscWork_eta,
                    heatFlux_eta,
                    heatFlux_eta_uncorrected,
                    heatFlux_eta_correction,
                    heatFlux_eta_laminar,
                    heatFlux_eta_turbulent,
                    heatFlux_eta_laminar_uncorrected,
                    heatFlux_eta_laminar_correction,
                    heatFlux_eta_turbulent_uncorrected,
                    heatFlux_eta_turbulent_correction,
                    heatFlux_eta_laminar_uncorrected_unitcoef,
                    heatFlux_eta_laminar_correction_unitcoef,
                    heatFlux_eta_laminar_uncorrected_unitcoef_x,
                    heatFlux_eta_laminar_uncorrected_unitcoef_y,
                    heatFlux_eta_laminar_correction_unitcoef_x,
                    heatFlux_eta_laminar_correction_unitcoef_y,
                    heatFlux_eta_laminar_uncorrected_unitcoef_qy,
                    heatFlux_eta_laminar_correction_unitcoef_qy,
                    heatFlux_eta_heat_q_dot_n_uncorrected,
                    heatFlux_eta_heat_delta_aa,
                    heatFlux_eta_heat_corr,
                ) = compute_energy_viscous_flux(
                    rho, u, v, p, rhoE,
                    du_dx_node, du_dy_node, dv_dx_node, dv_dy_node,
                    daa_dx_node, daa_dy_node,
                    tau_xx, tau_xy, tau_yy,
                    mu_l, mu_t, face_geom, direction='eta',
                    gamma=gamma, Pr_laminar=Pr_laminar, Pr_turbulent=Pr_turbulent,
                    # ✅ ADFLOW对齐：启用halo参数
                    halo_u_wall=halo_u_wall,
                    halo_v_wall=halo_v_wall,
                    halo_mu_t_wall=halo_mu_t_wall,
                    halo_u_farfield=halo_u_farfield,
                    halo_v_farfield=halo_v_farfield,
                    # ✅ ADFLOW对齐：启用热通量法向修正
                    apply_heat_flux_normal_correction=True,
                    aa=aa,
                    ssx_sep=face_geom.get('ssx_eta'),
                    ssy_sep=face_geom.get('ssy_eta'),
                    inv_d=face_geom.get('inv_d_eta'),
                    halo_aa_wall=halo_aa_wall,
                    halo_aa_farfield=halo_aa_farfield,
                    return_heat_parts=True,
                )

                fluxes['VE_xi'] = VE_xi

                # 注意：移除壁面强制置零逻辑（review3.md对齐）
                # ADFLOW实际输出壁面η向粘性通量非零（约1e-11）
                # porK=noFlux的处理应该由边界条件本身决定，而非后处理强制置零
                # 保留原始通量值，与ADFLOW输出一致

                fluxes['VE_eta'] = VE_eta
                fluxes['face_A_y_eta'] = face_geom['A_y_eta']
                fluxes['face_ssx_eta'] = face_geom['ssx_eta']
                fluxes['face_ssy_eta'] = face_geom['ssy_eta']
                fluxes['face_inv_d_eta'] = face_geom['inv_d_eta']
                fluxes['heat_q_dot_n_uncorrected_eta'] = heatFlux_eta_heat_q_dot_n_uncorrected
                fluxes['heat_delta_aa_eta'] = heatFlux_eta_heat_delta_aa
                fluxes['heat_corr_eta'] = heatFlux_eta_heat_corr
                fluxes['face_aa_left_eta'] = aa_left_eta
                fluxes['face_aa_right_eta'] = aa_right_eta
                # 保存分解用于调试
                fluxes['viscWork_eta'] = viscWork_eta
                fluxes['heatFlux_eta'] = heatFlux_eta
                fluxes['heatFlux_eta_uncorrected'] = heatFlux_eta_uncorrected
                fluxes['heatFlux_eta_correction'] = heatFlux_eta_correction
                fluxes['heatFlux_eta_laminar'] = heatFlux_eta_laminar
                fluxes['heatFlux_eta_turbulent'] = heatFlux_eta_turbulent
                fluxes['heatFlux_eta_laminar_uncorrected'] = heatFlux_eta_laminar_uncorrected
                fluxes['heatFlux_eta_laminar_correction'] = heatFlux_eta_laminar_correction
                fluxes['heatFlux_eta_turbulent_uncorrected'] = heatFlux_eta_turbulent_uncorrected
                fluxes['heatFlux_eta_turbulent_correction'] = heatFlux_eta_turbulent_correction
                fluxes['heatFlux_eta_laminar_uncorrected_unitcoef'] = heatFlux_eta_laminar_uncorrected_unitcoef
                fluxes['heatFlux_eta_laminar_correction_unitcoef'] = heatFlux_eta_laminar_correction_unitcoef
                fluxes['heatFlux_eta_laminar_uncorrected_unitcoef_x'] = heatFlux_eta_laminar_uncorrected_unitcoef_x
                fluxes['heatFlux_eta_laminar_uncorrected_unitcoef_y'] = heatFlux_eta_laminar_uncorrected_unitcoef_y
                fluxes['heatFlux_eta_laminar_correction_unitcoef_x'] = heatFlux_eta_laminar_correction_unitcoef_x
                fluxes['heatFlux_eta_laminar_correction_unitcoef_y'] = heatFlux_eta_laminar_correction_unitcoef_y
                fluxes['heatFlux_eta_laminar_uncorrected_unitcoef_qy'] = heatFlux_eta_laminar_uncorrected_unitcoef_qy
                fluxes['heatFlux_eta_laminar_correction_unitcoef_qy'] = heatFlux_eta_laminar_correction_unitcoef_qy
                fluxes['heatFlux_uncorrected_eta'] = heatFlux_eta_uncorrected
                fluxes['heatFlux_correction_eta'] = heatFlux_eta_correction
                fluxes['heatFlux_laminar_eta'] = heatFlux_eta_laminar
                fluxes['heatFlux_turbulent_eta'] = heatFlux_eta_turbulent
                fluxes['heatFlux_laminar_uncorrected_eta'] = heatFlux_eta_laminar_uncorrected
                fluxes['heatFlux_laminar_correction_eta'] = heatFlux_eta_laminar_correction
                fluxes['heatFlux_laminar_uncorrected_unitcoef_eta'] = heatFlux_eta_laminar_uncorrected_unitcoef
                fluxes['heatFlux_laminar_correction_unitcoef_eta'] = heatFlux_eta_laminar_correction_unitcoef
                fluxes['heatFlux_laminar_uncorrected_unitcoef_x_eta'] = heatFlux_eta_laminar_uncorrected_unitcoef_x
                fluxes['heatFlux_laminar_uncorrected_unitcoef_y_eta'] = heatFlux_eta_laminar_uncorrected_unitcoef_y
                fluxes['heatFlux_laminar_correction_unitcoef_x_eta'] = heatFlux_eta_laminar_correction_unitcoef_x
                fluxes['heatFlux_laminar_correction_unitcoef_y_eta'] = heatFlux_eta_laminar_correction_unitcoef_y
                fluxes['heatFlux_laminar_uncorrected_unitcoef_qy_eta'] = heatFlux_eta_laminar_uncorrected_unitcoef_qy
                fluxes['heatFlux_laminar_correction_unitcoef_qy_eta'] = heatFlux_eta_laminar_correction_unitcoef_qy
                fluxes['heatFlux_turbulent_uncorrected_eta'] = heatFlux_eta_turbulent_uncorrected
                fluxes['heatFlux_turbulent_correction_eta'] = heatFlux_eta_turbulent_correction
                # ✅ ADFLOW对齐：保存ξ方向分解用于调试
                fluxes['viscWork_xi'] = viscWork_xi
                fluxes['heatFlux_xi'] = heatFlux_xi
                fluxes['heatFlux_xi_uncorrected'] = heatFlux_xi_uncorrected
                fluxes['heatFlux_xi_correction'] = heatFlux_xi_correction
                fluxes['heatFlux_xi_laminar'] = heatFlux_xi_laminar
                fluxes['heatFlux_xi_turbulent'] = heatFlux_xi_turbulent
                fluxes['heatFlux_xi_laminar_uncorrected'] = heatFlux_xi_laminar_uncorrected
                fluxes['heatFlux_xi_laminar_correction'] = heatFlux_xi_laminar_correction
                fluxes['heatFlux_xi_turbulent_uncorrected'] = heatFlux_xi_turbulent_uncorrected
                fluxes['heatFlux_xi_turbulent_correction'] = heatFlux_xi_turbulent_correction
                fluxes['heatFlux_xi_laminar_uncorrected_unitcoef'] = heatFlux_xi_laminar_uncorrected_unitcoef
                fluxes['heatFlux_xi_laminar_correction_unitcoef'] = heatFlux_xi_laminar_correction_unitcoef
                fluxes['heatFlux_xi_laminar_uncorrected_unitcoef_x'] = heatFlux_xi_laminar_uncorrected_unitcoef_x
                fluxes['heatFlux_xi_laminar_uncorrected_unitcoef_y'] = heatFlux_xi_laminar_uncorrected_unitcoef_y
                fluxes['heatFlux_xi_laminar_correction_unitcoef_x'] = heatFlux_xi_laminar_correction_unitcoef_x
                fluxes['heatFlux_xi_laminar_correction_unitcoef_y'] = heatFlux_xi_laminar_correction_unitcoef_y
                fluxes['heatFlux_xi_laminar_uncorrected_unitcoef_qy'] = heatFlux_xi_laminar_uncorrected_unitcoef_qy
                fluxes['heatFlux_xi_laminar_correction_unitcoef_qy'] = heatFlux_xi_laminar_correction_unitcoef_qy
                fluxes['heatFlux_uncorrected_xi'] = heatFlux_xi_uncorrected
                fluxes['heatFlux_correction_xi'] = heatFlux_xi_correction
                fluxes['heatFlux_laminar_xi'] = heatFlux_xi_laminar
                fluxes['heatFlux_turbulent_xi'] = heatFlux_xi_turbulent
                fluxes['heatFlux_laminar_uncorrected_xi'] = heatFlux_xi_laminar_uncorrected
                fluxes['heatFlux_laminar_correction_xi'] = heatFlux_xi_laminar_correction
                fluxes['heatFlux_laminar_uncorrected_unitcoef_xi'] = heatFlux_xi_laminar_uncorrected_unitcoef
                fluxes['heatFlux_laminar_correction_unitcoef_xi'] = heatFlux_xi_laminar_correction_unitcoef
                fluxes['heatFlux_laminar_uncorrected_unitcoef_x_xi'] = heatFlux_xi_laminar_uncorrected_unitcoef_x
                fluxes['heatFlux_laminar_uncorrected_unitcoef_y_xi'] = heatFlux_xi_laminar_uncorrected_unitcoef_y
                fluxes['heatFlux_laminar_correction_unitcoef_x_xi'] = heatFlux_xi_laminar_correction_unitcoef_x
                fluxes['heatFlux_laminar_correction_unitcoef_y_xi'] = heatFlux_xi_laminar_correction_unitcoef_y
                fluxes['heatFlux_laminar_uncorrected_unitcoef_qy_xi'] = heatFlux_xi_laminar_uncorrected_unitcoef_qy
                fluxes['heatFlux_laminar_correction_unitcoef_qy_xi'] = heatFlux_xi_laminar_correction_unitcoef_qy
                fluxes['heatFlux_turbulent_uncorrected_xi'] = heatFlux_xi_turbulent_uncorrected
                fluxes['heatFlux_turbulent_correction_xi'] = heatFlux_xi_turbulent_correction

    # ========== DEBUG: 通量统计 ==========
    if debug:
        print(f"\n[DEBUG fluxes.py] 通量统计:")
        print(f"  ξ面对流通量:")
        print(f"    Fc_xi: shape={Fc_xi.shape}, min={Fc_xi.min():.6e}, max={Fc_xi.max():.6e}, mean={Fc_xi.mean():.6e}")
        print(f"    Fmx_xi: shape={Fmx_xi.shape}, min={Fmx_xi.min():.6e}, max={Fmx_xi.max():.6e}, mean={Fmx_xi.mean():.6e}")
        print(f"    Fmy_xi: shape={Fmy_xi.shape}, min={Fmy_xi.min():.6e}, max={Fmy_xi.max():.6e}, mean={Fmy_xi.mean():.6e}")
        print(f"  η面对流通量:")
        print(f"    Fc_eta: shape={Fc_eta.shape}, min={Fc_eta.min():.6e}, max={Fc_eta.max():.6e}, mean={Fc_eta.mean():.6e}")
        print(f"    Fmx_eta: shape={Fmx_eta.shape}, min={Fmx_eta.min():.6e}, max={Fmx_eta.max():.6e}, mean={Fmx_eta.mean():.6e}")
        print(f"    Fmy_eta: shape={Fmy_eta.shape}, min={Fmy_eta.min():.6e}, max={Fmy_eta.max():.6e}, mean={Fmy_eta.mean():.6e}")
        if include_viscous and mu_eff is not None:
            print(f"  黏性通量:")
            print(f"    Vmx_xi: min={Vmx_xi.min():.6e}, max={Vmx_xi.max():.6e}, mean={Vmx_xi.mean():.6e}")
            print(f"    Vmy_xi: min={Vmy_xi.min():.6e}, max={Vmy_xi.max():.6e}, mean={Vmy_xi.mean():.6e}")
            print(f"    Vmx_eta: min={Vmx_eta.min():.6e}, max={Vmx_eta.max():.6e}, mean={Vmx_eta.mean():.6e}")
            print(f"    Vmy_eta: min={Vmy_eta.min():.6e}, max={Vmy_eta.max():.6e}, mean={Vmy_eta.mean():.6e}")
        # 动量通量组件统计（对流、耗散分离）
        if return_dissipation:
            print(f"  [动量通量组件分解]:")
            print(f"    ξ方向x动量:")
            print(f"      Fmx_conv_xi (纯对流): RMS={torch.sqrt((Fmx_conv_xi**2).mean()):.6e}")
            print(f"      D_rhou_xi (耗散): RMS={torch.sqrt((D_rhou_xi**2).mean()):.6e}")
            print(f"      Fmx_xi (对流-耗散): RMS={torch.sqrt((Fmx_xi**2).mean()):.6e}")
            if include_viscous and mu_eff is not None:
                print(f"      Vmx_xi (黏性): RMS={torch.sqrt((Vmx_xi**2).mean()):.6e}")
            print(f"    ξ方向y动量:")
            print(f"      Fmy_conv_xi (纯对流): RMS={torch.sqrt((Fmy_conv_xi**2).mean()):.6e}")
            print(f"      D_rhov_xi (耗散): RMS={torch.sqrt((D_rhov_xi**2).mean()):.6e}")
            print(f"      Fmy_xi (对流-耗散): RMS={torch.sqrt((Fmy_xi**2).mean()):.6e}")
            if include_viscous and mu_eff is not None:
                print(f"      Vmy_xi (黏性): RMS={torch.sqrt((Vmy_xi**2).mean()):.6e}")
            print(f"    η方向x动量:")
            print(f"      Fmx_conv_eta (纯对流): RMS={torch.sqrt((Fmx_conv_eta**2).mean()):.6e}")
            print(f"      D_rhou_eta (耗散): RMS={torch.sqrt((D_rhou_eta**2).mean()):.6e}")
            print(f"      Fmx_eta (对流-耗散): RMS={torch.sqrt((Fmx_eta**2).mean()):.6e}")
            if include_viscous and mu_eff is not None:
                print(f"      Vmx_eta (黏性): RMS={torch.sqrt((Vmx_eta**2).mean()):.6e}")
            print(f"    η方向y动量:")
            print(f"      Fmy_conv_eta (纯对流): RMS={torch.sqrt((Fmy_conv_eta**2).mean()):.6e}")
            print(f"      D_rhov_eta (耗散): RMS={torch.sqrt((D_rhov_eta**2).mean()):.6e}")
            print(f"      Fmy_eta (对流-耗散): RMS={torch.sqrt((Fmy_eta**2).mean()):.6e}")
            if include_viscous and mu_eff is not None:
                print(f"      Vmy_eta (黏性): RMS={torch.sqrt((Vmy_eta**2).mean()):.6e}")
    # ========== END DEBUG ==========

    # ========== DEBUG: Save xi flux to npz file ==========
    import os
    if os.environ.get('SURROGATE_DEBUG_FLUX', '') == '1':
        import numpy as np
        # 获取 xi 面几何信息用于对比
        A_x_xi = face_geom.get('A_x_xi', None)
        A_y_xi = face_geom.get('A_y_xi', None)

        # 从 Fc_xi 获取维度
        if Fc_xi.ndim == 3:
            _, H_dbg, W_dbg = Fc_xi.shape
        else:
            H_dbg, W_dbg = Fc_xi.shape

        xi_debug_data = {
            # ξ 方向总通量 (batch, H, W) -> squeeze to (H, W)
            'Fc_xi': Fc_xi[0].detach().cpu().numpy() if Fc_xi.ndim == 3 else Fc_xi.detach().cpu().numpy(),
            'Fmx_xi': Fmx_xi[0].detach().cpu().numpy() if Fmx_xi.ndim == 3 else Fmx_xi.detach().cpu().numpy(),
            'Fmy_xi': Fmy_xi[0].detach().cpu().numpy() if Fmy_xi.ndim == 3 else Fmy_xi.detach().cpu().numpy(),
            # 维度信息
            'H': H_dbg,
            'W': W_dbg,
        }

        # 添加耗散通量（如果启用且有返回）
        if 'D_rho_xi' in fluxes:
            xi_debug_data['D_rho_xi'] = fluxes['D_rho_xi'][0].detach().cpu().numpy() if fluxes['D_rho_xi'].ndim == 3 else fluxes['D_rho_xi'].detach().cpu().numpy()
            xi_debug_data['D_rhou_xi'] = fluxes['D_rhou_xi'][0].detach().cpu().numpy() if fluxes['D_rhou_xi'].ndim == 3 else fluxes['D_rhou_xi'].detach().cpu().numpy()
            xi_debug_data['D_rhov_xi'] = fluxes['D_rhov_xi'][0].detach().cpu().numpy() if fluxes['D_rhov_xi'].ndim == 3 else fluxes['D_rhov_xi'].detach().cpu().numpy()
        if 'Fc_conv_xi' in fluxes:
            # 对流通量已经在 return_dissipation 时返回
            xi_debug_data['Fc_conv_xi'] = fluxes['Fc_conv_xi'][0].detach().cpu().numpy() if fluxes['Fc_conv_xi'].ndim == 3 else fluxes['Fc_conv_xi'].detach().cpu().numpy()

        # 添加面积向量（如果存在）
        if A_x_xi is not None:
            xi_debug_data['A_x_xi'] = A_x_xi[0].detach().cpu().numpy() if A_x_xi.ndim == 3 else A_x_xi.detach().cpu().numpy()
        if A_y_xi is not None:
            xi_debug_data['A_y_xi'] = A_y_xi[0].detach().cpu().numpy() if A_y_xi.ndim == 3 else A_y_xi.detach().cpu().numpy()

        # 添加黏性通量（如果存在）
        if include_viscous and mu_eff is not None:
            xi_debug_data['Vmx_xi'] = Vmx_xi[0].detach().cpu().numpy() if Vmx_xi.ndim == 3 else Vmx_xi.detach().cpu().numpy()
            xi_debug_data['Vmy_xi'] = Vmy_xi[0].detach().cpu().numpy() if Vmy_xi.ndim == 3 else Vmy_xi.detach().cpu().numpy()

        # ========== 添加中间量用于深度对比 (vnp, vnm, rqsp, rqsm) ==========
        # 这些中间量与ADflow导出的对应，用于定位对流通量差异来源
        if A_x_xi is not None and A_y_xi is not None:
            # 确保有batch维度
            _rho = rho.unsqueeze(0) if rho.ndim == 2 else rho
            _u = u.unsqueeze(0) if u.ndim == 2 else u
            _v = v.unsqueeze(0) if v.ndim == 2 else v
            _A_x = A_x_xi.unsqueeze(0) if A_x_xi.ndim == 2 else A_x_xi
            _A_y = A_y_xi.unsqueeze(0) if A_y_xi.ndim == 2 else A_y_xi

            # 左右状态 (周期边界: W个面)
            rho_L = torch.cat([_rho[..., :, :-1], _rho[..., :, -1:]], dim=-1)
            rho_R = torch.cat([_rho[..., :, 1:], _rho[..., :, :1]], dim=-1)
            u_L = torch.cat([_u[..., :, :-1], _u[..., :, -1:]], dim=-1)
            u_R = torch.cat([_u[..., :, 1:], _u[..., :, :1]], dim=-1)
            v_L = torch.cat([_v[..., :, :-1], _v[..., :, -1:]], dim=-1)
            v_R = torch.cat([_v[..., :, 1:], _v[..., :, :1]], dim=-1)

            # 法向速度×面积 (ADflow: vnp/vnm)
            vnp = u_R * _A_x + v_R * _A_y  # 右侧
            vnm = u_L * _A_x + v_L * _A_y  # 左侧

            # 质量流率 (ADflow: rqsp/rqsm)
            rqsp = 0.5 * rho_R * vnp
            rqsm = 0.5 * rho_L * vnm

            # 保存中间量
            xi_debug_data['vnp'] = vnp[0].detach().cpu().numpy()
            xi_debug_data['vnm'] = vnm[0].detach().cpu().numpy()
            xi_debug_data['rqsp'] = rqsp[0].detach().cpu().numpy()
            xi_debug_data['rqsm'] = rqsm[0].detach().cpu().numpy()
            # 左右状态也保存（用于验证）
            xi_debug_data['rho_L'] = rho_L[0].detach().cpu().numpy()
            xi_debug_data['rho_R'] = rho_R[0].detach().cpu().numpy()
            xi_debug_data['u_L'] = u_L[0].detach().cpu().numpy()
            xi_debug_data['u_R'] = u_R[0].detach().cpu().numpy()
            xi_debug_data['v_L'] = v_L[0].detach().cpu().numpy()
            xi_debug_data['v_R'] = v_R[0].detach().cpu().numpy()

        np.savez('pytorch_xi_flux_debug.npz', **xi_debug_data)
        print(f"[DEBUG] Saved xi flux to: pytorch_xi_flux_debug.npz")
        print(f"        Xi flux shape: Fc_xi={Fc_xi.shape}")

        if include_viscous and mu_eff is not None:
            eta_viscous_data = {
                'Vmx_eta': Vmx_eta[0].detach().cpu().numpy() if Vmx_eta.ndim == 3 else Vmx_eta.detach().cpu().numpy(),
                'Vmy_eta': Vmy_eta[0].detach().cpu().numpy() if Vmy_eta.ndim == 3 else Vmy_eta.detach().cpu().numpy(),
            }
            if 'VE_eta' in fluxes and fluxes['VE_eta'] is not None:
                eta_viscous_data['VE_eta'] = (
                    fluxes['VE_eta'][0].detach().cpu().numpy()
                    if fluxes['VE_eta'].ndim == 3
                    else fluxes['VE_eta'].detach().cpu().numpy()
                )
            np.savez('pytorch_eta_viscous_debug.npz', **eta_viscous_data)
            print(f"[DEBUG] Saved eta viscous flux to: pytorch_eta_viscous_debug.npz")

        # ========== 添加能量通量调试文件（eta方向）==========
        # 保存能量对流通量和能量粘性通量用于与ADFLOW对比
        if 'FE_eta' in fluxes:
            energy_debug_data = {
                'FE_eta': fluxes['FE_eta'][0].detach().cpu().numpy() if fluxes['FE_eta'].ndim == 3 else fluxes['FE_eta'].detach().cpu().numpy(),
            }
            # 添加能量粘性通量（如果存在）
            if 'VE_eta' in fluxes and fluxes['VE_eta'] is not None:
                energy_debug_data['VE_eta'] = fluxes['VE_eta'][0].detach().cpu().numpy() if fluxes['VE_eta'].ndim == 3 else fluxes['VE_eta'].detach().cpu().numpy()
            # 添加粘性功分量
            if 'viscWork_eta' in fluxes and fluxes['viscWork_eta'] is not None:
                energy_debug_data['viscWork_eta'] = fluxes['viscWork_eta'][0].detach().cpu().numpy() if fluxes['viscWork_eta'].ndim == 3 else fluxes['viscWork_eta'].detach().cpu().numpy()
            # 添加热通量分量
            if 'heatFlux_eta' in fluxes and fluxes['heatFlux_eta'] is not None:
                energy_debug_data['heatFlux_eta'] = fluxes['heatFlux_eta'][0].detach().cpu().numpy() if fluxes['heatFlux_eta'].ndim == 3 else fluxes['heatFlux_eta'].detach().cpu().numpy()
            # 添加能量耗散通量（如果存在）
            if 'D_rhoE_eta' in fluxes and fluxes['D_rhoE_eta'] is not None:
                energy_debug_data['D_rhoE_eta'] = fluxes['D_rhoE_eta'][0].detach().cpu().numpy() if fluxes['D_rhoE_eta'].ndim == 3 else fluxes['D_rhoE_eta'].detach().cpu().numpy()
            # 添加能量对流通量（不含耗散）
            if 'FE_conv_eta' in fluxes and fluxes['FE_conv_eta'] is not None:
                energy_debug_data['FE_conv_eta'] = fluxes['FE_conv_eta'][0].detach().cpu().numpy() if fluxes['FE_conv_eta'].ndim == 3 else fluxes['FE_conv_eta'].detach().cpu().numpy()

            # 添加 ξ 向能量通量（如果存在）
            if 'FE_xi' in fluxes and fluxes['FE_xi'] is not None:
                energy_debug_data['FE_xi'] = fluxes['FE_xi'][0].detach().cpu().numpy() if fluxes['FE_xi'].ndim == 3 else fluxes['FE_xi'].detach().cpu().numpy()
            if 'VE_xi' in fluxes and fluxes['VE_xi'] is not None:
                energy_debug_data['VE_xi'] = fluxes['VE_xi'][0].detach().cpu().numpy() if fluxes['VE_xi'].ndim == 3 else fluxes['VE_xi'].detach().cpu().numpy()
            if 'D_rhoE_xi' in fluxes and fluxes['D_rhoE_xi'] is not None:
                energy_debug_data['D_rhoE_xi'] = fluxes['D_rhoE_xi'][0].detach().cpu().numpy() if fluxes['D_rhoE_xi'].ndim == 3 else fluxes['D_rhoE_xi'].detach().cpu().numpy()
            if 'FE_conv_xi' in fluxes and fluxes['FE_conv_xi'] is not None:
                energy_debug_data['FE_conv_xi'] = fluxes['FE_conv_xi'][0].detach().cpu().numpy() if fluxes['FE_conv_xi'].ndim == 3 else fluxes['FE_conv_xi'].detach().cpu().numpy()
            # 添加 ξ方向粘性通量分量（用于定位偏差来源）
            if 'viscWork_xi' in fluxes and fluxes['viscWork_xi'] is not None:
                energy_debug_data['viscWork_xi'] = fluxes['viscWork_xi'][0].detach().cpu().numpy() if fluxes['viscWork_xi'].ndim == 3 else fluxes['viscWork_xi'].detach().cpu().numpy()
            if 'heatFlux_xi' in fluxes and fluxes['heatFlux_xi'] is not None:
                energy_debug_data['heatFlux_xi'] = fluxes['heatFlux_xi'][0].detach().cpu().numpy() if fluxes['heatFlux_xi'].ndim == 3 else fluxes['heatFlux_xi'].detach().cpu().numpy()

            np.savez('pytorch_energy_flux_debug.npz', **energy_debug_data)
            print(f"[DEBUG] Saved energy flux to: pytorch_energy_flux_debug.npz")
            print(f"        Keys: {list(energy_debug_data.keys())}")
    # ========== END DEBUG: xi flux ==========

    return fluxes


# ========== 辅助函数：节点梯度四节点平均 ==========

def _interpolate_nodal_to_face_xi(
    dphi_dx_node: torch.Tensor,
    dphi_dy_node: torch.Tensor,
    periodic: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    伪3D节点梯度到ξ面插值（2节点平均）

    对应ADFLOW blockette.F90:5724-5727四节点平均在thin axis=j情况下的退化

    物理意义：
    - ξ面（i-向面）在i+1/2位置，法向垂直于i
    - 真3D四个角点：(i, j, k), (i, j+1, k), (i, j, k+1), (i, j+1, k+1)
    - 伪3D退化（j=0和j=1重合）：仅剩2个不同点 (i, k) 和 (i, k+1)
    - 因此：同一i-face上沿k（即Dataset的H轴）的2节点平均

    Args:
        dphi_dx_node, dphi_dy_node: 节点梯度 (batch, H+1, W+1)
            - H+1: k方向节点数（wall-normal）
            - W+1: i方向节点数（streamwise，周期）
        periodic: ξ方向（i方向）是否周期性

    Returns:
        (dphi_dx_face, dphi_dy_face): ξ面梯度
            - periodic=True: (batch, H, W)
            - periodic=False: (batch, H, W-1)
    """
    batch, H_plus_1, W_plus_1 = dphi_dx_node.shape
    H, W = H_plus_1 - 1, W_plus_1 - 1

    if periodic:
        # ξ面 i+1/2（周期）：W个面
        # 伪3D 2节点平均：沿k方向（H轴），i固定
        #
        # 索引对齐说明（与review3.md一致）：
        # - A_x_xi来自vertex列1..W（seam在W列）
        # - ξ面 face i 对应节点列 i+1
        # - 因此使用节点列 1:W+1 而非 0:W
        #
        # 节点 (k, i+1) 和 (k+1, i+1) 对应ξ面 i
        dphi_dx_face = 0.5 * (
            dphi_dx_node[:, :H, 1:W+1] +    # 节点 (k=0..H-1, i=1..W)
            dphi_dx_node[:, 1:H+1, 1:W+1]   # 节点 (k=1..H, i=1..W)
        )  # (batch, H, W)

        dphi_dy_face = 0.5 * (
            dphi_dy_node[:, :H, 1:W+1] +
            dphi_dy_node[:, 1:H+1, 1:W+1]
        )
    else:
        # 非周期：W-1个内部面
        # 伪3D 2节点平均：沿k方向（H轴），i固定
        # 使用节点列 1:W 而非 0:W-1
        dphi_dx_face = 0.5 * (
            dphi_dx_node[:, :H, 1:W] +      # 节点 (k=0..H-1, i=1..W-1)
            dphi_dx_node[:, 1:H+1, 1:W]     # 节点 (k=1..H, i=1..W-1)
        )  # (batch, H, W-1)

        dphi_dy_face = 0.5 * (
            dphi_dy_node[:, :H, 1:W] +
            dphi_dy_node[:, 1:H+1, 1:W]
        )

    return dphi_dx_face, dphi_dy_face


def _interpolate_nodal_to_face_eta(
    dphi_dx_node: torch.Tensor,
    dphi_dy_node: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    伪3D节点梯度到η面插值（2节点平均）

    对应ADFLOW blockette.F90:5724-5727四节点平均在thin axis=j情况下的退化

    物理意义：
    - η面（k-向面）在k+1/2位置，法向垂直于k
    - 真3D四个角点：(i, j, k), (i+1, j, k), (i, j+1, k), (i+1, j+1, k)
    - 伪3D退化（j=0和j=1重合）：仅剩2个不同点 (i, k) 和 (i+1, k)
    - 因此：同一k-face上沿i（即Dataset的W轴）的2节点平均

    Args:
        dphi_dx_node, dphi_dy_node: 节点梯度 (batch, H+1, W+1)
            - H+1: k方向节点数（wall-normal）
            - W+1: i方向节点数（streamwise，周期）

    Returns:
        (dphi_dx_face, dphi_dy_face): η面梯度 (batch, H+1, W)
            - 包含完整边界面（k=0到k=H，共H+1个面）
    """
    batch, H_plus_1, W_plus_1 = dphi_dx_node.shape
    H, W = H_plus_1 - 1, W_plus_1 - 1

    # η面 k+1/2：H+1个面（包含边界k=0和k=H）
    # 伪3D 2节点平均：沿i方向（W轴），k固定
    # 节点 (k, i) 和 (k, i+1)
    # 注意：k方向不变！仅在i方向取相邻节点
    dphi_dx_face = 0.5 * (
        dphi_dx_node[:, :H+1, :W] +      # 节点 (k=0..H, i=0..W-1)
        dphi_dx_node[:, :H+1, 1:W+1]     # 节点 (k=0..H, i=1..W) - 仅i变化
    )  # (batch, H+1, W)

    dphi_dy_face = 0.5 * (
        dphi_dy_node[:, :H+1, :W] +
        dphi_dy_node[:, :H+1, 1:W+1]
    )

    return dphi_dx_face, dphi_dy_face
