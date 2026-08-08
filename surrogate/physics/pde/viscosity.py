"""
粘度模型模块 - Phase 3

提供多种粘度模型：
1. constant_Re: 常粘度，μ = (Ma√γ)/Re
2. sutherland: Sutherland变粘度，μ(T)
3. rans: RANS常数湍粘度，μ_eff = μ_lam + c_turb
4. rans_smagorinsky: Smagorinsky湍流粘度，μ_turb = ρ(CsΔ)²|S|

从 residual backend 中提取，保持功能不变。
"""

import torch
import numpy as np
from typing import Union, Dict, Optional


def compute_mu_constant_Re(
    Ma: Union[float, torch.Tensor],
    Re: Union[float, torch.Tensor],
    gamma: float = 1.4
) -> Union[float, torch.Tensor]:
    """
    常粘度模型（无量纲化）

    μ' = (Ma√γ) / Re

    Args:
        Ma: 马赫数（标量或张量）
        Re: 雷诺数（标量或张量）
        gamma: 比热比（默认1.4）

    Returns:
        mu: 无量纲粘度
    """
    if isinstance(Ma, torch.Tensor) or isinstance(Re, torch.Tensor):
        if isinstance(Ma, (int, float)):
            Ma = torch.tensor(Ma, dtype=torch.float32)
        if isinstance(Re, (int, float)):
            Re = torch.tensor(Re, dtype=torch.float32)

        sqrt_gamma = torch.tensor(gamma, dtype=torch.float32).sqrt()
        mu = (Ma * sqrt_gamma) / (Re + 1e-12)
    else:
        sqrt_gamma = np.sqrt(gamma)
        mu = (Ma * sqrt_gamma) / (Re + 1e-12)

    return mu


def compute_mu_sutherland(
    T_nondim: torch.Tensor,
    T_ref: float = 288.15,
    S_ref: float = 110.4,
    gamma: float = 1.4
) -> torch.Tensor:
    """
    Sutherland变粘度模型

    μ(T) / μ_ref = (T/T_ref)^(3/2) * (T_ref + S) / (T + S)

    Args:
        T_nondim: 无量纲温度场 T' = T/T_∞ (batch, H, W)
        T_ref: 参考温度（K）
        S_ref: Sutherland常数（K）
        gamma: 比热比

    Returns:
        mu: 粘度场 (batch, H, W)
    """
    # 恢复有量纲温度
    T = T_nondim * T_ref

    # Sutherland公式
    mu = torch.pow(T / T_ref, 1.5) * (T_ref + S_ref) / (T + S_ref + 1e-12)

    return mu


def compute_mu_rans_constant(
    Ma: Union[float, torch.Tensor],
    Re: Union[float, torch.Tensor],
    c_turb: float = 0.1,
    gamma: float = 1.4
) -> Union[float, torch.Tensor]:
    """
    RANS常数湍粘度模型

    μ_eff = μ_lam + μ_turb
    μ_turb = c_turb * μ_lam (简化模型)

    Args:
        Ma: 马赫数
        Re: 雷诺数
        c_turb: 湍粘度系数（默认0.1）
        gamma: 比热比

    Returns:
        mu_eff: 有效粘度
    """
    mu_lam = compute_mu_constant_Re(Ma, Re, gamma)
    mu_eff = mu_lam * (1.0 + c_turb)

    return mu_eff


def compute_mu_rans_smagorinsky(
    rho: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    face_geom: Dict[str, torch.Tensor],
    vol: torch.Tensor,
    Cs: float = 0.12,
    Ma: Optional[float] = None,
    Re: Optional[float] = None,
    gamma: float = 1.4,
    wall_damp: bool = True,
    clamp_ratio: float = 100.0
) -> torch.Tensor:
    """
    Smagorinsky湍流粘度模型（空间变化）

    μ_turb = ρ(CsΔ)²|S|

    其中：
    - Δ = √Vol (特征长度尺度)
    - |S| = √(2S_ij S_ij) (应变率张量模)
    - S_ij = 0.5(∂u_i/∂x_j + ∂u_j/∂x_i)

    Args:
        rho: 密度场 (batch, H, W) 或 (H, W)
        u, v: 速度场
        face_geom: 面几何（来自geometry.compute_face_geometry）
        vol: 单元体积（来自geometry.compute_cell_volume）
        Cs: Smagorinsky常数（0.1-0.17，默认0.12）
        Ma, Re: 马赫数和雷诺数（用于计算层流粘度）
        gamma: 比热比
        wall_damp: 是否启用近壁抑制（van Driest damping）
        clamp_ratio: 湍粘度/层流粘度的最大比值

    Returns:
        mu_eff: 有效粘度场 (batch, H, W) 或 (H, W)
    """
    # 添加batch维度（如果没有）
    if rho.ndim == 2:
        rho = rho.unsqueeze(0)
        u = u.unsqueeze(0)
        v = v.unsqueeze(0)
        vol = vol.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size, H, W = rho.shape

    # 1. 计算速度梯度（使用面几何信息）
    # 简化版：使用中心差分近似
    du_dx = torch.zeros_like(u)
    du_dy = torch.zeros_like(u)
    dv_dx = torch.zeros_like(u)
    dv_dy = torch.zeros_like(u)

    # ξ方向梯度（简化为差分）
    periodic_xi = face_geom.get('periodic_xi', True)

    if periodic_xi:
        du_dx[:, :, 1:-1] = 0.5 * (u[:, :, 2:] - u[:, :, :-2])
        dv_dx[:, :, 1:-1] = 0.5 * (v[:, :, 2:] - v[:, :, :-2])

        du_dx[:, :, 0] = 0.5 * (u[:, :, 1] - u[:, :, -1])
        du_dx[:, :, -1] = 0.5 * (u[:, :, 0] - u[:, :, -2])

        dv_dx[:, :, 0] = 0.5 * (v[:, :, 1] - v[:, :, -1])
        dv_dx[:, :, -1] = 0.5 * (v[:, :, 0] - v[:, :, -2])
    else:
        du_dx[:, :, 1:-1] = 0.5 * (u[:, :, 2:] - u[:, :, :-2])
        dv_dx[:, :, 1:-1] = 0.5 * (v[:, :, 2:] - v[:, :, :-2])

        du_dx[:, :, 0] = u[:, :, 1] - u[:, :, 0]
        du_dx[:, :, -1] = u[:, :, -1] - u[:, :, -2]

        dv_dx[:, :, 0] = v[:, :, 1] - v[:, :, 0]
        dv_dx[:, :, -1] = v[:, :, -1] - v[:, :, -2]

    # η方向梯度
    if H <= 1:
        du_dy.zero_()
        dv_dy.zero_()
    else:
        du_dy[:, 1:-1, :] = 0.5 * (u[:, 2:, :] - u[:, :-2, :])
        dv_dy[:, 1:-1, :] = 0.5 * (v[:, 2:, :] - v[:, :-2, :])

        du_dy[:, 0, :] = u[:, 1, :] - u[:, 0, :]
        du_dy[:, -1, :] = u[:, -1, :] - u[:, -2, :]

        dv_dy[:, 0, :] = v[:, 1, :] - v[:, 0, :]
        dv_dy[:, -1, :] = v[:, -1, :] - v[:, -2, :]

    # 2. 计算应变率张量模
    # S_11 = du/dx, S_22 = dv/dy, S_12 = 0.5(du/dy + dv/dx)
    S_11 = du_dx
    S_22 = dv_dy
    S_12 = 0.5 * (du_dy + dv_dx)

    # |S| = sqrt(2 * (S_11² + S_22² + 2*S_12²))
    S_mag = torch.sqrt(2.0 * (S_11**2 + S_22**2 + 2.0 * S_12**2) + 1e-12)

    # 3. 特征长度尺度 Δ = sqrt(Vol)
    Delta = torch.sqrt(vol + 1e-12)

    # 4. 湍流粘度
    mu_turb = rho * (Cs * Delta)**2 * S_mag

    # 5. 近壁抑制（van Driest damping）
    if wall_damp:
        # 简化：j方向（η）作为壁面法向
        # 壁面在j=0，抑制函数 f = 1 - exp(-y+/A+)，A+≈26
        # 这里简化为基于j索引的线性抑制
        j_coords = torch.arange(H, dtype=torch.float32, device=rho.device).view(1, H, 1)
        denom = max(H - 1, 1)
        wall_distance_norm = j_coords / denom  # 归一化壁距 0→1

        # van Driest damping (简化版)
        A_plus = 26.0
        y_plus_approx = wall_distance_norm * 100.0  # 粗略估计
        damp_factor = 1.0 - torch.exp(-y_plus_approx / A_plus)

        mu_turb = mu_turb * damp_factor

    # 6. 计算层流粘度（如果提供Ma和Re）
    if Ma is not None and Re is not None:
        mu_lam = compute_mu_constant_Re(Ma, Re, gamma)
    else:
        mu_lam = 0.0

    # 7. 有效粘度
    mu_eff = mu_lam + mu_turb

    # 8. Clamp湍粘度比（防止过大）
    if Ma is not None and Re is not None and clamp_ratio > 0:
        mu_turb_clamped = torch.clamp(mu_turb, max=clamp_ratio * mu_lam)
        mu_eff = mu_lam + mu_turb_clamped

    # 移除batch维度（如果输入没有）
    if squeeze_output:
        mu_eff = mu_eff.squeeze(0)

    return mu_eff


def get_turb_ratio_adaptive(Re: float) -> float:
    """
    根据雷诺数自适应选择湍粘度比（来自 residual backend）

    Args:
        Re: 雷诺数

    Returns:
        turb_ratio: μ_turb / μ_lam
    """
    # 湍粘度比查表（根据Re）
    # 这是经验表，来自原始实现
    TURB_RATIO_TABLE = {
        1e5: 0.5,
        2e5: 1.0,
        5e5: 2.0,
        1e6: 5.0,
        2e6: 10.0,
        5e6: 20.0,
        1e7: 50.0
    }

    # 查找最近的Re
    Re_keys = sorted(TURB_RATIO_TABLE.keys())

    if Re <= Re_keys[0]:
        return TURB_RATIO_TABLE[Re_keys[0]]
    elif Re >= Re_keys[-1]:
        return TURB_RATIO_TABLE[Re_keys[-1]]
    else:
        # 线性插值
        for i in range(len(Re_keys) - 1):
            if Re_keys[i] <= Re <= Re_keys[i+1]:
                Re_low, Re_high = Re_keys[i], Re_keys[i+1]
                ratio_low = TURB_RATIO_TABLE[Re_low]
                ratio_high = TURB_RATIO_TABLE[Re_high]

                # 对数插值（Re在对数尺度上）
                log_Re = np.log10(Re)
                log_Re_low = np.log10(Re_low)
                log_Re_high = np.log10(Re_high)

                weight = (log_Re - log_Re_low) / (log_Re_high - log_Re_low + 1e-12)
                turb_ratio = ratio_low + weight * (ratio_high - ratio_low)

                return turb_ratio

    return 1.0  # fallback
