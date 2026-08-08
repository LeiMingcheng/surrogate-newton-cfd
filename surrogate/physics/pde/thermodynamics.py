"""
热力学闭式关系模块

实现ADFLOW对齐的热力学公式，用于能量方程残差计算。

参考ADFLOW源码：
- flowUtils.F90:871 (computePressureSimple)
- flowUtils.F90:488 (computeSpeedOfSoundSquared)
- flowUtils.F90:1205 (computeLamViscosity)
- fluxes.F90:3148 (heatCoef calculation)
"""

import torch
from typing import Optional, Union


def compute_pressure_from_conservatives(
    rho: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    rhoE: torch.Tensor,
    gamma: float = 1.4,
    p_min: Optional[float] = None,
    pInfCorr: float = 1.0
) -> torch.Tensor:
    """
    从守恒变量计算压力（ADFLOW对齐）

    参考：
        - flowUtils.F90:871 (computePressureSimple)
        - flowUtils.F90:923-925 (pressure floor protection)

    公式：p = (gamma-1) * [rhoE - 0.5*rho*(u^2 + v^2)]

    Args:
        rho: 密度 (H, W) 或 (batch, H, W)
        u: x方向速度 (H, W) 或 (batch, H, W)
        v: y方向速度 (H, W) 或 (batch, H, W)
        rhoE: 能量密度 (H, W) 或 (batch, H, W)
        gamma: 比热比（默认1.4）
        p_min: 压力下限（默认：None，使用ADFLOW默认值 1e-4 * pInfCorr）
        pInfCorr: 修正的无穷远压力（ADFLOW默认1.0，无量纲化）

    Returns:
        p: 压力场

    注意：
        - 2D流场：w速度=0，动能仅包含u^2+v^2
        - ADFLOW强制 p = max(p, 1e-4*pInfCorr)（flowUtils.F90:923）
        - 此函数支持float64精度，确保数值稳定性
    """
    # 动能（2D：仅u和v分量）
    kinetic_energy = 0.5 * rho * (u**2 + v**2)

    # 内能
    internal_energy = rhoE - kinetic_energy

    # 压力（理想气体）
    p = (gamma - 1.0) * internal_energy

    # ✅ ADFLOW对齐：压力下限保护（flowUtils.F90:923-925）
    # ADFLOW强制 p = max(p, 1e-4*pInfCorr) 避免负压或过小压力
    if p_min is None:
        p_min = 1e-4 * pInfCorr
    p = torch.clamp(p, min=p_min)

    return p


def compute_speed_of_sound_squared(
    rho: torch.Tensor,
    p: torch.Tensor,
    gamma: float = 1.4
) -> torch.Tensor:
    """
    计算声速平方：a² = gamma * p / rho

    参考：flowUtils.F90:488 (computeSpeedOfSoundSquared)

    注意：
        - ADFLOW粘性通量使用grad(a²)而非grad(T)
        - 原因：数值稳定性更好，避免温度计算的中间步骤

    Args:
        rho: 密度
        p: 压力
        gamma: 比热比（默认1.4）

    Returns:
        aa: 声速平方 (a² = c²)
    """
    # 避免除零
    rho_safe = rho + 1e-14

    # 声速平方
    aa = gamma * p / rho_safe

    return aa


def compute_temperature_from_state(
    rho: torch.Tensor,
    p: torch.Tensor,
    R_gas: float = 287.05
) -> torch.Tensor:
    """
    理想气体状态方程：T = p / (rho * R)

    用于Sutherland粘度公式。

    Args:
        rho: 密度（量纲化）
        p: 压力（量纲化）
        R_gas: 比气体常数（J/(kg·K)，空气默认287.05）

    Returns:
        T: 温度（K）

    注意：
        - 输入必须是量纲化的物理量
        - 对于无量纲化的流场，需要先恢复量纲
    """
    # 避免除零
    rho_safe = rho + 1e-14

    # 温度
    T = p / (rho_safe * R_gas)

    return T


def compute_laminar_viscosity_sutherland(
    T: torch.Tensor,
    T_ref: float = 300.0,
    mu_ref: float = 1.716e-5,
    S_sutherland: float = 110.4
) -> torch.Tensor:
    """
    Sutherland粘度公式

    参考：flowUtils.F90:1205 (computeLamViscosity)

    公式：mu_l = mu_ref * (T/T_ref)^1.5 * (T_ref + S) / (T + S)

    Args:
        T: 温度场（K）
        T_ref: 参考温度（K，默认300K）
        mu_ref: 参考粘度（Pa·s，默认1.716e-5）
        S_sutherland: Sutherland常数（K，默认110.4）

    Returns:
        mu_l: 层流粘度（Pa·s）

    注意：
        - Sutherland公式适用于温度范围：100K ~ 1900K
        - 对于无量纲化流场，需要使用无量纲化的Sutherland公式
    """
    # 温度比
    T_ratio = T / T_ref

    # Sutherland公式
    mu_l = mu_ref * torch.pow(T_ratio, 1.5) * (T_ref + S_sutherland) / (T + S_sutherland)

    return mu_l


def compute_laminar_viscosity_nondimensional(
    Ma: float,
    Re: float,
    gamma: float = 1.4
) -> float:
    """
    无量纲层流粘度（基于马赫数和雷诺数）

    参考：ADFLOW无量纲化约定

    公式：mu_l = (Ma * sqrt(gamma)) / Re

    Args:
        Ma: 马赫数
        Re: 雷诺数
        gamma: 比热比（默认1.4）

    Returns:
        mu_l: 无量纲层流粘度

    注意：
        - 这是无量纲化流场下的粘度计算
        - 与Sutherland公式等效，但直接使用无量纲参数
        - Surrogate-Newton CFD surrogate训练中使用此公式（flow_conditions包含Ma, Re）
    """
    import math
    mu_l = (Ma * math.sqrt(gamma)) / Re
    return mu_l


def compute_heat_conduction_coefficient(
    mu_l: torch.Tensor,
    mu_t: torch.Tensor,
    gamma: float = 1.4,
    Pr_laminar: float = 0.72,
    Pr_turbulent: float = 0.9
) -> torch.Tensor:
    """
    热传导系数（双Prandtl数）

    参考：fluxes.F90:3148 (heatCoef in viscousFlux)

    公式：k_heat = mu_l/(Pr_l*(gamma-1)) + mu_t/(Pr_t*(gamma-1))

    Args:
        mu_l: 层流粘度
        mu_t: 湍流粘度
        gamma: 比热比（默认1.4）
        Pr_laminar: 层流Prandtl数（默认0.72）
        Pr_turbulent: 湍流Prandtl数（默认0.9）

    Returns:
        k_heat: 热传导系数

    注意：
        - 层流和湍流使用不同的Prandtl数
        - k_heat用于计算热传导通量：q = -k_heat * grad(a²)
        - ADFLOW使用grad(a²)而非grad(T)
    """
    # 层流热传导系数
    k_laminar = mu_l / (Pr_laminar * (gamma - 1.0))

    # 湍流热传导系数
    k_turbulent = mu_t / (Pr_turbulent * (gamma - 1.0))

    # 总热传导系数
    k_heat = k_laminar + k_turbulent

    return k_heat


def compute_effective_viscosity(
    mu_l: torch.Tensor,
    mu_t: torch.Tensor
) -> torch.Tensor:
    """
    有效粘度（层流+湍流）

    公式：mu_eff = mu_l + mu_t

    用于：
        - 动量方程粘性应力
        - 能量方程粘性功

    Args:
        mu_l: 层流粘度
        mu_t: 湍流粘度（由SA模型计算）

    Returns:
        mu_eff: 有效粘度
    """
    return mu_l + mu_t


def decompose_effective_viscosity(
    mu_eff: torch.Tensor,
    rho: torch.Tensor,
    nuTilde: torch.Tensor,
    T: torch.Tensor,
    T_ref: float = 300.0,
    mu_ref: float = 1.716e-5,
    S_sutherland: float = 110.4,
    cv1: float = 7.1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    分解有效粘度为层流和湍流部分

    用于需要分离层流/湍流粘度的场景（如双Prandtl数热传导）

    策略：
        1. 从温度计算层流粘度（Sutherland公式）
        2. mu_t = mu_eff - mu_l

    Args:
        mu_eff: 有效粘度
        rho: 密度
        nuTilde: SA变量
        T: 温度
        T_ref: 参考温度
        mu_ref: 参考粘度
        S_sutherland: Sutherland常数
        cv1: SA常数

    Returns:
        (mu_l, mu_t): 层流和湍流粘度

    注意：
        - mu_t保证非负（clamp）
        - 一致性检查：mu_t应与SA公式计算的值接近
    """
    # 计算层流粘度
    mu_l = compute_laminar_viscosity_sutherland(T, T_ref, mu_ref, S_sutherland)

    # 计算湍流粘度
    mu_t = mu_eff - mu_l

    # 确保非负
    mu_t = torch.clamp(mu_t, min=0.0)

    return mu_l, mu_t


# ========== 批量计算工具函数 ==========

def compute_thermodynamic_properties(
    rho: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    rhoE: torch.Tensor,
    gamma: float = 1.4,
    R_gas: float = 287.05,
    Ma: Optional[float] = None,
    Re: Optional[float] = None
) -> dict:
    """
    批量计算热力学性质（一站式接口）

    Args:
        rho, u, v, rhoE: 守恒变量
        gamma: 比热比
        R_gas: 比气体常数
        Ma, Re: 马赫数和雷诺数（用于无量纲粘度）

    Returns:
        properties: {
            'p': 压力,
            'aa': 声速平方,
            'T': 温度（如果提供了R_gas）,
            'mu_l': 层流粘度（如果提供了Ma, Re）
        }

    用途：
        - 简化能量方程残差计算流程
        - 一次性计算所有需要的热力学量
    """
    properties = {}

    # 压力
    p = compute_pressure_from_conservatives(rho, u, v, rhoE, gamma)
    properties['p'] = p

    # 声速平方
    aa = compute_speed_of_sound_squared(rho, p, gamma)
    properties['aa'] = aa

    # 温度（可选）
    if R_gas is not None:
        T = compute_temperature_from_state(rho, p, R_gas)
        properties['T'] = T

    # 层流粘度（无量纲化）
    if Ma is not None and Re is not None:
        mu_l_scalar = compute_laminar_viscosity_nondimensional(Ma, Re, gamma)
        # 广播到与rho相同的形状
        mu_l = torch.full_like(rho, mu_l_scalar)
        properties['mu_l'] = mu_l

    return properties


# ========== 验证工具函数 ==========

def compute_rhoE_from_primitives(
    rho: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    gamma: float = 1.4
) -> torch.Tensor:
    """
    从原始变量计算总能量密度（ADFLOW对齐）

    参考：与compute_pressure_from_conservatives互为逆运算

    公式：rhoE = p/(gamma-1) + 0.5*rho*(u^2 + v^2)

    Args:
        rho: 密度 (H, W) 或 (batch, H, W)
        u: x方向速度 (H, W) 或 (batch, H, W)
        v: y方向速度 (H, W) 或 (batch, H, W)
        p: 压力 (H, W) 或 (batch, H, W)
        gamma: 比热比（默认1.4）

    Returns:
        rhoE: 总能量密度

    注意：
        - 2D流场：w速度=0，动能仅包含u^2+v^2
        - 用于边界条件halo单元的rhoE计算
        - 应与compute_pressure_from_conservatives循环一致

    用途：
        - halo.py中从原始变量计算rhoE边界值
        - 验证EOS一致性测试
    """
    # 内能（理想气体）
    internal_energy = p / (gamma - 1.0)

    # 动能（2D：仅u和v分量）
    kinetic_energy = 0.5 * rho * (u**2 + v**2)

    # 总能量密度
    rhoE = internal_energy + kinetic_energy

    return rhoE


def validate_thermodynamic_consistency(
    rho: torch.Tensor,
    p: torch.Tensor,
    T: torch.Tensor,
    R_gas: float = 287.05,
    tol: float = 1e-6
) -> bool:
    """
    验证热力学一致性：p = rho * R * T

    用于调试和测试。

    Args:
        rho, p, T: 热力学变量
        R_gas: 比气体常数
        tol: 相对容差

    Returns:
        is_consistent: 是否满足理想气体状态方程
    """
    # 从p和rho计算T
    T_computed = compute_temperature_from_state(rho, p, R_gas)

    # 相对误差
    rel_error = torch.abs(T_computed - T) / (torch.abs(T) + 1e-14)

    # 检查最大误差
    max_error = rel_error.max().item()

    return max_error < tol
