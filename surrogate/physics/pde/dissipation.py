"""
Jameson数值耗散模块

实现ADflow的Jameson 2阶+4阶混合人工耗散格式，用于中央差分稳定化。
完全对标ADflow标量方案 (inviscidDissFluxScalar)。

参考：
- ADflow fluxes.F90: inviscidDissFluxScalar subroutine (line 1351-1738)
- ADflow solverUtils.F90: timeStep_block subroutine (line 43-244)
- Jameson, Schmidt, Turkel (1981): "Numerical Solution of the Euler Equations..."

核心公式（ADflow标量方案）：
1. 压力传感器：dss = |p_{i+1} - 2p_i + p_{i-1}| / (p_{i+1} + 2p_i + p_{i-1} + sslim)
2. 单元中心谱半径：rad = 0.5 * (|u·S| + c·|S|)，其中S为两侧面面积向量之和
3. 面谱半径：rrad = 0.5 * (rad_L + rad_R)
4. 2阶耗散：dis2 = vis2 · rrad · min(dss_max, max(dss_L, dss_R))
5. 4阶耗散：dis4 = max(0, vis4 · rrad - dis2)
6. 耗散通量：D = dis2·(W_R - W_L) - dis4·(W_{i+2} - W_{i-1} - 3·(W_R - W_L))

当前live ADFLOW对齐参数：
- vis2 = 0.25（2阶耗散系数）
- vis4 = 0.0156（4阶耗散系数）
- dss_max = 0.25（传感器上限）
- sslim = 0.001 * pInfCorr（压力下限）
"""

import math

import torch
from typing import Dict, Tuple, Optional, Union


def _compute_effective_jameson_coefficients(
    *,
    vis2: float,
    vis4: float,
    rfil: float = 1.0,
    use_dissipation_continuation: bool = False,
    diss_cont_magnitude: float = 0.0,
    diss_cont_midpoint: float = 20.0,
    diss_cont_sharpness: float = 3.0,
    total_r: Optional[Union[float, torch.Tensor]] = None,
    total_r0: Optional[Union[float, torch.Tensor]] = None,
) -> tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]]:
    fis2 = float(rfil) * float(vis2)
    fis4 = float(rfil) * float(vis4)
    if not bool(use_dissipation_continuation):
        return fis2, fis4

    def _needs_tensor_path(value: Optional[Union[float, torch.Tensor]]) -> bool:
        return torch.is_tensor(value) or isinstance(value, (list, tuple)) or (
            value is not None and hasattr(value, "shape") and not isinstance(value, (int, float))
        )

    fallback = float(diss_cont_magnitude) / (
        1.0 + math.exp(-float(diss_cont_sharpness) * float(diss_cont_midpoint))
    )
    if _needs_tensor_path(total_r) or _needs_tensor_path(total_r0):
        device = None
        if torch.is_tensor(total_r):
            device = total_r.device
        elif torch.is_tensor(total_r0):
            device = total_r0.device

        total_r_t = None if total_r is None else torch.as_tensor(total_r, dtype=torch.float64, device=device)
        total_r0_t = None if total_r0 is None else torch.as_tensor(total_r0, dtype=torch.float64, device=device)
        if total_r_t is None and total_r0_t is None:
            continuation_t = torch.full((1,), fallback, dtype=torch.float64, device=device)
        else:
            ref_t = total_r_t if total_r_t is not None else total_r0_t
            assert ref_t is not None
            continuation_t = torch.full_like(ref_t, fallback, dtype=torch.float64)
            if total_r_t is not None and total_r0_t is not None:
                total_r_t, total_r0_t = torch.broadcast_tensors(total_r_t, total_r0_t)
                continuation_t = torch.full_like(total_r_t, fallback, dtype=torch.float64)
                valid = (
                    torch.isfinite(total_r_t)
                    & torch.isfinite(total_r0_t)
                    & (total_r_t != 0.0)
                    & (total_r0_t != 0.0)
                )
                if bool(valid.any().item()):
                    ratio = torch.clamp(total_r_t[valid] / total_r0_t[valid], min=1e-300)
                    continuation_t = continuation_t.clone()
                    continuation_t[valid] = float(diss_cont_magnitude) / (
                        1.0
                        + torch.exp(
                            -float(diss_cont_sharpness)
                            * (torch.log10(ratio) + float(diss_cont_midpoint))
                        )
                    )
        fis2_t = float(rfil) * (float(vis2) + continuation_t)
        return fis2_t, fis4

    if (
        total_r is None
        or total_r0 is None
        or float(total_r) == 0.0
        or float(total_r0) == 0.0
    ):
        continuation = fallback
    else:
        ratio = max(float(total_r) / float(total_r0), 1e-300)
        continuation = float(diss_cont_magnitude) / (
            1.0
            + math.exp(
                -float(diss_cont_sharpness)
                * (math.log10(ratio) + float(diss_cont_midpoint))
            )
        )
    fis2 = float(rfil) * (float(vis2) + continuation)
    return fis2, fis4


def _broadcast_batch_coeff_like(
    coeff: Union[float, torch.Tensor],
    target: torch.Tensor,
) -> Union[float, torch.Tensor]:
    if not torch.is_tensor(coeff):
        return coeff
    if coeff.ndim == 0 or coeff.ndim >= target.ndim:
        return coeff
    coeff_batch = int(coeff.shape[0])
    target_batch = int(target.shape[0])
    if coeff_batch == 1 and target_batch > 1:
        coeff = coeff.expand(target_batch)
    elif coeff_batch == target_batch:
        pass
    elif target_batch % coeff_batch == 0:
        coeff = coeff.repeat(int(target_batch // coeff_batch))
    else:
        raise ValueError(
            "batch coefficient mismatch in dissipation broadcast: "
            f"got coeff batch {coeff_batch} for target batch {target_batch}"
        )
    return coeff.reshape(target_batch, *([1] * (target.ndim - 1)))


def _match_batch_tensor(
    value: Optional[torch.Tensor],
    *,
    target_batch: int,
    name: str,
) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if value.ndim == 2:
        value = value.unsqueeze(0)
    value_batch = int(value.shape[0])
    if value_batch == int(target_batch):
        return value
    if value_batch == 1:
        return value.expand(int(target_batch), *value.shape[1:])
    if int(target_batch) % value_batch == 0:
        reps = [int(target_batch // value_batch)] + [1] * (value.ndim - 1)
        return value.repeat(*reps)
    raise ValueError(
        f"{name} batch mismatch: got {value_batch} for target batch {int(target_batch)}"
    )


def _stack_stencil_fields(*fields: torch.Tensor) -> torch.Tensor:
    """Pack multiple fields so stencil extraction is performed once."""
    return torch.stack(fields, dim=1)


def _extract_stencil_xi(
    stacked_fields: torch.Tensor,
    *,
    periodic_xi: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if periodic_xi:
        return (
            torch.roll(stacked_fields, shifts=1, dims=-1),
            stacked_fields,
            torch.roll(stacked_fields, shifts=-1, dims=-1),
            torch.roll(stacked_fields, shifts=-2, dims=-1),
        )
    left_edge = stacked_fields[..., :1]
    right_edge = stacked_fields[..., -1:]
    return (
        torch.cat([left_edge, stacked_fields[..., :-2]], dim=-1),
        stacked_fields[..., :-1],
        stacked_fields[..., 1:],
        torch.cat([stacked_fields[..., 2:], right_edge], dim=-1),
    )


def _extract_stencil_eta(
    stacked_fields: torch.Tensor,
    *,
    lower_halo: Optional[torch.Tensor] = None,
    upper_halo: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if lower_halo is None:
        lower_halo = stacked_fields[..., :1, :]
    if upper_halo is None:
        upper_halo = stacked_fields[..., -1:, :]
    return (
        torch.cat([lower_halo, stacked_fields[..., :-2, :]], dim=-2),
        stacked_fields[..., :-1, :],
        stacked_fields[..., 1:, :],
        torch.cat([stacked_fields[..., 2:, :], upper_halo], dim=-2),
    )


def compute_ss_with_halo(
    p: torch.Tensor,
    rho: torch.Tensor,
    gamma: float,
    halo_wall: Optional[torch.Tensor] = None,
    halo_farfield: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    构造包含halo层的熵场 ss_halo (batch, H+2, W)

    与plan74.md第3点一致：让dss在η方向对边界halo友好

    ADflow dss计算(blockette.F90:3210-3223)访问k+1和k-1单元，
    包括边界处的halo单元。PyTorch需要构造包含halo的熵场来正确计算边界dss。

    Args:
        p: 压力场 (batch, H, W) 或 (H, W)
        rho: 密度场，形状与p相同
        gamma: 比热比
        halo_wall: 壁面halo单元物理量 (batch, C, W) 或 (C, W)，C=4 [rho,u,v,p]
        halo_farfield: 远场halo单元物理量（可选，不提供则使用自由流状态）

    Returns:
        ss_halo: 包含halo层的熵场 (batch, H+2, W)
            - ss_halo[:, 0, :]: 壁面halo的熵
            - ss_halo[:, 1:H+1, :]: 物理单元的熵
            - ss_halo[:, H+1, :]: 远场halo的熵 = 1.0（自由流状态）
    """
    # 添加batch维度
    if p.ndim == 2:
        p = p.unsqueeze(0)
        rho = rho.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch, H, W = p.shape

    # 物理单元熵: ss = p / rho^gamma (ADflow RANS模式)
    # 数值安全：神经网络预测可能出现极少量 rho<=0 的点，直接做非整数幂会产生NaN。
    # 这里对 rho 做下限裁剪，避免在耗散/残差诊断中传播NaN（物理上 rho<=0 本身已不可接受）。
    rho_safe = torch.clamp(rho, min=1e-12)
    ss_physical = p / (rho_safe ** gamma)  # (batch, H, W)

    # 壁面halo熵
    if halo_wall is not None:
        # 确保halo_wall有batch维度
        if halo_wall.ndim == 2:  # (C, W)
            halo_wall = halo_wall.unsqueeze(0)  # (1, C, W)

        # 提取halo物理量: [rho, u, v, p]
        rho_wall_halo = halo_wall[:, 0:1, :]  # (batch, 1, W)
        p_wall_halo = halo_wall[:, 3:4, :]    # (batch, 1, W)

        # 计算halo熵
        rho_wall_safe = torch.clamp(rho_wall_halo, min=1e-12)
        ss_wall = p_wall_halo / (rho_wall_safe ** gamma)
    else:
        # 回退：使用物理层第一层值（简单复制，与原逻辑一致）
        ss_wall = ss_physical[:, 0:1, :]

    # 远场halo熵
    if halo_farfield is not None:
        # 确保halo_farfield有batch维度
        if halo_farfield.ndim == 2:  # (C, W)
            halo_farfield = halo_farfield.unsqueeze(0)

        rho_far_halo = halo_farfield[:, 0:1, :]
        p_far_halo = halo_farfield[:, 3:4, :]
        rho_far_safe = torch.clamp(rho_far_halo, min=1e-12)
        ss_farfield = p_far_halo / (rho_far_safe ** gamma)
    else:
        # 自由流状态: p_inf = 1.0, rho_inf = 1.0 → ss_inf = 1.0
        # 这是ADflow的标准做法（无量纲参考状态）
        ss_farfield = torch.ones((batch, 1, W), device=p.device, dtype=p.dtype)

    # 拼接: [壁面halo, 物理单元, 远场halo] → (batch, H+2, W)
    ss_halo = torch.cat([ss_wall, ss_physical, ss_farfield], dim=-2)

    if squeeze_output:
        ss_halo = ss_halo.squeeze(0)

    return ss_halo


def compute_ss_with_two_halos(
    p: torch.Tensor,
    rho: torch.Tensor,
    gamma: float,
    *,
    halo_wall: Optional[torch.Tensor],
    halo_farfield: Optional[torch.Tensor],
    basis: str = "entropy",
) -> torch.Tensor:
    """
    Build the eta-direction shock-sensor field including two halo layers.

    This matches the eta Jameson dissipation path where the frozen sensor is used
    on `[halo2, halo1, physical, halo1, halo2]`, i.e. shape `(batch, H+4, W)`.
    """
    if p.ndim == 2:
        p = p.unsqueeze(0)
        rho = rho.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    if halo_wall is None or halo_farfield is None:
        raise ValueError("compute_ss_with_two_halos requires halo_wall and halo_farfield.")

    if halo_wall.ndim == 2:
        halo_wall = halo_wall.unsqueeze(0)
    if halo_farfield.ndim == 2:
        halo_farfield = halo_farfield.unsqueeze(0)

    rho_bot = rho[:, 0:1, :]
    p_bot = p[:, 0:1, :]
    rho_top = rho[:, -1:, :]
    p_top = p[:, -1:, :]

    rho_h1_w = halo_wall[:, 0:1, :]
    p_h1_w = halo_wall[:, 3:4, :]
    rho_h1_f = halo_farfield[:, 0:1, :]
    p_h1_f = halo_farfield[:, 3:4, :]

    factor = 0.5
    rho_h2_w = torch.maximum(factor * rho_h1_w, 2.0 * rho_h1_w - rho_bot)
    p_h2_w = torch.maximum(factor * p_h1_w, 2.0 * p_h1_w - p_bot)
    rho_h2_f = torch.maximum(factor * rho_h1_f, 2.0 * rho_h1_f - rho_top)
    p_h2_f = torch.maximum(factor * p_h1_f, 2.0 * p_h1_f - p_top)

    if basis == "entropy":
        rho_safe = torch.clamp(rho, min=1e-12)
        rho_h1_w_safe = torch.clamp(rho_h1_w, min=1e-12)
        rho_h1_f_safe = torch.clamp(rho_h1_f, min=1e-12)
        rho_h2_w_safe = torch.clamp(rho_h2_w, min=1e-12)
        rho_h2_f_safe = torch.clamp(rho_h2_f, min=1e-12)
        ss = p / (rho_safe ** gamma)
        ss_h1_w = p_h1_w / (rho_h1_w_safe ** gamma)
        ss_h1_f = p_h1_f / (rho_h1_f_safe ** gamma)
        ss_h2_w = p_h2_w / (rho_h2_w_safe ** gamma)
        ss_h2_f = p_h2_f / (rho_h2_f_safe ** gamma)
    elif basis == "pressure":
        ss = p
        ss_h1_w = p_h1_w
        ss_h1_f = p_h1_f
        ss_h2_w = p_h2_w
        ss_h2_f = p_h2_f
    else:
        raise ValueError(f"Unsupported shock sensor basis: {basis}")

    ss_ext = torch.cat([ss_h2_w, ss_h1_w, ss, ss_h1_f, ss_h2_f], dim=-2)
    if squeeze_output:
        ss_ext = ss_ext.squeeze(0)
    return ss_ext


def compute_pressure_sensor(
    p: torch.Tensor,
    rho: Optional[torch.Tensor] = None,
    gamma: float = 1.4,
    direction: str = 'xi',
    periodic_xi: bool = False,
    sslim: float = 1e-3,
    basis: str = 'entropy',
    ss_halo: Optional[torch.Tensor] = None,
    shock_sensor: Optional[torch.Tensor] = None,
    shock_sensor_halo: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    计算激波传感器（ADflow标量方案的dss数组）

    ADflow blockette.F90 行3173-3223:
    - Euler方程: 使用压力 ss = p, sslim = 0.001 * pInfCorr
    - NS/RANS方程: 使用熵 ss = p / rho^gamma, sslim = 0.001 * pInfCorr / rhoInf^gammaInf

    公式：dss = |ss_{i+1} - 2ss_i + ss_{i-1}| / (ss_{i+1} + 2ss_i + ss_{i-1} + sslim)

    **η方向边界修复（plan74.md第3点）**：
    当direction='eta'且提供ss_halo时，使用预计算的(H+2, W)熵场来正确处理边界，
    避免简单的边界复制导致远场dss计算错误。

    Args:
        p: 压力场 (batch, H, W) 或 (H, W)
        rho: 密度场，与p形状相同。仅basis='entropy'时需要
        gamma: 比热比（默认1.4）
        direction: 'xi' 或 'eta'
        periodic_xi: ξ方向是否周期（仅direction='xi'时有效）
        sslim: 传感器下限
            - 压力基底: 0.001 * pInfCorr
            - 熵基底: 0.001 * pInfCorr / rhoInf^gammaInf
        basis: 'entropy'（RANS/NS，默认）或 'pressure'（Euler）
        ss_halo: 预计算的熵场 (batch, H+2, W)，包含壁面和远场halo层
            - 仅direction='eta'时使用
            - 由compute_ss_with_halo()生成

    Returns:
        dss: 激波传感器，形状与p相同
            - 值域：[0, 1]，激波/间断处接近1，光滑区域接近0
    """
    # 添加batch维度
    if p.ndim == 2:
        p = p.unsqueeze(0)
        if rho is not None:
            rho = rho.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch, H, W = p.shape

    # 计算传感器基底变量 ss
    if shock_sensor is not None:
        ss = _match_batch_tensor(
            shock_sensor,
            target_batch=batch,
            name="shock_sensor",
        )
        assert ss is not None
        shock_sensor_halo = _match_batch_tensor(
            shock_sensor_halo,
            target_batch=batch,
            name="shock_sensor_halo",
        )
    elif basis == 'entropy' and rho is not None:
        # RANS/NS模式：使用熵 ss = p / rho^gamma (ADflow blockette.F90:3203)
        rho_safe = torch.clamp(rho, min=1e-12)
        ss = p / (rho_safe ** gamma)
    else:
        # Euler模式或回退：使用压力 (ADflow blockette.F90:3187)
        ss = p

    ss_halo = _match_batch_tensor(
        ss_halo,
        target_batch=batch,
        name="ss_halo",
    )

    # 计算邻居值：ss_{i+1} 和 ss_{i-1}
    if direction == 'xi':
        if periodic_xi:
            # 周期边界：使用环绕索引
            ss_plus = torch.cat([ss[..., :, 1:], ss[..., :, :1]], dim=-1)
            ss_minus = torch.cat([ss[..., :, -1:], ss[..., :, :-1]], dim=-1)
        else:
            # 非周期：边界填充（ADflow使用halo单元，这里用边界值近似）
            ss_plus = torch.cat([ss[..., :, 1:], ss[..., :, -1:]], dim=-1)
            ss_minus = torch.cat([ss[..., :, :1], ss[..., :, :-1]], dim=-1)

    elif direction == 'eta':
        # η方向：使用ss_halo（如果提供）或边界填充
        if shock_sensor_halo is not None:
            ss = shock_sensor_halo[:, 1:-1, :]
            ss_plus = shock_sensor_halo[:, 2:, :]
            ss_minus = shock_sensor_halo[:, :-2, :]
        elif ss_halo is not None:
            # **plan74.md修复**：使用预计算的ss_halo包含正确的边界halo值
            # ss_halo形状: (batch, H+2, W)
            # - ss_halo[:, 0, :]: 壁面halo
            # - ss_halo[:, 1:H+1, :]: 物理单元
            # - ss_halo[:, H+1, :]: 远场halo

            # 确保ss_halo有正确的batch维度
            # 从ss_halo提取：ss是物理单元部分
            ss = ss_halo[:, 1:-1, :]  # (batch, H, W)

            # ss_plus = ss[j+1]，对于j=H-1（最后一层），访问远场halo
            ss_plus = ss_halo[:, 2:, :]  # (batch, H, W)

            # ss_minus = ss[j-1]，对于j=0（第一层），访问壁面halo
            ss_minus = ss_halo[:, :-2, :]  # (batch, H, W)
        else:
            # 回退：简单边界填充（原逻辑，仅用于无halo时）
            ss_plus = torch.cat([ss[..., 1:, :], ss[..., -1:, :]], dim=-2)
            ss_minus = torch.cat([ss[..., :1, :], ss[..., :-1, :]], dim=-2)

    else:
        raise ValueError(f"Invalid direction: {direction}")

    # ADflow标量方案公式 (blockette.F90:3213-3214)
    # 分子：二阶差分
    d2ss = ss_plus - 2.0 * ss + ss_minus

    # 分母：三点之和 + sslim
    sum_ss = ss_plus + 2.0 * ss + ss_minus
    denom = sum_ss + sslim

    # 传感器
    dss = torch.abs(d2ss) / denom

    # 移除batch维度
    if squeeze_output:
        dss = dss.squeeze(0)

    return dss


def apply_anisotropic_scaling_3d(
    ri: torch.Tensor,
    rj: torch.Tensor,
    rk: torch.Tensor,
    adis: float = 0.67
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ADflow 3D 各向异性谱半径缩放（solverUtils.F90:213-224）

    公式：
        rij = (ri / rj)^adis
        rjk = (rj / rk)^adis
        rki = (rk / ri)^adis

        radI = ri * (1 + 1/rij + rki)
        radJ = rj * (1 + 1/rjk + rij)
        radK = rk * (1 + 1/rki + rjk)

    对于 Surrogate-Newton CFD surrogate pseudo-2D 网格的方向映射：
        - ri: ξ方向（沿翼型周向）
        - rj: 薄向（spanwise，z方向，厚度=1）
        - rk: η方向（壁面法向）

    Args:
        ri: ξ方向单元中心谱半径 (batch, H, W) 或 (H, W)
        rj: 薄向（j方向）谱半径，形状与ri相同
        rk: η方向单元中心谱半径，形状与ri相同
        adis: 各向异性缩放指数（ADflow实际值 0.67）

    Returns:
        (radI, radJ, radK): 缩放后的三方向谱半径
    """
    # ADflow eps=1e-25 (constants.F90:19)，PyTorch直接对齐
    eps = 1e-25

    # 避免除零（仅分母保护）
    ri_safe = torch.clamp(ri, min=eps)
    rj_safe = torch.clamp(rj, min=eps)
    rk_safe = torch.clamp(rk, min=eps)

    # 计算缩放比例
    rij = (ri_safe / rj_safe) ** adis
    rjk = (rj_safe / rk_safe) ** adis
    rki = (rk_safe / ri_safe) ** adis

    # ADflow 3D 公式：严格除法 1/rij，无+eps保护
    # solverUtils.F90:213-224
    radI = ri_safe * (1.0 + 1.0 / rij + rki)
    radJ = rj_safe * (1.0 + 1.0 / rjk + rij)
    radK = rk_safe * (1.0 + 1.0 / rki + rjk)

    return radI, radJ, radK


def compute_rj_thin(
    rho: torch.Tensor,
    p: torch.Tensor,
    vol: torch.Tensor,
    gamma: float = 1.4,
    acoustic_scale_factor: float = 1.0
) -> torch.Tensor:
    """
    计算薄向（spanwise j方向）谱半径

    对于 Surrogate-Newton CFD surrogate pseudo-2D 网格（z厚度=1）：
        sj(j-1) + sj(j) ≈ 2 * vol（两侧j面面积向量之和）
        qsj ≈ 0（2D流动，w=0）
        rj = 0.5 * acousticScaleFactor * c * |sj_sum| = c * vol

    Args:
        rho: 密度场 (batch, H, W) 或 (H, W)
        p: 压力场，形状与rho相同
        vol: 单元体积，形状与rho相同
        gamma: 比热比
        acoustic_scale_factor: 声学缩放因子

    Returns:
        rj: 薄向谱半径
    """
    c = torch.sqrt(gamma * p / (rho + 1e-12))
    # ADflow: rj = 0.5 * (|qsj| + c * |sj_sum|)
    # 对于 2D: qsj ≈ 0, |sj_sum| ≈ 2*vol
    rj = acoustic_scale_factor * c * vol
    return rj


def _compute_raw_spectral_radius_single_direction(
    rho: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    face_geom: Dict[str, torch.Tensor],
    direction: str,
    gamma: float = 1.4,
    acoustic_scale_factor: float = 1.0
) -> torch.Tensor:
    """
    计算单个方向的原始谱半径（未缩放）

    这是从compute_cell_centered_spectral_radius提取的核心逻辑。

    Args:
        rho, u, v, p: 物理场 (batch, H, W)，已有batch维度
        face_geom: 面几何字典
        direction: 'xi' 或 'eta'
        gamma: 比热比
        acoustic_scale_factor: 声学贡献因子

    Returns:
        rad: 原始单元中心谱半径 (batch, H, W)
    """
    batch, H, W = rho.shape

    # 计算音速平方：cc2 = gamma * p / rho
    cc2 = gamma * p / (rho + 1e-12)

    if direction == 'xi':
        A_x = face_geom['A_x_xi']
        A_y = face_geom['A_y_xi']
        periodic = face_geom.get('periodic_xi', False)

        # 确保有batch维度
        if A_x.ndim == 2:
            A_x = A_x.unsqueeze(0)
            A_y = A_y.unsqueeze(0)

        # ADflow: sx = si(i-1) + si(i)
        if periodic:
            if int(A_x.shape[-1]) != W:
                raise ValueError(
                    f"Periodic xi expects W faces, got A_x width={int(A_x.shape[-1])}, W={W}"
                )
            A_x_eff = A_x
            A_y_eff = A_y
        else:
            if int(A_x.shape[-1]) == W - 1:
                A_x_eff = torch.zeros(batch, H, W, device=A_x.device, dtype=A_x.dtype)
                A_y_eff = torch.zeros(batch, H, W, device=A_y.device, dtype=A_y.dtype)
                A_x_eff[..., :W-1] = A_x
                A_y_eff[..., :W-1] = A_y
            elif int(A_x.shape[-1]) == W:
                A_x_eff = A_x
                A_y_eff = A_y
            else:
                raise ValueError(
                    "Non-periodic xi expects W or W-1 faces, "
                    f"got A_x width={int(A_x.shape[-1])}, W={W}"
                )

        A_x_left = torch.roll(A_x_eff, shifts=1, dims=-1)
        A_y_left = torch.roll(A_y_eff, shifts=1, dims=-1)

        sx = A_x_left + A_x_eff
        sy = A_y_left + A_y_eff

    elif direction == 'eta':
        A_x_full = face_geom['A_x_eta']
        A_y_full = face_geom['A_y_eta']

        if A_x_full.ndim == 2:
            A_x_full = A_x_full.unsqueeze(0)
            A_y_full = A_y_full.unsqueeze(0)

        H_faces = A_x_full.shape[-2]

        if H_faces == H + 1:
            # 完整面：包含边界面
            A_x_left = A_x_full[..., :-1, :]
            A_y_left = A_y_full[..., :-1, :]
            A_x_right = A_x_full[..., 1:, :]
            A_y_right = A_y_full[..., 1:, :]
        elif H_faces == H - 1:
            # 内部面：需要扩展
            A_x_left = torch.cat([A_x_full[..., :1, :], A_x_full[..., :-1, :]], dim=-2)
            A_y_left = torch.cat([A_y_full[..., :1, :], A_y_full[..., :-1, :]], dim=-2)
            A_x_right = torch.cat([A_x_full[..., 1:, :], A_x_full[..., -1:, :]], dim=-2)
            A_y_right = torch.cat([A_y_full[..., 1:, :], A_y_full[..., -1:, :]], dim=-2)
        else:
            A_x_left = torch.cat([A_x_full[..., :1, :], A_x_full[..., :-1, :]], dim=-2)
            A_y_left = torch.cat([A_y_full[..., :1, :], A_y_full[..., :-1, :]], dim=-2)
            A_x_right = A_x_full
            A_y_right = A_y_full

        sx = A_x_left + A_x_right
        sy = A_y_left + A_y_right

    else:
        raise ValueError(f"Invalid direction: {direction}")

    # ADflow公式：qsi = u*sx + v*sy
    qsi = u * sx + v * sy

    # 面积向量之和的模平方
    s_mag_sq = sx**2 + sy**2

    # ADflow物理下限（solverUtils.F90:139-140）
    clim2 = 1e-6 * gamma
    cc2_safe = torch.clamp(cc2, min=clim2)
    s_mag_sq_safe = torch.clamp(s_mag_sq, min=0.0)

    # 单元中心谱半径
    rad = 0.5 * (torch.abs(qsi) + acoustic_scale_factor * torch.sqrt(cc2_safe * s_mag_sq_safe))

    return rad


def compute_raw_spectral_radius_all_directions(
    rho: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    vol: torch.Tensor,
    face_geom: Dict[str, torch.Tensor],
    gamma: float = 1.4,
    acoustic_scale_factor: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算三个方向的原始谱半径（未缩放），用于 ADflow 3D 各向异性缩放

    方向映射（Surrogate-Newton CFD surrogate pseudo-2D 网格）：
        - ri: ξ方向（沿翼型周向）
        - rj: 薄向（spanwise j方向，z厚度=1）
        - rk: η方向（壁面法向）

    Args:
        rho, u, v, p: 物理场 (batch, H, W) 或 (H, W)
        vol: 单元体积，形状与物理场相同
        face_geom: 面几何字典
        gamma: 比热比
        acoustic_scale_factor: 声学贡献因子

    Returns:
        (ri, rj, rk): 三个方向的单元中心原始谱半径
    """
    # 添加batch维度
    squeeze_output = False
    if rho.ndim == 2:
        rho = rho.unsqueeze(0)
        u = u.unsqueeze(0)
        v = v.unsqueeze(0)
        p = p.unsqueeze(0)
        vol = vol.unsqueeze(0)
        squeeze_output = True

    # ξ方向谱半径
    ri = _compute_raw_spectral_radius_single_direction(
        rho, u, v, p, face_geom, 'xi', gamma, acoustic_scale_factor
    )

    # 薄向（j方向）谱半径
    rj = compute_rj_thin(rho, p, vol, gamma, acoustic_scale_factor)

    # η方向谱半径
    rk = _compute_raw_spectral_radius_single_direction(
        rho, u, v, p, face_geom, 'eta', gamma, acoustic_scale_factor
    )

    # 移除batch维度
    if squeeze_output:
        ri = ri.squeeze(0)
        rj = rj.squeeze(0)
        rk = rk.squeeze(0)

    return ri, rj, rk


def compute_cell_centered_spectral_radius_pair(
    rho: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    face_geom: Dict[str, torch.Tensor],
    vol: Optional[torch.Tensor] = None,
    gamma: float = 1.4,
    acoustic_scale_factor: float = 1.0,
    adis: float = 2.0 / 3.0,
    apply_anisotropic: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the xi/eta cell-centered radii once for a shared state."""
    if rho.ndim == 2:
        rho = rho.unsqueeze(0)
        u = u.unsqueeze(0)
        v = v.unsqueeze(0)
        p = p.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    if apply_anisotropic:
        if vol is None:
            raise ValueError(
                "vol (cell volume) is required for 3D anisotropic scaling. "
                "Use compute_cell_volume_adflow() to compute volumes from coords_vertex."
            )
        vol_batch = vol.unsqueeze(0) if vol.ndim == 2 else vol
        ri, rj_thin, rk = compute_raw_spectral_radius_all_directions(
            rho,
            u,
            v,
            p,
            vol_batch,
            face_geom,
            gamma,
            acoustic_scale_factor,
        )
        if ri.ndim == 2:
            ri = ri.unsqueeze(0)
            rj_thin = rj_thin.unsqueeze(0)
            rk = rk.unsqueeze(0)
        rad_xi, _, rad_eta = apply_anisotropic_scaling_3d(ri, rj_thin, rk, adis)
    else:
        rad_xi = _compute_raw_spectral_radius_single_direction(
            rho,
            u,
            v,
            p,
            face_geom,
            'xi',
            gamma,
            acoustic_scale_factor,
        )
        rad_eta = _compute_raw_spectral_radius_single_direction(
            rho,
            u,
            v,
            p,
            face_geom,
            'eta',
            gamma,
            acoustic_scale_factor,
        )

    if squeeze_output:
        rad_xi = rad_xi.squeeze(0)
        rad_eta = rad_eta.squeeze(0)

    return rad_xi, rad_eta


def compute_cell_centered_spectral_radius(
    rho: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    face_geom: Dict[str, torch.Tensor],
    vol: torch.Tensor = None,
    direction: str = 'xi',
    gamma: float = 1.4,
    acoustic_scale_factor: float = 1.0,
    adis: float = 2.0 / 3.0,
    apply_anisotropic: bool = True
) -> torch.Tensor:
    """
    计算单元中心谱半径（ADflow 3D 各向异性缩放）

    ADflow公式（solverUtils.F90:152-159）：
    sx = si(i-1,j,k,1) + si(i,j,k,1)  # 两侧面面积向量x分量之和
    sy = si(i-1,j,k,2) + si(i,j,k,2)  # 两侧面面积向量y分量之和
    qsi = uux*sx + uuy*sy             # 法向速度乘以面积向量之和
    ri = 0.5 * (|qsi| + acousticScaleFactor * sqrt(cc2 * (sx^2+sy^2)))

    ADflow 3D 各向异性缩放（solverUtils.F90:213-224）：
    rij = (ri/rj)^adis, rjk = (rj/rk)^adis, rki = (rk/ri)^adis
    radI = ri * (1 + 1/rij + rki)
    radJ = rj * (1 + 1/rjk + rij)
    radK = rk * (1 + 1/rki + rjk)

    注意：ADflow的si数组是面面积向量，已包含面积大小

    Args:
        rho, u, v, p: 物理场 (batch, H, W) 或 (H, W)
        face_geom: 面几何字典
        vol: 单元体积（3D各向异性缩放必需）
        direction: 'xi' 或 'eta'
        gamma: 比热比（默认1.4）
        acoustic_scale_factor: 声学贡献因子（ADflow默认1.0）
        adis: 各向异性缩放指数（ADflow默认 2/3）
        apply_anisotropic: 是否应用各向异性缩放（默认True）

    Returns:
        rad: 单元中心谱半径（含各向异性缩放），形状与输入场相同
    """
    # 添加batch维度
    if rho.ndim == 2:
        rho = rho.unsqueeze(0)
        u = u.unsqueeze(0)
        v = v.unsqueeze(0)
        p = p.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch, H, W = rho.shape

    # 计算音速平方：cc2 = gamma * p / rho
    cc2 = gamma * p / (rho + 1e-12)

    if direction == 'xi':
        A_x = face_geom['A_x_xi']  # (H, W) 或 (batch, H, W)
        A_y = face_geom['A_y_xi']
        periodic = face_geom.get('periodic_xi', False)

        # 确保有batch维度
        if A_x.ndim == 2:
            A_x = A_x.unsqueeze(0)
            A_y = A_y.unsqueeze(0)

        # ADflow: sx = si(i-1) + si(i)
        # si(i)是单元i右侧面的面积向量
        # 对于单元i，需要左侧面si(i-1)和右侧面si(i)
        if periodic:
            # 周期边界
            A_x_left = torch.cat([A_x[..., :, -1:], A_x[..., :, :-1]], dim=-1)  # si(i-1)
            A_y_left = torch.cat([A_y[..., :, -1:], A_y[..., :, :-1]], dim=-1)
        else:
            # 非周期：边界用最近面近似
            A_x_left = torch.cat([A_x[..., :, :1], A_x[..., :, :-1]], dim=-1)
            A_y_left = torch.cat([A_y[..., :, :1], A_y[..., :, :-1]], dim=-1)

        # 两侧面面积向量之和
        sx = A_x_left + A_x  # (batch, H, W)
        sy = A_y_left + A_y

    elif direction == 'eta':
        A_x_full = face_geom['A_x_eta']  # 可能是 (H+1, W) 或 (H-1, W)
        A_y_full = face_geom['A_y_eta']

        # 确保有batch维度
        if A_x_full.ndim == 2:
            A_x_full = A_x_full.unsqueeze(0)
            A_y_full = A_y_full.unsqueeze(0)

        # 判断面数量
        H_faces = A_x_full.shape[-2]

        if H_faces == H + 1:
            # 完整面：包含边界面
            # sj(j-1)和sj(j)的索引对应
            A_x_left = A_x_full[..., :-1, :]  # sj(j-1): j=0..H-1
            A_y_left = A_y_full[..., :-1, :]
            A_x_right = A_x_full[..., 1:, :]   # sj(j): j=1..H
            A_y_right = A_y_full[..., 1:, :]
        elif H_faces == H - 1:
            # 内部面：需要扩展
            A_x_left = torch.cat([A_x_full[..., :1, :], A_x_full[..., :-1, :]], dim=-2)
            A_y_left = torch.cat([A_y_full[..., :1, :], A_y_full[..., :-1, :]], dim=-2)
            A_x_right = torch.cat([A_x_full[..., 1:, :], A_x_full[..., -1:, :]], dim=-2)
            A_y_right = torch.cat([A_y_full[..., 1:, :], A_y_full[..., -1:, :]], dim=-2)
        else:
            # 假设与单元数相同
            A_x_left = torch.cat([A_x_full[..., :1, :], A_x_full[..., :-1, :]], dim=-2)
            A_y_left = torch.cat([A_y_full[..., :1, :], A_y_full[..., :-1, :]], dim=-2)
            A_x_right = A_x_full
            A_y_right = A_y_full

        # 两侧面面积向量之和
        sx = A_x_left + A_x_right
        sy = A_y_left + A_y_right

    else:
        raise ValueError(f"Invalid direction: {direction}")

    # ADflow公式：
    # qsi = u*sx + v*sy (法向速度乘以面积向量之和)
    qsi = u * sx + v * sy

    # 面积向量之和的模：|S| = sqrt(sx^2 + sy^2)
    s_mag_sq = sx**2 + sy**2

    # ADflow物理下限（solverUtils.F90:139-140）
    clim2 = 1e-6 * gamma  # 无量纲化后 pInfCorr=1, rhoInfCorr=1
    cc2_safe = torch.clamp(cc2, min=clim2)
    s_mag_sq_safe = torch.clamp(s_mag_sq, min=0.0)

    # 单元中心谱半径：rad = 0.5 * (|qsi| + acousticScaleFactor * sqrt(cc2 * S²))
    rad = 0.5 * (torch.abs(qsi) + acoustic_scale_factor * torch.sqrt(cc2_safe * s_mag_sq_safe))

    # 应用 ADflow 3D 各向异性缩放（如果启用）
    if apply_anisotropic:
        if vol is None:
            raise ValueError(
                "vol (cell volume) is required for 3D anisotropic scaling. "
                "Use compute_cell_volume_adflow() to compute volumes from coords_vertex."
            )

        # 确保 vol 有正确的 batch 维度
        vol_batch = vol.unsqueeze(0) if vol.ndim == 2 else vol

        # 计算三个方向的原始谱半径
        ri, rj_thin, rk = compute_raw_spectral_radius_all_directions(
            rho, u, v, p, vol_batch, face_geom, gamma, acoustic_scale_factor
        )

        # 确保有正确的 batch 维度
        if ri.ndim == 2:
            ri = ri.unsqueeze(0)
            rj_thin = rj_thin.unsqueeze(0)
            rk = rk.unsqueeze(0)

        # 应用 ADflow 3D 各向异性缩放
        radI, radJ, radK = apply_anisotropic_scaling_3d(ri, rj_thin, rk, adis)

        # 返回请求方向的缩放后谱半径
        if direction == 'xi':
            rad = radI
        else:  # eta
            rad = radK

    # 移除batch维度
    if squeeze_output:
        rad = rad.squeeze(0)

    return rad


def compute_spectral_radius(
    rho: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    face_geom: Dict[str, torch.Tensor],
    vol: torch.Tensor = None,
    direction: str = 'xi',
    gamma: float = 1.4,
    acoustic_scale_factor: float = 1.0,
    adis: float = 2.0 / 3.0,
    apply_anisotropic: bool = True,
    *,
    precomputed_cell_radius: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    计算面谱半径（ADflow 3D 各向异性缩放 + 面聚合）

    ADflow公式（fluxes.F90:1514）：
    rrad = ppor * (radI(i,j,k) + radI(i+1,j,k))

    其中 ppor = 0.5 for normalFlux，所以：
    rrad = 0.5 * (rad_L + rad_R)

    注意：radI/radK 已经过 ADflow 3D 各向异性缩放（如果启用）

    Args:
        rho, u, v, p: 物理场 (batch, H, W) 或 (H, W)
        face_geom: 面几何字典
        vol: 单元体积（3D各向异性缩放必需）
        direction: 'xi' 或 'eta'
        gamma: 比热比（默认1.4）
        acoustic_scale_factor: 声学贡献因子（默认1.0）
        adis: 各向异性缩放指数（ADflow默认 2/3）
        apply_anisotropic: 是否应用各向异性缩放（默认True）

    Returns:
        rrad: 面谱半径（含各向异性缩放）
            - direction='xi': (batch, H, W) if periodic else (batch, H, W-1)
            - direction='eta': (batch, H-1, W)
    """
    # 添加batch维度
    if rho.ndim == 2:
        rho = rho.unsqueeze(0)
        u = u.unsqueeze(0)
        v = v.unsqueeze(0)
        p = p.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    # 计算单元中心谱半径（含 ADflow 3D 各向异性缩放）
    if precomputed_cell_radius is None:
        rad_cell = compute_cell_centered_spectral_radius(
            rho, u, v, p, face_geom, vol=vol, direction=direction,
            gamma=gamma, acoustic_scale_factor=acoustic_scale_factor,
            adis=adis, apply_anisotropic=apply_anisotropic
        )
    else:
        rad_cell = precomputed_cell_radius

    # 确保rad_cell有batch维度
    if rad_cell.ndim == 2:
        rad_cell = rad_cell.unsqueeze(0)

    # 面聚合：取两侧单元谱半径的平均
    if direction == 'xi':
        periodic = face_geom.get('periodic_xi', False)
        if periodic:
            # 周期边界：W个面
            rad_L = rad_cell  # radI(i)
            rad_R = torch.cat([rad_cell[..., :, 1:], rad_cell[..., :, :1]], dim=-1)  # radI(i+1)
        else:
            # 非周期：W-1个面
            rad_L = rad_cell[..., :, :-1]  # radI(i)
            rad_R = rad_cell[..., :, 1:]   # radI(i+1)

    elif direction == 'eta':
        # η方向：H-1个面
        rad_L = rad_cell[..., :-1, :]  # radJ(j)
        rad_R = rad_cell[..., 1:, :]   # radJ(j+1)

    else:
        raise ValueError(f"Invalid direction: {direction}")

    # ADflow: rrad = ppor * (rad_L + rad_R), ppor=0.5 for normalFlux
    rrad = 0.5 * (rad_L + rad_R)

    # 移除batch维度
    if squeeze_output:
        rrad = rrad.squeeze(0)

    return rrad


def compute_jameson_dissipation(
    rho: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    face_geom: Dict[str, torch.Tensor],
    vol: torch.Tensor = None,
    direction: str = 'xi',
    vis2: float = 0.25,
    vis4: float = 0.0156,
    gamma: float = 1.4,
    dss_max: float = 0.25,
    periodic_xi: bool = False,
    sslim: float = 1e-3,
    acoustic_scale_factor: float = 1.0,
    basis: str = 'entropy',
    adis: float = 0.67,
    apply_anisotropic: bool = True,
    ss_halo: Optional[torch.Tensor] = None,
    halo_wall: Optional[torch.Tensor] = None,
    halo_farfield: Optional[torch.Tensor] = None,
    rhoE: Optional[torch.Tensor] = None,  # ✅ 新增：能量方程支持
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
    rfil: float = 1.0,
    *,
    precomputed_cell_radius: Optional[torch.Tensor] = None,
    return_coeffs: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
           Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    计算Jameson 2阶+4阶混合人工耗散通量（ADflow标量方案 + 各向异性缩放）

    ADflow实现（blockette.F90:3190-3270）：
    1. 计算传感器：dss = f(ss)，其中 ss = p/ρ^γ (RANS) 或 p (Euler)
    2. 计算面谱半径：rrad = 0.5 * (radI(i) + radI(i+1))（含各向异性缩放）
    3. 2阶耗散系数：dis2 = vis2 · rrad · min(dss_max, max(dss_L, dss_R))
    4. 4阶耗散系数：dis4 = max(0, vis4 · rrad - dis2)
    5. 耗散通量：
       - ddw1 = W_R - W_L
       - fs = dis2·ddw1 - dis4·(W_{i+2} - W_{i-1} - 3·ddw1)

    当前live ADFLOW对齐参数（含anisotropic scaling）：
    - vis2 = 0.25
    - vis4 = 0.0156
    - adis = 2/3（各向异性缩放指数）

    Args:
        rho, u, v, p: 物理场 (batch, H, W) 或 (H, W)
        face_geom: 面几何字典
        direction: 'xi' 或 'eta'
        vis2: 2阶耗散系数（ADflow默认0.5）
        vis4: 4阶耗散系数（当前live ADFLOW默认0.0156）
        gamma: 比热比（默认1.4）
        dss_max: 传感器上限（ADflow默认0.25）
        periodic_xi: ξ方向是否周期
        sslim: 传感器下限
            - RANS (entropy): 0.001 * pInfCorr / rhoInf^gammaInf
            - Euler (pressure): 0.001 * pInfCorr
        acoustic_scale_factor: 声学贡献因子（默认1.0）
        basis: 传感器基底 (ADflow blockette.F90:3190-3207)
            - 'entropy': RANS/NS模式，ss = p / rho^gamma（默认）
            - 'pressure': Euler模式，ss = p
        adis: 各向异性缩放指数（ADflow默认 2/3）
        apply_anisotropic: 是否应用各向异性谱半径缩放（默认True）
        ss_halo: 预计算的熵场 (batch, H+2, W)，用于η方向边界修复
            - 仅direction='eta'时使用
            - 由compute_ss_with_halo()生成
        halo_wall: 壁面halo单元物理量 (batch, C, W) 或 (C, W)
            - C=4: [rho,u,v,p] (3方程模式)
            - C=5: [rho,u,v,p,rhoE] (4方程模式，含能量)
            - 仅direction='eta'时使用
            - 用于4阶差分的壁面边界处理（反射BC计算守恒变量）
        rhoE: 能量密度场 (batch, H, W) 或 (H, W)，可选
            - 如果提供，则计算能量方程的耗散通量D_rhoE
            - 与rho完全独立计算（不使用比例近似）

    Returns:
        - 3方程模式（rhoE=None）: (D_rho, D_rhou, D_rhov)
        - 4方程模式（rhoE!=None）: (D_rho, D_rhou, D_rhov, D_rhoE)
        - 当 return_coeffs=True 时，额外返回 (dis2, dis4)：
            * dis2/dis4 形状与耗散面通量一致
            * 常用于局部装配 Jacobian（frozen-coefficient 线性化）

        耗散通量形状：
            - direction='xi': (batch, H, W) if periodic else (batch, H, W-1)
            - direction='eta': (batch, H+1, W) when提供halo_wall和halo_farfield（与ADflow一致）
            - 符号约定：fw(i+1) += D, fw(i) -= D（与ADflow一致）
    """
    # 添加batch维度
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
    frozen_shock_sensor = _match_batch_tensor(
        frozen_shock_sensor,
        target_batch=batch,
        name="frozen_shock_sensor",
    )
    frozen_ss_halo = _match_batch_tensor(
        frozen_ss_halo,
        target_batch=batch,
        name="frozen_ss_halo",
    )
    precomputed_cell_radius = _match_batch_tensor(
        precomputed_cell_radius,
        target_batch=batch,
        name="precomputed_cell_radius",
    )

    # ========== η方向：使用ADflow halo 单元计算完整 H+1 个面耗散 ==========
    # ADflow inviscidDissFluxScalar 的 k-direction 循环包含边界面 (k=1..kl)，
    # 因此需要在 η 方向计算包含 wall 与 farfield 的耗散面通量。
    if direction == 'eta' and halo_wall is not None and halo_farfield is not None:
        # 确保 halo 有 batch 维度: (batch, 4, W)
        if halo_wall.ndim == 2:
            halo_wall_b = halo_wall.unsqueeze(0)
        else:
            halo_wall_b = halo_wall
        if halo_farfield.ndim == 2:
            halo_far_b = halo_farfield.unsqueeze(0)
        else:
            halo_far_b = halo_farfield

        # 物理边界相邻单元（第一层 / 最后一层）
        rho_bot = rho[:, 0:1, :]
        u_bot = u[:, 0:1, :]
        v_bot = v[:, 0:1, :]
        p_bot = p[:, 0:1, :]

        rho_top = rho[:, -1:, :]
        u_top = u[:, -1:, :]
        v_top = v[:, -1:, :]
        p_top = p[:, -1:, :]

        # 第一层 halo（由BC计算得到）
        rho_h1_w = halo_wall_b[:, 0:1, :]
        u_h1_w = halo_wall_b[:, 1:2, :]
        v_h1_w = halo_wall_b[:, 2:3, :]
        p_h1_w = halo_wall_b[:, 3:4, :]

        rho_h1_f = halo_far_b[:, 0:1, :]
        u_h1_f = halo_far_b[:, 1:2, :]
        v_h1_f = halo_far_b[:, 2:3, :]
        p_h1_f = halo_far_b[:, 3:4, :]

        # ✅ 新增：如果halo包含rhoE（通道5，索引4），则提取
        if rhoE is not None and halo_wall_b.shape[1] >= 5:
            rhoE_h1_w = halo_wall_b[:, 4:5, :]
            rhoE_h1_f = halo_far_b[:, 4:5, :]
            rhoE_bot = rhoE[:, 0:1, :]
            rhoE_top = rhoE[:, -1:, :]
        else:
            rhoE_h1_w = None

        # 第二层 halo：ADflow extrapolate2ndHalo (BCRoutines.F90:1870-1918)
        factor = 0.5
        rho_h2_w = torch.maximum(factor * rho_h1_w, 2.0 * rho_h1_w - rho_bot)
        p_h2_w = torch.maximum(factor * p_h1_w, 2.0 * p_h1_w - p_bot)
        u_h2_w = 2.0 * u_h1_w - u_bot
        v_h2_w = 2.0 * v_h1_w - v_bot

        rho_h2_f = torch.maximum(factor * rho_h1_f, 2.0 * rho_h1_f - rho_top)
        p_h2_f = torch.maximum(factor * p_h1_f, 2.0 * p_h1_f - p_top)
        u_h2_f = 2.0 * u_h1_f - u_top
        v_h2_f = 2.0 * v_h1_f - v_top

        # ✅ ADFLOW对齐：halo2 rhoE 从外推的原始变量用EOS重算
        # 参考：BCRoutines.F90:1915-1916 (computeEtot)
        # 注意：能量耗散使用 H = rhoE + p，所以这里计算 H_h2
        if rhoE_h1_w is not None:
            from .thermodynamics import compute_rhoE_from_primitives

            # 使用已外推的原始变量(rho_h2, u_h2, v_h2, p_h2)重算rhoE
            rhoE_h2_w = compute_rhoE_from_primitives(
                rho_h2_w, u_h2_w, v_h2_w, p_h2_w, gamma=gamma
            )
            rhoE_h2_f = compute_rhoE_from_primitives(
                rho_h2_f, u_h2_f, v_h2_f, p_h2_f, gamma=gamma
            )

            # 计算焓 H = rhoE + p（用于能量耗散）
            H_h2_w = rhoE_h2_w + p_h2_w
            H_h2_f = rhoE_h2_f + p_h2_f

            # halo1 的焓（从已有的 rhoE_h1 和 p_h1 计算）
            H_h1_w = rhoE_h1_w + p_h1_w
            H_h1_f = rhoE_h1_f + p_h1_f

            # 物理场的焓
            H = rhoE + p

        # 构造扩展原始变量（两层 halo）
        rho_ext = torch.cat([rho_h2_w, rho_h1_w, rho, rho_h1_f, rho_h2_f], dim=-2)  # (B, H+4, W)
        u_ext = torch.cat([u_h2_w, u_h1_w, u, u_h1_f, u_h2_f], dim=-2)
        v_ext = torch.cat([v_h2_w, v_h1_w, v, v_h1_f, v_h2_f], dim=-2)
        p_ext = torch.cat([p_h2_w, p_h1_w, p, p_h1_f, p_h2_f], dim=-2)

        # ✅ ADFLOW对齐：构造焓H扩展数组（用于能量耗散）
        # 变量名保持 rhoE_ext 以减少下游代码改动，但实际存储的是 H = rhoE + p
        if rhoE_h1_w is not None:
            rhoE_ext = torch.cat([H_h2_w, H_h1_w, H, H_h1_f, H_h2_f], dim=-2)  # (B, H+4, W)

        # 1) 传感器：为 halo1 + 物理 + farfield halo1 计算 dss (H+2, W)
        if frozen_ss_halo is not None:
            ss_center = frozen_ss_halo[:, 1:-1, :]
            ss_plus = frozen_ss_halo[:, 2:, :]
            ss_minus = frozen_ss_halo[:, :-2, :]
        else:
            if basis == 'entropy':
                ss_ext = p_ext / (rho_ext + 1e-12) ** gamma
            else:
                ss_ext = p_ext

            ss_center = ss_ext[:, 1:-1, :]  # (B, H+2, W) : halo1..halo1
            ss_plus = ss_ext[:, 2:, :]      # (B, H+2, W)
            ss_minus = ss_ext[:, :-2, :]    # (B, H+2, W)
        d2ss = ss_plus - 2.0 * ss_center + ss_minus
        denom = ss_plus + 2.0 * ss_center + ss_minus + sslim
        dss_cells = torch.abs(d2ss) / denom  # (B, H+2, W)

        # 2) 面谱半径 rrad：使用物理单元 rad_eta，并对 halo 单元做近似复制（足以匹配ADflow量级）
        if precomputed_cell_radius is None:
            rad_eta_phys = compute_cell_centered_spectral_radius(
                rho, u, v, p, face_geom,
                vol=vol,  # 3D谱半径计算必需
                direction='eta',
                gamma=gamma,
                acoustic_scale_factor=acoustic_scale_factor,
                adis=adis,
                apply_anisotropic=apply_anisotropic,
            )
        else:
            rad_eta_phys = precomputed_cell_radius
        if rad_eta_phys.ndim == 2:
            rad_eta_phys = rad_eta_phys.unsqueeze(0)

        # cells (halo1_wall, physical[0..H-1], halo1_far) => H+2
        rad_cells = torch.cat([rad_eta_phys[:, :1, :], rad_eta_phys, rad_eta_phys[:, -1:, :]], dim=-2)
        rrad = 0.5 * (rad_cells[:, :-1, :] + rad_cells[:, 1:, :])  # (B, H+1, W)

        # ADflow: wall face porK=boundFlux -> ppor=0 -> rrad=0
        rrad[:, 0:1, :] = 0.0

        # 3) 耗散系数
        dss_L = dss_cells[:, :-1, :]  # (B, H+1, W)
        dss_R = dss_cells[:, 1:, :]
        dss_face = torch.clamp(torch.maximum(dss_L, dss_R), max=dss_max)

        fis2, fis4 = _compute_effective_jameson_coefficients(
            vis2=vis2,
            vis4=vis4,
            rfil=rfil,
            use_dissipation_continuation=use_dissipation_continuation,
            diss_cont_magnitude=diss_cont_magnitude,
            diss_cont_midpoint=diss_cont_midpoint,
            diss_cont_sharpness=diss_cont_sharpness,
            total_r=diss_cont_total_r,
            total_r0=diss_cont_total_r0,
        )
        fis2 = _broadcast_batch_coeff_like(fis2, rrad)
        fis4 = _broadcast_batch_coeff_like(fis4, rrad)
        if lumped_dissipation:
            dis2 = fis2 * rrad * dss_face + float(lumped_sigma) * fis4 * rrad
            dis4 = torch.zeros_like(dis2)
        else:
            dis2 = fis2 * rrad * dss_face
            dis4 = torch.clamp(fis4 * rrad - dis2, min=0.0)

        # 4) 守恒量（扩展）
        W_rho_ext = rho_ext
        W_rhou_ext = rho_ext * u_ext
        W_rhov_ext = rho_ext * v_ext

        # ✅ 新增：能量守恒量（如果提供rhoE）
        if rhoE_h1_w is not None:
            W_rhoE_ext = rhoE_ext  # rhoE本身就是守恒变量
            ext_stacked = _stack_stencil_fields(
                W_rho_ext,
                W_rhou_ext,
                W_rhov_ext,
                W_rhoE_ext,
            )
        else:
            ext_stacked = _stack_stencil_fields(
                W_rho_ext,
                W_rhou_ext,
                W_rhov_ext,
            )

        # 面索引 f=0..H: L=f+1, R=f+2, LL=f, RR=f+3
        ext_LL = ext_stacked[..., :-3, :]
        ext_L = ext_stacked[..., 1:-2, :]
        ext_R = ext_stacked[..., 2:-1, :]
        ext_RR = ext_stacked[..., 3:, :]

        W_rho_LL = ext_LL[:, 0, ...]
        W_rho_L = ext_L[:, 0, ...]
        W_rho_R = ext_R[:, 0, ...]
        W_rho_RR = ext_RR[:, 0, ...]

        W_rhou_LL = ext_LL[:, 1, ...]
        W_rhou_L = ext_L[:, 1, ...]
        W_rhou_R = ext_R[:, 1, ...]
        W_rhou_RR = ext_RR[:, 1, ...]

        W_rhov_LL = ext_LL[:, 2, ...]
        W_rhov_L = ext_L[:, 2, ...]
        W_rhov_R = ext_R[:, 2, ...]
        W_rhov_RR = ext_RR[:, 2, ...]

        # ✅ 新增：能量守恒量的面索引提取
        if rhoE_h1_w is not None:
            W_rhoE_LL = ext_LL[:, 3, ...]
            W_rhoE_L = ext_L[:, 3, ...]
            W_rhoE_R = ext_R[:, 3, ...]
            W_rhoE_RR = ext_RR[:, 3, ...]

        ddw_rho = W_rho_R - W_rho_L
        ddw_rhou = W_rhou_R - W_rhou_L
        ddw_rhov = W_rhov_R - W_rhov_L

        nabla4_rho = W_rho_RR - W_rho_LL - 3.0 * ddw_rho
        nabla4_rhou = W_rhou_RR - W_rhou_LL - 3.0 * ddw_rhou
        nabla4_rhov = W_rhov_RR - W_rhov_LL - 3.0 * ddw_rhov

        if lumped_dissipation:
            D_rho = dis2 * ddw_rho
            D_rhou = dis2 * ddw_rhou
            D_rhov = dis2 * ddw_rhov
        else:
            D_rho = dis2 * ddw_rho - dis4 * nabla4_rho
            D_rhou = dis2 * ddw_rhou - dis4 * nabla4_rhou
            D_rhov = dis2 * ddw_rhov - dis4 * nabla4_rhov

        # ✅ 新增：能量方程耗散（独立计算，与ADFLOW完全对齐）
        if rhoE_h1_w is not None:
            ddw_rhoE = W_rhoE_R - W_rhoE_L
            if lumped_dissipation:
                D_rhoE = dis2 * ddw_rhoE
            else:
                nabla4_rhoE = W_rhoE_RR - W_rhoE_LL - 3.0 * ddw_rhoE
                D_rhoE = dis2 * ddw_rhoE - dis4 * nabla4_rhoE

        if squeeze_output:
            D_rho = D_rho.squeeze(0)
            D_rhou = D_rhou.squeeze(0)
            D_rhov = D_rhov.squeeze(0)
            if rhoE_h1_w is not None:
                D_rhoE = D_rhoE.squeeze(0)
            dis2 = dis2.squeeze(0)
            dis4 = dis4.squeeze(0)

        if rhoE_h1_w is not None:
            if return_coeffs:
                return D_rho, D_rhou, D_rhov, D_rhoE, dis2, dis4
            return D_rho, D_rhou, D_rhov, D_rhoE

        if return_coeffs:
            return D_rho, D_rhou, D_rhov, dis2, dis4
        return D_rho, D_rhou, D_rhov

    # ========== 默认路径：原有实现（主要用于 ξ 方向，以及无完整 halo 的 η 方向调试） ==========
    # 1. 计算激波传感器（单元中心）- 使用ADflow标量方案
    dss = compute_pressure_sensor(
        p, rho=rho, gamma=gamma,
        direction=direction, periodic_xi=periodic_xi,
        sslim=sslim, basis=basis,
        ss_halo=ss_halo,
        shock_sensor=frozen_shock_sensor,
        shock_sensor_halo=frozen_ss_halo,
    )

    # 2. 计算守恒变量
    W_rho = rho
    W_rhou = rho * u
    W_rhov = rho * v

    # ✅ ADFLOW对齐：能量耗散使用焓 H = rhoE + p
    # 参考：blockette.F90:3310-3314
    # ddw5 = (rhoE + p)_R - (rhoE + p)_L
    # 注意：变量名保持 W_rhoE 以减少代码改动，但实际存储的是 H = rhoE + p
    if rhoE is not None:
        W_rhoE = rhoE + p  # 使用焓 H = rhoE + p 进行耗散计算（ADFLOW标准）

    # 3. 提取左右状态（单元中心）
    if direction == 'xi':
        if rhoE is not None:
            conservative_stacked = _stack_stencil_fields(W_rho, W_rhou, W_rhov, W_rhoE)
        else:
            conservative_stacked = _stack_stencil_fields(W_rho, W_rhou, W_rhov)

        cons_LL, cons_L, cons_R, cons_RR = _extract_stencil_xi(
            conservative_stacked,
            periodic_xi=periodic_xi,
        )
        _, dss_L_stacked, dss_R_stacked, _ = _extract_stencil_xi(
            dss.unsqueeze(1),
            periodic_xi=periodic_xi,
        )

        W_rho_LL = cons_LL[:, 0, ...]
        W_rho_L = cons_L[:, 0, ...]
        W_rho_R = cons_R[:, 0, ...]
        W_rho_RR = cons_RR[:, 0, ...]

        W_rhou_LL = cons_LL[:, 1, ...]
        W_rhou_L = cons_L[:, 1, ...]
        W_rhou_R = cons_R[:, 1, ...]
        W_rhou_RR = cons_RR[:, 1, ...]

        W_rhov_LL = cons_LL[:, 2, ...]
        W_rhov_L = cons_L[:, 2, ...]
        W_rhov_R = cons_R[:, 2, ...]
        W_rhov_RR = cons_RR[:, 2, ...]

        dss_L = dss_L_stacked[:, 0, ...]
        dss_R = dss_R_stacked[:, 0, ...]

        if rhoE is not None:
            W_rhoE_LL = cons_LL[:, 3, ...]
            W_rhoE_L = cons_L[:, 3, ...]
            W_rhoE_R = cons_R[:, 3, ...]
            W_rhoE_RR = cons_RR[:, 3, ...]

    elif direction == 'eta':
        if rhoE is not None:
            conservative_stacked = _stack_stencil_fields(W_rho, W_rhou, W_rhov, W_rhoE)
        else:
            conservative_stacked = _stack_stencil_fields(W_rho, W_rhou, W_rhov)

        lower_halo = None
        if halo_wall is not None:
            # **plan修复**：使用halo_wall计算壁面边界的守恒变量
            # halo_wall格式: [rho, u, v, p] 或 [rho, u, v, p, rhoE]（新格式）
            if halo_wall.ndim == 2:  # (C, W)
                halo_wall = halo_wall.unsqueeze(0)  # (1, C, W)

            rho_halo = halo_wall[:, 0:1, :]
            u_halo = halo_wall[:, 1:2, :]
            v_halo = halo_wall[:, 2:3, :]
            rhou_halo = rho_halo * u_halo
            rhov_halo = rho_halo * v_halo

            if rhoE is not None and halo_wall.shape[1] >= 5:
                rhoE_halo = halo_wall[:, 4:5, :]
                lower_halo = _stack_stencil_fields(rho_halo, rhou_halo, rhov_halo, rhoE_halo)
            elif rhoE is not None:
                lower_halo = _stack_stencil_fields(
                    rho_halo,
                    rhou_halo,
                    rhov_halo,
                    W_rhoE[..., :1, :],
                )
            else:
                lower_halo = _stack_stencil_fields(rho_halo, rhou_halo, rhov_halo)

        cons_LL, cons_L, cons_R, cons_RR = _extract_stencil_eta(
            conservative_stacked,
            lower_halo=lower_halo,
        )
        _, dss_L_stacked, dss_R_stacked, _ = _extract_stencil_eta(dss.unsqueeze(1))

        W_rho_LL = cons_LL[:, 0, ...]
        W_rho_L = cons_L[:, 0, ...]
        W_rho_R = cons_R[:, 0, ...]
        W_rho_RR = cons_RR[:, 0, ...]

        W_rhou_LL = cons_LL[:, 1, ...]
        W_rhou_L = cons_L[:, 1, ...]
        W_rhou_R = cons_R[:, 1, ...]
        W_rhou_RR = cons_RR[:, 1, ...]

        W_rhov_LL = cons_LL[:, 2, ...]
        W_rhov_L = cons_L[:, 2, ...]
        W_rhov_R = cons_R[:, 2, ...]
        W_rhov_RR = cons_RR[:, 2, ...]

        dss_L = dss_L_stacked[:, 0, ...]
        dss_R = dss_R_stacked[:, 0, ...]

        if rhoE is not None:
            W_rhoE_LL = cons_LL[:, 3, ...]
            W_rhoE_L = cons_L[:, 3, ...]
            W_rhoE_R = cons_R[:, 3, ...]
            W_rhoE_RR = cons_RR[:, 3, ...]

    else:
        raise ValueError(f"Invalid direction: {direction}")

    # 4. 计算面谱半径（ADflow 3D 各向异性缩放）
    rrad = compute_spectral_radius(
        rho, u, v, p, face_geom, vol=vol, direction=direction,
        gamma=gamma, acoustic_scale_factor=acoustic_scale_factor,
        adis=adis, apply_anisotropic=apply_anisotropic,
        precomputed_cell_radius=precomputed_cell_radius,
    )

    # 确保rrad有正确的batch维度
    if rrad.ndim == 2:
        rrad = rrad.unsqueeze(0)

    # 5. 计算耗散系数
    # ADflow: dis2 = fis2 * rrad * min(dssMax, max(dss(i), dss(i+1)))
    # 其中 fis2 = rFil * vis2，这里假设 rFil = 1
    dss_face = torch.clamp(torch.maximum(dss_L, dss_R), max=dss_max)
    fis2, fis4 = _compute_effective_jameson_coefficients(
        vis2=vis2,
        vis4=vis4,
        rfil=rfil,
        use_dissipation_continuation=use_dissipation_continuation,
        diss_cont_magnitude=diss_cont_magnitude,
        diss_cont_midpoint=diss_cont_midpoint,
        diss_cont_sharpness=diss_cont_sharpness,
        total_r=diss_cont_total_r,
        total_r0=diss_cont_total_r0,
    )
    fis2 = _broadcast_batch_coeff_like(fis2, rrad)
    fis4 = _broadcast_batch_coeff_like(fis4, rrad)
    if lumped_dissipation:
        dis2 = fis2 * rrad * dss_face + float(lumped_sigma) * fis4 * rrad
        dis4 = torch.zeros_like(dis2)
    else:
        dis2 = fis2 * rrad * dss_face

        # ADflow: dis4 = myDim(fis4 * rrad, dis2) = max(0, fis4*rrad - dis2)
        dis4 = torch.clamp(fis4 * rrad - dis2, min=0.0)

    # 6. 计算一阶差分：ddw1 = W_R - W_L
    ddw_rho = W_rho_R - W_rho_L
    ddw_rhou = W_rhou_R - W_rhou_L
    ddw_rhov = W_rhov_R - W_rhov_L

    # ✅ 新增：能量方程一阶差分
    if rhoE is not None:
        ddw_rhoE = W_rhoE_R - W_rhoE_L

    # 7. 计算4阶差分项：W_{i+2} - W_{i-1} - 3·ddw1
    # ADflow (fluxes.F90:1525): w(i+2) - w(i-1) - three*ddw1
    nabla4_rho = W_rho_RR - W_rho_LL - 3.0 * ddw_rho
    nabla4_rhou = W_rhou_RR - W_rhou_LL - 3.0 * ddw_rhou
    nabla4_rhov = W_rhov_RR - W_rhov_LL - 3.0 * ddw_rhov

    # ✅ 新增：能量方程4阶差分
    if rhoE is not None:
        nabla4_rhoE = W_rhoE_RR - W_rhoE_LL - 3.0 * ddw_rhoE

    # 8. 总耗散通量：fs = dis2·ddw1 - dis4·nabla4
    if lumped_dissipation:
        D_rho = dis2 * ddw_rho
        D_rhou = dis2 * ddw_rhou
        D_rhov = dis2 * ddw_rhov
    else:
        D_rho = dis2 * ddw_rho - dis4 * nabla4_rho
        D_rhou = dis2 * ddw_rhou - dis4 * nabla4_rhou
        D_rhov = dis2 * ddw_rhov - dis4 * nabla4_rhov

    # ✅ 新增：能量方程总耗散（独立计算，与ADFLOW完全对齐）
    if rhoE is not None:
        if lumped_dissipation:
            D_rhoE = dis2 * ddw_rhoE
        else:
            D_rhoE = dis2 * ddw_rhoE - dis4 * nabla4_rhoE

    # 移除batch维度
    if squeeze_output:
        D_rho = D_rho.squeeze(0)
        D_rhou = D_rhou.squeeze(0)
        D_rhov = D_rhov.squeeze(0)
        if rhoE is not None:
            D_rhoE = D_rhoE.squeeze(0)
        dis2 = dis2.squeeze(0)
        dis4 = dis4.squeeze(0)

    if return_coeffs:
        if rhoE is not None:
            return D_rho, D_rhou, D_rhov, D_rhoE, dis2, dis4
        return D_rho, D_rhou, D_rhov, dis2, dis4

    # ========== DEBUG: 保存耗散中间量 ==========
    import os
    if os.environ.get('SURROGATE_DEBUG_DISSIPATION') == '1' and direction == 'xi':
        import numpy as np

        # 输出wake区域（j=0..59, i=280..303）
        # 使用rrad的shape（面谱半径）来推断维度
        if rrad.ndim == 3:  # (batch, H, W)
            j_max = min(60, rrad.shape[1])
            i_start, i_end = 280, 304
            i_end_actual = min(i_end, rrad.shape[2])

            debug_data = {
                'rrad': rrad[0, :j_max, i_start:i_end_actual].detach().cpu().numpy(),
                'dss': dss[0, :j_max, i_start:i_end_actual].detach().cpu().numpy(),
                'dss_face': dss_face[0, :j_max, i_start:i_end_actual].detach().cpu().numpy(),
                'dis2': dis2[0, :j_max, i_start:i_end_actual].detach().cpu().numpy(),
                'dis4': dis4[0, :j_max, i_start:i_end_actual].detach().cpu().numpy(),
                'ddw1_rho': ddw_rho[0, :j_max, i_start:i_end_actual].detach().cpu().numpy(),
                'nabla4_rho': nabla4_rho[0, :j_max, i_start:i_end_actual].detach().cpu().numpy(),
                'D_rho': (D_rho[0, :j_max, i_start:i_end_actual] if D_rho.ndim == 3 else D_rho[:j_max, i_start:i_end_actual]).detach().cpu().numpy(),
            }
        else:  # (H, W)
            j_max = min(60, rrad.shape[0])
            i_start, i_end = 280, 304
            i_end_actual = min(i_end, rrad.shape[1])

            debug_data = {
                'rrad': rrad[:j_max, i_start:i_end_actual].detach().cpu().numpy(),
                'dss': dss[:j_max, i_start:i_end_actual].detach().cpu().numpy(),
                'dss_face': dss_face[:j_max, i_start:i_end_actual].detach().cpu().numpy(),
                'dis2': dis2[:j_max, i_start:i_end_actual].detach().cpu().numpy(),
                'dis4': dis4[:j_max, i_start:i_end_actual].detach().cpu().numpy(),
                'ddw1_rho': ddw_rho[:j_max, i_start:i_end_actual].detach().cpu().numpy(),
                'nabla4_rho': nabla4_rho[:j_max, i_start:i_end_actual].detach().cpu().numpy(),
                'D_rho': D_rho[:j_max, i_start:i_end_actual].detach().cpu().numpy(),
            }

        # 保存维度信息
        debug_data['j_max'] = j_max
        debug_data['i_range'] = (i_start, i_end_actual)

        np.savez_compressed('pytorch_dissipation_debug_detail_xi.npz', **debug_data)
        print(f"[DEBUG dissipation.py] Saved dissipation details to: pytorch_dissipation_debug_detail_xi.npz")
        print(f"                       Region: j=0..{j_max-1}, i={i_start}..{i_end_actual-1}")
    # ========== END DEBUG ==========

    # ✅ 根据是否提供rhoE决定返回值数量（ADFLOW对齐）
    if rhoE is not None:
        return D_rho, D_rhou, D_rhov, D_rhoE
    else:
        return D_rho, D_rhou, D_rhov
