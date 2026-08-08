"""
Halo元素处理模块 - 与ADflow边界条件对齐

实现ADflow风格的halo单元值计算，用于边界面通量计算。

ADflow边界处理机制：
1. 壁面(bcNSWallAdiabatic): 密度复制，速度反射，压力复制/外推
2. 远场(bcFarfield): 特征边界条件
3. 对称面: 法向速度反射

索引对应关系：
- ADflow k=1: halo单元
- ADflow k=2: 第一层物理单元 (PyTorch j=0)
- ADflow k=3: 第二层物理单元 (PyTorch j=1)
"""

import torch
from typing import Optional, Tuple, Union


def apply_wall_bc(
    fields_wall: torch.Tensor,
    fields_second_layer: Optional[torch.Tensor] = None,
    slip_velocity: Optional[torch.Tensor] = None,
    normal_direction: Optional[torch.Tensor] = None,
    wall_pressure_treatment: str = 'constant_pressure',
    gamma: float = 1.4
) -> torch.Tensor:
    """
    计算壁面halo单元值（与ADflow bcNSWallAdiabatic对齐）

    ADflow壁面BC规则（BCRoutines.F90:549-565）：
    - ww1(irho) = ww2(irho)                    # 密度：直接复制
    - ww1(ivx)  = -ww2(ivx) + 2*uSlip          # 速度：反射
    - ww1(ivy)  = -ww2(ivy) + 2*vSlip
    - 压力：根据viscWallBCTreatment选项
        * constantPressure（默认）: pp1 = pp2（零梯度）
        * 其他（细网格）: pp1 = 2*pp2 - pp3（线性外推）

    对于no-slip壁面（uSlip=0）：
    - halo_u = -physical_u
    - halo_v = -physical_v
    这使得壁面处速度为 (halo + physical)/2 = 0

    Args:
        fields_wall: 壁面层(j=0)物理量 (C, W) 或 (batch, C, W)
                     C=4时: [rho, u, v, p] (Euler方程)
                     C=5时: [rho, u, v, p, rhoE] (Euler+能量方程)
                     C=6时: [rho, u, v, p, rhoE, nuTilde] (完整RANS，ADFLOW标准格式)
        fields_second_layer: 第二层(j=1)物理量，用于压力外推（仅linear_extrapolation需要）
        slip_velocity: 滑移速度 (2, W) 或 (batch, 2, W)，默认为0（no-slip）
        normal_direction: 壁面法向量（用于投影，目前未使用）
        wall_pressure_treatment: 壁面压力BC处理方式（默认'constant_pressure'）
            - 'constant_pressure': p_halo = p_wall（对齐ADflow默认，BCRoutines.F90:554-557）
            - 'linear_extrapolation': p_halo = 2*p_wall - p_second（细网格模式，BCRoutines.F90:559-563）
        gamma: 比热比（用于由原始变量重建rhoE）

    Returns:
        halo_fields: halo单元物理量，与输入形状相同
    """
    # 确定维度
    is_batched = fields_wall.ndim == 3
    if not is_batched:
        fields_wall = fields_wall.unsqueeze(0)  # (1, C, W)

    batch_size, n_channels, W = fields_wall.shape

    # 提取物理量
    rho_wall = fields_wall[:, 0, :]  # (batch, W)
    u_wall = fields_wall[:, 1, :]
    v_wall = fields_wall[:, 2, :]
    p_wall = fields_wall[:, 3, :]

    # 处理滑移速度
    if slip_velocity is None:
        u_slip = torch.zeros_like(u_wall)
        v_slip = torch.zeros_like(v_wall)
    else:
        if slip_velocity.ndim == 2:
            slip_velocity = slip_velocity.unsqueeze(0)
        u_slip = slip_velocity[:, 0, :]
        v_slip = slip_velocity[:, 1, :]

    # 计算halo单元值
    # ADflow公式: ww1 = -ww2 + 2*uSlip
    # 对于no-slip (uSlip=0): ww1 = -ww2
    halo_rho = rho_wall.clone()                      # 密度：直接复制
    halo_u = -u_wall + 2 * u_slip                    # 速度：反射
    halo_v = -v_wall + 2 * v_slip

    # 压力处理：根据wall_pressure_treatment选项
    if wall_pressure_treatment == 'constant_pressure':
        # ADFLOW默认：constantPressure（BCRoutines.F90:554-557）
        # pp1 = pp2（零梯度，忽略4/3*rhok修正项，因SA模型rhok≈0）
        halo_p = p_wall.clone()

    elif wall_pressure_treatment == 'linear_extrapolation':
        # 细网格模式：线性外推（BCRoutines.F90:559-563）
        if fields_second_layer is not None:
            # 处理第二层数据维度
            if fields_second_layer.ndim == 2:
                fields_second_layer = fields_second_layer.unsqueeze(0)
            p_second = fields_second_layer[:, 3, :]  # 第二层压力 (batch, W)
            # 线性外推: pp1 = 2*pp2 - pp3
            halo_p = 2.0 * p_wall - p_second
            # 负压保护: if (pp1 <= zero) pp1 = pp2
            halo_p = torch.where(halo_p <= 0, p_wall, halo_p)
        else:
            # 无第二层数据时fallback到零梯度
            halo_p = p_wall.clone()
    else:
        raise ValueError(f"未知的wall_pressure_treatment: {wall_pressure_treatment}，"
                        f"支持的选项：'constant_pressure', 'linear_extrapolation'")

    # 组装halo场
    halo_fields = torch.zeros_like(fields_wall)
    halo_fields[:, 0, :] = halo_rho
    halo_fields[:, 1, :] = halo_u
    halo_fields[:, 2, :] = halo_v
    halo_fields[:, 3, :] = halo_p

    # ✅ 新增：第5通道 rhoE（ADFLOW标准格式）
    if n_channels >= 5:
        # 壁面rhoE从halo原始变量重新计算（EOS）
        # 公式：rhoE = p/(gamma-1) + 0.5*rho*(u^2 + v^2)
        from .thermodynamics import compute_rhoE_from_primitives

        halo_rhoE = compute_rhoE_from_primitives(
            rho=halo_rho,
            u=halo_u,      # 已反射（no-slip时为-u_wall）
            v=halo_v,      # 已反射（no-slip时为-v_wall）
            p=halo_p,      # 已处理（零梯度或线性外推）
            gamma=gamma
        )
        halo_fields[:, 4, :] = halo_rhoE

        # ✅ 新增：第6通道 nuTilde（如果存在）
        # ADFLOW壁面BC: SA模型使用反射条件
        # 参考：adflow/src/turbulence/turbBCRoutines.F90:834-877 (bmt=1, bvt=0)
        # 参考：adflow/src/turbulence/turbBCRoutines.F90:177-187 (halo应用)
        # 公式: halo_nuTilde = bvt - bmt * nuTilde_internal = 0 - 1 * nuTilde[j=0] = -nuTilde[j=0]
        # 这使得壁面处 nuTilde_face = (halo + physical) / 2 = 0
        if n_channels >= 6:
            nuTilde_wall = fields_wall[:, 5, :]
            halo_fields[:, 5, :] = -nuTilde_wall

    # 移除batch维度（如果输入没有）
    if not is_batched:
        halo_fields = halo_fields.squeeze(0)

    return halo_fields


def apply_wall_bc_second_halo(
    first_halo: torch.Tensor
) -> torch.Tensor:
    """
    计算SA二阶格式所需的第二层壁面halo（与ADflow turb2ndHalo对齐）

    ADflow实现（turbBCRoutines.F90:1131-1230）使用常值外推：
        w(i, j, 0, l) = w(i, j, 1, l)
    即第二层halo = 第一层halo

    Args:
        first_halo: 第一层halo单元物理量 (C, W) 或 (batch, C, W)

    Returns:
        second_halo: 第二层halo单元物理量（与first_halo相同）
    """
    return first_halo.clone()


def apply_farfield_bc_second_halo(
    first_halo: torch.Tensor
) -> torch.Tensor:
    """
    计算SA二阶格式所需的第二层远场halo（与ADflow turb2ndHalo对齐）

    ADflow实现（turbBCRoutines.F90:1131-1230）使用常值外推：
        w(i, j, kb, l) = w(i, j, ke, l)
    即第二层halo = 第一层halo

    Args:
        first_halo: 第一层halo单元物理量 (C, W) 或 (batch, C, W)

    Returns:
        second_halo: 第二层halo单元物理量（与first_halo相同）
    """
    return first_halo.clone()


def apply_farfield_bc(
    fields_farfield: torch.Tensor,
    normal: Optional[torch.Tensor] = None,
    mesh_normal_velocity: Optional[torch.Tensor] = None,
    Ma: Optional[torch.Tensor] = None,
    AoA: Optional[torch.Tensor] = None,
    gamma: float = 1.4,
    nuTilde_inf: Optional[Union[float, torch.Tensor]] = None,
    Re: Optional[Union[float, torch.Tensor]] = None,
    *,
    allow_extrapolation_fallback: bool = False,
) -> torch.Tensor:
    """
    计算远场halo单元值（与ADflow bcFarfield对齐）

    ADflow实现（BCRoutines.F90: bcFarfield）使用Riemann不变量构造halo单元的原始变量：
      - 使用自由流状态与内部单元状态，依据自由流法向速度决定入/出流特征
      - 需要边界面单位法向量 normal = (n_x, n_y)
      - 网格静止时 mesh_normal_velocity = 0

    Args:
        fields_farfield: 远场层物理量 (C, W) 或 (batch, C, W)
                         C=4时: [rho, u, v, p] (Euler方程)
                         C=5时: [rho, u, v, p, rhoE] (Euler+能量方程)
                         C=6时: [rho, u, v, p, rhoE, nuTilde] (完整RANS)
        normal: 远场边界面单位法向量 (2, W) 或 (batch, 2, W)
        mesh_normal_velocity: 网格法向速度 rface (W,) 或 (batch, W) 或标量（默认0）
        Ma: 马赫数（标量或batch tensor）
        AoA: 攻角（度，标量或batch tensor）
        gamma: 比热比
        nuTilde_inf: 自由流nuTilde值（入流边界使用，由saNuKnownEddyRatio计算）
                     - 支持标量float或(batch,)张量
                     - 若为None且有Re，则使用ADFLOW默认eddyVisInfRatio=0.009计算
        Re: 雷诺数（用于在nuTilde_inf=None时计算默认值，支持标量或(batch,)张量）

    Returns:
        halo_fields: halo单元物理量
    """
    # 确定维度
    is_batched = fields_farfield.ndim == 3
    if not is_batched:
        fields_farfield = fields_farfield.unsqueeze(0)

    batch_size, n_channels, W = fields_farfield.shape

    # 如果无法构造bcFarfield所需信息，则禁止 silent fallback（避免悄悄改变PDE问题）
    if normal is None or Ma is None or AoA is None:
        missing = []
        if normal is None:
            missing.append("normal")
        if Ma is None:
            missing.append("Ma")
        if AoA is None:
            missing.append("AoA")
        msg = (
            "apply_farfield_bc requires farfield face normal + flow conditions to build "
            f"ADflow-style characteristic BC, but missing: {', '.join(missing)}. "
            "This would change the discrete PDE (previously fell back to zero-order extrapolation). "
            "Pass the missing inputs, or set allow_extrapolation_fallback=True to keep the old behavior."
        )
        if not allow_extrapolation_fallback:
            raise ValueError(msg)

        # Legacy behavior: zero-order extrapolation (strongly discouraged for Newton/PTC consistency)
        halo_fields = fields_farfield.clone()
        if not is_batched:
            halo_fields = halo_fields.squeeze(0)
        return halo_fields

    # ===== ADflow bcFarfield (BCRoutines.F90:1282-1409) =====
    # 处理 normal 维度，期望 (batch, 2, W)
    if normal.ndim == 2:  # (2, W)
        normal = normal.unsqueeze(0)  # (1, 2, W)
    if normal.shape[0] == 1 and batch_size > 1:
        normal = normal.expand(batch_size, -1, -1)

    n_x = normal[:, 0, :]  # (batch, W)
    n_y = normal[:, 1, :]

    # 网格法向速度 rface，静态网格默认0
    if mesh_normal_velocity is None:
        rface = torch.zeros((batch_size, W), device=fields_farfield.device, dtype=fields_farfield.dtype)
    else:
        if isinstance(mesh_normal_velocity, (int, float)):
            rface = torch.full((batch_size, W), float(mesh_normal_velocity),
                               device=fields_farfield.device, dtype=fields_farfield.dtype)
        else:
            rface = mesh_normal_velocity.to(device=fields_farfield.device, dtype=fields_farfield.dtype)
            if rface.ndim == 1:  # (W,)
                rface = rface.unsqueeze(0).expand(batch_size, -1)
            elif rface.ndim == 0:  # scalar tensor
                rface = rface.view(1, 1).expand(batch_size, W)
            elif rface.ndim == 2:
                if rface.shape[0] == 1 and batch_size > 1:
                    rface = rface.expand(batch_size, -1)
            else:
                raise ValueError(f"Invalid mesh_normal_velocity shape: {rface.shape}")

    gm1 = gamma - 1.0
    ovgm1 = 1.0 / gm1

    # 处理 Ma/AoA（支持标量或(B,)）
    if isinstance(Ma, (int, float)):
        Ma_t = torch.full((batch_size,), float(Ma), device=fields_farfield.device, dtype=fields_farfield.dtype)
    else:
        Ma_t = Ma.to(device=fields_farfield.device, dtype=fields_farfield.dtype)
        if Ma_t.ndim == 0:
            Ma_t = Ma_t.view(1).expand(batch_size)
        elif Ma_t.ndim == 1 and Ma_t.shape[0] == batch_size:
            pass
        else:
            raise ValueError(f"Invalid Ma shape: {Ma_t.shape}, expected scalar or ({batch_size},)")

    if isinstance(AoA, (int, float)):
        AoA_t = torch.full((batch_size,), float(AoA), device=fields_farfield.device, dtype=fields_farfield.dtype)
    else:
        AoA_t = AoA.to(device=fields_farfield.device, dtype=fields_farfield.dtype)
        if AoA_t.ndim == 0:
            AoA_t = AoA_t.view(1).expand(batch_size)
        elif AoA_t.ndim == 1 and AoA_t.shape[0] == batch_size:
            pass
        else:
            raise ValueError(f"Invalid AoA shape: {AoA_t.shape}, expected scalar or ({batch_size},)")

    AoA_rad = AoA_t * (torch.pi / 180.0)

    # 自由流参考状态（与数据无量纲一致）：rho_inf=1, pInfCorr=1
    rho_inf = torch.ones((batch_size,), device=fields_farfield.device, dtype=fields_farfield.dtype)
    p_inf_corr = torch.ones((batch_size,), device=fields_farfield.device, dtype=fields_farfield.dtype)

    c0 = torch.sqrt(gamma * p_inf_corr / rho_inf)  # (batch,)
    s0 = rho_inf ** gamma / p_inf_corr             # (batch,)

    u0 = Ma_t * c0 * torch.cos(AoA_rad)  # (batch,)
    v0 = Ma_t * c0 * torch.sin(AoA_rad)  # (batch,)

    qn0 = u0.unsqueeze(-1) * n_x + v0.unsqueeze(-1) * n_y  # (batch, W)
    vn0 = qn0 - rface  # (batch, W)

    # 内部单元状态（最外层物理单元）
    rho_e = fields_farfield[:, 0, :]  # (batch, W)
    u_e = fields_farfield[:, 1, :]
    v_e = fields_farfield[:, 2, :]
    p_e = fields_farfield[:, 3, :]

    re = 1.0 / rho_e
    qne = u_e * n_x + v_e * n_y
    c_e = torch.sqrt(gamma * p_e * re)

    # Riemann invariants
    two_ovgm1 = 2.0 * ovgm1
    ac1 = torch.where(
        vn0 > (-c0.unsqueeze(-1)),
        qne + two_ovgm1 * c_e,
        qn0 + two_ovgm1 * c0.unsqueeze(-1),
    )
    ac2 = torch.where(
        vn0 > c0.unsqueeze(-1),
        qne - two_ovgm1 * c_e,
        qn0 - two_ovgm1 * c0.unsqueeze(-1),
    )

    qnf = 0.5 * (ac1 + ac2)
    cf = 0.25 * (ac1 - ac2) * gm1

    # 出/入流判断基于自由流相对法向速度 vn0
    outflow = vn0 > 0.0

    # Outflow: tangential from internal; entropy from internal
    uf_out = u_e + (qnf - qne) * n_x
    vf_out = v_e + (qnf - qne) * n_y
    sf_out = rho_e ** gamma / p_e

    # Inflow: tangential from freestream; entropy from freestream
    u0_2d = u0.unsqueeze(-1)
    v0_2d = v0.unsqueeze(-1)
    uf_in = u0_2d + (qnf - qn0) * n_x
    vf_in = v0_2d + (qnf - qn0) * n_y
    sf_in = s0.unsqueeze(-1).expand_as(sf_out)

    uf = torch.where(outflow, uf_out, uf_in)
    vf = torch.where(outflow, vf_out, vf_in)
    sf = torch.where(outflow, sf_out, sf_in)

    cc = (cf * cf) / gamma
    cc = torch.clamp(cc, min=1e-30)
    rho_halo = torch.clamp((sf * cc) ** ovgm1, min=1e-30)
    p_halo = rho_halo * cc

    halo_fields = fields_farfield.clone()
    halo_fields[:, 0, :] = rho_halo
    halo_fields[:, 1, :] = uf
    halo_fields[:, 2, :] = vf
    halo_fields[:, 3, :] = p_halo

    # ✅ 新增：第5通道 rhoE（ADFLOW标准格式）
    if n_channels >= 5:
        # 远场rhoE从halo原始变量重新计算（EOS）
        from .thermodynamics import compute_rhoE_from_primitives

        halo_rhoE = compute_rhoE_from_primitives(
            rho=rho_halo,
            u=uf,
            v=vf,
            p=p_halo,
            gamma=gamma
        )
        halo_fields[:, 4, :] = halo_rhoE

        # ✅ 新增：第6通道 nuTilde（如果存在）
        # ADFLOW远场BC: inflow指定wInf(itu1)，outflow外推
        # 参考：adflow/src/turbulence/turbBCRoutines.F90:bcTurbFarfield:403-455
        if n_channels >= 6:
            nuTilde_e = fields_farfield[:, 5, :]  # 内部单元的nuTilde

            # 计算入流边界的nuTilde_inf值
            if nuTilde_inf is not None:
                # 使用传入的nuTilde_inf（支持标量或(batch,)）
                if isinstance(nuTilde_inf, torch.Tensor):
                    nuTilde_inflow_t = nuTilde_inf.to(device=fields_farfield.device, dtype=fields_farfield.dtype)
                    if nuTilde_inflow_t.ndim == 0:
                        nuTilde_inflow_t = nuTilde_inflow_t.view(1, 1).expand(batch_size, W)
                    elif nuTilde_inflow_t.ndim == 1:
                        if nuTilde_inflow_t.shape[0] == 1 and batch_size > 1:
                            nuTilde_inflow_t = nuTilde_inflow_t.expand(batch_size)
                        if nuTilde_inflow_t.shape[0] != batch_size:
                            raise ValueError(
                                f"nuTilde_inf tensor must be scalar or (batch,), got {tuple(nuTilde_inflow_t.shape)} "
                                f"for batch_size={batch_size}"
                            )
                        nuTilde_inflow_t = nuTilde_inflow_t.unsqueeze(-1).expand(batch_size, W)
                    elif nuTilde_inflow_t.ndim == 2:
                        if nuTilde_inflow_t.shape[0] == 1 and batch_size > 1:
                            nuTilde_inflow_t = nuTilde_inflow_t.expand(batch_size, -1)
                        if nuTilde_inflow_t.shape != (batch_size, W):
                            raise ValueError(
                                f"nuTilde_inf tensor must be (batch, W), got {tuple(nuTilde_inflow_t.shape)} "
                                f"expected {(batch_size, W)}"
                            )
                    else:
                        raise ValueError(f"Invalid nuTilde_inf tensor shape: {tuple(nuTilde_inflow_t.shape)}")
                else:
                    nuTilde_inflow_t = torch.full(
                        (batch_size, W),
                        float(nuTilde_inf),
                        device=fields_farfield.device,
                        dtype=fields_farfield.dtype
                    )
            elif Re is not None and Ma is not None:
                # 使用ADFLOW默认: eddyVisInfRatio=0.009 计算
                from .sa_utils import compute_sa_nuTilde_inf_tensor
                import math

                # Re: 标量或(batch,)
                if isinstance(Re, torch.Tensor):
                    Re_t = Re.to(device=fields_farfield.device, dtype=fields_farfield.dtype)
                    if Re_t.ndim == 0:
                        Re_t = Re_t.view(1).expand(batch_size)
                    elif Re_t.ndim == 1:
                        if Re_t.shape[0] == 1 and batch_size > 1:
                            Re_t = Re_t.expand(batch_size)
                        if Re_t.shape[0] != batch_size:
                            raise ValueError(
                                f"Re tensor must be scalar or (batch,), got {tuple(Re_t.shape)} for batch_size={batch_size}"
                            )
                    else:
                        raise ValueError(f"Invalid Re tensor shape: {tuple(Re_t.shape)}")
                else:
                    Re_t = torch.full((batch_size,), float(Re), device=fields_farfield.device, dtype=fields_farfield.dtype)

                sqrt_gamma = math.sqrt(gamma)
                nuLam_t = (Ma_t.to(dtype=fields_farfield.dtype) * sqrt_gamma) / (Re_t + 1e-30)
                nuTilde_inflow_1d = compute_sa_nuTilde_inf_tensor(
                    nuLam=nuLam_t,
                    eddyVisInfRatio=0.009,  # ADFLOW SA默认值
                    cv1=7.1
                )
                nuTilde_inflow_t = nuTilde_inflow_1d.unsqueeze(-1).expand(batch_size, W)
            else:
                # 无法计算，使用0（不推荐）
                nuTilde_inflow_t = torch.zeros((batch_size, W), device=fields_farfield.device, dtype=fields_farfield.dtype)

            # Inflow: nuTilde = nuTilde_inf（ADFLOW标准）
            # Outflow: nuTilde = 内部值（外推）
            nuTilde_halo = torch.where(
                outflow,
                nuTilde_e,
                nuTilde_inflow_t
            )
            halo_fields[:, 5, :] = nuTilde_halo

    # 移除batch维度（如果输入没有）
    if not is_batched:
        halo_fields = halo_fields.squeeze(0)

    return halo_fields


def create_halo_extended_fields(
    fields: torch.Tensor,
    halo_wall: torch.Tensor,
    halo_farfield: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    创建包含halo层的扩展物理场

    将halo层附加到物理场的边界，用于通量计算。

    Args:
        fields: 原始物理场 (C, H, W) 或 (batch, C, H, W)
        halo_wall: 壁面halo层 (C, W) 或 (batch, C, W)
        halo_farfield: 远场halo层（可选）

    Returns:
        extended_fields: 扩展后的物理场
            - 如果只有壁面halo: (C, H+1, W) - j=0是halo
            - 如果有远场halo: (C, H+2, W) - j=0是壁面halo, j=H+1是远场halo
    """
    # 确定维度
    is_batched = fields.ndim == 4
    if not is_batched:
        fields = fields.unsqueeze(0)
        halo_wall = halo_wall.unsqueeze(0)
        if halo_farfield is not None:
            halo_farfield = halo_farfield.unsqueeze(0)

    batch_size, C, H, W = fields.shape

    if halo_farfield is not None:
        # 两侧都有halo
        extended = torch.zeros((batch_size, C, H + 2, W),
                               device=fields.device, dtype=fields.dtype)
        extended[:, :, 0, :] = halo_wall         # j=0: 壁面halo
        extended[:, :, 1:H+1, :] = fields        # j=1:H+1: 物理层
        extended[:, :, H+1, :] = halo_farfield   # j=H+1: 远场halo
    else:
        # 只有壁面halo
        extended = torch.zeros((batch_size, C, H + 1, W),
                               device=fields.device, dtype=fields.dtype)
        extended[:, :, 0, :] = halo_wall         # j=0: 壁面halo
        extended[:, :, 1:, :] = fields           # j=1:: 物理层

    if not is_batched:
        extended = extended.squeeze(0)

    return extended


def compute_halo_volume(
    vol: torch.Tensor,
    bc_type: str = 'wall'
) -> torch.Tensor:
    """
    计算halo层的体积（通常为0，与ADflow对齐）

    ADflow中，halo单元的体积初始化为0，不参与残差统计。

    Args:
        vol: 物理层体积 (H, W) 或 (batch, H, W)
        bc_type: 边界类型

    Returns:
        halo_vol: halo层体积 (W,) 或 (batch, W)，全为0
    """
    is_batched = vol.ndim == 3
    if not is_batched:
        vol = vol.unsqueeze(0)

    batch_size, H, W = vol.shape

    # ADflow: halo单元体积为0
    halo_vol = torch.zeros((batch_size, W), device=vol.device, dtype=vol.dtype)

    if not is_batched:
        halo_vol = halo_vol.squeeze(0)

    return halo_vol
