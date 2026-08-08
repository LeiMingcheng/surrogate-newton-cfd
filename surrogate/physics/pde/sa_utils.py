"""
Spalart-Allmaras湍流模型辅助函数模块

提供SA湍流模型所需的辅助计算函数：
1. SA涡粘计算（eddy viscosity）
2. 涡量模计算（vorticity magnitude）
3. 逆变速度计算（contravariant velocities）

参考ADFLOW源码：
- turbUtils.F90:656 (saEddyViscosity)
- sa.F90:196-238 (速度梯度 → 涡量)
- turbUtils.F90:825-840 (turbAdvection中的uu计算)
"""

import torch
from typing import Dict, Tuple


def compute_sa_eddy_viscosity(
    rho: torch.Tensor,
    nuTilde: torch.Tensor,
    mu_l: torch.Tensor,
    cv1: float = 7.1
) -> torch.Tensor:
    """
    SA涡粘计算

    参考：turbUtils.F90:656 (saEddyViscosity)

    公式：
        mu_t = rho * nuTilde * fv1
        fv1 = chi^3 / (chi^3 + cv1^3)
        chi = rho * nuTilde / mu_l

    Args:
        rho: 密度 (H, W) 或 (batch, H, W)
        nuTilde: SA湍流变量
        mu_l: 层流粘度
        cv1: SA常数（默认7.1）

    Returns:
        mu_t: 涡粘度
    """
    # Chi参数
    chi = rho * nuTilde / (mu_l + 1e-14)

    # fv1函数
    chi3 = chi**3
    cv13 = cv1**3
    fv1 = chi3 / (chi3 + cv13)

    # 涡粘
    mu_t = fv1 * rho * nuTilde

    return mu_t


def compute_vorticity_magnitude(
    du_dx: torch.Tensor,
    du_dy: torch.Tensor,
    dv_dx: torch.Tensor,
    dv_dy: torch.Tensor
) -> torch.Tensor:
    """
    涡量模计算（2D）

    参考：sa.F90:196-238 (速度梯度 → 涡量)

    2D情况：
        omega_z = dv/dx - du/dy
        |omega| = |omega_z|

    Args:
        du_dx, du_dy, dv_dx, dv_dy: 速度梯度 (H, W) 或 (batch, H, W)

    Returns:
        omega_mag: 涡量模
    """
    # 涡量z分量
    omega_z = dv_dx - du_dy

    # 涡量模（2D情况）
    omega_mag = torch.abs(omega_z)

    return omega_mag


def compute_contravariant_velocities(
    u: torch.Tensor,
    v: torch.Tensor,
    face_geom: Dict[str, torch.Tensor],
    vol: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    计算逆变速度（用于SA对流项）

    参考：turbUtils.F90:825-840 (turbAdvection中的uu计算)

    公式：
        uu_xi = (u*si_x + v*si_y) / vol
        uu_eta = (u*sj_x + v*sj_y) / vol

    其中si, sj是metric terms（与面面积向量相关）

    逆变速度具有velocity/length量纲，用于SA对流项的迎风格式。

    Args:
        u, v: 速度分量 (H, W) 或 (batch, H, W)
        face_geom: 面几何字典（包含A_x_xi, A_y_xi, A_x_eta, A_y_eta）
        vol: 单元体积

    Returns:
        contravariant_vel: {'uu_xi', 'uu_eta'} 逆变速度字典
    """
    # 从face_geom提取metric terms
    # 注意：面面积向量 A = (A_x, A_y) 与metric terms成正比
    # 简化：使用面面积向量作为metric terms
    si_x = face_geom['A_x_xi']  # ξ方向面面积向量x分量
    si_y = face_geom['A_y_xi']  # ξ方向面面积向量y分量
    sj_x = face_geom['A_x_eta']  # η方向面面积向量x分量
    sj_y = face_geom['A_y_eta']  # η方向面面积向量y分量

    # 逆变速度（velocity/length单位）
    # ξ方向
    # si_x, si_y shape: (H, W+1)
    # u, v shape: (H, W)
    # 需要将u, v平均到ξ面
    if u.ndim == 2:
        # (H, W) → (H, W+1)
        u_face_xi = torch.cat([
            u[:, 0:1],  # 左边界
            0.5 * (u[:, :-1] + u[:, 1:]),  # 内部面
            u[:, -1:]  # 右边界
        ], dim=1)
        v_face_xi = torch.cat([
            v[:, 0:1],
            0.5 * (v[:, :-1] + v[:, 1:]),
            v[:, -1:]
        ], dim=1)
    else:
        # (batch, H, W) → (batch, H, W+1)
        u_face_xi = torch.cat([
            u[:, :, 0:1],
            0.5 * (u[:, :, :-1] + u[:, :, 1:]),
            u[:, :, -1:]
        ], dim=2)
        v_face_xi = torch.cat([
            v[:, :, 0:1],
            0.5 * (v[:, :, :-1] + v[:, :, 1:]),
            v[:, :, -1:]
        ], dim=2)

    # vol也需要平均到ξ面
    if vol.ndim == 2:
        vol_face_xi = torch.cat([
            vol[:, 0:1],
            0.5 * (vol[:, :-1] + vol[:, 1:]),
            vol[:, -1:]
        ], dim=1)
    else:
        vol_face_xi = torch.cat([
            vol[:, :, 0:1],
            0.5 * (vol[:, :, :-1] + vol[:, :, 1:]),
            vol[:, :, -1:]
        ], dim=2)

    uu_xi = (u_face_xi * si_x + v_face_xi * si_y) / (vol_face_xi + 1e-14)

    # η方向
    # sj_x, sj_y shape: (H+1, W)
    # u, v shape: (H, W)
    # 需要将u, v平均到η面
    if u.ndim == 2:
        # (H, W) → (H+1, W)
        u_face_eta = torch.cat([
            u[0:1, :],  # 下边界
            0.5 * (u[:-1, :] + u[1:, :]),  # 内部面
            u[-1:, :]  # 上边界
        ], dim=0)
        v_face_eta = torch.cat([
            v[0:1, :],
            0.5 * (v[:-1, :] + v[1:, :]),
            v[-1:, :]
        ], dim=0)
    else:
        # (batch, H, W) → (batch, H+1, W)
        u_face_eta = torch.cat([
            u[:, 0:1, :],
            0.5 * (u[:, :-1, :] + u[:, 1:, :]),
            u[:, -1:, :]
        ], dim=1)
        v_face_eta = torch.cat([
            v[:, 0:1, :],
            0.5 * (v[:, :-1, :] + v[:, 1:, :]),
            v[:, -1:, :]
        ], dim=1)

    # vol也需要平均到η面
    if vol.ndim == 2:
        vol_face_eta = torch.cat([
            vol[0:1, :],
            0.5 * (vol[:-1, :] + vol[1:, :]),
            vol[-1:, :]
        ], dim=0)
    else:
        vol_face_eta = torch.cat([
            vol[:, 0:1, :],
            0.5 * (vol[:, :-1, :] + vol[:, 1:, :]),
            vol[:, -1:, :]
        ], dim=1)

    uu_eta = (u_face_eta * sj_x + v_face_eta * sj_y) / (vol_face_eta + 1e-14)

    return {
        'uu_xi': uu_xi,
        'uu_eta': uu_eta
    }


def compute_wall_distance(
    coords_vertex: torch.Tensor,
    wall_boundary: str = 'j_min'
) -> torch.Tensor:
    """
    计算壁面距离（简化版）

    对于O型网格，壁面通常是j=0（内边界）

    Args:
        coords_vertex: 顶点坐标 (2, H+1, W+1) 或 (batch, 2, H+1, W+1)
                      coords_vertex[0] = x, coords_vertex[1] = y
        wall_boundary: 壁面边界类型 ('j_min' 或 'j_max')

    Returns:
        d_wall: 壁面距离 (H, W) 或 (batch, H, W)
    """
    # 提取单元中心坐标
    if coords_vertex.ndim == 3:
        # (2, H+1, W+1) → (2, H, W)
        x_center = 0.25 * (
            coords_vertex[0, :-1, :-1] + coords_vertex[0, :-1, 1:] +
            coords_vertex[0, 1:, :-1] + coords_vertex[0, 1:, 1:]
        )
        y_center = 0.25 * (
            coords_vertex[1, :-1, :-1] + coords_vertex[1, :-1, 1:] +
            coords_vertex[1, 1:, :-1] + coords_vertex[1, 1:, 1:]
        )
    else:
        # (batch, 2, H+1, W+1) → (batch, 2, H, W)
        x_center = 0.25 * (
            coords_vertex[:, 0, :-1, :-1] + coords_vertex[:, 0, :-1, 1:] +
            coords_vertex[:, 0, 1:, :-1] + coords_vertex[:, 0, 1:, 1:]
        )
        y_center = 0.25 * (
            coords_vertex[:, 1, :-1, :-1] + coords_vertex[:, 1, :-1, 1:] +
            coords_vertex[:, 1, 1:, :-1] + coords_vertex[:, 1, 1:, 1:]
        )

    # 壁面坐标（O型网格，j=0是壁面）
    if wall_boundary == 'j_min':
        # 壁面：j=0
        if coords_vertex.ndim == 3:
            # (2, H+1, W+1) → 取j=0的点，平均到单元中心
            x_wall = 0.5 * (coords_vertex[0, 0, :-1] + coords_vertex[0, 0, 1:])  # (W,)
            y_wall = 0.5 * (coords_vertex[1, 0, :-1] + coords_vertex[1, 0, 1:])
            # 扩展到(H, W)
            x_wall = x_wall.unsqueeze(0).expand(x_center.shape[0], -1)
            y_wall = y_wall.unsqueeze(0).expand(y_center.shape[0], -1)
        else:
            # (batch, 2, H+1, W+1)
            x_wall = 0.5 * (coords_vertex[:, 0, 0, :-1] + coords_vertex[:, 0, 0, 1:])  # (batch, W)
            y_wall = 0.5 * (coords_vertex[:, 1, 0, :-1] + coords_vertex[:, 1, 0, 1:])
            # 扩展到(batch, H, W)
            x_wall = x_wall.unsqueeze(1).expand(-1, x_center.shape[1], -1)
            y_wall = y_wall.unsqueeze(1).expand(-1, y_center.shape[1], -1)
    else:
        raise ValueError(f"Unsupported wall_boundary: {wall_boundary}")

    # 计算距离
    d_wall = torch.sqrt((x_center - x_wall)**2 + (y_center - y_wall)**2)

    return d_wall


def compute_sa_nuTilde_inf(
    eddyVisInfRatio: float = 0.009,
    nuLam: float = None,
    cv1: float = 7.1,
    max_iter: int = 50,
    tol: float = 1e-10
) -> float:
    """
    计算自由流nuTilde值（ADFLOW saNuKnownEddyRatio对齐）

    参考：turbUtils.F90:333-407 (saNuKnownEddyRatio)

    使用Newton迭代求解：
        chi^4 - eddyRatio * (chi^3 + cv1^3) = 0

    然后：nuTilde_inf = nuLam * chi

    Args:
        eddyVisInfRatio: 自由流涡粘比 νt/ν（SA默认0.009）
        nuLam: 自由流层流动力粘度（无量纲）
               对于无量纲流场：nuLam = (Ma * sqrt(gamma)) / Re
        cv1: SA常数（默认7.1）
        max_iter: 最大迭代次数（默认50）
        tol: 收敛容差（默认1e-10）

    Returns:
        nuTilde_inf: 自由流nuTilde值

    用途：
        - 远场入流边界条件中的nuTilde设置
        - 与ADFLOW bcTurbFarfield对齐
    """
    if nuLam is None or nuLam <= 0:
        return 0.0

    if eddyVisInfRatio <= 0:
        return 0.0

    cv13 = cv1 ** 3  # = 7.1^3 ≈ 357.911

    # 初始值选择（ADFLOW策略 turbUtils.F90:344-356）
    if eddyVisInfRatio < 1e-4:
        chi = 0.5
    elif eddyVisInfRatio < 1.0:
        chi = 5.0
    elif eddyVisInfRatio < 10.0:
        chi = 10.0
    else:
        chi = eddyVisInfRatio

    # Newton迭代
    for _ in range(max_iter):
        chi2 = chi * chi
        chi3 = chi * chi2
        chi4 = chi * chi3

        # 函数值：f = chi^4 - eddyRatio*(chi^3 + cv1^3)
        f = chi4 - eddyVisInfRatio * (chi3 + cv13)

        # 导数：f' = 4*chi^3 - 3*eddyRatio*chi^2
        df = 4.0 * chi3 - 3.0 * eddyVisInfRatio * chi2

        # Newton更新
        if abs(df) < 1e-30:
            break
        dchi = f / df
        chi = chi - dchi

        # 收敛检查
        if abs(dchi / (chi + 1e-30)) <= tol:
            break

    # 最终：nuTilde_inf = nuLam * chi
    return nuLam * chi


def compute_sa_nuTilde_inf_tensor(
    nuLam: torch.Tensor,
    eddyVisInfRatio: float = 0.009,
    cv1: float = 7.1,
    max_iter: int = 50,
    tol: float = 1e-10
) -> torch.Tensor:
    """
    向量化版本：计算自由流nuTilde值（ADFLOW saNuKnownEddyRatio对齐）

    参考：turbUtils.F90:333-407 (saNuKnownEddyRatio)

    与 compute_sa_nuTilde_inf 的区别：
    - 输入nuLam为torch张量，可批量计算
    - Newton迭代在torch上向量化执行

    Args:
        nuLam: 自由流层流运动粘度（无量纲），形状任意
        eddyVisInfRatio: 自由流涡粘比 νt/ν（SA默认0.009）
        cv1: SA常数（默认7.1）
        max_iter: 最大迭代次数（默认50）
        tol: 收敛容差（默认1e-10）

    Returns:
        nuTilde_inf: 与nuLam同shape的张量
    """
    if eddyVisInfRatio <= 0.0:
        return torch.zeros_like(nuLam)

    # 有效mask：nuLam>0
    active = nuLam > 0.0
    if not torch.any(active):
        return torch.zeros_like(nuLam)

    cv13 = cv1 ** 3

    # 初始chi（ADFLOW策略：依赖eddyRatio，且eddyRatio在本函数中为常数）
    if eddyVisInfRatio < 1e-4:
        chi0 = 0.5
    elif eddyVisInfRatio < 1.0:
        chi0 = 5.0
    elif eddyVisInfRatio < 10.0:
        chi0 = 10.0
    else:
        chi0 = float(eddyVisInfRatio)

    chi = torch.full_like(nuLam, float(chi0))
    chi = torch.where(active, chi, torch.zeros_like(chi))

    # Newton迭代：chi^4 - ratio*(chi^3 + cv1^3) = 0
    for _ in range(max_iter):
        chi2 = chi * chi
        chi3 = chi * chi2
        chi4 = chi * chi3

        f = chi4 - eddyVisInfRatio * (chi3 + cv13)
        df = 4.0 * chi3 - 3.0 * eddyVisInfRatio * chi2

        # 避免除0：对df极小的点停止更新
        safe = active & (torch.abs(df) > 1e-30)
        if not torch.any(safe):
            break

        dchi = f / df
        chi_new = chi - dchi

        # 收敛判据：abs(dchi/chi) <= thresholdReal
        conv = safe & (torch.abs(dchi) / (torch.abs(chi_new) + 1e-30) <= tol)

        chi = torch.where(safe, chi_new, chi)
        active = active & ~conv
        if not torch.any(active):
            break

    return nuLam * chi


def compute_production_term(
    du_dx: torch.Tensor,
    du_dy: torch.Tensor,
    dv_dx: torch.Tensor,
    dv_dy: torch.Tensor,
    mode: str = 'strain'
) -> torch.Tensor:
    """
    计算SA源项中的生成项基底（ADFLOW turbProd对齐）

    参考：
    - inputParamRoutines.F90:3971 (turbProd默认为strain)
    - sa.F90:196-237 (应变率和涡量计算)

    Args:
        du_dx, du_dy, dv_dx, dv_dy: 速度梯度 (H, W) 或 (batch, H, W)
        mode: 生成项模式
            - 'strain' (默认，ADFLOW默认): 应变率模
            - 'vorticity': 涡量模

    Returns:
        S: 生成项基底 |S| 或 |Ω|

    注意：
        - ADFLOW默认使用应变率（turbProd=strain）
        - 2D情况下，应变率为 sqrt(2 * S_ij * S_ij - (2/3)*(div u)^2)
        - 2D情况下，涡量为 |dv/dx - du/dy|
    """
    if mode == 'vorticity':
        # ADFLOW对齐（sa.F90:229-236, 2D静止网格）：
        #   vortz = 2*fact*(vvx - uuy) = 2*(dv/dx - du/dy)
        omega_z = dv_dx - du_dy
        return 2.0 * torch.abs(omega_z)

    elif mode == 'strain':
        # ADFLOW对齐（sa.F90:205-223, strain分支）
        # ADFLOW内部使用的应变分量为：
        #   sxx = 2*du/dx, syy = 2*dv/dy, szz = 0
        #   sxy = du/dy + dv/dx
        # 然后：
        #   div2       = (2/3) * (sxx+syy+szz)^2
        #   strainMag2 = 2*(sxy^2) + sxx^2 + syy^2 + szz^2
        #   strainProd = 2*strainMag2 - div2
        #   ss         = sqrt(strainProd)

        sxx = 2.0 * du_dx
        syy = 2.0 * dv_dy
        szz = torch.zeros_like(sxx)
        sxy = du_dy + dv_dx

        div2 = (2.0 / 3.0) * (sxx + syy + szz) ** 2
        strainMag2 = 2.0 * (sxy ** 2) + sxx ** 2 + syy ** 2 + szz ** 2
        strainProd = 2.0 * strainMag2 - div2
        strainProd = torch.clamp(strainProd, min=0.0)

        # Gradient-safe sqrt: 避免 sqrt(0) 产生 inf 导数导致 NaN
        # - 推理模式（no_grad）：保持原始 ADFLOW 对齐
        # - 梯度模式（requires_grad）：对 0 点使用安全替换
        if strainProd.requires_grad:
            m = (strainProd > 0.0).to(strainProd.dtype)  # mask: 1 for positive, 0 for zero
            strainProd_safe = strainProd + (1.0 - m)  # 将 0 点替换成 1，避免 sqrt 反传 inf
            return torch.sqrt(strainProd_safe) * m  # 输出仍为 0 at zero points
        return torch.sqrt(strainProd)

    else:
        raise ValueError(f"未知的mode: {mode}，支持的选项：'strain', 'vorticity'")


def compute_contravariant_velocities_adflow(
    u: torch.Tensor,
    v: torch.Tensor,
    face_geom: Dict[str, torch.Tensor],
    vol: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    计算逆变速度（ADFLOW turbUtils.F90:902-914 cell-center定义）

    与面平均版本(compute_contravariant_velocities)的关键区别：
    - 面平均版本：将速度插值到面，再与面向量计算
    - ADFLOW版本：使用cell-center速度，与相邻面向量平均计算

    ADFLOW公式 (turbUtils.F90:902-914):
        voli = half / vol(i, j, k)
        xa = (sk(i, j, k, 1) + sk(i, j, k - 1, 1)) * voli
        ya = (sk(i, j, k, 2) + sk(i, j, k - 1, 2)) * voli
        za = (sk(i, j, k, 3) + sk(i, j, k - 1, 3)) * voli
        uu = xa * w(i, j, k, ivx) + ya * w(i, j, k, ivy) + za * w(i, j, k, ivz) - qs

    其中 qs = 0（静止网格）

    注意：结果具有 velocity/length 量纲，用于SA对流项的迎风格式。

    Args:
        u, v: 速度分量 (H, W) 或 (batch, H, W)，cell-center值
        face_geom: 面几何字典（包含A_x_xi, A_y_xi, A_x_eta, A_y_eta）
        vol: 单元体积 (H, W) 或 (batch, H, W)

    Returns:
        contravariant_vel: {'uu_xi', 'uu_eta'} 逆变速度字典，形状均为(batch, H, W)
    """
    # 从face_geom提取面向量
    # sj = A_eta: η方向面向量 (H+1, W) 或 (batch, H+1, W)
    # si = A_xi: ξ方向面向量
    #   - periodic_xi=True:  (H, W)
    #   - periodic_xi=False: (H, W-1) internal faces only
    sj_x = face_geom['A_x_eta']  # (H+1, W) 或 (batch, H+1, W)
    sj_y = face_geom['A_y_eta']
    si_x = face_geom['A_x_xi']   # (H, W) 或 (batch, H, W)
    si_y = face_geom['A_y_xi']
    periodic_xi = bool(face_geom.get('periodic_xi', True))

    # 处理batch维度
    is_batched = u.ndim == 3
    if not is_batched:
        u = u.unsqueeze(0)        # (1, H, W)
        v = v.unsqueeze(0)
        vol = vol.unsqueeze(0)
        sj_x = sj_x.unsqueeze(0)  # (1, H+1, W)
        sj_y = sj_y.unsqueeze(0)
        si_x = si_x.unsqueeze(0)  # (1, H, W)
        si_y = si_y.unsqueeze(0)

    batch, H, W = u.shape

    # ========== η方向逆变速度 (uu_eta) ==========
    # ADFLOW: xa = (sj[k] + sj[k-1]) * (0.5 / vol)
    # 对于单元 j=0..H-1，使用面 j (下方) 和面 j+1 (上方) 的平均
    # sj shape: (batch, H+1, W)

    # voli = 0.5 / vol （ADFLOW无epsilon）
    voli = 0.5 / vol  # (batch, H, W)

    # 面向量平均：sj[j] + sj[j+1] 对应单元 j 的上下两个面
    # sj[:, 0:H, :] 是面 0..H-1（每个单元的下方面）
    # sj[:, 1:H+1, :] 是面 1..H（每个单元的上方面）
    sj_x_sum = sj_x[:, :-1, :] + sj_x[:, 1:, :]  # (batch, H, W)
    sj_y_sum = sj_y[:, :-1, :] + sj_y[:, 1:, :]

    # xa, ya = (sj_plus + sj_minus) * voli
    xa_eta = sj_x_sum * voli  # (batch, H, W)
    ya_eta = sj_y_sum * voli

    # uu_eta = xa * u + ya * v  (使用cell-center速度，不是面平均！)
    uu_eta = xa_eta * u + ya_eta * v  # (batch, H, W)

    # ========== ξ方向逆变速度 (uu_xi) ==========
    # 周期网格使用 W 个面；非周期局部窗口只提供 W-1 个内部面。
    # 对于 non-periodic xi，将内部面放到 [:W-1]，缺失外边界面保留为 0，
    # 这样左/右面平均在内部单元上仍然成立，外层 halo 单元吸收边界误差。
    if periodic_xi:
        if int(si_x.shape[-1]) != W:
            raise ValueError(
                f"Periodic xi expects W faces, got si_x width={int(si_x.shape[-1])}, W={W}"
            )
        si_x_eff = si_x
        si_y_eff = si_y
    else:
        if int(si_x.shape[-1]) != W - 1:
            raise ValueError(
                "Non-periodic xi expects W-1 internal faces, "
                f"got si_x width={int(si_x.shape[-1])}, W={W}"
            )
        si_x_eff = torch.zeros(batch, H, W, device=si_x.device, dtype=si_x.dtype)
        si_y_eff = torch.zeros(batch, H, W, device=si_y.device, dtype=si_y.dtype)
        si_x_eff[:, :, :W-1] = si_x
        si_y_eff[:, :, :W-1] = si_y

    si_x_left = torch.roll(si_x_eff, 1, dims=-1)  # (batch, H, W)
    si_y_left = torch.roll(si_y_eff, 1, dims=-1)

    # 面向量平均
    si_x_sum = si_x_left + si_x_eff  # (batch, H, W)
    si_y_sum = si_y_left + si_y_eff

    xa_xi = si_x_sum * voli
    ya_xi = si_y_sum * voli

    # uu_xi = xa * u + ya * v
    uu_xi = xa_xi * u + ya_xi * v  # (batch, H, W)

    # 去除batch维度（如果输入没有）
    if not is_batched:
        uu_eta = uu_eta.squeeze(0)
        uu_xi = uu_xi.squeeze(0)

    return {
        'uu_xi': uu_xi,
        'uu_eta': uu_eta
    }
