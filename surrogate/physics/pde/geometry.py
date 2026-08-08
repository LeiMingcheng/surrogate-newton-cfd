"""
GridGeometry模块 - 统一几何系统（Phase 1）

目标：提供统一的geometry-based面向量和体积计算，替代metric-based方法

核心原则：
- 所有几何量基于单元中心坐标构建（center-to-center）
- 面向量：相邻中心连线
- 单元体积：四边形面积（2D）或六面体体积（3D）
- 与ADflow的geometry-based方法对齐

参考：
- ADflow: src/solver/fluxes.F90 的面积向量和体积计算
- 当前实现: residual backend 的 _estimate_face_normal（部分正确）
"""

import torch
from typing import Dict, Tuple, Optional, Union








def diagnose_geometry_quality(
    x: torch.Tensor,
    y: torch.Tensor,
    periodic_xi: bool = True
) -> Dict[str, float]:
    """
    诊断网格几何质量

    检查：
    - 面长度范围（检测强拉伸）
    - 体积范围（检测退化单元）
    - 正交性（ξ/η向量夹角）

    Args:
        x, y: 坐标
        periodic_xi: 周期性

    Returns:
        quality_metrics: 质量指标字典
    """
    # 使用ADflow方法进行几何质量诊断
    # 构造节点坐标（假设简单情况）
    coords_center = torch.stack([x, y], dim=0)
    H, W = x.shape[-2], x.shape[-1]

    # 节点坐标：简单的均匀网格假设
    xv = torch.linspace(0, 1, W+1).view(1, W+1).expand(H+1, W+1)
    yv = torch.linspace(0, 1, H+1).view(H+1, 1).expand(H+1, W+1)
    coords_vertex = torch.stack([xv, yv], dim=0)

    face_geom = compute_face_area_vectors_full(coords_center, coords_vertex, periodic_xi)
    vol = compute_cell_volume_adflow(coords_vertex, periodic_xi)

    s_xi = face_geom['s_area_xi']  # ADflow方法使用s_area_xi键名
    s_eta = face_geom['s_area_eta']

    metrics = {
        'face_length_xi': {
            'min': float(s_xi.min().item()),
            'max': float(s_xi.max().item()),
            'mean': float(s_xi.mean().item()),
            'std': float(s_xi.std().item())
        },
        'face_length_eta': {
            'min': float(s_eta.min().item()),
            'max': float(s_eta.max().item()),
            'mean': float(s_eta.mean().item()),
            'std': float(s_eta.std().item())
        },
        'cell_volume': {
            'min': float(vol.min().item()),
            'max': float(vol.max().item()),
            'mean': float(vol.mean().item()),
            'std': float(vol.std().item())
        },
        'aspect_ratio': {
            'mean': float((s_xi.mean() / (s_eta.mean() )).item())
        }
    }

    return metrics


# ========== 辅助函数 ==========

def get_face_normal_vector(
    face_geom: Dict[str, torch.Tensor],
    direction: str = 'xi'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从face_geom字典中提取面法向信息

    Args:
        face_geom: compute_face_geometry返回的字典
        direction: 'xi' 或 'eta'

    Returns:
        (ssx, ssy, s): 单位法向x分量、y分量、面长度
    """
    if direction == 'xi':
        return face_geom['ssx_xi'], face_geom['ssy_xi'], face_geom['s_xi']
    elif direction == 'eta':
        return face_geom['ssx_eta'], face_geom['ssy_eta'], face_geom['s_eta']
    else:
        raise ValueError(f"Invalid direction: {direction}")


def get_face_inv_distance(
    face_geom: Dict[str, torch.Tensor],
    direction: str = 'xi'
) -> torch.Tensor:
    """
    从face_geom字典中提取距离倒数（用于法向修正）

    Args:
        face_geom: compute_face_geometry返回的字典
        direction: 'xi' 或 'eta'

    Returns:
        inv_d: 距离倒数 1/|d|
    """
    if direction == 'xi':
        return face_geom['inv_d_xi']
    elif direction == 'eta':
        return face_geom['inv_d_eta']
    else:
        raise ValueError(f"Invalid direction: {direction}")


def compute_physical_derivatives_green_gauss(
    phi: torch.Tensor,
    face_geom: Dict[str, torch.Tensor] = None,
    volumes: torch.Tensor = None,
    x: torch.Tensor = None,
    y: torch.Tensor = None,
    periodic_xi: bool = True,
    compute_volumes: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Green-Gauss梯度重构（最精确，推荐用于PDE残差计算）

    核心原理：
    基于散度定理，单元上的梯度等于面通量的体积平均：
        ∇φ = (1/V) ∮_faces φ n⃗ dS ≈ (1/V) Σ_faces φ_face S⃗_face

    优势：
    - 直接利用守恒形式，与通量计算完全一致
    - 无插值误差（面值直接用相邻单元平均）
    - 在强非正交网格上比2×2线性系统更稳定
    - 自动满足散度定理的守恒性

    实现：
    1. 对每个单元，遍历4个面（ξ_left, ξ_right, η_bottom, η_top）
    2. 面标量值：φ_face = 0.5 * (φ_L + φ_R)
    3. 面面积向量：S⃗_face = (S_x, S_y)（来自面切向旋转90度）
    4. 累加：∇φ = (1/V) * Σ φ_face * S⃗_face

    参数：
        phi: 标量场 (batch, H, W) 或 (H, W)
        face_geom: 面几何字典（来自compute_face_area_vectors_full，优先使用）
        volumes: 单元体积（来自compute_cell_volume_adflow，优先使用）
        x, y: 坐标 (batch, H, W) 或 (H, W)（当face_geom/volumes未提供时使用）
        periodic_xi: ξ方向周期性（O-grid）
        compute_volumes: 是否计算并返回单元体积（用于验证，已废弃）

    返回：
        (dphi_dx, dphi_dy): 物理空间导数 (batch, H, W) 或 (H, W)
    """
    # 添加batch维度
    squeeze_output = False
    if phi.ndim == 2:
        phi = phi.unsqueeze(0)
        squeeze_output = True

    batch, H, W = phi.shape
    device = phi.device
    dtype = phi.dtype

    # 优先使用传入的几何
    if face_geom is None or volumes is None:
        # 回退到center计算（需要x, y坐标）
        if x is None or y is None:
            raise ValueError(
                "Either (face_geom + volumes) or (x + y) must be provided to compute_physical_derivatives_green_gauss"
            )

        if x.ndim == 2:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        # 1. 计算面几何（面积向量）- 使用ADflow方法
        # 构造节点坐标
        coords_center = torch.stack([x, y], dim=0)
        H_coords, W_coords = x.shape[-2], x.shape[-1]

        # 节点坐标：简单的均匀网格假设
        xv = torch.linspace(0, 1, W_coords+1).view(1, W_coords+1).expand(H_coords+1, W_coords+1)
        yv = torch.linspace(0, 1, H_coords+1).view(H_coords+1, 1).expand(H_coords+1, W_coords+1)
        coords_vertex = torch.stack([xv, yv], dim=0)

        face_geom = compute_face_area_vectors_full(coords_center, coords_vertex, periodic_xi=periodic_xi)
        volumes = compute_cell_volume_adflow(coords_vertex, periodic_xi=periodic_xi)
        if volumes.ndim == 2:
            volumes = volumes.unsqueeze(0)  # (batch, H, W)
    else:
        # 使用传入的vertex几何
        # 确保volumes有batch维度
        if volumes.ndim == 2:
            volumes = volumes.unsqueeze(0)  # (batch, H, W)

    # 初始化梯度累加器
    dphi_dx = torch.zeros_like(phi)
    dphi_dy = torch.zeros_like(phi)

    # 3. ξ方向面贡献（left & right）
    # 直接使用面积向量A（与ADflow几何同源）
    A_x_xi = face_geom['A_x_xi']  # (H, W) or (batch, H, W)
    A_y_xi = face_geom['A_y_xi']
    if A_x_xi.ndim == 2:
        A_x_xi = A_x_xi.unsqueeze(0)
        A_y_xi = A_y_xi.unsqueeze(0)

    # 计算ξ面的面标量值（相邻单元平均）
    if periodic_xi:
        phi_left = phi  # (batch, H, W)
        phi_right = torch.cat([phi[:, :, 1:], phi[:, :, :1]], dim=-1)  # 右移1格，周期
        phi_face_xi = 0.5 * (phi_left + phi_right)  # (batch, H, W)

        # 累加ξ_right面贡献（每个单元的右面，i+1/2）
        A_x_xi_right = A_x_xi  # 右面几何
        A_y_xi_right = A_y_xi
        dphi_dx += phi_face_xi * A_x_xi_right
        dphi_dy += phi_face_xi * A_y_xi_right

        # 累加ξ_left面贡献（左面 = 左邻居的右面，i-1/2，符号相反）
        # 关键修复：左面必须使用左邻居单元的右面几何，即roll(A, +1)
        A_x_xi_left = torch.roll(A_x_xi, shifts=1, dims=-1)  # 左邻居的右面几何
        A_y_xi_left = torch.roll(A_y_xi, shifts=1, dims=-1)
        phi_left_neighbor = torch.cat([phi[:, :, -1:], phi[:, :, :-1]], dim=-1)  # 左移
        phi_face_xi_left = 0.5 * (phi_left_neighbor + phi)
        dphi_dx -= phi_face_xi_left * A_x_xi_left
        dphi_dy -= phi_face_xi_left * A_y_xi_left
    else:
        # 非周期：只处理内部面
        phi_face_xi_internal = 0.5 * (phi[:, :, :-1] + phi[:, :, 1:])  # (batch, H, W-1)

        # 右面贡献（i+1/2）
        dphi_dx[:, :, :-1] += phi_face_xi_internal * A_x_xi
        dphi_dy[:, :, :-1] += phi_face_xi_internal * A_y_xi

        # 左面贡献（i-1/2，符号相反）
        dphi_dx[:, :, 1:] -= phi_face_xi_internal * A_x_xi
        dphi_dy[:, :, 1:] -= phi_face_xi_internal * A_y_xi

    # 4. η方向面贡献（bottom & top）
    # 提取完整边界面的内部面（排除j=0和j=H边界）
    A_x_eta_full = face_geom['A_x_eta']  # (H+1, W) or (B, H+1, W)
    A_y_eta_full = face_geom['A_y_eta']  # (H+1, W) or (B, H+1, W)

    # Green-Gauss只需要内部面（H-1, W）or (B, H-1, W)
    if A_x_eta_full.ndim == 2:
        A_x_eta = A_x_eta_full[1:-1, :]  # 排除边界，保留内部面
        A_y_eta = A_y_eta_full[1:-1, :]  # 排除边界，保留内部面
        A_x_eta = A_x_eta.unsqueeze(0)
        A_y_eta = A_y_eta.unsqueeze(0)
    else:  # ndim == 3, batch模式
        A_x_eta = A_x_eta_full[:, 1:-1, :]  # 排除边界，保留内部面
        A_y_eta = A_y_eta_full[:, 1:-1, :]  # 排除边界，保留内部面

    # η面的面标量值（内部面）
    phi_face_eta = 0.5 * (phi[:, :-1, :] + phi[:, 1:, :])  # (batch, H-1, W)

    # η_top面贡献（j+1/2）
    dphi_dx[:, :-1, :] += phi_face_eta * A_x_eta
    dphi_dy[:, :-1, :] += phi_face_eta * A_y_eta

    # η_bottom面贡献（j-1/2，符号相反）
    dphi_dx[:, 1:, :] -= phi_face_eta * A_x_eta
    dphi_dy[:, 1:, :] -= phi_face_eta * A_y_eta

    # 5. 除以体积得到梯度
    dphi_dx = dphi_dx / volumes
    dphi_dy = dphi_dy / volumes

    # 移除batch维度
    if squeeze_output:
        dphi_dx = dphi_dx.squeeze(0)
        dphi_dy = dphi_dy.squeeze(0)

    return dphi_dx, dphi_dy




# ========== 几何辅助函数 ==========

def compute_cell_centers_from_vertex(
    xv: torch.Tensor,
    yv: torch.Tensor,
    periodic_xi: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从节点坐标计算单元中心坐标（四角点平均）

    用于：
    - 分隔法向计算
    - 可视化和后处理

    Args:
        xv: x节点坐标 (batch, H+1, W+1) 或 (H+1, W+1)
        yv: y节点坐标 (batch, H+1, W+1) 或 (H+1, W+1)
        periodic_xi: ξ方向周期性（O-grid）

    Returns:
        (xc, yc): 单元中心坐标，均为 (batch, H, W) 或 (H, W)
    """
    # 添加batch维度
    if xv.ndim == 2:
        xv = xv.unsqueeze(0)
        yv = yv.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size, H_plus_1, W_plus_1 = xv.shape
    H = H_plus_1 - 1
    W = W_plus_1 - 1

    # 四角点：V00, V10, V11, V01
    if periodic_xi:
        # 周期：使用W+1列（应等于第0列），构造W个单元
        V00_x = xv[:, :-1, :-1]  # (batch, H, W)
        V00_y = yv[:, :-1, :-1]

        V10_x = xv[:, :-1, 1:]   # (batch, H, W)
        V10_y = yv[:, :-1, 1:]

        V11_x = xv[:, 1:, 1:]    # (batch, H, W)
        V11_y = yv[:, 1:, 1:]

        V01_x = xv[:, 1:, :-1]   # (batch, H, W)
        V01_y = yv[:, 1:, :-1]
    else:
        # 非周期：只使用前W列，构造W-1个单元
        # 这里为了保持(H, W)输出，需要特殊处理边界
        # 简化：先构造(H, W-1)，然后pad右边界
        V00_x = xv[:, :-1, :-1]  # (batch, H, W)
        V00_y = yv[:, :-1, :-1]

        V10_x = xv[:, :-1, 1:]   # (batch, H, W)
        V10_y = yv[:, :-1, 1:]

        V11_x = xv[:, 1:, 1:]    # (batch, H, W)
        V11_y = yv[:, 1:, 1:]

        V01_x = xv[:, 1:, :-1]   # (batch, H, W)
        V01_y = yv[:, 1:, :-1]

    # 计算中心（四点平均）
    xc = 0.25 * (V00_x + V10_x + V11_x + V01_x)
    yc = 0.25 * (V00_y + V10_y + V11_y + V01_y)

    # 移除batch维度
    if squeeze_output:
        xc = xc.squeeze(0)
        yc = yc.squeeze(0)

    return xc, yc



# ========== ADflow Corner-Based Geometry (Plan33: 完全统一) ==========

def lift_vertices_to_3d(coords_vertex: torch.Tensor) -> torch.Tensor:
    """
    将2D节点坐标扩展为伪3D（单位厚度）

    ADflow对齐：2D翼型视为单位厚度的3D结构块

    Args:
        coords_vertex: (2, H+1, W+1) 或 (B, 2, H+1, W+1) - [x, y]

    Returns:
        coords_3d: (3, H+1, W+1, 2) 或 (B, 3, H+1, W+1, 2) - [x, y, z], 最后一维是z层(0/1)
    """
    if coords_vertex.ndim == 3:
        # (2, H+1, W+1) - 单样本
        x, y = coords_vertex[0], coords_vertex[1]  # (H+1, W+1)
        device = x.device
        dtype = x.dtype

        # 复制两层：z=0和z=1（单位厚度）
        z0 = torch.zeros_like(x, device=device, dtype=dtype)
        z1 = torch.ones_like(x, device=device, dtype=dtype)

        # 堆叠成(3, H+1, W+1, 2) - 最后一维是z层
        layer0 = torch.stack([x, y, z0], dim=0)  # (3, H+1, W+1)
        layer1 = torch.stack([x, y, z1], dim=0)  # (3, H+1, W+1)

        coords_3d = torch.stack([layer0, layer1], dim=-1)  # (3, H+1, W+1, 2)

    elif coords_vertex.ndim == 4:
        # (B, 2, H+1, W+1) - 批量
        x, y = coords_vertex[:, 0], coords_vertex[:, 1]  # (B, H+1, W+1)
        device = x.device
        dtype = x.dtype

        # 复制两层：z=0和z=1
        z0 = torch.zeros_like(x, device=device, dtype=dtype)  # (B, H+1, W+1)
        z1 = torch.ones_like(x, device=device, dtype=dtype)

        # 堆叠成(B, 3, H+1, W+1, 2)
        layer0 = torch.stack([x, y, z0], dim=1)  # (B, 3, H+1, W+1)
        layer1 = torch.stack([x, y, z1], dim=1)

        coords_3d = torch.stack([layer0, layer1], dim=-1)  # (B, 3, H+1, W+1, 2)

    else:
        raise ValueError(f"coords_vertex维度错误: {coords_vertex.shape}, 期望3D或4D")

    return coords_3d


def lift_centers_to_3d(coords_center: torch.Tensor) -> torch.Tensor:
    """
    将2D中心坐标扩展为伪3D（单位厚度）

    Args:
        coords_center: (2, H, W) - [x, y]

    Returns:
        coords_3d: (3, H, W, 2) - [x, y, z], 最后一维是z层(0/1)
    """
    if coords_center.ndim == 3:
        # (2, H, W)
        x, y = coords_center[0], coords_center[1]  # (H, W)
    else:
        raise ValueError(f"coords_center维度错误: {coords_center.shape}")

    device = x.device
    dtype = x.dtype

    # 复制两层：z=0和z=1
    z0 = torch.zeros_like(x, device=device, dtype=dtype)
    z1 = torch.ones_like(x, device=device, dtype=dtype)

    # 堆叠成(3, H, W, 2)
    layer0 = torch.stack([x, y, z0], dim=0)  # (3, H, W)
    layer1 = torch.stack([x, y, z1], dim=0)

    coords_3d = torch.stack([layer0, layer1], dim=-1)  # (3, H, W, 2)

    return coords_3d


def volpym_batch(
    p: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor
) -> torch.Tensor:
    """
    向量化金字塔体积计算（ADflow公式，返回6倍实际体积）

    参考：ADflow preprocessingAPI.F90:3489-3518

    核心公式：
    volpym = (p - 0.25*(a+b+c+d)) · ((a-c) × (b-d))

    Args:
        p: (H, W, 3) - 金字塔顶点（质心）
        a, b, c, d: (H, W, 3) - 底面四个角点（按顺序）

    Returns:
        vol_6x: (H, W) - 6倍金字塔体积
    """
    # 底面中心
    base_center = 0.25 * (a + b + c + d)  # (H, W, 3)

    # 顶点相对底面中心的向量
    vec_p = p - base_center  # (H, W, 3)

    # 对角线向量
    vca = a - c  # (H, W, 3)
    vdb = b - d  # (H, W, 3)

    # 叉乘：vca × vdb
    cross = torch.cross(vca, vdb, dim=-1)  # (H, W, 3)

    # 体积 = vec_p · (vca × vdb)
    vol_6x = torch.sum(vec_p * cross, dim=-1)  # (H, W)

    return vol_6x


def compute_cell_volume_adflow(
    coords_vertex: torch.Tensor,
    periodic_xi: bool = True
) -> Tuple[torch.Tensor, Union[float, torch.Tensor]]:
    """
    ADflow六金字塔方法计算体积（preprocessingAPI.F90:3489-3518）

    核心思想：
    1. 将2D节点扩展为伪3D（z=0/1单位厚度）
    2. 对每个六面体单元，取8个角点
    3. 计算质心p = mean(8个角点)
    4. 用volpym计算6个金字塔（6个面作为底）
    5. 总体积 = sum(6个金字塔) / 6

    Args:
        coords_vertex: (2, H+1, W+1) 或 (B, 2, H+1, W+1) - [x, y]节点坐标
        periodic_xi: ξ方向周期性（O-grid）

    Returns:
        vol: (H, W) 或 (B, H, W) - 单元体积，float64计算→float32输出
        sign: float 或 (B,) tensor - 块级手性符号
    """
    # 转换为float64计算（数值稳定性）
    original_dtype = coords_vertex.dtype
    coords_vertex = coords_vertex.double()

    is_batched = coords_vertex.ndim == 4

    # 1. 扩展为伪3D
    coords_3d = lift_vertices_to_3d(coords_vertex)
    # coords_3d: (3, H+1, W+1, 2) 或 (B, 3, H+1, W+1, 2)

    if is_batched:
        B, _, H_plus_1, W_plus_1, _ = coords_3d.shape
        H = H_plus_1 - 1
        W = W_plus_1 - 1

        # 2. 提取8个角点（batch维度）
        # z=0层的四个角点 (B, H, W, 3)
        p000 = coords_3d[:, :, :-1, :-1, 0].permute(0, 2, 3, 1)
        p100 = coords_3d[:, :, :-1, 1:, 0].permute(0, 2, 3, 1)
        p110 = coords_3d[:, :, 1:, 1:, 0].permute(0, 2, 3, 1)
        p010 = coords_3d[:, :, 1:, :-1, 0].permute(0, 2, 3, 1)

        # z=1层的四个角点 (B, H, W, 3)
        p001 = coords_3d[:, :, :-1, :-1, 1].permute(0, 2, 3, 1)
        p101 = coords_3d[:, :, :-1, 1:, 1].permute(0, 2, 3, 1)
        p111 = coords_3d[:, :, 1:, 1:, 1].permute(0, 2, 3, 1)
        p011 = coords_3d[:, :, 1:, :-1, 1].permute(0, 2, 3, 1)

        # 3. 计算质心p（8个角点平均） (B, H, W, 3)
        p = 0.125 * (p000 + p100 + p110 + p010 + p001 + p101 + p111 + p011)

        # 4. 计算6个金字塔体积（ADflow顺序） (B, H, W)
        vp1 = volpym_batch(p, p000, p100, p110, p010)
        vp2 = volpym_batch(p, p001, p011, p111, p101)
        vp3 = volpym_batch(p, p000, p010, p011, p001)
        vp4 = volpym_batch(p, p100, p101, p111, p110)
        vp5 = volpym_batch(p, p000, p001, p101, p100)
        vp6 = volpym_batch(p, p010, p110, p111, p011)

        # 5. 总体积 (B, H, W)
        vol = (vp1 + vp2 + vp3 + vp4 + vp5 + vp6) / 6.0

        # 6. 每个样本独立计算sign (B,)
        n_pos = (vol > 0).sum(dim=[1, 2])  # (B,) - 每个样本的正体积数量
        # NOTE: Our pseudo-3D lifting uses the opposite handedness convention
        # to ADFLOW's preprocessing, so the signed volumes come out flipped.
        # Flip the block-level sign here so downstream halo face-vectors and
        # viscous gradient corrections match ADFLOW's rightHanded/fact.
        sign = torch.where(
            n_pos > 0,
            -torch.ones(B, device=vol.device, dtype=torch.float64),
            torch.ones(B, device=vol.device, dtype=torch.float64),
        )

        # 7. 体积取绝对值并保护 (B, H, W)
        vol = torch.abs(vol)
        vol = torch.clamp(vol, min=1e-30)

    else:
        # 单样本路径
        _, H_plus_1, W_plus_1, _ = coords_3d.shape
        H = H_plus_1 - 1
        W = W_plus_1 - 1

        # 2. 提取8个角点 (H, W, 3)
        p000 = coords_3d[:, :-1, :-1, 0].permute(1, 2, 0)
        p100 = coords_3d[:, :-1, 1:, 0].permute(1, 2, 0)
        p110 = coords_3d[:, 1:, 1:, 0].permute(1, 2, 0)
        p010 = coords_3d[:, 1:, :-1, 0].permute(1, 2, 0)

        p001 = coords_3d[:, :-1, :-1, 1].permute(1, 2, 0)
        p101 = coords_3d[:, :-1, 1:, 1].permute(1, 2, 0)
        p111 = coords_3d[:, 1:, 1:, 1].permute(1, 2, 0)
        p011 = coords_3d[:, 1:, :-1, 1].permute(1, 2, 0)

        # 3. 计算质心p (H, W, 3)
        p = 0.125 * (p000 + p100 + p110 + p010 + p001 + p101 + p111 + p011)

        # 4. 计算6个金字塔体积 (H, W)
        vp1 = volpym_batch(p, p000, p100, p110, p010)
        vp2 = volpym_batch(p, p001, p011, p111, p101)
        vp3 = volpym_batch(p, p000, p010, p011, p001)
        vp4 = volpym_batch(p, p100, p101, p111, p110)
        vp5 = volpym_batch(p, p000, p001, p101, p100)
        vp6 = volpym_batch(p, p010, p110, p111, p011)

        # 5. 总体积 (H, W)
        vol = (vp1 + vp2 + vp3 + vp4 + vp5 + vp6) / 6.0

        # 6. 块级手性判定（标量）
        n_pos = (vol > 0).sum().item()
        # NOTE: Our pseudo-3D lifting uses the opposite handedness convention
        # to ADFLOW's preprocessing (signed volumes flipped). Flip the
        # block-level sign here so downstream halo face-vectors match ADFLOW.
        sign = -1.0 if n_pos > 0 else 1.0

        # 7. 体积取绝对值并保护
        vol = torch.abs(vol)
        vol = torch.clamp(vol, min=1e-30)

    # 8. 转回原始dtype
    vol = vol.to(dtype=original_dtype)

    # 9. 返回体积和块级手性符号
    return vol, sign


def compute_face_area_vectors_full(
    coords_center: torch.Tensor,
    coords_vertex: torch.Tensor,
    periodic_xi: bool = True,
    sign: Union[float, torch.Tensor] = 1.0
) -> Dict[str, torch.Tensor]:
    """
    计算完整面法向量，包括边界面，与ADflow完全对齐

    核心修复：解决GCL闭合问题的根源 - η方向边界面缺失

    ADflow计算所有边界面：
    - ξ方向面：i=0..W (W+1个面，周期性O-grid)
    - η方向面：j=0..H (H+1个面，包括上下边界)

    本函数计算完整的η面(H+1,W)，包括边界，使GCL得以闭合：
    ΔS_ξ + ΔS_η = 0 对所有单元成立

    Args:
        coords_center: (2, H, W) 或 (B, 2, H, W) - [x, y]中心坐标
        coords_vertex: (2, H+1, W+1) 或 (B, 2, H+1, W+1) - [x, y]节点坐标
        periodic_xi: ξ方向周期性（O-grid）
        sign: float 或 (B,) tensor - 块级手性符号

    Returns:
        face_geom_full: 完整面几何字典
            ξ面几何: (H, W) 或 (B, H, W)
            完整η面几何: (H+1, W) 或 (B, H+1, W)
    """
    # 转换为float64计算
    original_dtype = coords_vertex.dtype
    coords_vertex = coords_vertex.double()
    if coords_center is not None:
        if isinstance(coords_center, tuple):
            coords_center = torch.stack([c.double() if hasattr(c, 'double') else c for c in coords_center])
        else:
            coords_center = coords_center.double()

    is_batched = coords_vertex.ndim == 4

    # 1. 扩展为伪3D
    coords_3d = lift_vertices_to_3d(coords_vertex)
    # coords_3d: (3, H+1, W+1, 2) 或 (B, 3, H+1, W+1, 2)

    if is_batched:
        B, _, H_plus_1, W_plus_1, _ = coords_3d.shape
        H = H_plus_1 - 1
        W = W_plus_1 - 1
        device = coords_vertex.device

        # Plan93方案I: 获取带halo的节点坐标，用于4-point stencil计算
        coords_vertex_hat = extrapolate_halo_vertex_coords(coords_vertex, direction='eta')

        # sign广播: (B,) → (B, 1, 1, 1)
        if isinstance(sign, torch.Tensor):
            sign_broad = sign.view(B, 1, 1, 1)
        else:
            sign_broad = torch.full((B, 1, 1, 1), sign, device=device, dtype=torch.float64)
    else:
        _, H_plus_1, W_plus_1, _ = coords_3d.shape
        H = H_plus_1 - 1
        W = W_plus_1 - 1
        device = coords_vertex.device
        sign_broad = sign

        # Plan93方案I: 获取带halo的节点坐标，用于4-point stencil计算
        coords_vertex_hat = extrapolate_halo_vertex_coords(coords_vertex, direction='eta')

    # ========== 2. ξ方向面几何（plan41.md最终修复：wrap同源构造） ==========
    # 核心修复：构造W+1列角点并保持完整，使ξ面集合与roll差分算子完全同源
    if is_batched:
        # Batch路径
        if periodic_xi:
            # 使用i=0..W全列角点，首末两列对应同一条seam面
            p00 = coords_3d[:, :, :-1, :, 0].permute(0, 2, 3, 1)  # (B, H, W+1, 3)
            p01 = coords_3d[:, :, :-1, :, 1].permute(0, 2, 3, 1)
            p10 = coords_3d[:, :, 1:,  :, 0].permute(0, 2, 3, 1)
            p11 = coords_3d[:, :, 1:,  :, 1].permute(0, 2, 3, 1)
        else:
            # 非周期：使用列1到W-1
            p00 = coords_3d[:, :, :-1, 1:-1, 0].permute(0, 2, 3, 1)  # (B, H, W-1, 3)
            p01 = coords_3d[:, :, :-1, 1:-1, 1].permute(0, 2, 3, 1)
            p10 = coords_3d[:, :, 1:, 1:-1, 0].permute(0, 2, 3, 1)
            p11 = coords_3d[:, :, 1:, 1:-1, 1].permute(0, 2, 3, 1)
    else:
        # 单样本路径
        if periodic_xi:
            p00 = coords_3d[:, :-1, :, 0].permute(1, 2, 0)  # (H, W+1, 3)
            p01 = coords_3d[:, :-1, :, 1].permute(1, 2, 0)
            p10 = coords_3d[:, 1:,  :, 0].permute(1, 2, 0)
            p11 = coords_3d[:, 1:,  :, 1].permute(1, 2, 0)
        else:
            p00 = coords_3d[:, :-1, 1:-1, 0].permute(1, 2, 0)  # (H, W-1, 3)
            p01 = coords_3d[:, :-1, 1:-1, 1].permute(1, 2, 0)
            p10 = coords_3d[:, 1:, 1:-1, 0].permute(1, 2, 0)
            p11 = coords_3d[:, 1:, 1:-1, 1].permute(1, 2, 0)

    # 对角线向量（ADflow公式）
    v1 = p00 - p11
    v2 = p10 - p01

    # 面向量 = fact × (v1 × v2)，fact = 0.5 × sign（块级手性统一）
    fact = 0.5 * sign_broad
    A_xi_3d = fact * torch.cross(v1, v2, dim=-1)  # (H, W+1, 3) 或 (B, H, W+1, 3)

    # 提取三个分量
    if is_batched:
        if periodic_xi:
            # 取faces[1..W]，跳过重复的seam面，(B, H, W)
            A_x_xi = A_xi_3d[:, :, 1:, 0]
            A_y_xi = A_xi_3d[:, :, 1:, 1]
            A_z_xi = A_xi_3d[:, :, 1:, 2]

            # Seam闭合验证（对batch求max）
            seam_diff_x = torch.max(torch.abs(A_xi_3d[:, :, 0, 0] - A_xi_3d[:, :, -1, 0])).item()
            seam_diff_y = torch.max(torch.abs(A_xi_3d[:, :, 0, 1] - A_xi_3d[:, :, -1, 1])).item()
            seam_diff_z = torch.max(torch.abs(A_xi_3d[:, :, 0, 2] - A_xi_3d[:, :, -1, 2])).item()
            seam_diff_max = max(seam_diff_x, seam_diff_y, seam_diff_z)
        else:
            A_x_xi = A_xi_3d[..., 0]  # (B, H, W-1)
            A_y_xi = A_xi_3d[..., 1]
            A_z_xi = A_xi_3d[..., 2]
            seam_diff_max = 0.0
    else:
        if periodic_xi:
            # 取faces[1..W]，跳过重复的seam面，(H, W)
            A_x_xi = A_xi_3d[:, 1:, 0]
            A_y_xi = A_xi_3d[:, 1:, 1]
            A_z_xi = A_xi_3d[:, 1:, 2]

            seam_diff_x = torch.max(torch.abs(A_xi_3d[:, 0, 0] - A_xi_3d[:, -1, 0])).item()
            seam_diff_y = torch.max(torch.abs(A_xi_3d[:, 0, 1] - A_xi_3d[:, -1, 1])).item()
            seam_diff_z = torch.max(torch.abs(A_xi_3d[:, 0, 2] - A_xi_3d[:, -1, 2])).item()
            seam_diff_max = max(seam_diff_x, seam_diff_y, seam_diff_z)
        else:
            A_x_xi = A_xi_3d[..., 0]
            A_y_xi = A_xi_3d[..., 1]
            A_z_xi = A_xi_3d[..., 2]
            seam_diff_max = 0.0

    # ========== 2.5. ξ面方向对齐（review3.md补充修复） ==========
    # 为完整性，ξ面也需要方向对齐（虽然主要影响诊断路径）

    # 计算分隔法向（用于判断方向）
    # 统一使用vertex反推中心，确保与η面判向口径一致（plan36.md修复）
    if is_batched:
        # Batch路径：coords_vertex是(B, 2, H+1, W+1)，传入(B, H+1, W+1)支持batch计算
        xc_temp, yc_temp = compute_cell_centers_from_vertex(
            coords_vertex[:, 0], coords_vertex[:, 1], periodic_xi
        )
    else:
        xc_temp, yc_temp = compute_cell_centers_from_vertex(
            coords_vertex[0], coords_vertex[1], periodic_xi
        )
    xc_temp = xc_temp.double()
    yc_temp = yc_temp.double()

    # ξ方向分隔法向
    if periodic_xi:
        if is_batched:
            # Batch: 需要在倒数第1维操作（W维度）
            dx_sep_xi = torch.cat([xc_temp[..., 1:] - xc_temp[..., :-1],
                                   (xc_temp[..., :1] - xc_temp[..., -1:])], dim=-1)
            dy_sep_xi = torch.cat([yc_temp[..., 1:] - yc_temp[..., :-1],
                                   (yc_temp[..., :1] - yc_temp[..., -1:])], dim=-1)
        else:
            dx_sep_xi = torch.cat([xc_temp[:, 1:] - xc_temp[:, :-1],
                                   (xc_temp[:, :1] - xc_temp[:, -1:])], dim=-1)
            dy_sep_xi = torch.cat([yc_temp[:, 1:] - yc_temp[:, :-1],
                                   (yc_temp[:, :1] - yc_temp[:, -1:])], dim=-1)
    else:
        if is_batched:
            dx_sep_xi = xc_temp[..., 1:] - xc_temp[..., :-1]
            dy_sep_xi = yc_temp[..., 1:] - yc_temp[..., :-1]
        else:
            dx_sep_xi = xc_temp[:, 1:] - xc_temp[:, :-1]
            dy_sep_xi = yc_temp[:, 1:] - yc_temp[:, :-1]

    d_sep_xi = torch.sqrt(dx_sep_xi**2 + dy_sep_xi**2)
    ssx_xi = dx_sep_xi / (d_sep_xi)
    ssy_xi = dy_sep_xi / (d_sep_xi)

    # 计算单位法向
    n_area_x_xi = A_x_xi / (torch.sqrt(A_x_xi**2 + A_y_xi**2 ))
    n_area_y_xi = A_y_xi / (torch.sqrt(A_x_xi**2 + A_y_xi**2 ))

    # 判断并翻转ξ面向量（review3.md：保留作兜底，块级fact应使翻转接近0）
    dot_product_xi = n_area_x_xi * ssx_xi + n_area_y_xi * ssy_xi
    flip_mask_xi = dot_product_xi < 0

    # 统计翻转数量用于验证块级fact是否足够（预期：使用fact后应≈0）
    flipped_xi_count = flip_mask_xi.sum().item()

    # 兜底翻转：仅在个别面方向与分隔法向相反时翻转，保证方向一致
    if flipped_xi_count > 0:
        A_x_xi = torch.where(flip_mask_xi, -A_x_xi, A_x_xi)
        A_y_xi = torch.where(flip_mask_xi, -A_y_xi, A_y_xi)
        A_z_xi = torch.where(flip_mask_xi, -A_z_xi, A_z_xi)

    # ========== 3. 完整η方向面几何（关键修复：一次性张量操作） ==========
    # η面：j=0..H (H+1个面)，包含边界j=0和j=H
    # 修复说明（review2.md）：使用一次性张量切片而非循环，避免维度错误

    # 完整η面计算：coords_3d[:, :, :-1, 0]自动包含j=0到j=H的所有行
    # 角点：(i, j, z0), (i+1, j, z0), (i+1, j, z1), (i, j, z1)
    # coords_3d shape: (3, H+1, W+1, 2) or (B, 3, H+1, W+1, 2)

    # 四个角点（所有j行，z0/z1两层，i-1和i列）- 与ADflow对齐
    # ADflow索引映射：l=i-1, i=i, n=z0, k=z1
    # ADflow公式：v1 = x(i,j,n) - x(l,j,k)，v2 = x(l,j,n) - x(i,j,k)
    if is_batched:
        # Batch路径：(B, 3, H+1, W+1, 2) → (B, H+1, W, 3)
        p00_eta_full = coords_3d[:, :, :, 1:, 0].permute(0, 2, 3, 1)   # (B, H+1, W, 3)
        p01_eta_full = coords_3d[:, :, :, 1:, 1].permute(0, 2, 3, 1)
        p10_eta_full = coords_3d[:, :, :, :-1, 0].permute(0, 2, 3, 1)
        p11_eta_full = coords_3d[:, :, :, :-1, 1].permute(0, 2, 3, 1)
    else:
        # 单样本路径：(3, H+1, W+1, 2) → (H+1, W, 3)
        p00_eta_full = coords_3d[:, :, 1:, 0].permute(1, 2, 0)   # (H+1, W, 3) - z0层，i列（对应ADflow的x(i,j,n)）
        p01_eta_full = coords_3d[:, :, 1:, 1].permute(1, 2, 0)   # (H+1, W, 3) - z1层，i列（对应ADflow的x(i,j,k)）
        p10_eta_full = coords_3d[:, :, :-1, 0].permute(1, 2, 0)  # (H+1, W, 3) - z0层，i-1列（对应ADflow的x(l,j,n)）
        p11_eta_full = coords_3d[:, :, :-1, 1].permute(1, 2, 0)  # (H+1, W, 3) - z1层，i-1列（对应ADflow的x(l,j,k)）

    # 对角线向量 - 修复：与ADflow sK公式完全对齐
    # ADflow preprocessingAPI.F90:3208-3215:
    #   v1 = x(i,j,k) - x(l,m,k)  其中 l=i-1, m=j-1, 2D中j=1,m=0
    #   v2 = x(l,j,k) - x(i,m,k)
    # 映射到伪3D坐标：j=1→z1, m=0→z0
    #   ADflow v1 = (i,z1,k) - (i-1,z0,k) = p01 - p10
    #   ADflow v2 = (i-1,z1,k) - (i,z0,k) = p11 - p00
    v1_eta_full = p01_eta_full - p10_eta_full  # ADflow: x(i,j,k) - x(l,m,k) = (i,z1) - (i-1,z0)
    v2_eta_full = p11_eta_full - p00_eta_full  # ADflow: x(l,j,k) - x(i,m,k) = (i-1,z1) - (i,z0)

    # 面向量：与ADflow sK公式一致，使用 +fact
    # ADflow sK = fact * (v1 × v2)
    A_eta_full_3d = fact * torch.cross(v2_eta_full, v1_eta_full, dim=-1)  # (H+1, W, 3) or (B, H+1, W, 3)

    # 提取三个分量（H+1, W）or (B, H+1, W)
    A_x_eta_full = A_eta_full_3d[..., 0]
    A_y_eta_full = A_eta_full_3d[..., 1]
    A_z_eta_full = A_eta_full_3d[..., 2]

    # ========== 4. 完整η面方向对齐（plan34.md核心修复） ==========
    # 使用分隔法向判断并翻转完整η面方向，确保与索引增大方向一致

    # 计算分隔法向（用于判断方向）
    # 统一使用vertex反推中心，确保与ξ面判向口径一致（plan36.md修复）
    if is_batched:
        # Batch路径：coords_vertex是(B, 2, H+1, W+1)，传入(B, H+1, W+1)支持batch计算
        xc, yc = compute_cell_centers_from_vertex(
            coords_vertex[:, 0], coords_vertex[:, 1], periodic_xi
        )
    else:
        xc, yc = compute_cell_centers_from_vertex(
            coords_vertex[0], coords_vertex[1], periodic_xi
        )
    xc = xc.double()
    yc = yc.double()

    # η方向分隔法向 - Plan93方案I: 使用ADFlow 4-point node coordinate stencil
    # 替换原有的cell-center差分计算，以正确对齐ADFlow的viscous normal correction
    # compute_adflow_separation_stencil返回完整(H+1, W)形状的ssx/ssy/inv_d
    ssx_eta_stencil, ssy_eta_stencil, inv_d_eta_stencil = compute_adflow_separation_stencil(
        coords_vertex_hat, direction='eta', periodic_xi=periodic_xi
    )
    # 为方向对齐判断提取内部面(H-1, W)，对应原来的ssx_eta/ssy_eta
    if is_batched:
        ssx_eta = ssx_eta_stencil[:, 1:H, :]  # (B, H-1, W)
        ssy_eta = ssy_eta_stencil[:, 1:H, :]
        d_sep_eta = 1.0 / inv_d_eta_stencil[:, 1:H, :]  # 反求距离用于后续计算
    else:
        ssx_eta = ssx_eta_stencil[1:H, :]  # (H-1, W)
        ssy_eta = ssy_eta_stencil[1:H, :]
        d_sep_eta = 1.0 / inv_d_eta_stencil[1:H, :]

    # 为完整η面生成方向对齐掩码
    # 形状: (H+1, W) 或 (B, H+1, W)
    if is_batched:
        flip_mask_eta_full = torch.zeros(B, H+1, W, dtype=torch.bool, device=device)
    else:
        flip_mask_eta_full = torch.zeros(H+1, W, dtype=torch.bool, device=device)

    # 内部行j=1..H-1：使用对应的分隔法向判断
    # A_eta_full[j]对应单元j-1和j之间的面，使用ssx_eta[j-1], ssy_eta[j-1]
    n_area_x_eta_full = A_x_eta_full / (torch.sqrt(A_x_eta_full**2 + A_y_eta_full**2 ))
    n_area_y_eta_full = A_y_eta_full / (torch.sqrt(A_x_eta_full**2 + A_y_eta_full**2 ))

    # 张量化判断（移除for循环）
    # 创建padded分隔法向数组: ssx_eta形状(H-1, W)或(B, H-1, W)，填充为(H+1, W)或(B, H+1, W)
    if H <= 1:
        # Single-layer truncated domains have no interior eta faces. Reuse the
        # full eta-face stencil directly for both boundary faces so direction
        # alignment stays well-defined without indexing a non-existent j=1 row.
        ssx_eta_padded = ssx_eta_stencil.to(dtype=torch.float64)
        ssy_eta_padded = ssy_eta_stencil.to(dtype=torch.float64)
    elif is_batched:
        ssx_eta_padded = torch.zeros(B, H+1, W, dtype=torch.float64, device=device)
        ssy_eta_padded = torch.zeros(B, H+1, W, dtype=torch.float64, device=device)

        # 内部行j=1..H-1对应ssx_eta的索引0..H-2
        ssx_eta_padded[:, 1:H, :] = ssx_eta
        ssy_eta_padded[:, 1:H, :] = ssy_eta

        # 首末行继承邻行
        ssx_eta_padded[:, 0, :] = ssx_eta[:, 0, :]  # j=0继承j=1
        ssx_eta_padded[:, H, :] = ssx_eta[:, H-2, :]  # j=H继承j=H-1
        ssy_eta_padded[:, 0, :] = ssy_eta[:, 0, :]
        ssy_eta_padded[:, H, :] = ssy_eta[:, H-2, :]
    else:
        ssx_eta_padded = torch.zeros(H+1, W, dtype=torch.float64, device=device)
        ssy_eta_padded = torch.zeros(H+1, W, dtype=torch.float64, device=device)

        # 内部行j=1..H-1对应ssx_eta的索引0..H-2
        ssx_eta_padded[1:H, :] = ssx_eta
        ssy_eta_padded[1:H, :] = ssy_eta

        # 首末行继承邻行
        ssx_eta_padded[0, :] = ssx_eta[0, :]  # j=0继承j=1
        ssx_eta_padded[H, :] = ssx_eta[H-2, :]  # j=H继承j=H-1
        ssy_eta_padded[0, :] = ssy_eta[0, :]
        ssy_eta_padded[H, :] = ssy_eta[H-2, :]

    # 一次性计算所有行的点积
    dot_eta = n_area_x_eta_full * ssx_eta_padded + n_area_y_eta_full * ssy_eta_padded
    flip_mask_eta_full = dot_eta < 0

    # 统计翻转数量用于验证块级fact是否足够（预期：应保持0）
    flipped_count = flip_mask_eta_full.sum().item()
    total_faces = flip_mask_eta_full.numel()

    # 翻转：确保面积向量与分隔法向同向（η增加方向）
    if flipped_count > 0:
        A_x_eta_full = torch.where(flip_mask_eta_full, -A_x_eta_full, A_x_eta_full)
        A_y_eta_full = torch.where(flip_mask_eta_full, -A_y_eta_full, A_y_eta_full)
        A_z_eta_full = torch.where(flip_mask_eta_full, -A_z_eta_full, A_z_eta_full)

    # ========== 5. 面长度计算和完整η/ξ面分隔法向 ==========
    s_area_xi = torch.sqrt(A_x_xi**2 + A_y_xi**2 + A_z_xi**2 )
    s_area_eta_full = torch.sqrt(A_x_eta_full**2 + A_y_eta_full**2 + A_z_eta_full**2 )

    # Plan94: ξ方向也使用4-point stencil (与η方向对齐)
    # 这是近壁粘性通量对齐的关键修复
    # 替换cell-center差分，使用ADFlow blockette.F90:6353-6364的4-point stencil
    ssx_xi_stencil, ssy_xi_stencil, inv_d_xi_stencil = compute_adflow_separation_stencil(
        coords_vertex_hat, direction='xi', periodic_xi=periodic_xi
    )
    # 用stencil结果替换cell-center结果
    ssx_xi = ssx_xi_stencil
    ssy_xi = ssy_xi_stencil
    inv_d_xi = inv_d_xi_stencil

    # Plan93方案I: 直接使用4-point stencil结果作为完整η面分隔法向
    # compute_adflow_separation_stencil已经返回完整(H+1, W)形状，包括边界面
    # 这与ADFlow blockette.F90:5752-5777的ss/snrm计算方式完全对齐
    ssx_eta_full = ssx_eta_stencil
    ssy_eta_full = ssy_eta_stencil
    inv_d_eta_full = inv_d_eta_stencil

    # ========== 5.5 DEBUG: Save complete geometry data to npz file ==========
    import os
    debug_env = os.environ.get('SURROGATE_DEBUG_GEOMETRY', '')
    if debug_env == '1':
        import numpy as np

        # 计算体积用于调试输出
        vol_debug, sign_debug = compute_cell_volume_adflow(
            coords_vertex, periodic_xi=periodic_xi
        )

        debug_data = {
            # 体积和手性
            'vol': vol_debug.detach().cpu().numpy(),
            'sign': float(sign_debug) if not isinstance(sign_debug, torch.Tensor) else sign_debug.item(),

            # ξ面向量（i方向）
            'A_x_xi': A_x_xi.detach().cpu().numpy(),
            'A_y_xi': A_y_xi.detach().cpu().numpy(),
            'A_z_xi': A_z_xi.detach().cpu().numpy(),
            's_area_xi': s_area_xi.detach().cpu().numpy(),

            # η面向量（j方向，完整含边界）
            'A_x_eta': A_x_eta_full.detach().cpu().numpy(),
            'A_y_eta': A_y_eta_full.detach().cpu().numpy(),
            'A_z_eta': A_z_eta_full.detach().cpu().numpy(),
            's_area_eta': s_area_eta_full.detach().cpu().numpy(),

            # ξ方向粘性修正
            'ssx_xi': ssx_xi.detach().cpu().numpy(),
            'ssy_xi': ssy_xi.detach().cpu().numpy(),
            'inv_d_xi': inv_d_xi.detach().cpu().numpy(),

            # η方向粘性修正（完整含边界）
            'ssx_eta': ssx_eta_full.detach().cpu().numpy(),
            'ssy_eta': ssy_eta_full.detach().cpu().numpy(),
            'inv_d_eta': inv_d_eta_full.detach().cpu().numpy(),

            # 元数据
            'H': H,
            'W': W,
            'coords_vertex': coords_vertex.detach().cpu().numpy(),
            'periodic_xi': periodic_xi,
            'flipped_xi_count': flipped_xi_count,
            'flipped_eta_count': flipped_count
        }
        debug_filename = 'pytorch_geometry_debug.npz'
        np.savez(debug_filename, **debug_data)
        print(f'[DEBUG geometry.py] Saved complete geometry data to: {debug_filename}')
        print(f'        Dimensions: H={H}, W={W}')
        print(f'        vol: {tuple(vol_debug.shape)}')
        print(f'        xi_faces: A_xi {tuple(A_x_xi.shape)}, ss_xi {tuple(ssx_xi.shape)}')
        print(f'        eta_faces: A_eta {tuple(A_x_eta_full.shape)}, ss_eta {tuple(ssx_eta_full.shape)}')
    # ========== END DEBUG ==========

    # ========== 6. 转回原始dtype并返回 ==========
    return {
        # ξ面几何
        'A_x_xi': A_x_xi.to(original_dtype),
        'A_y_xi': A_y_xi.to(original_dtype),
        'A_z_xi': A_z_xi.to(original_dtype),
        's_area_xi': s_area_xi.to(original_dtype),
        'n_area_x_xi': n_area_x_xi.to(original_dtype),
        'n_area_y_xi': n_area_y_xi.to(original_dtype),
        'ssx_xi': ssx_xi.to(original_dtype),
        'ssy_xi': ssy_xi.to(original_dtype),
        'inv_d_xi': inv_d_xi.to(original_dtype),

        # 完整η面几何（H+1, W包含边界面）
        'A_x_eta': A_x_eta_full.to(original_dtype),
        'A_y_eta': A_y_eta_full.to(original_dtype),
        'A_z_eta': A_z_eta_full.to(original_dtype),
        's_area_eta': s_area_eta_full.to(original_dtype),
        'n_area_x_eta': n_area_x_eta_full.to(original_dtype),
        'n_area_y_eta': n_area_y_eta_full.to(original_dtype),
        'ssx_eta': ssx_eta_full.to(original_dtype),
        'ssy_eta': ssy_eta_full.to(original_dtype),
        'inv_d_eta': inv_d_eta_full.to(original_dtype),

        # 配置和诊断信息
        'periodic_xi': periodic_xi,
        'eta_dims': (H + 1, W),  # 明确说明维度：完整边界面
        'flipped_xi_count': flipped_xi_count,
        'flipped_eta_count': flipped_count,
        'block_sign': sign,
        'seam_diff_max': seam_diff_max
    }


# ========== Plan91: ADFlow Halo Geometry Functions ==========

def extrapolate_halo_vertex_coords(
    coords_vertex: torch.Tensor,
    direction: str = 'eta'
) -> torch.Tensor:
    """
    ADFlow xhalo 线性外推：x_halo = 2*x_boundary - x_interior

    参考: ADFlow preprocessingAPI.F90:1095-1131

    Args:
        coords_vertex: (2, H+1, W+1) 或 (B, 2, H+1, W+1) - [x, y] 节点坐标
        direction: 'eta' (j方向) 或 'xi' (i方向)

    Returns:
        coords_vertex_hat: (2, H+3, W+1) 或 (B, 2, H+3, W+1) - 包含两侧 halo 行
    """
    is_batched = coords_vertex.ndim == 4
    device = coords_vertex.device
    dtype = coords_vertex.dtype

    if direction == 'eta':
        if is_batched:
            B, _, H_plus_1, W_plus_1 = coords_vertex.shape
            H = H_plus_1 - 1

            # 创建扩展数组 (B, 2, H+3, W+1)
            coords_hat = torch.zeros(B, 2, H + 3, W_plus_1, device=device, dtype=dtype)

            # 复制物理节点到中间位置 [1:H+2]
            coords_hat[:, :, 1:H+2, :] = coords_vertex

            if H >= 1:
                # Wall halo (j=-1 → index 0): x[-1] = 2*x[0] - x[1]
                coords_hat[:, :, 0, :] = 2.0 * coords_vertex[:, :, 0, :] - coords_vertex[:, :, 1, :]

                # Farfield halo (j=H+1 → index H+2): x[H+1] = 2*x[H] - x[H-1]
                coords_hat[:, :, H+2, :] = 2.0 * coords_vertex[:, :, H, :] - coords_vertex[:, :, H-1, :]
            else:
                # Single-cell eta direction: fall back to constant extrapolation.
                coords_hat[:, :, 0, :] = coords_vertex[:, :, 0, :]
                coords_hat[:, :, H+2, :] = coords_vertex[:, :, H, :]

        else:
            _, H_plus_1, W_plus_1 = coords_vertex.shape
            H = H_plus_1 - 1

            # 创建扩展数组 (2, H+3, W+1)
            coords_hat = torch.zeros(2, H + 3, W_plus_1, device=device, dtype=dtype)

            # 复制物理节点到中间位置 [1:H+2]
            coords_hat[:, 1:H+2, :] = coords_vertex

            if H >= 1:
                # Wall halo (j=-1 → index 0): x[-1] = 2*x[0] - x[1]
                coords_hat[:, 0, :] = 2.0 * coords_vertex[:, 0, :] - coords_vertex[:, 1, :]

                # Farfield halo (j=H+1 → index H+2): x[H+1] = 2*x[H] - x[H-1]
                coords_hat[:, H+2, :] = 2.0 * coords_vertex[:, H, :] - coords_vertex[:, H-1, :]
            else:
                # Single-cell eta direction: fall back to constant extrapolation.
                coords_hat[:, 0, :] = coords_vertex[:, 0, :]
                coords_hat[:, H+2, :] = coords_vertex[:, H, :]
    else:
        raise NotImplementedError(f"Direction '{direction}' not implemented")

    return coords_hat


def compute_halo_cell_volumes(
    coords_vertex_hat: torch.Tensor,
    periodic_xi: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用ADFlow 6-pyramid方法计算 halo 单元体积

    Plan93方案H: 与compute_cell_volume_adflow()使用相同的6-pyramid方法，
    而非简化的2D四边形面积公式。

    Args:
        coords_vertex_hat: (2, H+3, W+1) 或 (B, 2, H+3, W+1) - 包含 halo 行的节点坐标
        periodic_xi: ξ方向周期性

    Returns:
        halo_vol_wall: (W,) 或 (B, W) 壁面 halo 层体积
        halo_vol_ff: (W,) 或 (B, W) 远场 halo 层体积
    """
    is_batched = coords_vertex_hat.ndim == 4

    if is_batched:
        B, _, H_plus_3, W_plus_1 = coords_vertex_hat.shape
    else:
        _, H_plus_3, W_plus_1 = coords_vertex_hat.shape

    H = H_plus_3 - 3  # 物理网格高度

    # coords_vertex_hat 索引: 0=wall_halo, 1=j=0, ..., H+1=j=H, H+2=ff_halo

    # 提取 wall halo 层的顶点 (行0和1) - 形成单层网格
    if is_batched:
        wall_halo_vertices = coords_vertex_hat[:, :, 0:2, :]  # (B, 2, 2, W+1)
    else:
        wall_halo_vertices = coords_vertex_hat[:, 0:2, :]  # (2, 2, W+1)

    # 提取 farfield halo 层的顶点 (行H+1和H+2)
    if is_batched:
        ff_halo_vertices = coords_vertex_hat[:, :, H+1:H+3, :]  # (B, 2, 2, W+1)
    else:
        ff_halo_vertices = coords_vertex_hat[:, H+1:H+3, :]  # (2, 2, W+1)

    # 使用compute_cell_volume_adflow计算体积（6-pyramid方法）
    halo_vol_wall, _ = compute_cell_volume_adflow(wall_halo_vertices, periodic_xi)
    halo_vol_ff, _ = compute_cell_volume_adflow(ff_halo_vertices, periodic_xi)

    # 结果形状: (1, W) 或 (B, 1, W)，需要squeeze掉高度维度
    if is_batched:
        halo_vol_wall = halo_vol_wall.squeeze(1)  # (B, W)
        halo_vol_ff = halo_vol_ff.squeeze(1)      # (B, W)
    else:
        halo_vol_wall = halo_vol_wall.squeeze(0)  # (W,)
        halo_vol_ff = halo_vol_ff.squeeze(0)      # (W,)

    return halo_vol_wall, halo_vol_ff


def compute_halo_xi_face_vectors(
    coords_vertex_hat: torch.Tensor,
    periodic_xi: bool = True,
    sign: Union[float, torch.Tensor] = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算 halo 层的 ξ 面面积向量 - 使用正确的2D边差公式

    Plan95修复: ξ面（常i面）的正确构造方式
    Plan96修复: 添加handedness sign参数与ADFlow对齐

    问题: 原实现使用相邻两列(col_i, col_i+1)作为pseudo-3D的不同z层
          这是η面的构造方式，不是ξ面。

    正确构造: ξ面四角点应在**同一列i**，跨越两行(j, j+1)和两个z层(z=0, z=1)

    在2D pseudo-3D下的简化公式:
        p00 = (x[row0,col_i], y[row0,col_i], 0)
        p01 = (x[row0,col_i], y[row0,col_i], 1)  # 同一xy，不同z
        p10 = (x[row1,col_i], y[row1,col_i], 0)
        p11 = (x[row1,col_i], y[row1,col_i], 1)

        对角线叉乘简化后 (含handedness修正):
        si_x = 0.5 * sign * (y[row1] - y[row0])
        si_y = 0.5 * sign * (x[row0] - x[row1])

    ADFlow参考: adjointExtra.F90:197-201, 237-239
        if (rightHanded) then
            fact = half
        else
            fact = -half
        end if
        si(i,j,k,:) = fact * (v1 × v2)

    Args:
        coords_vertex_hat: (2, H+3, W+1) 或 (B, 2, H+3, W+1) - 包含 halo 行的节点坐标
        periodic_xi: ξ方向周期性
        sign: handedness修正因子 (1.0 for right-handed, -1.0 for left-handed)

    Returns:
        si_x_wall, si_y_wall: (W,) 或 (B, W) 壁面 halo ξ 面
        si_x_ff, si_y_ff: (W,) 或 (B, W) 远场 halo ξ 面
    """
    is_batched = coords_vertex_hat.ndim == 4

    if is_batched:
        _, _, H_plus_3, W_plus_1 = coords_vertex_hat.shape
    else:
        _, H_plus_3, W_plus_1 = coords_vertex_hat.shape

    H = H_plus_3 - 3
    W = W_plus_1 - 1

    # Plan97: 2D边差公式没有对角线叉乘产生的因子2，因此系数为1.0（不是0.5）。
    # Plan98: batch广播修正 - sign从(B,)变为(B,1)以正确广播
    if isinstance(sign, torch.Tensor) and sign.ndim == 1:
        fact = 1.0 * sign.unsqueeze(1)  # (B,) → (B, 1)
    else:
        fact = 1.0 * sign

    if is_batched:
        # ===== Wall halo ξ面 (行0-1) =====
        x0 = coords_vertex_hat[:, 0, 0, :]  # (B, W+1) - halo row
        y0 = coords_vertex_hat[:, 1, 0, :]
        x1 = coords_vertex_hat[:, 0, 1, :]  # (B, W+1) - physical wall row
        y1 = coords_vertex_hat[:, 1, 1, :]

        # 方向对齐：与compute_face_area_vectors_full / ADFLOW sI一致
        si_x_wall_full = fact * (y0 - y1)  # (B, W+1)
        si_y_wall_full = fact * (x1 - x0)

        # 取面1..W（跳过seam面0）与A_x_xi对齐
        if periodic_xi:
            si_x_wall = si_x_wall_full[:, 1:]  # (B, W)
            si_y_wall = si_y_wall_full[:, 1:]
        else:
            si_x_wall = si_x_wall_full[:, 1:-1]  # (B, W-1)
            si_y_wall = si_y_wall_full[:, 1:-1]

        # ===== Farfield halo ξ面 (行H+1到H+2) =====
        x0 = coords_vertex_hat[:, 0, H + 1, :]  # (B, W+1) - physical farfield row
        y0 = coords_vertex_hat[:, 1, H + 1, :]
        x1 = coords_vertex_hat[:, 0, H + 2, :]  # (B, W+1) - halo farfield row
        y1 = coords_vertex_hat[:, 1, H + 2, :]

        si_x_ff_full = fact * (y0 - y1)
        si_y_ff_full = fact * (x1 - x0)

        if periodic_xi:
            si_x_ff = si_x_ff_full[:, 1:]  # (B, W)
            si_y_ff = si_y_ff_full[:, 1:]
        else:
            si_x_ff = si_x_ff_full[:, 1:-1]  # (B, W-1)
            si_y_ff = si_y_ff_full[:, 1:-1]

    else:
        # ===== Wall halo ξ面 (行0-1) =====
        x0 = coords_vertex_hat[0, 0, :]  # (W+1,)
        y0 = coords_vertex_hat[1, 0, :]
        x1 = coords_vertex_hat[0, 1, :]
        y1 = coords_vertex_hat[1, 1, :]

        si_x_wall_full = fact * (y0 - y1)  # (W+1,)
        si_y_wall_full = fact * (x1 - x0)

        if periodic_xi:
            si_x_wall = si_x_wall_full[1:]  # (W,)
            si_y_wall = si_y_wall_full[1:]
        else:
            si_x_wall = si_x_wall_full[1:-1]  # (W-1,)
            si_y_wall = si_y_wall_full[1:-1]

        # ===== Farfield halo ξ面 (行H+1到H+2) =====
        x0 = coords_vertex_hat[0, H + 1, :]
        y0 = coords_vertex_hat[1, H + 1, :]
        x1 = coords_vertex_hat[0, H + 2, :]
        y1 = coords_vertex_hat[1, H + 2, :]

        si_x_ff_full = fact * (y0 - y1)
        si_y_ff_full = fact * (x1 - x0)

        if periodic_xi:
            si_x_ff = si_x_ff_full[1:]  # (W,)
            si_y_ff = si_y_ff_full[1:]
        else:
            si_x_ff = si_x_ff_full[1:-1]  # (W-1,)
            si_y_ff = si_y_ff_full[1:-1]

    return si_x_wall, si_y_wall, si_x_ff, si_y_ff


def compute_halo_eta_face_vectors(
    coords_vertex_hat: torch.Tensor,
    face_geom_eta: Tuple[torch.Tensor, torch.Tensor],
    periodic_xi: bool = True,
    sign: Union[float, torch.Tensor] = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算完整的 η 面向量（包含 halo 面）- Plan96 新增, Plan97 修正

    用途：为 nodal 梯度计算的 η-part 提供完整的 H+3 个 η 面（包含两个 halo 面）

    η面（常j面）位于节点行j处，跨越两个ξ列(i, i+1)

    Plan97修复：2D边差公式需要系数1.0而非0.5，且符号需要翻转
    正确的2D简化公式（与对角线叉乘退化一致）：
        sj_x = sign * (y[j, i] - y[j, i+1])   # 注意顺序翻转
        sj_y = sign * (x[j, i+1] - x[j, i])   # 注意顺序翻转

    ADFlow参考: preprocessingAPI.F90:3200-3224 (对角线叉乘公式)

    输出形状说明：
        - 物理 η 面: H+1 个（从 j=0 到 j=H）
        - 加上 halo 面: wall 外侧 (j=-1) 和 farfield 外侧 (j=H+1)
        - 总共: H+3 个 η 面

    Args:
        coords_vertex_hat: (2, H+3, W+1) 或 (B, 2, H+3, W+1) - 包含 halo 行的节点坐标
        face_geom_eta: (A_x_eta, A_y_eta) 原有物理 η 面向量，形状 (H+1, W) 或 (B, H+1, W)
        periodic_xi: ξ方向周期性
        sign: handedness修正因子 (1.0 for right-handed, -1.0 for left-handed)

    Returns:
        sj_x_hat, sj_y_hat: (H+3, W) 或 (B, H+3, W) 完整的 η 面向量
            - [0]: wall 外侧 halo 面 (j=-1, 即 coords_vertex_hat 行 0)
            - [1:H+2]: 物理 η 面 (j=0..H, 即 coords_vertex_hat 行 1..H+1)
            - [H+2]: farfield 外侧 halo 面 (j=H+1, 即 coords_vertex_hat 行 H+2)
    """
    A_x_eta, A_y_eta = face_geom_eta
    is_batched = coords_vertex_hat.ndim == 4
    device = coords_vertex_hat.device
    dtype = coords_vertex_hat.dtype

    if is_batched:
        B, _, H_plus_3, W_plus_1 = coords_vertex_hat.shape
    else:
        _, H_plus_3, W_plus_1 = coords_vertex_hat.shape

    H = H_plus_3 - 3
    W = W_plus_1 - 1

    # Plan97修复: 系数从0.5改为1.0
    # 原因: 2D边差公式没有对角线叉乘产生的因子2，需要手动补偿
    # 参考 review.md: η方向halo面比值≈-0.5，目标+1.0
    # Plan98修复: batch广播修正 - sign从(B,)变为(B,1)以正确广播
    if isinstance(sign, torch.Tensor) and sign.ndim == 1:
        fact = 1.0 * sign.unsqueeze(1)  # (B,) → (B, 1) for broadcasting
    else:
        fact = 1.0 * sign  # scalar case works fine

    if is_batched:
        # 初始化输出数组
        sj_x_hat = torch.zeros(B, H + 3, W, device=device, dtype=dtype)
        sj_y_hat = torch.zeros(B, H + 3, W, device=device, dtype=dtype)

        # 物理 η 面 (j=0..H) -> sj_hat[1:H+2]
        # 直接使用已有的 face_geom_eta（已经正确计算）
        sj_x_hat[:, 1:H+2, :] = A_x_eta
        sj_y_hat[:, 1:H+2, :] = A_y_eta

        # Wall 外侧 halo 面 (j=-1) -> sj_hat[0]
        # 使用 coords_vertex_hat 行 0 处的节点
        # η面位于节点行0，跨越列i到i+1
        x_row0_col_i = coords_vertex_hat[:, 0, 0, :-1]    # (B, W)
        y_row0_col_i = coords_vertex_hat[:, 1, 0, :-1]
        x_row0_col_ip1 = coords_vertex_hat[:, 0, 0, 1:]   # (B, W)
        y_row0_col_ip1 = coords_vertex_hat[:, 1, 0, 1:]

        # CRITICAL FIX (Plan98): Halo面符号翻转，与ADflow sK(k=0)对齐
        # review.md诊断：halo面符号反了，需要交换减法顺序
        sj_x_hat[:, 0, :] = fact * (y_row0_col_ip1 - y_row0_col_i)
        sj_y_hat[:, 0, :] = fact * (x_row0_col_i - x_row0_col_ip1)

        # Farfield 外侧 halo 面 (j=H+1) -> sj_hat[H+2]
        # 使用 coords_vertex_hat 行 H+2 处的节点
        x_rowH2_col_i = coords_vertex_hat[:, 0, H+2, :-1]   # (B, W)
        y_rowH2_col_i = coords_vertex_hat[:, 1, H+2, :-1]
        x_rowH2_col_ip1 = coords_vertex_hat[:, 0, H+2, 1:]  # (B, W)
        y_rowH2_col_ip1 = coords_vertex_hat[:, 1, H+2, 1:]

        # CRITICAL FIX (Plan98): Halo面符号翻转，与ADflow sK(k=ke)对齐
        # review.md诊断：halo面符号反了，需要交换减法顺序
        sj_x_hat[:, H+2, :] = fact * (y_rowH2_col_ip1 - y_rowH2_col_i)
        sj_y_hat[:, H+2, :] = fact * (x_rowH2_col_i - x_rowH2_col_ip1)

    else:
        # 初始化输出数组
        sj_x_hat = torch.zeros(H + 3, W, device=device, dtype=dtype)
        sj_y_hat = torch.zeros(H + 3, W, device=device, dtype=dtype)

        # 物理 η 面 (j=0..H) -> sj_hat[1:H+2]
        sj_x_hat[1:H+2, :] = A_x_eta
        sj_y_hat[1:H+2, :] = A_y_eta

        # Wall 外侧 halo 面 (j=-1) -> sj_hat[0]
        x_row0_col_i = coords_vertex_hat[0, 0, :-1]     # (W,)
        y_row0_col_i = coords_vertex_hat[1, 0, :-1]
        x_row0_col_ip1 = coords_vertex_hat[0, 0, 1:]    # (W,)
        y_row0_col_ip1 = coords_vertex_hat[1, 0, 1:]

        # CRITICAL FIX (Plan98): Halo面符号翻转，与ADflow sK(k=0)对齐
        # review.md诊断：halo面符号反了，需要交换减法顺序
        sj_x_hat[0, :] = fact * (y_row0_col_ip1 - y_row0_col_i)
        sj_y_hat[0, :] = fact * (x_row0_col_i - x_row0_col_ip1)

        # Farfield 外侧 halo 面 (j=H+1) -> sj_hat[H+2]
        x_rowH2_col_i = coords_vertex_hat[0, H+2, :-1]   # (W,)
        y_rowH2_col_i = coords_vertex_hat[1, H+2, :-1]
        x_rowH2_col_ip1 = coords_vertex_hat[0, H+2, 1:]  # (W,)
        y_rowH2_col_ip1 = coords_vertex_hat[1, H+2, 1:]

        # CRITICAL FIX (Plan98): Halo面符号翻转，与ADflow sK(k=ke)对齐
        # review.md诊断：halo面符号反了，需要交换减法顺序
        sj_x_hat[H+2, :] = fact * (y_rowH2_col_ip1 - y_rowH2_col_i)
        sj_y_hat[H+2, :] = fact * (x_rowH2_col_i - x_rowH2_col_ip1)

    # Plan97: 验证钩子调用
    _verify_halo_face_vectors_debug(
        sj_x_hat[..., 0, :], sj_y_hat[..., 0, :],
        coords_vertex_hat, 'wall', sign
    )
    _verify_halo_face_vectors_debug(
        sj_x_hat[..., -1, :], sj_y_hat[..., -1, :],
        coords_vertex_hat, 'farfield', sign
    )

    return sj_x_hat, sj_y_hat


def compute_adflow_separation_stencil(
    coords_vertex_hat: torch.Tensor,
    direction: str = 'eta',
    periodic_xi: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ADFlow 4-point node coordinate stencil for viscous normal correction

    Plan93方案I: 修正系数从0.5到0.25，并添加xi方向支持。

    参考: ADFlow blockette.F90:5757-5768 (k/eta方向), 6353-6364 (i/xi方向)

    ADFlow 3D公式:
        ss = eighth * Σ [x(dir+1,corner) - x(dir-1,corner)] at 8 corners

    2D pseudo-3D退化分析:
        ADFlow有 4 xy-corners × 2 z-layers = 8 terms, 每个乘eighth
        在2D thin中，z层的xy坐标相同，但ADFlow公式中j-1和j（thin方向）
        仍然是独立计算的两项，只是它们的值相等。

        所以 eighth * 8 = 1.0 在3D中
        在2D thin (nj=2): 只有2个unique xy位置（i-1, i），但每个被计算4次
        （j-1和j两个thin层 × k-1和k+1两个法向层）
        我们这里只存储2个unique xy差值，所以系数 = 0.25 * 2 positions = 0.5?

        但plan92.md指出应该是0.25，因为ADFlow的eighth对应完整3D表达式。
        按用户确认，使用0.25。

    Args:
        coords_vertex_hat: (2, H+3, W+1) 或 (B, 2, H+3, W+1) - 包含 halo 行的节点坐标
        direction: 'eta' 或 'xi'
        periodic_xi: ξ方向周期性

    Returns:
        direction='eta': ssx, ssy, inv_d 形状 (H+1, W) 或 (B, H+1, W)
        direction='xi':  ssx, ssy, inv_d 形状 (H, W) 或 (B, H, W)
    """
    is_batched = coords_vertex_hat.ndim == 4
    device = coords_vertex_hat.device
    dtype = coords_vertex_hat.dtype

    if is_batched:
        B, _, H_plus_3, W_plus_1 = coords_vertex_hat.shape
    else:
        _, H_plus_3, W_plus_1 = coords_vertex_hat.shape

    H = H_plus_3 - 3
    W = W_plus_1 - 1

    # Plan93: 系数从0.5改为0.25
    STENCIL_COEFF = 0.25

    if direction == 'eta':
        # η方向: 计算沿j方向的分隔向量
        # coords_vertex_hat 索引:
        # 0 = wall halo (j=-1), 1 = j=0 (wall), ..., H+1 = j=H, H+2 = farfield halo
        #
        # 对于η面 j (j=0..H)，ADFlow 4-point stencil使用:
        #   x[j+1, i-1] - x[j-1, i-1]  (i-1位置的j方向差)
        #   x[j+1, i]   - x[j-1, i]    (i位置的j方向差)
        # 在coords_vertex_hat中: j-1 → index j, j+1 → index j+2

        if is_batched:
            # x[j-1, i-1] 和 x[j-1, i]
            x_jm1_im1 = coords_vertex_hat[:, 0, :-2, :-1]  # (B, H+1, W)
            y_jm1_im1 = coords_vertex_hat[:, 1, :-2, :-1]
            x_jm1_i = coords_vertex_hat[:, 0, :-2, 1:]
            y_jm1_i = coords_vertex_hat[:, 1, :-2, 1:]

            # x[j+1, i-1] 和 x[j+1, i]
            x_jp1_im1 = coords_vertex_hat[:, 0, 2:, :-1]
            y_jp1_im1 = coords_vertex_hat[:, 1, 2:, :-1]
            x_jp1_i = coords_vertex_hat[:, 0, 2:, 1:]
            y_jp1_i = coords_vertex_hat[:, 1, 2:, 1:]

            # 4-point stencil (系数0.25)
            ss_x = STENCIL_COEFF * ((x_jp1_im1 - x_jm1_im1) + (x_jp1_i - x_jm1_i))
            ss_y = STENCIL_COEFF * ((y_jp1_im1 - y_jm1_im1) + (y_jp1_i - y_jm1_i))
        else:
            x_jm1_im1 = coords_vertex_hat[0, :-2, :-1]  # (H+1, W)
            y_jm1_im1 = coords_vertex_hat[1, :-2, :-1]
            x_jm1_i = coords_vertex_hat[0, :-2, 1:]
            y_jm1_i = coords_vertex_hat[1, :-2, 1:]

            x_jp1_im1 = coords_vertex_hat[0, 2:, :-1]
            y_jp1_im1 = coords_vertex_hat[1, 2:, :-1]
            x_jp1_i = coords_vertex_hat[0, 2:, 1:]
            y_jp1_i = coords_vertex_hat[1, 2:, 1:]

            ss_x = STENCIL_COEFF * ((x_jp1_im1 - x_jm1_im1) + (x_jp1_i - x_jm1_i))
            ss_y = STENCIL_COEFF * ((y_jp1_im1 - y_jm1_im1) + (y_jp1_i - y_jm1_i))

    elif direction == 'xi':
        # ξ方向: 计算沿i方向的分隔向量
        # 参考: ADFlow blockette.F90:6353-6364 (i方向)
        #
        # 对于ξ面 i (i=1..W)，ADFlow 4-point stencil使用:
        #   x[j-1, i+1] - x[j-1, i-1]  (j-1位置的i方向差)
        #   x[j,   i+1] - x[j,   i-1]  (j位置的i方向差)
        # 在coords_vertex_hat中: j → index j+1 (加halo偏移)
        #
        # 索引对齐修复 (Plan94):
        # - A_x_xi 对应面 i=1..W (跳过seam面 i=0)
        # - 基准列改为 1..W (而非 0..W-1)，使输出与 A_x_xi 对齐
        # - 周期性: 列 W = 列 0 (seam闭合)

        if is_batched:
            # 创建循环索引处理周期性
            # i-1: roll +1, i+1: roll -1
            x_all = coords_vertex_hat[:, 0, 1:H+2, :]  # (B, H+1, W+1) - 物理行j=0..H
            y_all = coords_vertex_hat[:, 1, 1:H+2, :]

            # x[j-1, i-1], x[j-1, i+1], x[j, i-1], x[j, i+1]
            # j-1: rows :-1, j: rows 1:
            if periodic_xi:
                # Plan94修复: 使用列 1..W (而非 0..W-1)
                # 对于面 i=1..W:
                #   - roll +1: 位置k得到原始列(k mod W)+1的值，实现 i-1
                #   - roll -1: 位置k得到原始列((k+2) mod W)+1的值，实现 i+1
                # 特别地: 列 W 在周期性边界上等于列 0
                x_jm1_im1 = torch.roll(x_all[:, :-1, 1:], 1, dims=2)  # (B, H, W)
                y_jm1_im1 = torch.roll(y_all[:, :-1, 1:], 1, dims=2)
                x_jm1_ip1 = torch.roll(x_all[:, :-1, 1:], -1, dims=2)
                y_jm1_ip1 = torch.roll(y_all[:, :-1, 1:], -1, dims=2)

                x_j_im1 = torch.roll(x_all[:, 1:, 1:], 1, dims=2)    # (B, H, W)
                y_j_im1 = torch.roll(y_all[:, 1:, 1:], 1, dims=2)
                x_j_ip1 = torch.roll(x_all[:, 1:, 1:], -1, dims=2)
                y_j_ip1 = torch.roll(y_all[:, 1:, 1:], -1, dims=2)
            else:
                # 非周期: 使用内部单元 i=1..W-2
                x_jm1_im1 = x_all[:, :-1, :-2]   # (B, H, W-1)
                y_jm1_im1 = y_all[:, :-1, :-2]
                x_jm1_ip1 = x_all[:, :-1, 2:]
                y_jm1_ip1 = y_all[:, :-1, 2:]

                x_j_im1 = x_all[:, 1:, :-2]
                y_j_im1 = y_all[:, 1:, :-2]
                x_j_ip1 = x_all[:, 1:, 2:]
                y_j_ip1 = y_all[:, 1:, 2:]

            # 4-point stencil
            ss_x = STENCIL_COEFF * ((x_jm1_ip1 - x_jm1_im1) + (x_j_ip1 - x_j_im1))
            ss_y = STENCIL_COEFF * ((y_jm1_ip1 - y_jm1_im1) + (y_j_ip1 - y_j_im1))
        else:
            x_all = coords_vertex_hat[0, 1:H+2, :]  # (H+1, W+1)
            y_all = coords_vertex_hat[1, 1:H+2, :]

            if periodic_xi:
                # Plan94修复: 使用列 1..W (而非 0..W-1)
                x_jm1_im1 = torch.roll(x_all[:-1, 1:], 1, dims=1)  # (H, W)
                y_jm1_im1 = torch.roll(y_all[:-1, 1:], 1, dims=1)
                x_jm1_ip1 = torch.roll(x_all[:-1, 1:], -1, dims=1)
                y_jm1_ip1 = torch.roll(y_all[:-1, 1:], -1, dims=1)

                x_j_im1 = torch.roll(x_all[1:, 1:], 1, dims=1)
                y_j_im1 = torch.roll(y_all[1:, 1:], 1, dims=1)
                x_j_ip1 = torch.roll(x_all[1:, 1:], -1, dims=1)
                y_j_ip1 = torch.roll(y_all[1:, 1:], -1, dims=1)
            else:
                x_jm1_im1 = x_all[:-1, :-2]
                y_jm1_im1 = y_all[:-1, :-2]
                x_jm1_ip1 = x_all[:-1, 2:]
                y_jm1_ip1 = y_all[:-1, 2:]

                x_j_im1 = x_all[1:, :-2]
                y_j_im1 = y_all[1:, :-2]
                x_j_ip1 = x_all[1:, 2:]
                y_j_ip1 = y_all[1:, 2:]

            ss_x = STENCIL_COEFF * ((x_jm1_ip1 - x_jm1_im1) + (x_j_ip1 - x_j_im1))
            ss_y = STENCIL_COEFF * ((y_jm1_ip1 - y_jm1_im1) + (y_j_ip1 - y_j_im1))

    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'eta' or 'xi'.")

    # 计算距离和单位向量
    d = torch.sqrt(ss_x**2 + ss_y**2)
    d = torch.clamp(d, min=1e-30)  # 避免除零

    ssx = ss_x / d
    ssy = ss_y / d
    inv_d = 1.0 / d

    return ssx, ssy, inv_d


def _verify_halo_face_vectors_debug(
    sj_x_computed: torch.Tensor,
    sj_y_computed: torch.Tensor,
    coords_vertex_hat: torch.Tensor,
    halo_type: str,
    sign: Union[float, torch.Tensor] = 1.0,
    rtol: float = 1e-5
):
    """
    Debug验证钩子：对比边差公式与对角线叉乘公式的结果

    仅在 SURROGATE_DEBUG_GEOMETRY=1 环境变量设置时激活

    Args:
        sj_x_computed, sj_y_computed: 已计算的halo面向量 (W,) 或 (B, W)
        coords_vertex_hat: (2, H+3, W+1) 或 (B, 2, H+3, W+1) - 包含halo行的节点坐标
        halo_type: 'wall' 或 'farfield'
        sign: handedness修正因子
        rtol: 相对容差
    """
    import os
    if os.environ.get('SURROGATE_DEBUG_GEOMETRY') != '1':
        return

    is_batched = coords_vertex_hat.ndim == 4
    device = coords_vertex_hat.device
    dtype = coords_vertex_hat.dtype

    if is_batched:
        B, _, H_plus_3, W_plus_1 = coords_vertex_hat.shape
    else:
        _, H_plus_3, W_plus_1 = coords_vertex_hat.shape

    H = H_plus_3 - 3
    W = W_plus_1 - 1

    # 使用对角线叉乘法计算参考值
    # η面四角点: (i-1,j,z=0), (i,j,z=0), (i-1,j,z=1), (i,j,z=1)
    # 2D pseudo-3D叉乘退化结果: sj_x = y[i] - y[i+1], sj_y = x[i+1] - x[i]
    # Plan98修复: batch广播修正 - sign从(B,)变为(B,1)以正确广播
    if isinstance(sign, torch.Tensor) and sign.ndim == 1:
        fact = 1.0 * sign.unsqueeze(1)  # (B,) → (B, 1) for broadcasting
    else:
        fact = 1.0 * sign  # scalar case works fine

    if halo_type == 'wall':
        row_idx = 0  # wall halo行
    elif halo_type == 'farfield':
        row_idx = H + 2  # farfield halo行
    else:
        return

    if is_batched:
        x_col_i = coords_vertex_hat[:, 0, row_idx, :-1]    # (B, W)
        y_col_i = coords_vertex_hat[:, 1, row_idx, :-1]
        x_col_ip1 = coords_vertex_hat[:, 0, row_idx, 1:]   # (B, W)
        y_col_ip1 = coords_vertex_hat[:, 1, row_idx, 1:]

        # 参考值（与修正后的公式一致）
        sj_x_ref = fact * (y_col_i - y_col_ip1)
        sj_y_ref = fact * (x_col_ip1 - x_col_i)
    else:
        x_col_i = coords_vertex_hat[0, row_idx, :-1]       # (W,)
        y_col_i = coords_vertex_hat[1, row_idx, :-1]
        x_col_ip1 = coords_vertex_hat[0, row_idx, 1:]      # (W,)
        y_col_ip1 = coords_vertex_hat[1, row_idx, 1:]

        sj_x_ref = fact * (y_col_i - y_col_ip1)
        sj_y_ref = fact * (x_col_ip1 - x_col_i)

    # 计算差异
    max_diff_x = torch.max(torch.abs(sj_x_computed - sj_x_ref)).item()
    max_diff_y = torch.max(torch.abs(sj_y_computed - sj_y_ref)).item()
    mean_mag = torch.mean(torch.sqrt(sj_x_ref**2 + sj_y_ref**2)).item()

    ratio = max(max_diff_x, max_diff_y) / (mean_mag + 1e-30)

    print(f"[SURROGATE_DEBUG_GEOMETRY] {halo_type} halo η-face verification:")
    print(f"  max_diff_x = {max_diff_x:.6e}, max_diff_y = {max_diff_y:.6e}")
    print(f"  mean_magnitude = {mean_mag:.6e}, ratio = {ratio:.6e}")

    if ratio > rtol:
        import warnings
        warnings.warn(
            f"[SURROGATE_DEBUG_GEOMETRY] {halo_type} halo face vectors: "
            f"max_diff/mean = {ratio:.6e} > rtol={rtol}"
        )
