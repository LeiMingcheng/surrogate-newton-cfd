"""
独立梯度计算模块（ADFLOW完全对齐）

提供统一的梯度计算接口，支持多种梯度重构方法：
- Green-Gauss方法（基于散度定理）
- ADFLOW六向量方法（cell-center中心差分，用于耗散/湍流）
- ADFLOW节点梯度方法（nodal体积法，用于粘性通量 - 强制使用）

设计原则：
- 解耦：梯度计算与残差计算完全分离
- 统一接口：多种方法通过抽象基类规范
- 灵活边界：边界条件外部化，支持halo值传入
- 可扩展：易于添加新的梯度方法
- 双梯度支持：必须同时使用cell-center和nodal梯度

应用场景（强制分离）：
- Cell-center梯度（Green-Gauss或ADFLOW六向量）：用于耗散计算、湍流模型
- Nodal梯度（ADFLOW体积法）：用于粘性通量计算（必须使用）

Plan86对齐：
- 粘性通量计算必须使用节点梯度四节点平均
- 不再支持cell-center梯度两侧单元平均的fallback
- 确保与ADFLOW blockette.F90:5350-5634完全对齐
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import torch


class GradientCalculatorBase(ABC):
    """
    梯度计算器抽象基类

    统一接口规范：
    - 输入: 标量场 + 几何信息 + 边界条件
    - 输出: 梯度场 (dphi_dx, dphi_dy)
    - 边界处理: 外部传入（halo值或边界配置）
    """

    def __init__(
        self,
        periodic_xi: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Args:
            periodic_xi: ξ方向是否周期性（O-grid为True）
            device: 计算设备
            dtype: 数据类型
        """
        self.periodic_xi = periodic_xi
        self.device = device or torch.device('cpu')
        self.dtype = dtype or torch.float32

    @abstractmethod
    def compute_gradient(
        self,
        phi: torch.Tensor,
        geometry: Dict[str, torch.Tensor],
        boundary_conditions: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算标量场的物理空间梯度

        Args:
            phi: 标量场 (batch, H, W) 或 (H, W)
            geometry: 几何信息字典
                必需: 'volumes' - 单元体积 (H, W) 或 (batch, H, W)
                可选: 'face_geom' - 面几何信息（具体结构由子类定义）
                可选: 'coords_center' - 单元中心坐标 (2, H, W)
                可选: 'coords_vertex' - 节点坐标 (2, H+1, W+1)
            boundary_conditions: 边界条件字典（可选）
                'halo_eta_bottom': η方向底部halo值 (batch, W) 或 (W,)
                'halo_eta_top': η方向顶部halo值 (batch, W) 或 (W,)
                'periodic_xi': 覆盖构造函数的周期性设置

        Returns:
            dphi_dx: x方向梯度 (batch, H, W) 或 (H, W)
            dphi_dy: y方向梯度 (batch, H, W) 或 (H, W)
        """
        pass

    @abstractmethod
    def get_required_geometry(self) -> Dict[str, str]:
        """
        返回此梯度方法需要的几何信息

        Returns:
            字典: {key: description}
        """
        pass

    @property
    @abstractmethod
    def method_name(self) -> str:
        """梯度方法名称"""
        pass

    def validate_geometry(self, geometry: Dict[str, torch.Tensor]) -> None:
        """
        验证几何信息完整性

        Args:
            geometry: 几何信息字典

        Raises:
            ValueError: 如果缺少必需的几何信息
        """
        required = self.get_required_geometry()
        for key in required.keys():
            if key not in geometry or geometry[key] is None:
                raise ValueError(
                    f"{self.method_name} requires geometry['{key}'] "
                    f"({required[key]})"
                )


# ⚠️ DEPRECATED: GreenGaussCalculator在当前配置下不可达
# 原因：gradient_method强制为'adflow_6pt'，Green-Gauss分支永不执行
# 状态：保留实现以备将来使用，但当前不维护
# 如需使用：需放松 residual backend 中的强制校验
class GreenGaussCalculator(GradientCalculatorBase):
    """
    Green-Gauss梯度重构

    原理: 基于散度定理的面通量体积平均
        ∇φ = (1/V) ∮_faces φ n⃗ dS ≈ (1/V) Σ_faces φ_face S⃗_face

    优势:
    - 无插值误差（面值直接用相邻单元平均）
    - 自动满足散度定理的守恒性
    - 在强非正交网格上稳定

    边界处理:
    - ξ方向: 周期边界自动处理
    - η方向: 仅使用内部面，自动排除边界
    """

    def __init__(self, periodic_xi: bool = True, **kwargs):
        super().__init__(periodic_xi=periodic_xi, **kwargs)

    @property
    def method_name(self) -> str:
        return "Green-Gauss"

    def get_required_geometry(self) -> Dict[str, str]:
        return {
            'volumes': '单元体积',
            'face_geom': '面几何（面积向量）'
        }

    def compute_gradient(
        self,
        phi: torch.Tensor,
        geometry: Dict[str, torch.Tensor],
        boundary_conditions: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Green-Gauss梯度计算

        实现细节:
        1. ξ方向: 使用周期边界处理（如果periodic_xi=True）
        2. η方向: 仅使用内部面（排除j=0和j=H边界）
        3. 面值: φ_face = 0.5 * (φ_L + φ_R)
        4. 梯度: ∇φ = (1/V) * Σ φ_face * S⃗_face
        """
        # 验证几何信息
        self.validate_geometry(geometry)

        # 提取几何信息
        volumes = geometry['volumes']
        face_geom = geometry['face_geom']

        # 导入底层实现
        from .geometry import compute_physical_derivatives_green_gauss

        # 调用底层实现
        dphi_dx, dphi_dy = compute_physical_derivatives_green_gauss(
            phi=phi,
            face_geom=face_geom,
            volumes=volumes,
            periodic_xi=self.periodic_xi,
            compute_volumes=False
        )

        return dphi_dx, dphi_dy


class NodalGradientCalculator(GradientCalculatorBase):
    """
    ADFLOW节点梯度计算器（体积法）

    原理: 节点控制体的散度定理（与ADFLOW allNodalGradients完全对齐）
        ∇φ_node = (1/V_node) * Σ_faces [φ_face * S⃗_face]

    用途:
    - 专用于粘性通量计算（与ADFLOW viscousFlux一致）
    - 不用于耗散计算（耗散使用cell-center六向量梯度）

    优势:
    - 与ADFLOW粘性通量计算精确对齐
    - 四节点平均插值到面（减少近壁误差）

    参考: ADFLOW blockette.F90:5350-5634
    """

    def __init__(self, periodic_xi: bool = True, **kwargs):
        super().__init__(periodic_xi=periodic_xi, **kwargs)

    @property
    def method_name(self) -> str:
        return "ADFLOW-Nodal"

    def get_required_geometry(self) -> Dict[str, str]:
        return {
            'volumes': '单元体积 (H, W)',
            'si_x': 'ξ方向面积向量x分量 (H, W)',
            'si_y': 'ξ方向面积向量y分量 (H, W)',
            'sj_x': 'η方向面积向量x分量 (H+1, W)',
            'sj_y': 'η方向面积向量y分量 (H+1, W)'
        }

    def compute_gradient(
        self,
        phi: torch.Tensor,
        geometry: Dict[str, torch.Tensor],
        boundary_conditions: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算节点梯度（ADFLOW体积法）

        Args:
            phi: 标量场 (batch, H, W) 或 (H, W)
            geometry: 几何信息
                - volumes: 单元体积
                - si_x, si_y: ξ方向面积向量
                - sj_x, sj_y: η方向面积向量
            boundary_conditions: 边界条件（必须提供！）
                - halo_eta_bottom: 壁面halo值 (batch, W) 或 (W,)
                - halo_eta_top: 远场halo值 (batch, W) 或 (W,)
                Plan91 新增（可选，用于精确ADFlow对齐）：
                - halo_vol_wall: 壁面halo单元体积 (W,)
                - halo_vol_ff: 远场halo单元体积 (W,)
                - si_x_halo_wall, si_y_halo_wall: 壁面halo层ξ面向量 (W,)
                - si_x_halo_ff, si_y_halo_ff: 远场halo层ξ面向量 (W,)

        Returns:
            dphi_dx_node: 节点x梯度 (batch, H+1, W+1) 或 (H+1, W+1)
            dphi_dy_node: 节点y梯度 (batch, H+1, W+1) 或 (H+1, W+1)

        Raises:
            ValueError: 如果boundary_conditions未提供或缺少halo值
        """
        # 验证几何信息
        self.validate_geometry(geometry)

        # 验证边界条件必须存在
        if boundary_conditions is None:
            raise ValueError(
                "NodalGradientCalculator requires boundary_conditions with halo_eta_bottom and halo_eta_top. "
                "These are needed for accurate near-wall gradient computation aligned with ADflow."
            )

        if 'halo_eta_bottom' not in boundary_conditions or 'halo_eta_top' not in boundary_conditions:
            raise ValueError(
                "boundary_conditions must contain 'halo_eta_bottom' and 'halo_eta_top' keys. "
                f"Got keys: {list(boundary_conditions.keys())}"
            )

        # 提取几何信息
        volumes = geometry['volumes']
        si_x = geometry['si_x']
        si_y = geometry['si_y']
        sj_x = geometry['sj_x']
        sj_y = geometry['sj_y']

        # 提取halo边界值
        halo_eta_bottom = boundary_conditions['halo_eta_bottom']
        halo_eta_top = boundary_conditions['halo_eta_top']

        # Plan91: 提取halo几何参数（用于精确ADFlow对齐）
        halo_vol_wall = boundary_conditions.get('halo_vol_wall', None)
        halo_vol_ff = boundary_conditions.get('halo_vol_ff', None)
        si_x_halo_wall = boundary_conditions.get('si_x_halo_wall', None)
        si_y_halo_wall = boundary_conditions.get('si_y_halo_wall', None)
        si_x_halo_ff = boundary_conditions.get('si_x_halo_ff', None)
        si_y_halo_ff = boundary_conditions.get('si_y_halo_ff', None)

        # Plan96: 提取完整η面向量（含halo面）
        sj_x_hat = boundary_conditions.get('sj_x_hat', None)
        sj_y_hat = boundary_conditions.get('sj_y_hat', None)

        # Plan96验证: sj_x_hat/sj_y_hat必须提供
        if sj_x_hat is None or sj_y_hat is None:
            raise ValueError(
                "NodalGradientCalculator requires sj_x_hat and sj_y_hat in boundary_conditions (Plan96). "
                "These must be computed using compute_halo_eta_face_vectors."
            )

        # 导入底层实现
        from .nodal_gradients import compute_nodal_gradients_volume_method

        # 调用底层实现（传递halo值和halo几何）
        dphi_dx_node, dphi_dy_node = compute_nodal_gradients_volume_method(
            phi=phi,
            si_x=si_x,
            si_y=si_y,
            sj_x=sj_x,
            sj_y=sj_y,
            volumes=volumes,
            halo_eta_bottom=halo_eta_bottom,
            halo_eta_top=halo_eta_top,
            periodic_xi=self.periodic_xi,
            # Plan91: 传递halo几何参数
            halo_vol_wall=halo_vol_wall,
            halo_vol_ff=halo_vol_ff,
            si_x_halo_wall=si_x_halo_wall,
            si_y_halo_wall=si_y_halo_wall,
            si_x_halo_ff=si_x_halo_ff,
            si_y_halo_ff=si_y_halo_ff,
            # Plan96: 传递完整η面向量
            sj_x_hat=sj_x_hat,
            sj_y_hat=sj_y_hat
        )

        return dphi_dx_node, dphi_dy_node


class ADFlowSixPointCalculator(GradientCalculatorBase):
    """
    ADFLOW六向量中心差分梯度（cell-center）

    原理: 六点中心差分（与ADFLOW blockette.F90完全对齐）
        u_x = u[i+1]*si_x[i] - u[i-1]*si_x[i-1]
            + u[j+1]*sj_x[j] - u[j-1]*sj_x[j-1]
        梯度 = u_x / (4*vol)

    用途:
    - 湍流模型计算（应变/涡量）
    - 耗散项计算
    - 不用于粘性通量（粘性通量使用nodal梯度）

    优势:
    - 与ADFLOW精确对齐
    - 高精度（二阶中心差分）

    边界处理:
    - 需要halo单元值（必须外部提供）
    - η方向边界: 使用halo_eta_bottom和halo_eta_top

    参考: ADFLOW blockette.F90:1070-1124
    """

    def __init__(self, periodic_xi: bool = True, **kwargs):
        super().__init__(periodic_xi=periodic_xi, **kwargs)

    @property
    def method_name(self) -> str:
        return "ADFLOW-6Point"

    def get_required_geometry(self) -> Dict[str, str]:
        return {
            'volumes': '单元体积',
            'si_x': 'ξ方向面面积向量x分量 (H, W)',
            'si_y': 'ξ方向面面积向量y分量 (H, W)',
            'sj_x': 'η方向面面积向量x分量 (H+1, W)',
            'sj_y': 'η方向面面积向量y分量 (H+1, W)'
        }

    def compute_gradient(
        self,
        phi: torch.Tensor,
        geometry: Dict[str, torch.Tensor],
        boundary_conditions: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ADFLOW六向量梯度计算

        重要: η方向需要halo单元值
        - 如果提供boundary_conditions['halo_eta_bottom/top']则使用
        - 否则使用简化边界处理（最近邻外推）
        """
        # 验证几何信息
        self.validate_geometry(geometry)

        # 提取几何信息
        volumes = geometry['volumes']
        si_x = geometry['si_x']
        si_y = geometry['si_y']
        sj_x = geometry['sj_x']
        sj_y = geometry['sj_y']

        # 处理batch维度
        squeeze_output = False
        if phi.ndim == 2:
            phi = phi.unsqueeze(0)
            squeeze_output = True

        batch, H, W = phi.shape
        device = phi.device
        dtype = phi.dtype

        # 确保几何有batch维度
        if si_x.ndim == 2:
            si_x = si_x.unsqueeze(0).expand(batch, -1, -1)
            si_y = si_y.unsqueeze(0).expand(batch, -1, -1)
        if sj_x.ndim == 2:
            sj_x = sj_x.unsqueeze(0).expand(batch, -1, -1)
            sj_y = sj_y.unsqueeze(0).expand(batch, -1, -1)
        if volumes.ndim == 2:
            volumes = volumes.unsqueeze(0).expand(batch, -1, -1)

        # ========== ξ方向贡献 ==========
        if self.periodic_xi:
            phi_right = torch.cat([phi[..., 1:], phi[..., :1]], dim=-1)
            phi_left = torch.cat([phi[..., -1:], phi[..., :-1]], dim=-1)
            si_x_left = torch.roll(si_x, shifts=1, dims=-1)
            si_y_left = torch.roll(si_y, shifts=1, dims=-1)

            dphi_dx_xi = phi_right * si_x - phi_left * si_x_left
            dphi_dy_xi = phi_right * si_y - phi_left * si_y_left
        else:
            # Non-periodic xi uses only internal xi faces. This is sufficient
            # for local-window + halo evaluations because the discarded halo
            # cells absorb the missing outer-boundary contribution.
            dphi_dx_xi = torch.zeros_like(phi)
            dphi_dy_xi = torch.zeros_like(phi)
            dphi_dx_xi[:, :, :-1] += phi[:, :, 1:] * si_x
            dphi_dy_xi[:, :, :-1] += phi[:, :, 1:] * si_y
            dphi_dx_xi[:, :, 1:] -= phi[:, :, :-1] * si_x
            dphi_dy_xi[:, :, 1:] -= phi[:, :, :-1] * si_y

        # ========== η方向贡献（需要halo） ==========
        # 构造padded场: (batch, H+2, W)
        if boundary_conditions and 'halo_eta_bottom' in boundary_conditions:
            halo_bottom = boundary_conditions['halo_eta_bottom']
            halo_top = boundary_conditions['halo_eta_top']

            if halo_bottom.ndim == 1:
                halo_bottom = halo_bottom.unsqueeze(0).expand(batch, -1)
                halo_top = halo_top.unsqueeze(0).expand(batch, -1)

            phi_padded = torch.cat([
                halo_bottom.unsqueeze(1),  # j=-1
                phi,                        # j=0..H-1
                halo_top.unsqueeze(1)       # j=H
            ], dim=1)
        else:
            # 简化边界处理（最近邻）
            phi_padded = torch.cat([
                phi[:, :1, :],
                phi,
                phi[:, -1:, :]
            ], dim=1)

        # 提取phi[j+1]和phi[j-1]
        phi_jp1 = phi_padded[:, 2:, :]
        phi_jm1 = phi_padded[:, :-2, :]

        # sj索引对齐
        sj_x_top = sj_x[:, 1:, :]
        sj_y_top = sj_y[:, 1:, :]
        sj_x_bottom = sj_x[:, :-1, :]
        sj_y_bottom = sj_y[:, :-1, :]

        # η方向梯度贡献
        dphi_dx_eta = phi_jp1 * sj_x_top - phi_jm1 * sj_x_bottom
        dphi_dy_eta = phi_jp1 * sj_y_top - phi_jm1 * sj_y_bottom

        # ========== 合并贡献并应用缩放 ==========
        dphi_dx_raw = dphi_dx_xi + dphi_dx_eta
        dphi_dy_raw = dphi_dy_xi + dphi_dy_eta

        # 真实梯度（除以4*vol，与ADFLOW一致）
        dphi_dx = dphi_dx_raw / (4.0 * volumes)
        dphi_dy = dphi_dy_raw / (4.0 * volumes)

        # 移除batch维度
        if squeeze_output:
            dphi_dx = dphi_dx.squeeze(0)
            dphi_dy = dphi_dy.squeeze(0)

        return dphi_dx, dphi_dy


def create_gradient_calculator(
    method: str = 'green_gauss',
    periodic_xi: bool = True,
    **kwargs
) -> GradientCalculatorBase:
    """
    工厂函数：创建梯度计算器

    Args:
        method: 梯度计算方法
            - 'green_gauss': Green-Gauss方法（默认）
            - 'adflow_6pt': ADFLOW六向量方法（cell-center，用于耗散/湍流）
            - 'nodal': ADFLOW节点梯度（用于粘性通量）
            - 'least_squares': 最小二乘法（未来扩展）
        periodic_xi: ξ方向是否周期性
        **kwargs: 传递给计算器构造函数的额外参数

    Returns:
        梯度计算器实例

    Raises:
        ValueError: 如果方法名未知

    Examples:
        >>> # Green-Gauss方法
        >>> grad_calc = create_gradient_calculator('green_gauss')
        >>> du_dx, du_dy = grad_calc.compute_gradient(u, geometry)

        >>> # ADFLOW六向量方法（cell-center，用于耗散/湍流）
        >>> grad_calc_cell = create_gradient_calculator('adflow_6pt')
        >>> boundary_conditions = {
        ...     'halo_eta_bottom': halo_wall[1, :],
        ...     'halo_eta_top': halo_farfield[1, :]
        ... }
        >>> du_dx_cell, du_dy_cell = grad_calc_cell.compute_gradient(
        ...     u, geometry, boundary_conditions
        ... )

        >>> # ADFLOW节点梯度方法（用于粘性通量）
        >>> grad_calc_nodal = create_gradient_calculator('nodal')
        >>> du_dx_node, du_dy_node = grad_calc_nodal.compute_gradient(u, geometry)
        >>> # 返回节点梯度 (batch, H+1, W+1)
    """
    method = method.lower().replace('-', '_')

    if method in ['green_gauss', 'gg']:
        return GreenGaussCalculator(periodic_xi=periodic_xi, **kwargs)
    elif method in ['adflow_6pt', 'adflow', 'six_point']:
        return ADFlowSixPointCalculator(periodic_xi=periodic_xi, **kwargs)
    elif method in ['nodal', 'adflow_nodal', 'node']:
        return NodalGradientCalculator(periodic_xi=periodic_xi, **kwargs)
    elif method in ['least_squares', 'ls']:
        raise NotImplementedError(f"Method '{method}' not implemented yet")
    else:
        raise ValueError(
            f"Unknown gradient method: '{method}'. "
            f"Available: ['green_gauss', 'adflow_6pt', 'nodal']"
        )


def extract_halo_for_gradient(
    halo_wall: torch.Tensor,
    halo_farfield: torch.Tensor,
    variable_index: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从完整halo数组中提取单个变量的halo值

    用于从多变量halo场 [rho, u, v, p] 中提取单个变量的边界值，
    传递给梯度计算器的 boundary_conditions 参数。

    Args:
        halo_wall: 壁面halo单元 (C, W) or (batch, C, W)
        halo_farfield: 远场halo单元 (C, W) or (batch, C, W)
        variable_index: 变量索引
            - 0: rho (密度)
            - 1: u (x方向速度)
            - 2: v (y方向速度)
            - 3: p (压力)

    Returns:
        (halo_bottom, halo_top): 标量场的边界值
            - halo_bottom: 壁面halo值 (batch, W) or (W,)
            - halo_top: 远场halo值 (batch, W) or (W,)

    Examples:
        >>> # 提取u速度的halo
        >>> halo_u_bottom, halo_u_top = extract_halo_for_gradient(
        ...     halo_wall, halo_farfield, variable_index=1
        ... )
        >>> boundary_conditions = {
        ...     'halo_eta_bottom': halo_u_bottom,
        ...     'halo_eta_top': halo_u_top
        ... }
    """
    if halo_wall.ndim == 2:
        # (C, W) 格式
        halo_bottom = halo_wall[variable_index, :]
        halo_top = halo_farfield[variable_index, :]
    else:
        # (batch, C, W) 格式
        halo_bottom = halo_wall[:, variable_index, :]
        halo_top = halo_farfield[:, variable_index, :]

    return halo_bottom, halo_top
