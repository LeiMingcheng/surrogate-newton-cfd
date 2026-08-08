"""
Force Coefficients Calculator
从cell-center流场数据计算力系数（CL, CD, Cm）

基于ADflow方法：压力积分 + 可选的粘性应力积分
适用于结构化网格的壁面线积分（默认假设 j=0 为壁面，i 方向沿壁面周向）。

兼容性增强：
- 自动处理壁面点顺序（顺/逆时针）以保证外法向方向一致
- 支持包含 wake cut 的 C-grid：当壁面线包含远场 wake cut 段（通常 y≈0 且 x≫尾缘）
  时，会自动剔除该段，仅对翼型表面段积分（否则力矩等会出现偏差）

无量纲化规范（与ADflow一致）：
- p' = p / p_inf  (无量纲压力)
- 参考面积 S_ref = chord * 1 (2D翼型单位展长)
- 参考弦长 c_ref = chord = 1.0

NOTE: 这是Surrogate-Newton CFD surrogate框架中所有力系数计算的统一模块
      所有CD/CL/Cm计算应使用此模块，不要使用经验公式
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Dict, Tuple, Optional, Union

if TYPE_CHECKING:
    import torch


def _is_tensor_like(value) -> bool:
    return hasattr(value, "detach") and hasattr(value, "cpu")


class ForceCoefficientsCalculator:
    """
    力系数计算器

    从cell-center流场数据计算气动力系数。
    支持纯压力积分（CDp）和完整阻力估计（CDp + CDv）。
    """

    def __init__(
        self,
        gamma: float = 1.4,
        chord_ref: float = 1.0,
        area_ref: Optional[float] = None,
        moment_center: Tuple[float, float] = (0.0, 0.0),
        device: str = 'cpu'
    ):
        """
        Args:
            gamma: 比热比
            chord_ref: 参考弦长
            area_ref: 参考面积；二维单位展向默认等于chord_ref
            moment_center: 与ADFLOW xRef/yRef一致的绝对坐标
            device: 计算设备
        """
        self.gamma = gamma
        self.chord_ref = chord_ref
        self.area_ref = chord_ref if area_ref is None else area_ref
        self.moment_center = moment_center
        self.device = device
        self._surface_force_geometry_cache_key = None
        self._surface_force_geometry_cache = None

    def _coords_cache_key(self, coords_vertex: np.ndarray) -> Tuple:
        array = np.asarray(coords_vertex)
        return (
            int(array.__array_interface__['data'][0]),
            array.shape,
            array.strides,
            array.dtype.str,
        )

    def _get_surface_force_geometry(self, coords_vertex: np.ndarray) -> Dict[str, 'torch.Tensor']:
        import torch
        from surrogate.physics.pde.surface_forces import prepare_surface_force_geometry

        cache_key = self._coords_cache_key(coords_vertex)
        if self._surface_force_geometry_cache_key == cache_key and self._surface_force_geometry_cache is not None:
            return self._surface_force_geometry_cache

        prepared = prepare_surface_force_geometry(
            coords_vertex,
            periodic_xi=True,
            device=self.device,
            dtype=torch.float64,
        )
        self._surface_force_geometry_cache_key = cache_key
        self._surface_force_geometry_cache = prepared
        return prepared

    def _compute_temperature(self, rho: np.ndarray, p: np.ndarray) -> np.ndarray:
        """从无量纲密度和压力计算无量纲温度

        理想气体: p = ρ R T。当前模块统一采用：
        - p' = p / p_inf，ρ' = ρ / ρ_inf，T' = T / T_inf
        - 因此 T' ≈ p'/ρ'（不额外乘以γ）。

        注意：benchmark/ref/coefficient.py 采用不同的压力标度（p/(ρ∞ a∞²)），
        该处推导中会出现γ系数；本实现与之不同，请勿混用。

        Args:
            rho: 无量纲密度 ρ'
            p: 无量纲压力 p'

        Returns:
            无量纲温度 T' = T/T_inf
        """
        return p / (rho + 1e-12)

    def _compute_sutherland_viscosity(self, T: np.ndarray, T_inf: float = 300.0) -> np.ndarray:
        """Sutherland公式计算无量纲粘度

        μ/μ_ref = T^1.5 × (1 + S/T_ref) / (T + S/T_ref)
        其中 S = 198.6 K (Sutherland常数，空气)

        Args:
            T: 无量纲温度 T' = T/T_inf
            T_inf: 参考温度 (K)

        Returns:
            无量纲粘度 μ' = μ/μ_inf
        """
        S = 198.6 / T_inf
        return T ** 1.5 * (1 + S) / (T + S)

    def compute_coefficients(
        self,
        fields: Union[np.ndarray, torch.Tensor],
        coords_vertex: Union[np.ndarray, torch.Tensor],
        flow_conditions: Union[np.ndarray, torch.Tensor, Dict],
        return_components: bool = False,
        compute_viscous: bool = True,
        T_inf: float = 300.0
    ) -> Dict[str, float]:
        """
        计算力系数 (CDp + CDv)

        Args:
            fields: 流场数据 (4, H, W) = [rho, u, v, p]
                - H: j方向（法向），j=0为壁面
                - W: i方向（周向），环绕翼型
            coords_vertex: 节点坐标 (2, H+1, W+1)，完整网格
            flow_conditions: 流动条件
                - ndarray/tensor: [Ma, AoA, Re]
                - dict: {'AOA': ..., 'Ma': ..., 'Re': ...}
            return_components: 是否返回分量详情
            compute_viscous: 是否计算粘性阻力CDv（默认True）
            T_inf: 参考温度(K)，用于Sutherland粘度公式

        Returns:
            dict: {
                'CL': lift coefficient,
                'CD': total drag coefficient (CDp + CDv),
                'CDp': pressure drag coefficient,
                'CDv': viscous drag coefficient,
                'Cm': pitching moment coefficient,
                # 如果return_components=True:
                'Fx_body': x方向力,
                'Fy_body': y方向力,
                'M': 力矩
            }
        """
        if _is_tensor_like(fields):
            fields = fields.detach().cpu().numpy()
        if _is_tensor_like(coords_vertex):
            coords_vertex = coords_vertex.detach().cpu().numpy()

        if isinstance(flow_conditions, dict):
            aoa_deg = flow_conditions.get('AOA', flow_conditions.get('aoa', 0.0))
            mach = flow_conditions.get('Ma', flow_conditions.get('mach', 0.3))
            Re = flow_conditions.get('Re', flow_conditions.get('re', 1e6))
        elif isinstance(flow_conditions, (np.ndarray, list, tuple)) or _is_tensor_like(flow_conditions):
            if _is_tensor_like(flow_conditions):
                flow_conditions = flow_conditions.detach().cpu().numpy()
            mach = flow_conditions[0]
            aoa_deg = flow_conditions[1]
            Re = flow_conditions[2] if len(flow_conditions) > 2 else 1e6
        else:
            aoa_deg, mach, Re = 0.0, 0.3, 1e6

        aoa_rad = np.radians(aoa_deg)

        p_wall = fields[3, 0, :]

        x_wall_v = coords_vertex[0, 0, :]
        y_wall_v = coords_vertex[1, 0, :]

        x_wall_c = 0.5 * (x_wall_v[:-1] + x_wall_v[1:])
        y_wall_c = 0.5 * (y_wall_v[:-1] + y_wall_v[1:])

        dx = x_wall_v[1:] - x_wall_v[:-1]
        dy = y_wall_v[1:] - y_wall_v[:-1]
        ds = np.sqrt(dx**2 + dy**2)

        # ------------------------------------------------------------
        # 外法向方向修正：根据壁面顶点顺/逆时针自动选择外法向
        # CCW(面积>0): outward = (dy, -dx)/ds
        # CW (面积<0): outward = (-dy, dx)/ds = -(dy, -dx)/ds
        # ------------------------------------------------------------
        signed_area = 0.5 * np.sum(x_wall_v[:-1] * y_wall_v[1:] - x_wall_v[1:] * y_wall_v[:-1])
        orient = 1.0 if signed_area > 0 else -1.0
        nx = orient * dy / (ds + 1e-12)
        ny = orient * (-dx) / (ds + 1e-12)

        # ------------------------------------------------------------
        # C-grid wake cut 自动剔除
        # 对于包含 wake cut 的网格，壁面线常包含一段 y≈0 且 x 明显大于尾缘的远场段，
        # 该段对力系数应不参与（尤其是力矩）；这里通过 x 阈值自动筛除。
        # ------------------------------------------------------------
        seg_mask = np.ones_like(ds, dtype=bool)
        y_eps = 1e-6
        surface_vmask = np.abs(y_wall_v) > y_eps
        if surface_vmask.any():
            x_max_surface = float(np.max(x_wall_v[surface_vmask]))
            x_min_surface = float(np.min(x_wall_v[surface_vmask]))
            chord_like = max(x_max_surface - x_min_surface, 1e-6)
            x_margin = max(1e-3, 0.01 * chord_like)
            x_thresh = x_max_surface + x_margin
            # 若存在明显远场段（x 远大于尾缘且 y≈0），则启用剔除
            has_cut = bool(((x_wall_v > x_thresh) & (np.abs(y_wall_v) <= y_eps)).any())
            if has_cut:
                seg_mask = (x_wall_v[:-1] <= x_thresh) & (x_wall_v[1:] <= x_thresh)

        q_nondim = 0.5 * self.gamma * mach**2

        Cp = (p_wall - 1.0) / (q_nondim + 1e-12)

        Fx_body = -np.sum(Cp[seg_mask] * nx[seg_mask] * ds[seg_mask])
        Fy_body = -np.sum(Cp[seg_mask] * ny[seg_mask] * ds[seg_mask])

        cos_a = np.cos(aoa_rad)
        sin_a = np.sin(aoa_rad)

        CL = Fy_body * cos_a - Fx_body * sin_a
        CD = Fx_body * cos_a + Fy_body * sin_a

        S_ref = self.area_ref
        CLp = CL / S_ref
        CDp = CD / S_ref

        CLv = 0.0
        CDv = 0.0
        Mv = 0.0
        if compute_viscous:
            from surrogate.physics.pde.surface_forces import compute_viscous_wall_force_adflow_like

            prepared_geometry = self._get_surface_force_geometry(coords_vertex)
            viscous_force = compute_viscous_wall_force_adflow_like(
                fields=fields,
                flow_conditions={'Ma': mach, 'AOA': aoa_deg, 'Re': Re},
                prepared_geometry=prepared_geometry,
                gamma=self.gamma,
                seg_mask=seg_mask.astype(np.float64, copy=False),
                return_details=True,
            )
            Fx_v = viscous_force['Fx_v']
            Fy_v = viscous_force['Fy_v']
            CLv = (Fy_v * cos_a - Fx_v * sin_a) / (q_nondim * S_ref)
            CDv = (Fx_v * cos_a + Fy_v * sin_a) / (q_nondim * S_ref)
            wall_vx = np.asarray(viscous_force['wall_vx'], dtype=np.float64)
            wall_vy = np.asarray(viscous_force['wall_vy'], dtype=np.float64)
        else:
            Fx_v = 0.0
            Fy_v = 0.0

        x_ref = self.moment_center[0]
        y_ref = self.moment_center[1]

        Mp = np.sum(
            (-Cp[seg_mask] * ds[seg_mask])
            * ((x_wall_c[seg_mask] - x_ref) * ny[seg_mask] - (y_wall_c[seg_mask] - y_ref) * nx[seg_mask])
        )
        if compute_viscous:
            Mv = np.sum(
                (
                    (x_wall_c[seg_mask] - x_ref) * wall_vy[seg_mask]
                    - (y_wall_c[seg_mask] - y_ref) * wall_vx[seg_mask]
                )
            )

        CL_total = CLp + CLv
        CD_total = CDp + CDv
        Cmp = Mp / (S_ref * self.chord_ref)
        Cmv = Mv / (q_nondim * S_ref * self.chord_ref)
        Cm_total = Cmp + Cmv

        result = {
            'CL': float(CL_total),
            'CD': float(CD_total),
            'CDp': float(CDp),
            'CDv': float(CDv),
            'Cm': float(Cm_total)
        }

        if return_components:
            result.update({
                'Fx_body': float(Fx_body),
                'Fy_body': float(Fy_body),
                'Fx_v': float(Fx_v),
                'Fy_v': float(Fy_v),
                'CLp': float(CLp),
                'CLv': float(CLv),
                'Cmp': float(Cmp),
                'Cmv': float(Cmv),
                'M': float(Mp + Mv),
                'M_p': float(Mp),
                'M_v': float(Mv),
                'aoa_deg': float(aoa_deg),
                'mach': float(mach),
                'Re': float(Re)
            })

        return result

    def compute_cp_distribution(
        self,
        fields: Union[np.ndarray, 'torch.Tensor'],
        coords_vertex: Union[np.ndarray, 'torch.Tensor'],
        flow_conditions: Union[np.ndarray, 'torch.Tensor', Dict],
    ) -> Dict[str, np.ndarray]:
        """
        提取壁面Cp分布并分离上下翼面

        Args:
            fields: (C, H, W) 物理空间流场 [rho, u, v, p, ...]
            coords_vertex: (2, H+1, W+1) 节点坐标
            flow_conditions: 流动条件 (dict or array [Ma, AoA, Re])

        Returns:
            dict: {
                'x_wall': (W,) 壁面x坐标,
                'cp_wall': (W,) 壁面Cp,
                'x_upper': upper surface x,
                'cp_upper': upper surface Cp,
                'x_lower': lower surface x,
                'cp_lower': lower surface Cp,
                'i_le': leading edge index,
                'mach': Mach number used,
            }
        """
        if _is_tensor_like(fields):
            fields = fields.detach().cpu().numpy()
        if _is_tensor_like(coords_vertex):
            coords_vertex = coords_vertex.detach().cpu().numpy()

        if isinstance(flow_conditions, dict):
            mach = flow_conditions.get('Ma', flow_conditions.get('mach', 0.3))
        elif isinstance(flow_conditions, (np.ndarray, list, tuple)) or _is_tensor_like(flow_conditions):
            if _is_tensor_like(flow_conditions):
                flow_conditions = flow_conditions.detach().cpu().numpy()
            mach = float(flow_conditions[0])
        else:
            mach = 0.3

        p_wall = fields[3, 0, :]
        q_nondim = 0.5 * self.gamma * mach**2
        cp_wall = (p_wall - 1.0) / (q_nondim + 1e-12)

        x_wall = 0.5 * (coords_vertex[0, 0, :-1] + coords_vertex[0, 0, 1:])

        i_le = int(np.argmin(x_wall))

        x_upper = x_wall[i_le:]
        cp_upper = cp_wall[i_le:]
        x_lower = x_wall[:i_le + 1][::-1].copy()
        cp_lower = cp_wall[:i_le + 1][::-1].copy()

        return {
            'x_wall': x_wall,
            'cp_wall': cp_wall,
            'x_upper': x_upper,
            'cp_upper': cp_upper,
            'x_lower': x_lower,
            'cp_lower': cp_lower,
            'i_le': i_le,
            'mach': float(mach),
        }

    def compute_coefficients_batch(
        self,
        fields_batch: Union[np.ndarray, torch.Tensor],
        coords_vertex_batch: Union[np.ndarray, torch.Tensor],
        flow_conditions_batch: Union[np.ndarray, torch.Tensor],
        compute_viscous: bool = True,
        T_inf: float = 300.0
    ) -> Dict[str, np.ndarray]:
        """
        批量计算力系数

        Args:
            fields_batch: (B, 4, H, W)
            coords_vertex_batch: (B, 2, H+1, W+1) 或 (2, H+1, W+1) 共享网格
            flow_conditions_batch: (B, 3) = [Ma, AoA, Re]
            compute_viscous: 是否计算粘性阻力CDv
            T_inf: 参考温度(K)

        Returns:
            dict: {
                'CL': (B,),
                'CD': (B,),
                'CDp': (B,),
                'CDv': (B,),
                'Cm': (B,)
            }
        """
        if _is_tensor_like(fields_batch):
            fields_batch = fields_batch.detach().cpu().numpy()
        if _is_tensor_like(coords_vertex_batch):
            coords_vertex_batch = coords_vertex_batch.detach().cpu().numpy()
        if _is_tensor_like(flow_conditions_batch):
            flow_conditions_batch = flow_conditions_batch.detach().cpu().numpy()

        B = fields_batch.shape[0]

        if coords_vertex_batch.ndim == 3:
            coords_shared = True
        else:
            coords_shared = False

        results = {'CL': [], 'CD': [], 'CDp': [], 'CDv': [], 'Cm': []}

        for i in range(B):
            coords_v = coords_vertex_batch if coords_shared else coords_vertex_batch[i]
            coefs = self.compute_coefficients(
                fields_batch[i],
                coords_v,
                flow_conditions_batch[i],
                compute_viscous=compute_viscous,
                T_inf=T_inf
            )
            for key in results:
                results[key].append(coefs[key])

        return {k: np.array(v) for k, v in results.items()}


def compute_force_coefficients(
    fields: np.ndarray,
    coords_vertex: np.ndarray,
    flow_conditions: np.ndarray,
    gamma: float = 1.4,
    chord_ref: float = 1.0,
    area_ref: Optional[float] = None,
    moment_center: Tuple[float, float] = (0.0, 0.0),
    compute_viscous: bool = True,
    T_inf: float = 300.0
) -> Dict[str, float]:
    """
    便捷函数：计算力系数（含粘性阻力）

    Args:
        fields: (4, H, W) = [rho, u, v, p]
        coords_vertex: (2, H+1, W+1) 完整网格节点坐标
        flow_conditions: [Ma, AoA, Re]
        gamma: 比热比
        chord_ref: ADFLOW chordRef
        area_ref: ADFLOW areaRef；默认二维单位展向面积
        moment_center: ADFLOW绝对参考点(xRef, yRef)
        compute_viscous: 是否计算粘性升力、阻力和力矩
        T_inf: 参考温度(K)，用于Sutherland粘度公式

    Returns:
        {'CL': ..., 'CD': ..., 'CDp': ..., 'CDv': ..., 'Cm': ...}
    """
    calc = ForceCoefficientsCalculator(
        gamma=gamma,
        chord_ref=chord_ref,
        area_ref=area_ref,
        moment_center=moment_center,
    )
    return calc.compute_coefficients(
        fields, coords_vertex, flow_conditions,
        compute_viscous=compute_viscous, T_inf=T_inf
    )
