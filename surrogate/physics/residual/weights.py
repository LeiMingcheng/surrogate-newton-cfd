"""
残差聚合权重统一配置模块

该模块提供统一的残差聚合权重配置，确保以下位置使用一致的权重：
1. residual score 计算 - 加权 RMS 范数
2. residual reporting - 聚合残差指标
3. 后续 ADflow 后端 - 聚合残差指标

设计原则：
- 连续性方程和动量方程使用独立权重
- 支持向后兼容（自动转换旧的 'momentum' 参数）
- 提供统一的聚合方法接口
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Mapping, Optional, Tuple, Union

if TYPE_CHECKING:
    import torch


class ResidualWeights:
    """
    统一的残差聚合权重配置

    默认权重：
    - wc (continuity): 1.0 - 连续性方程权重
    - wmx (momentum_x): 0.5 - x动量方程权重
    - wmy (momentum_y): 0.5 - y动量方程权重

    物理意义：
    - 连续性方程：质量守恒，基础约束
    - 动量方程：动量守恒，决定流动结构
    - 当前配置 (1.0, 0.5, 0.5)：平衡连续性和动量项影响

    Attributes:
        wc: 连续性方程权重
        wmx: x动量方程权重
        wmy: y动量方程权重

    Examples:
        >>> # 使用默认权重
        >>> weights = ResidualWeights()
        >>> print(weights.wc, weights.wmx, weights.wmy)
        1.0 0.5 0.5

        >>> # 自定义权重
        >>> weights = ResidualWeights(wc=1.0, wmx=2.0, wmy=2.0)

        >>> # 聚合3通道残差
        >>> Rc = torch.randn(8, 84, 304)
        >>> Rmx = torch.randn(8, 84, 304)
        >>> Rmy = torch.randn(8, 84, 304)
        >>> R_agg = weights.aggregate_3channel(Rc, Rmx, Rmy)
        >>> print(R_agg.shape)
        torch.Size([8, 84, 304])
    """

    # 默认权重配置
    DEFAULT_CONTINUITY = 1.0
    DEFAULT_MOMENTUM_X = 0.5
    DEFAULT_MOMENTUM_Y = 0.5

    def __init__(self,
                 wc: Optional[float] = None,
                 wmx: Optional[float] = None,
                 wmy: Optional[float] = None):
        """
        初始化残差权重

        Args:
            wc: 连续性方程权重（默认1.0）
            wmx: x动量方程权重（默认0.5）
            wmy: y动量方程权重（默认0.5）
        """
        self.wc = wc if wc is not None else self.DEFAULT_CONTINUITY
        self.wmx = wmx if wmx is not None else self.DEFAULT_MOMENTUM_X
        self.wmy = wmy if wmy is not None else self.DEFAULT_MOMENTUM_Y

    @classmethod
    def from_dict(cls, config: Dict[str, float]) -> 'ResidualWeights':
        """
        从配置字典创建权重对象

        支持的键名：
        - 'continuity': 连续性权重
        - 'momentum_x': x动量权重
        - 'momentum_y': y动量权重

        Args:
            config: 权重配置字典

        Returns:
            ResidualWeights对象

        Examples:
            >>> config = {'continuity': 1.0, 'momentum_x': 0.5, 'momentum_y': 0.5}
            >>> weights = ResidualWeights.from_dict(config)
        """
        wc = config.get('continuity', cls.DEFAULT_CONTINUITY)
        wmx = config.get('momentum_x', cls.DEFAULT_MOMENTUM_X)
        wmy = config.get('momentum_y', cls.DEFAULT_MOMENTUM_Y)

        return cls(wc=wc, wmx=wmx, wmy=wmy)

    def to_dict(self) -> Dict[str, float]:
        """
        转换为字典格式

        Returns:
            权重字典，包含 'continuity', 'momentum_x', 'momentum_y' 键

        Examples:
            >>> weights = ResidualWeights()
            >>> weights.to_dict()
            {'continuity': 1.0, 'momentum_x': 0.5, 'momentum_y': 0.5}
        """
        return {
            'continuity': self.wc,
            'momentum_x': self.wmx,
            'momentum_y': self.wmy
        }

    def aggregate_3channel(self,
                          Rc: torch.Tensor,
                          Rmx: torch.Tensor,
                          Rmy: torch.Tensor) -> torch.Tensor:
        """
        聚合3通道残差为加权标量

        公式：R_agg = wc * Rc + wmx * Rmx + wmy * Rmy

        Args:
            Rc: 连续性方程残差，shape (B, H, W) 或 (H, W)
            Rmx: x动量方程残差，shape与Rc相同
            Rmy: y动量方程残差，shape与Rc相同

        Returns:
            聚合后的残差，shape与输入相同

        Examples:
            >>> weights = ResidualWeights(wc=1.0, wmx=0.5, wmy=0.5)
            >>> Rc = torch.ones(2, 84, 304)
            >>> Rmx = torch.ones(2, 84, 304) * 2
            >>> Rmy = torch.ones(2, 84, 304) * 3
            >>> R_agg = weights.aggregate_3channel(Rc, Rmx, Rmy)
            >>> print(R_agg[0, 0, 0])  # 1.0*1 + 0.5*2 + 0.5*3 = 3.5
            tensor(3.5000)
        """
        return self.wc * Rc + self.wmx * Rmx + self.wmy * Rmy

    def aggregate_spatial_map(self, spatial_map: torch.Tensor) -> torch.Tensor:
        """
        从spatial_map聚合残差

        Args:
            spatial_map: 残差空间图，shape (B, C, H, W) 或 (C, H, W)
                         通道0: Rc, 通道1: Rmx, 通道2: Rmy
                         C>=3即可，多余通道(RE, RSA等)被忽略

        Returns:
            聚合后的残差，shape (B, H, W) 或 (H, W)

        Examples:
            >>> weights = ResidualWeights()
            >>> spatial_map = torch.randn(8, 5, 84, 304)
            >>> R_agg = weights.aggregate_spatial_map(spatial_map)
            >>> print(R_agg.shape)
            torch.Size([8, 84, 304])
        """
        if spatial_map.ndim == 3:
            # (C, H, W)
            if spatial_map.shape[0] < 3:
                raise ValueError(f"Expected at least 3 channels, got {spatial_map.shape[0]}")
            Rc = spatial_map[0]
            Rmx = spatial_map[1]
            Rmy = spatial_map[2]
        elif spatial_map.ndim == 4:
            # (B, C, H, W)
            if spatial_map.shape[1] < 3:
                raise ValueError(f"Expected at least 3 channels, got {spatial_map.shape[1]}")
            Rc = spatial_map[:, 0]
            Rmx = spatial_map[:, 1]
            Rmy = spatial_map[:, 2]
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {spatial_map.ndim}D")

        return self.aggregate_3channel(Rc, Rmx, Rmy)

    def __repr__(self) -> str:
        """字符串表示"""
        return (f"ResidualWeights(wc={self.wc}, wmx={self.wmx}, wmy={self.wmy})")

    def __eq__(self, other) -> bool:
        """相等性比较"""
        if not isinstance(other, ResidualWeights):
            return False
        return (self.wc == other.wc and
                self.wmx == other.wmx and
                self.wmy == other.wmy)


def validate_weights_consistency(weights: ResidualWeights,
                                 expected_wc: float = 1.0,
                                 expected_wmx: float = 0.5,
                                 expected_wmy: float = 0.5,
                                 strict: bool = False) -> bool:
    """
    验证权重配置的一致性

    Args:
        weights: 待验证的权重对象
        expected_wc: 期望的连续性权重
        expected_wmx: 期望的x动量权重
        expected_wmy: 期望的y动量权重
        strict: 是否严格检查（抛出异常）

    Returns:
        是否通过验证

    Raises:
        ValueError: 如果strict=True且验证失败

    Examples:
        >>> weights = ResidualWeights()
        >>> validate_weights_consistency(weights)
        True

        >>> weights_custom = ResidualWeights(wc=1.0, wmx=2.0, wmy=2.0)
        >>> validate_weights_consistency(weights_custom, strict=False)
        False
    """
    is_valid = (weights.wc == expected_wc and
                weights.wmx == expected_wmx and
                weights.wmy == expected_wmy)

    if not is_valid:
        msg = (f"Weight mismatch: expected ({expected_wc}, {expected_wmx}, {expected_wmy}), "
               f"got ({weights.wc}, {weights.wmx}, {weights.wmy})")
        if strict:
            raise ValueError(msg)
        else:
            import warnings
            warnings.warn(msg)

    return is_valid


def parse_residual_weights(
    weights: Optional[Mapping[str, float]],
    *,
    wc_default: float = ResidualWeights.DEFAULT_CONTINUITY,
    wmx_default: float = ResidualWeights.DEFAULT_MOMENTUM_X,
    wmy_default: float = ResidualWeights.DEFAULT_MOMENTUM_Y,
    energy_default: float = 1.0,
    turbulence_default: float = 1.0,
) -> Tuple[float, float, float, float, float]:
    """
    Parse residual aggregation weights with backward-compatible keys.

    Supported keys:
    - continuity: 'continuity' | 'wc'
    - momentum (per-component): 'momentum_x' | 'wmx', 'momentum_y' | 'wmy'
    - momentum (legacy single): 'momentum' | 'wm' (applies to both x/y)
    - energy: 'energy' | 'w_energy'
    - turbulence (SA): 'turbulence' | 'w_SA' | 'w_turbulence'

    Returns:
        (wc, wmx, wmy, w_energy, w_turbulence)
    """
    if weights is None:
        return wc_default, wmx_default, wmy_default, energy_default, turbulence_default

    wc = float(weights.get('continuity', weights.get('wc', wc_default)))

    # Prefer unified per-component momentum weights; fall back to legacy single momentum.
    wmx = weights.get('momentum_x', weights.get('wmx', None))
    wmy = weights.get('momentum_y', weights.get('wmy', None))

    if wmx is None and wmy is None:
        wm_legacy = weights.get('momentum', weights.get('wm', None))
        if wm_legacy is not None:
            wmx = wm_legacy
            wmy = wm_legacy
        else:
            wmx = wmx_default
            wmy = wmy_default
    else:
        if wmx is None:
            wmx = wmx_default
        if wmy is None:
            wmy = wmx

    w_energy = float(weights.get('energy', weights.get('w_energy', energy_default)))
    w_turbulence = float(
        weights.get(
            'turbulence',
            weights.get('w_SA', weights.get('w_turbulence', turbulence_default)),
        )
    )

    return float(wc), float(wmx), float(wmy), w_energy, w_turbulence
