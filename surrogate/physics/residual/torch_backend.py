"""
PyTorch Native Residual Calculator

PyTorch实现的PDE残差计算器，使用modular backend（surrogate/pde/）。

核心功能：
1. 基于PyTorch的可微分PDE残差计算
2. 支持多种粘度模型（constant_Re, Sutherland, RANS）
3. 灵活的加权策略（uniform, area, area_wall_boost）
4. GPU加速和自动微分支持

接口说明：
- compute_residual(): 统一接口，与ADflow backend对齐
- 支持return_spatial控制是否返回残差图
- 支持return_components控制是否返回残差分量
"""

import os
import time
import torch
import numpy as np
import weakref
from collections import OrderedDict
from typing import Dict, Tuple, Optional, Union


# ========== 湍流模型参数 ==========
# TURB_RATIO_TABLE 和 get_turb_ratio_adaptive() 已删除
# 原因：
# - 全仓无任何调用
# - 用于废弃的 viscosity_mode='rans' 模式
# - 当前强制使用 viscosity_mode='laminar+SA'，不需要这些函数


# ========== 几何缓存辅助函数 ==========
def _tensor_hash(tensor: torch.Tensor) -> int:
    """
    计算tensor内容的hash值（用于缓存key）

    与data_ptr()不同，此hash基于tensor的实际内容，不受内存地址复用影响。
    使用numpy.tobytes()确保相同内容的tensor有相同hash。

    Args:
        tensor: 输入tensor（任意shape）

    Returns:
        基于tensor内容的hash值

    注意：
    - 计算hash有一定开销（需要CPU同步），仅在缓存场景使用
    - 对于训练中的单样本重复计算，hash开销远小于几何计算开销
    """
    # 将tensor转到CPU并转为bytes计算hash
    # 使用.contiguous()确保内存布局一致
    return hash(tensor.detach().cpu().contiguous().numpy().tobytes())


class TorchResidualBackend:
    """
    Torch 原生残差计算 backend。

    封装surrogate.physics.pde模块，提供统一接口。
    """

    def __init__(
        self,
        viscosity_source: str = 'none',
        viscosity_mode: Optional[str] = None,
        residual_norm_mode: str = 'standard',
        gamma: float = 1.4,
        device: str = 'cpu',
        periodic_xi: bool = True,
        dissipation_mode: str = 'jameson',
        vis2: float = 0.25,
        vis4: float = 0.0156,
        dss_max: float = 0.25,
        sslim: Optional[float] = None,
        pInfCorr: float = 1.0,
        rhoInf: float = 1.0,
        rfl_b: float = 2.0,
        debug: bool = False,
        geometry_fast_cache_max_entries: int = 1,
        basis: Optional[str] = None,
        gradient_method: str = 'adflow_6pt',
        adis: float = 0.67,
        acoustic_scale_factor: float = 1.0,
        compute_only_momentum: bool = False,
        eddyVisInfRatio: float = 0.009,
        sa_clamp_nu_tilde: bool = True,
        clamp_rho_p: bool = True,
        rho_floor: float = 1e-6,
        p_floor: float = 1e-6,
        **kwargs
    ):
        """
        初始化PyTorch backend（与pde模块和ADflow对齐）

        Args:
            viscosity_source: 粘性来源（与pde模块统一）
                - 'none': Euler方程
                - 'laminar+SA': 层流粘度+SA涡粘（与ADflow对齐）
            viscosity_mode: 兼容参数，映射到viscosity_source（调度层使用）
            residual_norm_mode: 残差归一化模式（默认：'standard'，与ADFlow对齐）
                - 'standard'/'adflow': per-volume残差 RMS = sqrt(mean(R^2))
                - 'plain'/'flux': 通量残差 RMS = sqrt(mean((R*vol)^2))
                - 'weighted': 加权残差 RMS = sqrt(mean((R*sqrt(vol))^2))
            gamma: 比热比
            device: 计算设备
            periodic_xi: ξ方向周期性
            dissipation_mode: 数值耗散模式
                - 'jameson': Jameson 2阶+4阶混合耗散（默认，与ADflow对齐）
                - 'none': 无耗散（纯中央差分）
            vis2: Jameson 2阶耗散系数（默认0.5，与ADflow对齐）
            vis4: Jameson 4阶耗散系数（默认0.0156，与当前live ADFLOW对齐）
            dss_max: 传感器上限（默认0.25）
            sslim: 传感器下限（如果为None则动态计算）
                - None（默认）: 根据basis自动计算:
                    * entropy: 0.001 * pInfCorr / rhoInf^gamma
                    * pressure: 0.001 * pInfCorr
                - float: 使用固定值（与ADflow默认1e-3一致）
            pInfCorr: 校正压力（无量纲，默认1.0）
                - 用于计算sslim
                - ADflow: pInfCorr = pInf (无k方程) 或 pInf + 2/3*rhoInf*k (有k方程)
            rhoInf: 自由流密度（无量纲，默认1.0）
                - 用于计算熵基底的sslim
            rfl_b: rfl抑制系数（默认2.0）
            basis: 激波传感器基底 (ADflow blockette.F90:3190-3207)
                - None: 自动根据viscosity_source选择
                - 'entropy': RANS/NS模式，ss = p / rho^gamma
                - 'pressure': Euler模式，ss = p
            gradient_method: Cell-center梯度计算方法（默认：'adflow_6pt'，与ADflow对齐）
                - 'adflow_6pt': ADFLOW六向量梯度（用于耗散、湍流模型，与ADflow blockette.F90:1070-1124对齐）
                - 'green_gauss': Green-Gauss梯度（基于散度定理）
                注意：粘性通量自动使用nodal梯度（ADflow blockette.F90:5350-5634），
                      不受此参数控制，确保与ADflow粘性通量完全对齐。
            compute_only_momentum: 只计算连续性和动量方程残差（Rc, Rmx, Rmy）
                - True: 强制跳过能量方程和SA湍流方程残差，即使输入有4/5通道
                - False: 根据输入通道数自动计算所有方程残差（默认）
            eddyVisInfRatio: SA湍流模型自由流涡粘比（默认0.009，与ADFLOW对齐）
                - 用于计算SA入流边界的nuTilde_inf值
                - ADFLOW默认值: 0.009 (inputParamRoutines.F90:3443-3445)
                - 通过saNuKnownEddyRatio Newton迭代求解nuTilde_inf
            **kwargs: 接受但忽略的废弃参数（weighting, allow_fallback_to_euler等）
        """
        # 配置验证：强制使用最佳实践配置
        if gradient_method != 'adflow_6pt':
            raise ValueError(
                f"gradient_method必须为'adflow_6pt'，不支持'{gradient_method}'。"
                f"其他梯度方法已废弃。"
            )

        # 确定viscosity_source（兼容旧参数viscosity_mode）
        if viscosity_mode is not None:
            self.viscosity_source = viscosity_mode
        else:
            self.viscosity_source = viscosity_source

        # 验证viscosity_source
        if self.viscosity_source != 'laminar+SA':
            raise ValueError(
                f"viscosity_source必须为'laminar+SA'，不支持'{self.viscosity_source}'。"
                f"Euler模式('none')已不再支持。"
            )

        self.residual_norm_mode = residual_norm_mode
        self.gamma = gamma
        self.device = device
        self.periodic_xi = periodic_xi

        # 数值耗散参数
        self.dissipation_mode = dissipation_mode
        self.vis2 = vis2
        self.vis4 = vis4
        self.dss_max = dss_max
        self.pInfCorr = pInfCorr
        self.rhoInf = rhoInf
        self.rfl_b = rfl_b

        # 激波传感器基底选择 (ADflow blockette.F90:3190-3207)
        # RANS/NS使用熵基底 ss = p/rho^gamma，Euler使用压力基底 ss = p
        if basis is not None:
            self.basis = basis
        else:
            # 自动选择：有粘性则用熵基底，无粘性（Euler）用压力基底
            if self.viscosity_source != 'none':
                self.basis = 'entropy'
            else:
                self.basis = 'pressure'

        # sslim动态计算（与ADflow blockette.F90:3182-3198对齐）
        # - Euler (压力基底): sslim = 0.001 * pInfCorr
        # - NS/RANS (熵基底): sslim = 0.001 * pInfCorr / rhoInf^gamma
        if sslim is not None:
            self.sslim = sslim
        else:
            if self.basis == 'entropy':
                self.sslim = 0.001 * self.pInfCorr / (self.rhoInf ** self.gamma)
            else:  # 'pressure'
                self.sslim = 0.001 * self.pInfCorr

        # Debug模式
        self.debug = debug

        # 梯度计算方法
        self.gradient_method = gradient_method

        # Jameson耗散参数（与ADflow对齐）
        self.adis = adis
        self.acoustic_scale_factor = acoustic_scale_factor

        # 方程计算控制
        self.compute_only_momentum = compute_only_momentum

        # SA湍流模型参数（ADFLOW对齐）
        self.eddyVisInfRatio = eddyVisInfRatio
        self.sa_clamp_nu_tilde = bool(sa_clamp_nu_tilde)
        self.clamp_rho_p = bool(clamp_rho_p)
        self.rho_floor = float(rho_floor)
        self.p_floor = float(p_floor)

        # 导入pde子模块
        try:
            from surrogate.physics.pde import geometry as geom_module
            from surrogate.physics.pde import fluxes as flux_module
            from surrogate.physics.pde import residual as res_module

            self._geom = geom_module
            self._flux = flux_module
            self._res = res_module
        except ImportError as e:
            raise RuntimeError(
                "Required PDE modules not available. "
                "Ensure surrogate.physics.pde is properly installed."
            ) from e

        self._logged_mode = False

        # 几何缓存（优化性能）
        # 使用tensor内容hash作为key，避免data_ptr()的指针复用问题
        # cache_key = (vertex_hash, center_hash, periodic_xi)
        self._geometry_cache = {}  # key: (vertex_hash, center_hash, periodic_xi), value: geometry_dict
        # 快速几何缓存：基于tensor对象身份(id + _version)的key，避免每次都做CPU同步的内容hash。
        # - 适用于同一process内“同一grid tensor对象”被频繁复用（例如 sparse-newton 构建J时的多次JVP）
        # - 若grid tensor发生in-place修改，_version会变化，从而自动miss并回退到内容hash路径
        self._geometry_fast_cache_max_entries = max(1, int(geometry_fast_cache_max_entries))
        self._geometry_cache_fast: "OrderedDict[tuple, dict]" = OrderedDict()
        self._residual_operator_profile_enabled = str(
            os.environ.get("SURROGATE_RESIDUAL_OPERATOR_PROFILE", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._residual_operator_profile_sync_cuda = str(
            os.environ.get("SURROGATE_RESIDUAL_OPERATOR_PROFILE_SYNC_CUDA", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._last_residual_operator_profile: Optional[Dict[str, float]] = None
        self._gradient_calculator_cache: Dict[Tuple[str, bool, str, torch.dtype], object] = {}
        self._sa_residual_calculator_cache: Dict[Tuple[float, bool, str, bool, str], object] = {}

    def _residual_profile_sync(self) -> None:
        if (not bool(self._residual_operator_profile_enabled)) or (not bool(self._residual_operator_profile_sync_cuda)):
            return
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(device=self.device)

    def _residual_profile_mark(self, profile_timing: Optional[Dict[str, float]], key: str, t0: Optional[float]) -> None:
        if profile_timing is None or t0 is None:
            return
        self._residual_profile_sync()
        profile_timing[str(key)] = float(profile_timing.get(str(key), 0.0) + (time.perf_counter() - t0))

    def _get_gradient_calculator(
        self,
        *,
        method: str,
        periodic_xi: bool,
        dtype: torch.dtype,
    ):
        from surrogate.physics.pde.gradient import create_gradient_calculator

        cache_key = (
            str(method).lower().replace("-", "_"),
            bool(periodic_xi),
            str(self.device),
            dtype,
        )
        gradient_calc = self._gradient_calculator_cache.get(cache_key)
        if gradient_calc is None:
            gradient_calc = create_gradient_calculator(
                method=method,
                periodic_xi=periodic_xi,
                device=self.device,
                dtype=dtype,
            )
            self._gradient_calculator_cache[cache_key] = gradient_calc
        return gradient_calc

    def _get_sa_residual_calculator(self, *, approx_sa: bool):
        from surrogate.physics.pde.sa_residual import SAResidualCalculator

        cache_key = (
            float(self.gamma),
            True,
            "strain",
            bool(approx_sa),
            str(self.device),
        )
        sa_calc = self._sa_residual_calculator_cache.get(cache_key)
        if sa_calc is None:
            sa_calc = SAResidualCalculator(
                gamma=self.gamma,
                cv1=7.1,
                cb1=0.1355,
                cb2=0.622,
                sigma=2.0 / 3.0,
                kappa=0.41,
                use_ft2=True,
                prod_mode="strain",
                approx_sa=bool(approx_sa),
                device=self.device,
            )
            self._sa_residual_calculator_cache[cache_key] = sa_calc
        return sa_calc

    def _build_adflow_consistent_state(
        self,
        rho: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        compute_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reconstruct the primitive/conservative state the same way the ADFLOW path sees it.

        ADFLOW's injected state is effectively defined by float32 primitive inputs promoted to
        conservative variables. In low-residual cells, the raw dataset pressure can differ from
        the pressure implied by that quantized conservative state at the 1e-7 level, which is
        enough to dominate momentum residual cancellation near tiny wall-adjacent volumes.
        """
        state_dtype = torch.float32

        rho_q = rho.to(dtype=state_dtype)
        u_q = u.to(dtype=state_dtype)
        v_q = v.to(dtype=state_dtype)
        p_q = p.to(dtype=state_dtype)

        gamma_minus_one_q = torch.tensor(
            self.gamma - 1.0,
            dtype=state_dtype,
            device=rho_q.device,
        )
        kinetic_q = 0.5 * (u_q * u_q + v_q * v_q)
        rhoE_q = rho_q * (p_q / (rho_q * gamma_minus_one_q) + kinetic_q)

        rho_state = rho_q.to(dtype=compute_dtype)
        u_state = u_q.to(dtype=compute_dtype)
        v_state = v_q.to(dtype=compute_dtype)
        rhoE_state = rhoE_q.to(dtype=compute_dtype)

        kinetic_energy = 0.5 * rho_state * (u_state * u_state + v_state * v_state)
        p_state = (self.gamma - 1.0) * (rhoE_state - kinetic_energy)

        if self.clamp_rho_p:
            p_state = torch.clamp(p_state, min=self.p_floor)

        return rho_state, u_state, v_state, p_state, rhoE_state

    def _aggregate_total_residual(
        self,
        residual_fields,
        wall_layers: Optional[int] = None,
    ) -> torch.Tensor:
        """Aggregate raw per-volume residual fields into a global TotalR-like norm."""
        total_sq = None
        for field in residual_fields:
            if field is None:
                continue
            if wall_layers is not None:
                field = field[..., :wall_layers, :]
            field64 = field.to(dtype=torch.float64)
            field_sq = torch.sum(field64 * field64, dim=(-2, -1))
            total_sq = field_sq if total_sq is None else (total_sq + field_sq)
        if total_sq is None:
            raise ValueError("No residual fields available to aggregate total residual.")
        return torch.sqrt(total_sq)

    def _build_freestream_fields_like(
        self,
        fields: Union[torch.Tensor, Dict[str, torch.Tensor]],
        flow_conditions: Optional[Union[torch.Tensor, Dict]] = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """Construct a batch of uniform free-stream primitive fields matching the input shape."""
        if flow_conditions is None:
            raise ValueError("flow_conditions is required to build free-stream reference fields.")

        if isinstance(fields, dict):
            rho = fields['rho']
            u = fields['u']
            v = fields['v']
            p = fields['p']
            nu_tilde = fields.get('nuTilde', fields.get('nu_tilde'))
            components = [rho, u, v, p] if nu_tilde is None else [rho, u, v, p, nu_tilde]
            stack_dim = 1 if rho.ndim == 3 else 0
            template = torch.stack(components, dim=stack_dim)
        else:
            template = fields

        template = template.to(device=self.device, dtype=dtype)
        is_batched = template.ndim == 4

        if is_batched:
            batch_size, n_channels, H, W = template.shape
        else:
            n_channels, H, W = template.shape
            batch_size = None

        if isinstance(flow_conditions, dict):
            Ma = flow_conditions['Ma']
            AoA = flow_conditions['AoA']
            Re = flow_conditions['Re']
        else:
            flow_tensor = flow_conditions.to(device=self.device, dtype=dtype)
            if flow_tensor.ndim == 1:
                Ma, AoA, Re = flow_tensor[0], flow_tensor[1], flow_tensor[2]
            elif flow_tensor.ndim == 2:
                Ma, AoA, Re = flow_tensor[:, 0], flow_tensor[:, 1], flow_tensor[:, 2]
            else:
                raise ValueError(f"Unexpected flow_conditions shape: {flow_tensor.shape}")

        sqrt_gamma = torch.sqrt(torch.tensor(self.gamma, device=self.device, dtype=dtype))

        if is_batched:
            Ma_t = torch.as_tensor(Ma, device=self.device, dtype=dtype).reshape(batch_size)
            AoA_t = torch.as_tensor(AoA, device=self.device, dtype=dtype).reshape(batch_size)
            Re_t = torch.as_tensor(Re, device=self.device, dtype=dtype).reshape(batch_size)

            aoa_rad = torch.deg2rad(AoA_t)
            u_inf = (Ma_t * sqrt_gamma * torch.cos(aoa_rad)).view(batch_size, 1, 1)
            v_inf = (Ma_t * sqrt_gamma * torch.sin(aoa_rad)).view(batch_size, 1, 1)

            free_fields = torch.zeros((batch_size, n_channels, H, W), device=self.device, dtype=dtype)
            free_fields[:, 0, :, :] = 1.0
            free_fields[:, 1, :, :] = u_inf
            free_fields[:, 2, :, :] = v_inf
            free_fields[:, 3, :, :] = 1.0

            if n_channels >= 5:
                from surrogate.physics.pde.sa_utils import compute_sa_nuTilde_inf_tensor

                nu_lam = (Ma_t * sqrt_gamma) / (Re_t + 1e-30)
                nu_tilde_inf = compute_sa_nuTilde_inf_tensor(
                    nu_lam,
                    eddyVisInfRatio=float(self.eddyVisInfRatio),
                ).view(batch_size, 1, 1)
                free_fields[:, 4, :, :] = nu_tilde_inf
            return free_fields

        Ma_t = torch.as_tensor(Ma, device=self.device, dtype=dtype)
        AoA_t = torch.as_tensor(AoA, device=self.device, dtype=dtype)
        Re_t = torch.as_tensor(Re, device=self.device, dtype=dtype)
        aoa_rad = torch.deg2rad(AoA_t)

        free_fields = torch.zeros((n_channels, H, W), device=self.device, dtype=dtype)
        free_fields[0, :, :] = 1.0
        free_fields[1, :, :] = Ma_t * sqrt_gamma * torch.cos(aoa_rad)
        free_fields[2, :, :] = Ma_t * sqrt_gamma * torch.sin(aoa_rad)
        free_fields[3, :, :] = 1.0

        if n_channels >= 5:
            from surrogate.physics.pde.sa_utils import compute_sa_nuTilde_inf_tensor

            nu_lam = (Ma_t * sqrt_gamma) / (Re_t + 1e-30)
            nu_tilde_inf = compute_sa_nuTilde_inf_tensor(
                nu_lam,
                eddyVisInfRatio=float(self.eddyVisInfRatio),
            )
            free_fields[4, :, :] = nu_tilde_inf

        return free_fields

    def _build_halo_geometry_cache(
        self,
        *,
        coords_vertex: torch.Tensor,
        face_geom: Dict[str, torch.Tensor],
        periodic_xi: bool,
        sign: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        coords_vertex_hat = self._geom.extrapolate_halo_vertex_coords(
            coords_vertex, direction='eta'
        )
        halo_vol_wall, halo_vol_ff = self._geom.compute_halo_cell_volumes(
            coords_vertex_hat, periodic_xi=periodic_xi
        )
        si_x_halo_wall, si_y_halo_wall, si_x_halo_ff, si_y_halo_ff = self._geom.compute_halo_xi_face_vectors(
            coords_vertex_hat, periodic_xi=periodic_xi, sign=sign
        )
        sj_x_hat, sj_y_hat = self._geom.compute_halo_eta_face_vectors(
            coords_vertex_hat,
            face_geom_eta=(face_geom['A_x_eta'], face_geom['A_y_eta']),
            periodic_xi=periodic_xi,
            sign=sign
        )
        return {
            'halo_vol_wall': halo_vol_wall,
            'halo_vol_ff': halo_vol_ff,
            'si_x_halo_wall': si_x_halo_wall,
            'si_y_halo_wall': si_y_halo_wall,
            'si_x_halo_ff': si_x_halo_ff,
            'si_y_halo_ff': si_y_halo_ff,
            'sj_x_hat': sj_x_hat,
            'sj_y_hat': sj_y_hat,
        }

    def _compute_multi_field_gradients(
        self,
        *,
        gradient_calculator,
        fields: list[torch.Tensor],
        geometry: Dict[str, torch.Tensor],
        boundary_conditions: Optional[list[Dict[str, torch.Tensor]]] = None,
        boundary_batch_ndims: Optional[Dict[str, int]] = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if not fields:
            return []

        if boundary_conditions is not None and len(boundary_conditions) != len(fields):
            raise ValueError(
                f"boundary_conditions length {len(boundary_conditions)} != fields length {len(fields)}"
            )

        batched_fields: list[torch.Tensor] = []
        squeeze_flags: list[bool] = []
        base_batch: Optional[int] = None
        for field in fields:
            if field.ndim == 2:
                field_batched = field.unsqueeze(0)
                squeeze_flags.append(True)
            elif field.ndim == 3:
                field_batched = field
                squeeze_flags.append(False)
            else:
                raise ValueError(
                    f"Gradient helper only supports (H, W) or (B, H, W) fields, got {tuple(field.shape)}"
                )

            if base_batch is None:
                base_batch = int(field_batched.shape[0])
            elif int(field_batched.shape[0]) != int(base_batch):
                raise ValueError(
                    f"All stacked gradient fields must share batch size; got {field_batched.shape[0]} and {base_batch}"
                )
            batched_fields.append(field_batched)

        assert base_batch is not None
        repeats = len(batched_fields)
        if repeats == 1:
            grad_dx, grad_dy = gradient_calculator.compute_gradient(
                fields[0],
                geometry,
                None if boundary_conditions is None else boundary_conditions[0],
            )
            return [(grad_dx, grad_dy)]

        stacked_field = torch.cat(batched_fields, dim=0)

        geometry_stacked: Dict[str, torch.Tensor] = {}
        for key, value in geometry.items():
            if isinstance(value, torch.Tensor) and value.ndim >= 3:
                if int(value.shape[0]) != int(base_batch):
                    raise ValueError(
                        f"Gradient geometry '{key}' batch {value.shape[0]} does not match field batch {base_batch}"
                    )
                geometry_stacked[key] = value.repeat(repeats, *([1] * (value.ndim - 1)))
            else:
                geometry_stacked[key] = value

        boundary_stacked = None
        if boundary_conditions is not None:
            if boundary_batch_ndims is None:
                raise ValueError("boundary_batch_ndims is required when stacking boundary conditions.")

            boundary_stacked = {}
            for key, batched_ndim in boundary_batch_ndims.items():
                values = []
                for bc in boundary_conditions:
                    if key not in bc or bc[key] is None:
                        raise ValueError(f"Missing required gradient boundary key '{key}'.")
                    value = bc[key]
                    if not isinstance(value, torch.Tensor):
                        raise TypeError(f"Gradient boundary key '{key}' must be a tensor, got {type(value)}.")

                    if value.ndim == batched_ndim:
                        if int(value.shape[0]) != int(base_batch):
                            raise ValueError(
                                f"Gradient boundary '{key}' batch {value.shape[0]} != field batch {base_batch}"
                            )
                        value_batched = value
                    elif value.ndim == batched_ndim - 1:
                        value_batched = value.unsqueeze(0)
                        if int(base_batch) > 1:
                            value_batched = value_batched.expand(base_batch, *value.shape)
                    else:
                        raise ValueError(
                            f"Gradient boundary '{key}' expected ndim {batched_ndim - 1} or {batched_ndim}, got {value.ndim}"
                        )
                    values.append(value_batched)
                boundary_stacked[key] = torch.cat(values, dim=0)

        grad_dx_all, grad_dy_all = gradient_calculator.compute_gradient(
            stacked_field,
            geometry_stacked,
            boundary_stacked,
        )

        outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for idx, squeeze_output in enumerate(squeeze_flags):
            start = idx * base_batch
            end = start + base_batch
            grad_dx = grad_dx_all[start:end]
            grad_dy = grad_dy_all[start:end]
            if squeeze_output:
                grad_dx = grad_dx.squeeze(0)
                grad_dy = grad_dy.squeeze(0)
            outputs.append((grad_dx, grad_dy))

        return outputs

    def compute_residual(
        self,
        fields: Union[torch.Tensor, Dict[str, torch.Tensor]],
        coords: Union[Dict[str, torch.Tensor]],
        flow_conditions: Optional[Union[torch.Tensor, Dict]] = None,
        cgns_basename: Optional[str] = None,  # 兼容性参数（PyTorch不使用）
        weights: Optional[Dict[str, float]] = None,
        return_spatial: bool = False,
        return_components: bool = True,
        return_signed_field: bool = False,  # 新增：返回有符号残差场
        wall_layers: Optional[int] = None,
        spatial_wall_layers: Optional[int] = None,
        periodic_xi: bool = True,
        wall_distance: Optional[torch.Tensor] = None,
        wall_segment_mask: Optional[torch.Tensor] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        preserve_residual_dtype: bool = False,
        state_is_adflow_consistent: bool = False,
        state_is_adflow_mixed: bool = False,
    ) -> Union[Tuple[float, Dict], float]:
        """
        统一接口：计算PyTorch PDE残差

        与ADflow backend接口对齐。

        Args:
            fields: 流场数据
                - Tensor (C, H, W): [ρ, u, v, p]
                - Dict: {'rho': ..., 'u': ..., 'v': ..., 'p': ...}
            coords: 网格坐标字典
                - coords['vertex']: (2, H+1, W+1) 节点坐标（必需）
                - coords['center']: (2, H, W) 格心坐标（可选）
            flow_conditions: 流动条件
                - Tensor (3,): [Ma, AoA, Re]
                - Dict: {'Ma': ..., 'AoA': ..., 'Re': ...}
            cgns_basename: CGNS文件名（兼容性参数，PyTorch不使用）
            weights: 残差权重字典 {'continuity': wc, 'momentum': wm}
            return_spatial: 是否返回残差空间分布 (3, H, W)
            return_components: 是否返回残差分量字典
            wall_layers: 近壁区层数（统计/奖励口径）。如果指定，residual_score/RMS 只在 j=0:wall_layers 区域统计
            spatial_wall_layers: 近壁区层数（空间残差图口径）。仅影响返回的 spatial_map / signed_residual_field 的裁剪层数
                - None: 默认与 wall_layers 一致（向后兼容）
            periodic_xi: 是否启用ξ向周期性边界
            wall_distance: 可选的预计算壁面距离 (H, W)，优先使用ADFLOW的
                TurbulentDistance；缺失或样本无效时自动使用批量Torch最近壁面线段距离
            wall_segment_mask: 可选的真实壁面线段mask，C-grid/多物体回退时使用
            state_is_adflow_consistent:
                - False: 输入是普通 primitive 场，先走一次当前的状态重建
                - True: 输入已经由 ADFLOW conservative state 恢复得到，跳过重复投影
            state_is_adflow_mixed:
                - True: tensor 输入使用 [rho,u,v,p,rhoE] 或
                  [rho,u,v,p,rhoE,nuTilde]，保留传入 rhoE

        Returns:
            如果return_components=True:
                (residual_score, result_dict)
                - residual_score: 残差分数（负加权残差）
                - result_dict: {
                    'method': 'pytorch',
                    'Rc': float, 'Rmx': float, 'Rmy': float,
                    'Rc_norm': float, 'Rmx_norm': float, 'Rmy_norm': float,
                    'mu_eff': tensor,
                    'weights': tensor,
                    'spatial_map': tensor (C, L, W) or (B, C, L, W) (仅当return_spatial=True)
                                   C=3/4/5, L=wall_layers or H
                  }
            否则:
                residual_score: 残差分数
        """
        if spatial_wall_layers is None:
            spatial_wall_layers = wall_layers
        # 调用内部实现（原_compute_residual_score_modular）
        result = self._compute_residual_score_modular(
            fields,
            coords,
            flow_conditions,
            weights,
            wall_layers,
            spatial_wall_layers,
            return_components=True,  # 总是获取分量
            periodic_xi=periodic_xi,
            wall_distance=wall_distance,
            wall_segment_mask=wall_segment_mask,
            dtype=dtype,
            preserve_residual_dtype=bool(preserve_residual_dtype),
            state_is_adflow_consistent=bool(state_is_adflow_consistent),
            state_is_adflow_mixed=bool(state_is_adflow_mixed),
        )
        residual_operator_profile = result.get('residual_operator_profile', None)
        self._last_residual_operator_profile = (
            dict(residual_operator_profile) if isinstance(residual_operator_profile, dict) else None
        )

        # 提取 residual score（支持 batch）
        residual_score = result['residual_score']
        if isinstance(residual_score, torch.Tensor):
            if residual_score.numel() == 1:
                # 单样本：如果需要梯度则保持tensor，否则转标量
                if not residual_score.requires_grad:
                    try:
                        residual_score = residual_score.item()
                    except RuntimeError as e:
                        msg = str(e)
                        if "tracing tensor" in msg and "aten._local_scalar_dense" in msg:
                            # FX tracing path (e.g. torch.func.linearize): keep tensor
                            pass
                        else:
                            raise
                # else: 需要梯度（pathwise RL），保持tensor
            # else: batch模式，保持tensor (B,)

        # 构建返回字典（支持batch）
        def _extract_value(val):
            """提取标量或保持batch tensor"""
            if isinstance(val, torch.Tensor):
                if val.numel() == 1:
                    try:
                        return val.item()
                    except RuntimeError as e:
                        msg = str(e)
                        if "tracing tensor" in msg and "aten._local_scalar_dense" in msg:
                            # FX tracing path (e.g. torch.func.linearize): keep tensor
                            return val
                        raise
                # else: batch tensor，保持原样
            return val

        def _build_signed_residual_field(residual_result):
            Rc_signed = residual_result['Rc']
            Rmx_signed = residual_result['Rmx']
            Rmy_signed = residual_result['Rmy']

            signed_channels = [Rc_signed, Rmx_signed, Rmy_signed]
            if 'RE' in residual_result and residual_result['RE'] is not None:
                signed_channels.append(residual_result['RE'])
            if 'RSA' in residual_result and residual_result['RSA'] is not None:
                signed_channels.append(residual_result['RSA'])

            if spatial_wall_layers is not None:
                signed_channels = [ch[..., :spatial_wall_layers, :] for ch in signed_channels]

            if Rc_signed.ndim == 3:
                return torch.stack(signed_channels, dim=1)
            return torch.stack(signed_channels, dim=0)

        if bool(return_signed_field) and (not bool(return_spatial)) and (not bool(return_components)):
            result_dict = {
                'method': 'pytorch',
                'signed_residual_field': _build_signed_residual_field(result),
            }
            if result.get('wall_distance_source') is not None:
                result_dict['wall_distance_source'] = result['wall_distance_source']
            if self._last_residual_operator_profile is not None:
                result_dict['residual_operator_profile'] = dict(self._last_residual_operator_profile)
            return residual_score, result_dict

        def _compute_flux_norm(residual_field):
            if residual_field is None:
                return None
            vol = result.get('vol', None)
            if vol is None:
                return None
            return self._res.compute_residual_norm_rms_scalar(
                residual_field,
                wall_layers=wall_layers,
                norm_mode='flux',
                vol=vol,
            )

        totalR = self._aggregate_total_residual(
            [
                result.get('Rc'),
                result.get('Rmx'),
                result.get('Rmy'),
                result.get('RE'),
                result.get('RSA'),
            ],
            wall_layers=wall_layers,
        )
        totalR0 = None
        l2_ratio = None
        if flow_conditions is not None:
            if dtype is None:
                ref_dtype = torch.float64
            elif isinstance(dtype, str):
                dtype_str = dtype.lower()
                if dtype_str in {"float32", "fp32"}:
                    ref_dtype = torch.float32
                elif dtype_str in {"float64", "fp64"}:
                    ref_dtype = torch.float64
                else:
                    raise ValueError(
                        f"Unknown dtype='{dtype}'. Expected float32/float64 (or fp32/fp64)."
                    )
            elif isinstance(dtype, torch.dtype):
                ref_dtype = dtype
            else:
                raise TypeError(f"dtype must be None, str, or torch.dtype. Got {type(dtype)}")

            freestream_fields = self._build_freestream_fields_like(
                fields=fields,
                flow_conditions=flow_conditions,
                dtype=ref_dtype,
            )
            freestream_result = self._compute_residual_score_modular(
                freestream_fields,
                coords,
                flow_conditions,
                weights,
                wall_layers,
                spatial_wall_layers,
                return_components=True,
                periodic_xi=periodic_xi,
                wall_distance=result.get('_resolved_wall_distance', wall_distance),
                wall_segment_mask=wall_segment_mask,
                dtype=dtype,
            )
            totalR0 = self._aggregate_total_residual(
                [
                    freestream_result.get('Rc'),
                    freestream_result.get('Rmx'),
                    freestream_result.get('Rmy'),
                    freestream_result.get('RE'),
                    freestream_result.get('RSA'),
                ],
                wall_layers=wall_layers,
            )
            l2_ratio = torch.where(
                totalR0 > 1e-30,
                totalR / totalR0,
                torch.full_like(totalR, float('nan')),
            )

        result_dict = {
            'method': 'pytorch',
            'Rc': _extract_value(result['Rc_norm']),
            'Rmx': _extract_value(result['Rmx_norm']),
            'Rmy': _extract_value(result['Rmy_norm']),
            'totalR': _extract_value(totalR),
            'totalR0': _extract_value(totalR0),
            'l2_ratio': _extract_value(l2_ratio),
            'Rc_flux': _extract_value(_compute_flux_norm(result.get('Rc', None))),
            'Rmx_flux': _extract_value(_compute_flux_norm(result.get('Rmx', None))),
            'Rmy_flux': _extract_value(_compute_flux_norm(result.get('Rmy', None))),
        }
        if result.get('wall_distance_source') is not None:
            result_dict['wall_distance_source'] = result['wall_distance_source']
        if self._last_residual_operator_profile is not None:
            result_dict['residual_operator_profile'] = dict(self._last_residual_operator_profile)

        # 始终返回能量和SA残差（如果计算了）- 三方程控制由 compute_only_momentum 决定
        if 'RE_norm' in result and result['RE_norm'] is not None:
            result_dict['RE_norm'] = _extract_value(result['RE_norm'])
            result_dict['RE_flux'] = _extract_value(_compute_flux_norm(result.get('RE', None)))
        if 'RSA_norm' in result and result['RSA_norm'] is not None:
            result_dict['RSA_norm'] = _extract_value(result['RSA_norm'])
            result_dict['RSA_flux'] = _extract_value(_compute_flux_norm(result.get('RSA', None)))

        # 可选：添加空间残差图（支持batch: (B, C, H, W) 或单样本: (C, H, W)）
        # C = 3 (3方程) / 4 (能量) / 5 (SA湍流)，与ADflow后端一致
        if return_spatial:
            Rc = torch.abs(result['Rc'])
            Rmx = torch.abs(result['Rmx'])
            Rmy = torch.abs(result['Rmy'])

            # 构建残差通道列表（动态添加能量和SA通道）
            residual_channels = [Rc, Rmx, Rmy]

            # 能量方程残差（如果计算了）
            if 'RE' in result and result['RE'] is not None:
                RE = torch.abs(result['RE'])
                residual_channels.append(RE)

            # SA湍流方程残差（如果计算了）
            if 'RSA' in result and result['RSA'] is not None:
                RSA = torch.abs(result['RSA'])
                residual_channels.append(RSA)

            # 应用 spatial_wall_layers 裁剪（与ADflow后端一致）
            if spatial_wall_layers is not None:
                residual_channels = [ch[..., :spatial_wall_layers, :] for ch in residual_channels]

            # 堆叠为spatial_map
            # Rc.ndim: 3 = batch模式 (B, H, W)，2 = 单样本 (H, W)
            if Rc.ndim == 3:
                spatial_map = torch.stack(residual_channels, dim=1)  # (B, C, L, W)
            else:
                spatial_map = torch.stack(residual_channels, dim=0)  # (C, L, W)

            # 保持tensor形式，不转numpy（masking.py需要tensor输入）
            result_dict['spatial_map'] = spatial_map

        # 新增：返回有符号残差场（用于 Neural LM Solver）
        # C = 3 (3方程) / 4 (能量) / 5 (SA湍流)，与ADflow后端一致
        if return_signed_field:
            result_dict['signed_residual_field'] = _build_signed_residual_field(result)

        # 可选：添加其他分量（支持batch）
        if return_components:
            result_dict.update({
                'Rc_norm': _extract_value(result['Rc_norm']),
                'Rmx_norm': _extract_value(result['Rmx_norm']),
                'Rmy_norm': _extract_value(result['Rmy_norm']),
                'mu_eff': result.get('mu_eff'),
                'weights': result.get('weights'),
                'vol': result.get('vol'),  # 单元体积用于残差归一化分析
            })
            if 'residual_operator_sa_context' in result and result['residual_operator_sa_context'] is not None:
                result_dict['residual_operator_sa_context'] = result['residual_operator_sa_context']
            if 'residual_operator_shock_sensor' in result and result['residual_operator_shock_sensor'] is not None:
                result_dict['residual_operator_shock_sensor'] = result['residual_operator_shock_sensor']
            if 'residual_operator_ss_halo' in result and result['residual_operator_ss_halo'] is not None:
                result_dict['residual_operator_ss_halo'] = result['residual_operator_ss_halo']

        # 始终返回tuple (residual_score, result_dict)
        return residual_score, result_dict

    def _compute_residual_score_modular(
        self,
        fields: Union[torch.Tensor, Dict[str, torch.Tensor]],
        grid: Union[torch.Tensor, Dict[str, torch.Tensor]],
        flow_conditions: Optional[Union[torch.Tensor, Dict]] = None,
        weights: Optional[Dict[str, float]] = None,
        wall_layers: Optional[int] = None,
        spatial_wall_layers: Optional[int] = None,
        return_components: bool = False,
        periodic_xi: bool = True,
        wall_distance: Optional[torch.Tensor] = None,
        wall_segment_mask: Optional[torch.Tensor] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        preserve_residual_dtype: bool = False,
        state_is_adflow_consistent: bool = False,
        state_is_adflow_mixed: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        内部实现：完整的PyTorch PDE残差计算

        从旧 residual 实现移入（完整272行实现）

        Args:
            wall_distance: 可选的ADFLOW TurbulentDistance；缺失或无效时自动几何回退
            wall_segment_mask: 可选的真实壁面线段mask
        """
        if spatial_wall_layers is None:
            spatial_wall_layers = wall_layers
        if dtype is None:
            compute_dtype = torch.float64
        elif isinstance(dtype, str):
            dtype_str = dtype.lower()
            if dtype_str in {"float32", "fp32"}:
                compute_dtype = torch.float32
            elif dtype_str in {"float64", "fp64"}:
                compute_dtype = torch.float64
            else:
                raise ValueError(
                    f"Unknown dtype='{dtype}'. Expected float32/float64 (or fp32/fp64)."
                )
        elif isinstance(dtype, torch.dtype):
            compute_dtype = dtype
        else:
            raise TypeError(f"dtype must be None, str, or torch.dtype. Got {type(dtype)}")
        wall_distance_source = None
        profile_timing: Optional[Dict[str, float]] = None
        if bool(self._residual_operator_profile_enabled):
            profile_timing = {}
            self._residual_profile_sync()
            t_total0 = time.perf_counter()
        else:
            t_total0 = None
        t_stage0 = time.perf_counter() if profile_timing is not None else None
        # 0. 检测batch模式（不再squeeze，直接支持batch）
        is_batched = False
        batch_size = None

        # 检测fields是否有batch维度
        if isinstance(fields, torch.Tensor):
            if fields.ndim == 4:  # (B, C, H, W)
                is_batched = True
                batch_size = fields.shape[0]
        elif isinstance(fields, dict):
            sample_field = fields.get('rho') or fields.get('u')
            if sample_field is not None and sample_field.ndim == 3:  # (B, H, W)
                is_batched = True
                batch_size = sample_field.shape[0]

        # 检测grid是否有batch维度
        if isinstance(grid, dict):
            for key in ['vertex', 'center']:
                if key in grid and isinstance(grid[key], torch.Tensor):
                    if grid[key].ndim == 4:  # (B, 2, H+1, W+1) or (B, 2, H, W)
                        is_batched = True
                        batch_size = grid[key].shape[0]
                        break

        # 检测flow_conditions是否有batch维度
        if flow_conditions is not None and isinstance(flow_conditions, torch.Tensor):
            if flow_conditions.ndim == 2:  # (B, 3)
                is_batched = True
                batch_size = flow_conditions.shape[0]

        # 1. 解析物理场（数据集格式：[rho, u, v, p, nuTilde]）
        # 注意：第4通道是压力p，不是rhoE！数据集已验证 mean(fields[3]) ≈ 0.98
        if isinstance(fields, dict):
            rho = fields['rho'].to(dtype=compute_dtype, device=self.device)
            u = fields['u'].to(dtype=compute_dtype, device=self.device)
            v = fields['v'].to(dtype=compute_dtype, device=self.device)
            p = fields['p'].to(dtype=compute_dtype, device=self.device)
            rhoE_from_input = fields.get('rhoE', None)
            if rhoE_from_input is not None:
                rhoE_from_input = rhoE_from_input.to(dtype=compute_dtype, device=self.device)
            # SA变量（如果存在）
            nu_tilde_field = fields.get('nuTilde', None)
            if nu_tilde_field is not None:
                nu_tilde_field = nu_tilde_field.to(dtype=compute_dtype, device=self.device)
        else:
            fields = fields.to(dtype=compute_dtype, device=self.device)
            n_channels = fields.shape[0] if fields.ndim == 3 else fields.shape[1]
            rhoE_from_input = None

            # 最少需要4个原始变量
            if n_channels < 4:
                raise ValueError(
                    f"需要至少4个原始变量 [rho, u, v, p]，但输入只有 {n_channels} 个通道。"
                    f"请确保输入格式为 [rho, u, v, p] 或 [rho, u, v, p, nuTilde]"
                )

            if fields.ndim == 3:  # (C, H, W)
                rho = fields[0]
                u = fields[1]
                v = fields[2]
                p = fields[3]  # 第4通道是压力p（索引3）
                if bool(state_is_adflow_mixed):
                    if n_channels < 5:
                        raise ValueError(
                            "state_is_adflow_mixed=True expects [rho,u,v,p,rhoE] "
                            f"or [rho,u,v,p,rhoE,nuTilde], got {n_channels} channels"
                        )
                    rhoE_from_input = fields[4]
                    nu_tilde_field = fields[5] if n_channels >= 6 else None
                else:
                    nu_tilde_field = fields[4] if n_channels >= 5 else None  # 第5通道是nuTilde（索引4）
            else:  # (batch, C, H, W)
                rho = fields[:, 0]
                u = fields[:, 1]
                v = fields[:, 2]
                p = fields[:, 3]  # 第4通道是压力p
                if bool(state_is_adflow_mixed):
                    if n_channels < 5:
                        raise ValueError(
                            "state_is_adflow_mixed=True expects [rho,u,v,p,rhoE] "
                            f"or [rho,u,v,p,rhoE,nuTilde], got {n_channels} channels"
                        )
                    rhoE_from_input = fields[:, 4]
                    nu_tilde_field = fields[:, 5] if n_channels >= 6 else None
                else:
                    nu_tilde_field = fields[:, 4] if n_channels >= 5 else None  # 第5通道是nuTilde

        # Numerical safety: neural predicted nuTilde can be slightly negative. In SA, fv1 uses
        #   fv1 = chi^3 / (chi^3 + cv1^3),  chi = rho * nuTilde / mu_l
        # which is singular at chi=-cv1. With small mu_l (high Re), this can happen even for
        # tiny negative nuTilde and will produce NaN/Inf gradients (training collapse) even if
        # The residual loss weight is extremely small. Clamp nuTilde to keep eddy viscosity >= 0.
        residual_operator_disable_sa_nu_clamp = bool(getattr(self, 'residual_operator_disable_sa_nu_clamp', False))
        if (
            nu_tilde_field is not None
            and self.sa_clamp_nu_tilde
            and (not residual_operator_disable_sa_nu_clamp)
        ):
            nu_tilde_field = torch.clamp(nu_tilde_field, min=0.0)

        # 数值安全：神经网络预测可能出现极少量 rho<=0 / p<=0 的异常点。
        # 下游热力学/耗散包含非整数幂与开方（例如 T=p/rho, c^2=gamma*p/rho, 熵 p/rho^gamma），
        # 若直接使用将产生NaN并污染整批残差诊断/训练。这里用正下限裁剪避免NaN传播。
        if self.clamp_rho_p:
            rho = torch.clamp(rho, min=self.rho_floor)
            p = torch.clamp(p, min=self.p_floor)
        self._residual_profile_mark(profile_timing, "parse_fields", t_stage0)

        t_stage0 = time.perf_counter() if profile_timing is not None else None
        if rhoE_from_input is not None:
            rhoE = rhoE_from_input
        elif bool(state_is_adflow_consistent):
            # Matrix-free authority path: upstream primitive variables were
            # already reconstructed from the active conservative state.
            rhoE = p / (self.gamma - 1.0) + 0.5 * rho * (u * u + v * v)
        else:
            # Generic primitive path: keep the existing projection that aligns
            # raw primitive inputs with the state seen after injection.
            rho, u, v, p, rhoE = self._build_adflow_consistent_state(
                rho, u, v, p, compute_dtype=compute_dtype
            )
        self._residual_profile_mark(profile_timing, "adflow_consistent_state", t_stage0)

        # Early non-finite guard (clearer diagnostics than deep in flux).
        #
        # NOTE: this is for *eager* diagnostics. Under FX proxy tracing (e.g.,
        # torch.func.linearize), Python truthiness of 0-dim tensors is disallowed
        # and would break tracing; in that case, skip this guard.
        finite_ok = True
        try:
            finite_ok = bool(
                torch.isfinite(rho).all()
                and torch.isfinite(u).all()
                and torch.isfinite(v).all()
                and torch.isfinite(p).all()
                and (torch.isfinite(nu_tilde_field).all() if nu_tilde_field is not None else True)
            )
        except RuntimeError as e:
            msg = str(e)
            if "tracing tensor" in msg and "aten._local_scalar_dense" in msg:
                finite_ok = True
            else:
                raise
        if not finite_ok:
            def _count_bad(x: torch.Tensor) -> int:
                return (~torch.isfinite(x)).sum().item()

            bad_rho = _count_bad(rho)
            bad_u = _count_bad(u)
            bad_v = _count_bad(v)
            bad_p = _count_bad(p)
            raise ValueError(
                f"Non-finite detected in input fields to TorchResidualBackend: "
                f"rho={bad_rho}, u={bad_u}, v={bad_v}, p={bad_p}. "
                f"Upstream should sanitize before residual computation."
            )

        t_stage0 = time.perf_counter() if profile_timing is not None else None
        # 2. 解析坐标（支持grid dict输入）
        top_boundary_donor = None
        top_boundary_second = None

        if isinstance(grid, dict):
            # 新格式：grid = {'center': (2,H,W), 'vertex': (2,H+1,W+1)}
            if 'vertex' not in grid:
                raise ValueError(
                    "Grid requires dict with 'vertex' key for GCL-compliant geometry. "
                    f"Got keys: {list(grid.keys())}"
                )

            coords_vertex_in = grid['vertex']
            coords_center_in = grid.get('center', None)  # 必需：用于几何/残差计算
            top_boundary_donor = grid.get('top_boundary_donor', None)
            top_boundary_second = grid.get('top_boundary_second', None)
            if coords_center_in is None:
                raise ValueError(
                    "coords_center为None，无法计算残差。"
                    "PyTorch后端要求必须提供coords_center，不支持从coords_vertex自动推导。"
                    "请在调用前使用geometry模块计算coords_center。"
                )

            # 几何缓存策略：
            # - Fast cache：基于 tensor 对象身份 (id + _version) 复用 cast 后的坐标与几何，
            #              避免每次 .to(dtype) 与内容 hash 的 CPU 同步（适合 sparse-newton 构建J的重复调用）
            # - Slow cache：基于内容 hash 的安全缓存，仅在单样本/无 batch 时启用（batch>1 做内容 hash 代价太高）
            use_fast_cache = coords_vertex_in.ndim in (3, 4)
            use_slow_cache = (
                coords_vertex_in.ndim == 3
                or (coords_vertex_in.ndim == 4 and coords_vertex_in.shape[0] == 1)
            )

            # Fast cache: 复用 “float64坐标 + 几何” 以避免每次 .to(float64) 和内容hash的CPU同步。
            # Key基于tensor对象身份(id + _version)，_version变化可捕捉in-place修改。
            fast_entry = None
            cache_key_fast = None
            cached_geom = None

            if use_fast_cache:
                # Inference-mode tensors don't track _version; use 0 as fallback
                # (in-place modification is impossible in inference mode anyway).
                def _ver(t):
                    return int(t._version) if not t.is_inference() else 0
                cache_key_fast = (
                    id(coords_vertex_in),
                    _ver(coords_vertex_in),
                    tuple(coords_vertex_in.shape),
                    id(coords_center_in),
                    _ver(coords_center_in),
                    tuple(coords_center_in.shape),
                    bool(periodic_xi),
                    compute_dtype,
                )
                fast_entry = self._geometry_cache_fast.get(cache_key_fast)
                if fast_entry is not None:
                    # Guard against Python id() reuse after the original tensors are freed.
                    # Without this, a later batch may collide on (id, _version, shape) and reuse
                    # geometry computed for a different batch, causing hard shape mismatches
                    # (e.g., farfield normals computed for B=2 used with B=16).
                    src_v = fast_entry.get('src_vertex_ref')
                    src_c = fast_entry.get('src_center_ref')
                    if (
                        src_v is None
                        or src_c is None
                        or src_v() is not coords_vertex_in
                        or src_c() is not coords_center_in
                    ):
                        # Stale entry: drop and treat as cache miss.
                        self._geometry_cache_fast.pop(cache_key_fast, None)
                        fast_entry = None

                if fast_entry is not None:
                    # Mark as recently used
                    self._geometry_cache_fast.move_to_end(cache_key_fast, last=True)
                    coords_vertex = fast_entry['coords_vertex']
                    coords_center = fast_entry['coords_center']
                    cached_geom = fast_entry['geom']
                else:
                    # Only cast on fast-miss; store casted tensors for reuse
                    coords_vertex = coords_vertex_in.to(dtype=compute_dtype, device=self.device)
                    coords_center = coords_center_in.to(dtype=compute_dtype, device=self.device)
            else:
                coords_vertex = coords_vertex_in.to(dtype=compute_dtype, device=self.device)
                coords_center = coords_center_in.to(dtype=compute_dtype, device=self.device)

            # 解包节点坐标
            if coords_vertex.ndim == 3:  # (2, H+1, W+1)
                xv, yv = coords_vertex[0], coords_vertex[1]
            else:  # (batch, 2, H+1, W+1)
                xv, yv = coords_vertex[:, 0], coords_vertex[:, 1]

            # 3. 计算几何（使用ADflow corner-based方法，完全统一）
            # ✨ 优化：使用缓存避免重复计算（对 batch>1 也启用 fast cache）

            # 使用外部提供的coords_center
            coords_center_for_geom = coords_center

            if use_fast_cache:
                if cached_geom is None:
                    if use_slow_cache:
                        # Slow-but-safe: 内容hash（需要CPU同步），仅在fast miss时走一次
                        _cv = coords_vertex.squeeze(0) if coords_vertex.ndim == 4 else coords_vertex
                        _cc = (
                            coords_center_for_geom.squeeze(0)
                            if coords_center_for_geom.ndim == 4
                            else coords_center_for_geom
                        )
                        vertex_hash = _tensor_hash(_cv)
                        center_hash = _tensor_hash(_cc)
                        cache_key = (vertex_hash, center_hash, bool(periodic_xi), compute_dtype)
                        cached_geom = self._geometry_cache.get(cache_key)
                    else:
                        cache_key = None

                    if cached_geom is None:
                        # 缓存未命中，重新计算
                        vol, sign = self._geom.compute_cell_volume_adflow(
                            coords_vertex, periodic_xi=periodic_xi
                        )
                        face_geom = self._geom.compute_face_area_vectors_full(
                            coords_center_for_geom,
                            coords_vertex,
                            periodic_xi=periodic_xi,
                            sign=sign,
                        )
                        x, y = self._geom.compute_cell_centers_from_vertex(
                            xv, yv, periodic_xi=periodic_xi
                        )
                        halo_geom = self._build_halo_geometry_cache(
                            coords_vertex=coords_vertex,
                            face_geom=face_geom,
                            periodic_xi=periodic_xi,
                            sign=sign,
                        )
                        cached_geom = {
                            'vol': vol,
                            'sign': sign,
                            'face_geom': face_geom,
                            'x': x,
                            'y': y,
                            'halo_geom': halo_geom,
                        }
                        if use_slow_cache and cache_key is not None:
                            self._geometry_cache[cache_key] = cached_geom

                    # 写入fast cache：复用cast后的coords与几何
                    assert cache_key_fast is not None
                    # ⚠️ Important: bound the fast cache to avoid GPU memory growth.
                    #
                    # fast_cache keys are based on tensor identity (id + _version). In a
                    # typical training loop, each batch produces new tensor objects, so
                    # keeping all entries would retain large GPU tensors indefinitely
                    # (coords + geometry), eventually causing OOM.
                    #
                    # Use a small LRU cache:
                    # - prevents unbounded growth during training (new batch -> new key),
                    # - still allows sparse-newton/JVP micro-batching to reuse a couple of
                    #   common batch shapes without thrashing (e.g., B=1 and B=4).
                    self._geometry_cache_fast[cache_key_fast] = {
                        'coords_vertex': coords_vertex,
                        'coords_center': coords_center_for_geom,
                        'geom': cached_geom,
                        'src_vertex_ref': weakref.ref(coords_vertex_in),
                        'src_center_ref': weakref.ref(coords_center_in),
                    }
                    self._geometry_cache_fast.move_to_end(cache_key_fast, last=True)
                    while len(self._geometry_cache_fast) > int(self._geometry_fast_cache_max_entries):
                        self._geometry_cache_fast.popitem(last=False)

                vol = cached_geom['vol']
                sign = cached_geom['sign']
                face_geom = cached_geom['face_geom']
                x = cached_geom['x']
                y = cached_geom['y']
                halo_geom = cached_geom.get('halo_geom')
            else:
                vol, sign = self._geom.compute_cell_volume_adflow(
                    coords_vertex, periodic_xi=periodic_xi
                )
                face_geom = self._geom.compute_face_area_vectors_full(
                    coords_center_for_geom, coords_vertex, periodic_xi=periodic_xi, sign=sign
                )
                x, y = self._geom.compute_cell_centers_from_vertex(xv, yv, periodic_xi=periodic_xi)
                halo_geom = self._build_halo_geometry_cache(
                    coords_vertex=coords_vertex,
                    face_geom=face_geom,
                    periodic_xi=periodic_xi,
                    sign=sign,
                )

            # ========== DIAGNOSIS: η面几何验证 ==========
            if self.debug:
                print(f"\n[DIAGNOSIS torch residual backend] η面几何检查:")
                print(f"  coords_center_for_geom shape: {coords_center_for_geom.shape}")
                H, W = coords_center_for_geom.shape[-2], coords_center_for_geom.shape[-1]
                print(f"  Grid: H={H}, W={W}")
                print(f"  face_geom['A_x_eta'] shape: {face_geom['A_x_eta'].shape}")
                print(f"  预期η面形状: ({H+1}, {W}) - 包含边界")
                s_area_eta = torch.sqrt(face_geom['A_x_eta']**2 + face_geom['A_y_eta']**2)
                print(f"  η面面积向量模长: mean={s_area_eta.mean():.6e}")
                print(f"  预期: 约0.132（与ADFlow |sK|一致）")
            # ========== END DIAGNOSIS ==========

            # cell-center 坐标在 cached_geom 中已复用；非缓存路径在上方已计算。

        else:
            # 旧格式不支持（不保向后兼容）
            raise TypeError(
                "Grid requires dict with 'vertex' (and optionally 'center') keys. "
                f"Got type: {type(grid)}. "
                "Please update your code to pass grid={{'vertex': coords_vertex, 'center': coords_center}}."
            )

        self._residual_profile_mark(profile_timing, "geometry", t_stage0)
        t_stage0 = time.perf_counter() if profile_timing is not None else None
        # 4. 提取Ma/AoA（用于边界条件和粘度计算）
        # 注意：为支持ADFLOW 6-point梯度方法，需要提前提取Ma/AoA用于halo计算
        Ma_val = None
        AoA_val = None
        Re_val = None
        flow_conditions_tensor = None
        if flow_conditions is not None:
            if isinstance(flow_conditions, dict):
                Ma_val = flow_conditions.get('Ma', None)
                AoA_val = flow_conditions.get('AoA', None)
                Re_val = flow_conditions.get('Re', None)
            else:
                # Tensor形式（支持批量和单样本）
                flow_conditions_tensor = flow_conditions.to(dtype=compute_dtype, device=self.device)
                if flow_conditions_tensor.ndim == 1:
                    # 单样本: (3,) -> 提取tensor（保持维度一致性）
                    Ma_val = flow_conditions_tensor[0]  # 标量tensor
                    AoA_val = flow_conditions_tensor[1]  # 标量tensor
                    Re_val = flow_conditions_tensor[2]  # 标量tensor
                elif flow_conditions_tensor.ndim == 2:
                    # 批量: (B, 3) -> 保持张量形式 (B,)，用于向量化
                    Ma_val = flow_conditions_tensor[:, 0]  # (B,)
                    AoA_val = flow_conditions_tensor[:, 1]  # (B,)
                    Re_val = flow_conditions_tensor[:, 2]  # (B,)
                else:
                    raise ValueError(f"Unexpected flow_conditions shape: {flow_conditions_tensor.shape}")

        # 4.5 创建halo层（仅在需要时，用于ADFLOW 6-point梯度、粘性通量或耗散计算）
        # 为梯度计算、通量计算和耗散提供边界条件
        halo_wall = None
        halo_farfield = None
        halo_wall_second = None
        halo_farfield_second = None
        if (self.gradient_method == 'adflow_6pt' or
            self.viscosity_source != 'none' or
            self.dissipation_mode != 'none'):
            from surrogate.physics.pde import halo as halo_module

            # SA湍流方程：若存在nuTilde，则halo必须包含第6通道（用于bcTurbFarfield + turb2ndHalo）
            include_sa_in_halo = (nu_tilde_field is not None) and (self.viscosity_source == 'laminar+SA') and (not self.compute_only_momentum)

            # 壁面halo：从第一层物理单元(j=0)推导
            # Plan96: 同时提取第二层(j=1)用于压力线性外推
            # ✅ 5通道: [rho, u, v, p, rhoE]（Euler+能量）
            # ✅ 6通道: [rho, u, v, p, rhoE, nuTilde]（RANS+SA，ADFLOW标准格式）
            second_idx = 1 if int(rho.shape[-2]) > 1 else 0
            if include_sa_in_halo:
                fields_wall = torch.stack([
                    rho[..., 0, :], u[..., 0, :], v[..., 0, :], p[..., 0, :], rhoE[..., 0, :], nu_tilde_field[..., 0, :]
                ], dim=-2)
                fields_second = torch.stack([
                    rho[..., second_idx, :], u[..., second_idx, :], v[..., second_idx, :], p[..., second_idx, :], rhoE[..., second_idx, :], nu_tilde_field[..., second_idx, :]
                ], dim=-2)
            else:
                fields_wall = torch.stack([
                    rho[..., 0, :], u[..., 0, :], v[..., 0, :], p[..., 0, :], rhoE[..., 0, :]
                ], dim=-2)
                fields_second = torch.stack([
                    rho[..., second_idx, :], u[..., second_idx, :], v[..., second_idx, :], p[..., second_idx, :], rhoE[..., second_idx, :]
                ], dim=-2)
            if fields_wall.ndim == 2:  # (C, W)
                fields_wall_for_halo = fields_wall
                fields_second_for_halo = fields_second
            else:  # (B, C, W)
                fields_wall_for_halo = fields_wall
                fields_second_for_halo = fields_second

            halo_wall = halo_module.apply_wall_bc(
                fields_wall=fields_wall_for_halo,
                fields_second_layer=fields_second_for_halo,  # 用于压力外推（仅linear_extrapolation需要）
                slip_velocity=None,  # no-slip壁面
                wall_pressure_treatment='constant_pressure',  # 对齐ADFlow默认（BCRoutines.F90:554）
                gamma=self.gamma
            )

            if top_boundary_donor is not None:
                halo_farfield = top_boundary_donor.to(dtype=compute_dtype, device=self.device)
                if top_boundary_second is not None:
                    halo_farfield_second = top_boundary_second.to(
                        dtype=compute_dtype,
                        device=self.device,
                    )
                elif include_sa_in_halo:
                    halo_farfield_second = halo_module.apply_farfield_bc_second_halo(halo_farfield)
            else:
                # 远场halo：从最后一层物理单元(j=H-1)推导
                # ✅ 5通道: [rho, u, v, p, rhoE]（Euler+能量）
                # ✅ 6通道: [rho, u, v, p, rhoE, nuTilde]（RANS+SA）
                if include_sa_in_halo:
                    fields_farfield = torch.stack([
                        rho[..., -1, :], u[..., -1, :], v[..., -1, :], p[..., -1, :], rhoE[..., -1, :], nu_tilde_field[..., -1, :]
                    ], dim=-2)
                else:
                    fields_farfield = torch.stack([
                        rho[..., -1, :], u[..., -1, :], v[..., -1, :], p[..., -1, :], rhoE[..., -1, :]
                    ], dim=-2)
                # ADflow bcFarfield 需要远场边界面单位法向量（由η顶边界面面积向量归一化得到）
                A_x_eta = face_geom['A_x_eta']
                A_y_eta = face_geom['A_y_eta']
                if A_x_eta.ndim == 2:
                    A_x_top = A_x_eta[-1, :]  # (W,)
                    A_y_top = A_y_eta[-1, :]
                    A_mag_top = torch.sqrt(A_x_top**2 + A_y_top**2) + 1e-30
                    normal_farfield = torch.stack([A_x_top / A_mag_top, A_y_top / A_mag_top], dim=0)  # (2, W)
                else:
                    # (B, H+1, W)
                    A_x_top = A_x_eta[:, -1, :]  # (B, W)
                    A_y_top = A_y_eta[:, -1, :]
                    A_mag_top = torch.sqrt(A_x_top**2 + A_y_top**2) + 1e-30
                    normal_farfield = torch.stack([A_x_top / A_mag_top, A_y_top / A_mag_top], dim=1)  # (B, 2, W)

                # SA自由流nuTilde（wInf(itu1)）：通过saNuKnownEddyRatio求解，保持与ADFLOW一致
                nuTilde_inf = None
                if include_sa_in_halo and (Ma_val is not None) and (Re_val is not None):
                    import math
                    from surrogate.physics.pde.sa_utils import compute_sa_nuTilde_inf

                    sqrt_gamma = math.sqrt(self.gamma)
                    if isinstance(Ma_val, torch.Tensor):
                        Ma_t = Ma_val.to(device=self.device, dtype=compute_dtype)
                    else:
                        Ma_t = torch.tensor(float(Ma_val), device=self.device, dtype=compute_dtype)

                    if isinstance(Re_val, torch.Tensor):
                        Re_t = Re_val.to(device=self.device, dtype=compute_dtype)
                    else:
                        Re_t = torch.tensor(float(Re_val), device=self.device, dtype=compute_dtype)

                    nuLam = (Ma_t * sqrt_gamma) / (Re_t + 1e-30)
                    chi_inf = getattr(self, "_sa_chi_inf", None)
                    if chi_inf is None:
                        chi_inf = compute_sa_nuTilde_inf(
                            eddyVisInfRatio=float(self.eddyVisInfRatio),
                            nuLam=1.0,
                            cv1=7.1,
                        )
                        setattr(self, "_sa_chi_inf", float(chi_inf))
                    chi_inf_f = float(getattr(self, "_sa_chi_inf"))
                    nuTilde_inf = torch.where(
                        nuLam > 0.0,
                        nuLam * chi_inf_f,
                        torch.zeros_like(nuLam),
                    )
                halo_farfield = halo_module.apply_farfield_bc(
                    fields_farfield=fields_farfield,
                    normal=normal_farfield,
                    Ma=Ma_val,
                    AoA=AoA_val,
                    gamma=self.gamma,
                    nuTilde_inf=nuTilde_inf,
                    Re=Re_val
                )

                # SA二阶格式需要第二层halo（ADFLOW turb2ndHalo：常值外推）
                if include_sa_in_halo:
                    halo_farfield_second = halo_module.apply_farfield_bc_second_halo(halo_farfield)

            if include_sa_in_halo:
                halo_wall_second = halo_module.apply_wall_bc_second_halo(halo_wall)

        self._residual_profile_mark(profile_timing, "flow_bc_halo", t_stage0)
        t_stage0 = time.perf_counter() if profile_timing is not None else None
        # 5. 计算物理导数（统一梯度接口，支持多种方法）
        # **Plan86双梯度对齐**：cell-center（耗散/湍流）+ nodal（粘性通量）
        # 5.1 创建cell-center梯度计算器（用于耗散、湍流模型）
        gradient_calc_cell = self._get_gradient_calculator(
            method=self.gradient_method,
            periodic_xi=periodic_xi,
            dtype=compute_dtype
        )

        # 准备几何参数（根据梯度方法选择）
        if self.gradient_method in ['adflow_6pt', 'adflow', 'six_point']:
            # ADFLOW 6-point需要面向量和体积
            geometry_params_cell = {
                'volumes': vol,
                'si_x': face_geom['A_x_xi'],
                'si_y': face_geom['A_y_xi'],
                'sj_x': face_geom['A_x_eta'],
                'sj_y': face_geom['A_y_eta']
            }

            # 提取u速度的边界条件
            if halo_wall is not None and halo_farfield is not None:
                bc_u = {
                    'halo_eta_bottom': halo_wall[1, :] if halo_wall.ndim == 2 else halo_wall[:, 1, :],
                    'halo_eta_top': halo_farfield[1, :] if halo_farfield.ndim == 2 else halo_farfield[:, 1, :]
                }
                bc_v = {
                    'halo_eta_bottom': halo_wall[2, :] if halo_wall.ndim == 2 else halo_wall[:, 2, :],
                    'halo_eta_top': halo_farfield[2, :] if halo_farfield.ndim == 2 else halo_farfield[:, 2, :]
                }

                if halo_geom is None:
                    halo_geom = self._build_halo_geometry_cache(
                        coords_vertex=coords_vertex,
                        face_geom=face_geom,
                        periodic_xi=periodic_xi,
                        sign=sign,
                    )

                # 将 halo 几何参数添加到边界条件字典（bc_u 和 bc_v 共享相同的几何）
                halo_geom_params = {
                    'halo_vol_wall': halo_geom['halo_vol_wall'],
                    'halo_vol_ff': halo_geom['halo_vol_ff'],
                    'si_x_halo_wall': halo_geom['si_x_halo_wall'],
                    'si_y_halo_wall': halo_geom['si_y_halo_wall'],
                    'si_x_halo_ff': halo_geom['si_x_halo_ff'],
                    'si_y_halo_ff': halo_geom['si_y_halo_ff'],
                    # Plan96: 完整η面向量
                    'sj_x_hat': halo_geom['sj_x_hat'],
                    'sj_y_hat': halo_geom['sj_y_hat']
                }
                bc_u.update(halo_geom_params)
                bc_v.update(halo_geom_params)
            else:
                raise ValueError(
                    "gradient_method='adflow_6pt' requires halo boundary conditions, "
                    "but halo values were not computed. This should not happen."
                )
        else:
            # Green-Gauss只需要面几何和体积
            geometry_params_cell = {
                'volumes': vol,
                'face_geom': face_geom
            }
            bc_u = None
            bc_v = None

        # 5.2 计算cell-center速度梯度（用于耗散等）
        if self.gradient_method in ['adflow_6pt', 'adflow', 'six_point']:
            cell_velocity_grads = self._compute_multi_field_gradients(
                gradient_calculator=gradient_calc_cell,
                fields=[u, v],
                geometry=geometry_params_cell,
                boundary_conditions=[bc_u, bc_v],
                boundary_batch_ndims={
                    'halo_eta_bottom': 2,
                    'halo_eta_top': 2,
                },
            )
            (du_dx, du_dy), (dv_dx, dv_dy) = cell_velocity_grads
        else:
            du_dx, du_dy = gradient_calc_cell.compute_gradient(u, geometry_params_cell, bc_u)
            dv_dx, dv_dy = gradient_calc_cell.compute_gradient(v, geometry_params_cell, bc_v)

        # 5.3 创建nodal梯度计算器（用于粘性通量）
        gradient_calc_nodal = self._get_gradient_calculator(
            method='nodal',  # ADFLOW节点梯度
            periodic_xi=periodic_xi,
            dtype=compute_dtype
        )

        # 准备nodal梯度几何参数（需要面向量和体积）
        geometry_params_nodal = {
            'volumes': vol,
            'si_x': face_geom['A_x_xi'],
            'si_y': face_geom['A_y_xi'],
            'sj_x': face_geom['A_x_eta'],
            'sj_y': face_geom['A_y_eta']
        }

        nodal_boundary_batch_ndims = {
            'halo_eta_bottom': 2,
            'halo_eta_top': 2,
            'halo_vol_wall': 2,
            'halo_vol_ff': 2,
            'si_x_halo_wall': 2,
            'si_y_halo_wall': 2,
            'si_x_halo_ff': 2,
            'si_y_halo_ff': 2,
            'sj_x_hat': 3,
            'sj_y_hat': 3,
        }

        # 5.4 计算nodal速度梯度（用于粘性通量）
        # 必须传递bc_u/bc_v以获得正确的近壁梯度（与ADflow对齐）
        nodal_velocity_grads = self._compute_multi_field_gradients(
            gradient_calculator=gradient_calc_nodal,
            fields=[u, v],
            geometry=geometry_params_nodal,
            boundary_conditions=[bc_u, bc_v],
            boundary_batch_ndims=nodal_boundary_batch_ndims,
        )
        (du_dx_node, du_dy_node), (dv_dx_node, dv_dy_node) = nodal_velocity_grads

        import os
        if os.environ.get('SURROGATE_DEBUG_GRADIENT', '') == '1':
            def _to_numpy(arr: torch.Tensor) -> np.ndarray:
                if arr.ndim == 3:
                    arr = arr[0]
                return arr.detach().cpu().numpy()

            np.savez(
                'pytorch_nodal_gradient_debug.npz',
                nodal_ux=_to_numpy(du_dx_node),
                nodal_uy=_to_numpy(du_dy_node),
                nodal_vx=_to_numpy(dv_dx_node),
                nodal_vy=_to_numpy(dv_dy_node),
            )
            print("[DEBUG] Saved nodal gradients to: pytorch_nodal_gradient_debug.npz")

        # 5.5 计算压力和密度的节点梯度（用于能量粘性通量）
        # 仅在需要能量方程粘性通量时计算
        dp_dx_node, dp_dy_node = None, None
        drho_dx_node, drho_dy_node = None, None
        daa_dx_node, daa_dy_node = None, None

        # ✅ ADFLOW对齐：rhoE已在前面统一提取，这里直接判断是否需要计算能量粘性通量
        # 通道4是rhoE（索引3），已在第412/418行提取
        has_energy_equation = n_channels >= 4 and not self.compute_only_momentum  # 至少4通道且未禁用能量方程

        if has_energy_equation and self.viscosity_source != 'none':
            # rhoE已在前面提取，无需重复提取
            # 确保rhoE有正确的维度（需要channel维度用于某些操作）
            if rhoE.ndim == 2:  # (H, W) -> (1, H, W)
                rhoE_with_channel = rhoE.unsqueeze(0)
            elif rhoE.ndim == 3 and rhoE.shape[0] != 1:  # (batch, H, W) -> (batch, 1, H, W)
                rhoE_with_channel = rhoE.unsqueeze(1)
            else:
                rhoE_with_channel = rhoE

            # 压力边界条件（与速度边界条件类似）
            bc_p = {
                'bc_type_wall': 'neumann',  # 壁面：Neumann BC（梯度=0）
                'bc_type_farfield': 'dirichlet',  # 远场：自由流压力
                'p_farfield': 1.0,  # 无量纲压力=1
            }

            # 密度边界条件
            bc_rho = {
                'bc_type_wall': 'neumann',  # 壁面：Neumann BC
                'bc_type_farfield': 'dirichlet',  # 远场：自由流密度
                'rho_farfield': 1.0,  # 无量纲密度=1
            }

            # 添加halo值（节点梯度计算器需要）
            if halo_wall is not None and halo_farfield is not None:
                bc_p['halo_eta_bottom'] = halo_wall[3, :] if halo_wall.ndim == 2 else halo_wall[:, 3, :]
                bc_p['halo_eta_top'] = halo_farfield[3, :] if halo_farfield.ndim == 2 else halo_farfield[:, 3, :]
                bc_rho['halo_eta_bottom'] = halo_wall[0, :] if halo_wall.ndim == 2 else halo_wall[:, 0, :]
                bc_rho['halo_eta_top'] = halo_farfield[0, :] if halo_farfield.ndim == 2 else halo_farfield[:, 0, :]

                # 添加halo几何参数（与bc_u/bc_v共享）
                bc_p.update(halo_geom_params)
                bc_rho.update(halo_geom_params)

            # ✅ 5.6 计算 q̂ = -grad(a²)（ADFLOW对齐：直接对aa使用nodal梯度算子）
            # ADFLOW方法（flowUtils.F90: allNodalGradients）：qx/qy 定义为 -∇(a²)
            # 不使用链式法则 grad(a²) = 2a*grad(a)

            # 1. 在单元中心计算声速平方
            # ADFLOW对齐：不要在rho分母中加入epsilon（会在近壁小Δx下放大到q̂与Δaa*inv_d里）
            aa = self.gamma * p / rho  # (H, W) 或 (batch, H, W)

            # 2. 声速平方的边界条件
            bc_aa = {
                'bc_type_wall': 'neumann',  # 壁面：grad(aa) = 0（绝热壁）
                'bc_type_farfield': 'dirichlet',  # 远场：aa = gamma * pInf / rhoInf
                'aa_farfield': self.gamma * 1.0 / 1.0,  # 无量纲远场值 = gamma
            }

            # 3. 添加halo值（如果存在）
            if halo_wall is not None and halo_farfield is not None:
                # 从halo计算aa = gamma * p_halo / rho_halo
                p_wall_halo = halo_wall[3, :] if halo_wall.ndim == 2 else halo_wall[:, 3, :]
                rho_wall_halo = halo_wall[0, :] if halo_wall.ndim == 2 else halo_wall[:, 0, :]
                aa_wall_halo = self.gamma * p_wall_halo / rho_wall_halo

                p_far_halo = halo_farfield[3, :] if halo_farfield.ndim == 2 else halo_farfield[:, 3, :]
                rho_far_halo = halo_farfield[0, :] if halo_farfield.ndim == 2 else halo_farfield[:, 0, :]
                aa_far_halo = self.gamma * p_far_halo / rho_far_halo

                bc_aa['halo_eta_bottom'] = aa_wall_halo
                bc_aa['halo_eta_top'] = aa_far_halo

                # 共享halo几何参数
                bc_aa.update(halo_geom_params)

            # 计算压力、密度和声速平方的节点梯度
            nodal_scalar_grads = self._compute_multi_field_gradients(
                gradient_calculator=gradient_calc_nodal,
                fields=[p, rho, aa],
                geometry=geometry_params_nodal,
                boundary_conditions=[bc_p, bc_rho, bc_aa],
                boundary_batch_ndims=nodal_boundary_batch_ndims,
            )
            (dp_dx_node, dp_dy_node), (drho_dx_node, drho_dy_node), (daa_dx_node, daa_dy_node) = nodal_scalar_grads

            # ✅ ADFLOW对齐：allNodalGradients中 qx/qy 定义为 -∇(a²)（见 flowUtils.F90 注释）
            # 这里将梯度取负，使 daa_*_node 与 ADFLOW 的 qx/qy 语义一致，供能量方程热通量计算使用。
            daa_dx_node = -daa_dx_node
            daa_dy_node = -daa_dy_node

        # 5.7 SA湍流方程支持（Phase 4）
        # ✅ ADFLOW对齐：nuTilde是第5个变量（索引4），已在前面提取为nu_tilde_field
        has_sa_equation = (nu_tilde_field is not None) and not self.compute_only_momentum  # 有nuTilde且未禁用SA方程

        dnu_dx, dnu_dy = None, None
        vort_mag = None
        contravariant_vel = None
        d_wall = None
        halo_nuTilde_wall = None
        halo_nuTilde_wall_second = None
        halo_nuTilde_ff = None
        halo_nuTilde_ff_second = None
        halo_rho_wall = None
        halo_rho_ff = None
        halo_p_wall = None
        halo_p_ff = None
        halo_vol_wall_sa = None
        halo_vol_ff_sa = None

        if has_sa_equation and self.viscosity_source == 'laminar+SA':
            # 使用前面提取的nu_tilde_field，确保正确的维度
            nuTilde = nu_tilde_field

            # 去除通道维度（如果有的话）
            if nuTilde.ndim == 3 and nuTilde.shape[0] == 1:
                nuTilde = nuTilde.squeeze(0)  # (H, W)
            elif nuTilde.ndim == 4 and nuTilde.shape[1] == 1:
                nuTilde = nuTilde.squeeze(1)  # (batch, H, W)

            # ✅ ADFLOW对齐：nuTilde halo 使用与流场相同的BC构造路径
            # - 壁面：turbBCRoutines.F90 (bmt=1,bvt=0) → halo = -nuTilde
            # - 远场：bcTurbFarfield 基于自由流方向dot判断入/出流
            if halo_wall is None or halo_farfield is None:
                raise ValueError("SA requires halo_wall and halo_farfield to be computed (ADFLOW bc alignment).")

            # halo必须包含nuTilde通道（第6通道，索引5）
            if halo_wall.ndim == 2:
                if halo_wall.shape[0] < 6 or halo_farfield.shape[0] < 6:
                    raise ValueError(
                        f"SA requires 6-channel halos [rho,u,v,p,rhoE,nuTilde], got "
                        f"halo_wall C={halo_wall.shape[0]}, halo_farfield C={halo_farfield.shape[0]}"
                    )
                halo_nuTilde_wall = halo_wall[5, :]
                halo_nuTilde_ff = halo_farfield[5, :]
                halo_rho_wall = halo_wall[0, :]
                halo_rho_ff = halo_farfield[0, :]
                halo_p_wall = halo_wall[3, :]
                halo_p_ff = halo_farfield[3, :]
                if halo_wall_second is not None and halo_farfield_second is not None:
                    halo_nuTilde_wall_second = halo_wall_second[5, :]
                    halo_nuTilde_ff_second = halo_farfield_second[5, :]
                else:
                    # turb2ndHalo: constant extrapolation
                    halo_nuTilde_wall_second = halo_nuTilde_wall
                    halo_nuTilde_ff_second = halo_nuTilde_ff
            else:
                if halo_wall.shape[1] < 6 or halo_farfield.shape[1] < 6:
                    raise ValueError(
                        f"SA requires 6-channel halos [rho,u,v,p,rhoE,nuTilde], got "
                        f"halo_wall C={halo_wall.shape[1]}, halo_farfield C={halo_farfield.shape[1]}"
                    )
                halo_nuTilde_wall = halo_wall[:, 5, :]
                halo_nuTilde_ff = halo_farfield[:, 5, :]
                halo_rho_wall = halo_wall[:, 0, :]
                halo_rho_ff = halo_farfield[:, 0, :]
                halo_p_wall = halo_wall[:, 3, :]
                halo_p_ff = halo_farfield[:, 3, :]
                if halo_wall_second is not None and halo_farfield_second is not None:
                    halo_nuTilde_wall_second = halo_wall_second[:, 5, :]
                    halo_nuTilde_ff_second = halo_farfield_second[:, 5, :]
                else:
                    halo_nuTilde_wall_second = halo_nuTilde_wall
                    halo_nuTilde_ff_second = halo_nuTilde_ff

            # saViscous 需要halo体积（Plan91/Plan96提供的几何外推体积）
            if isinstance(bc_u, dict) and 'halo_vol_wall' in bc_u and 'halo_vol_ff' in bc_u:
                halo_vol_wall_sa = bc_u['halo_vol_wall']
                halo_vol_ff_sa = bc_u['halo_vol_ff']

            # 计算nuTilde梯度（cell-center，用于SA粘性项）
            # 注意：SA使用cell-center梯度，不是nodal梯度
            bc_nu = {
                'halo_eta_bottom': halo_nuTilde_wall,
                'halo_eta_top': halo_nuTilde_ff,
            }
            dnu_dx, dnu_dy = gradient_calc_cell.compute_gradient(
                nuTilde, geometry_params_cell, bc_nu
            )

            # 计算涡量模（SA源项需要）
            from surrogate.physics.pde.sa_utils import compute_vorticity_magnitude
            vort_mag = compute_vorticity_magnitude(du_dx, du_dy, dv_dx, dv_dy)

            # 计算逆变速度（SA对流项需要）
            from surrogate.physics.pde.sa_utils import compute_contravariant_velocities_adflow
            contravariant_vel = compute_contravariant_velocities_adflow(u, v, face_geom, vol)

            # Prefer a supplied ADFLOW TurbulentDistance. Missing, non-finite,
            # non-positive, or per-sample placeholder fields are replaced by
            # the batched exact point-to-wall-segment Torch geometry kernel.
            from surrogate.physics.pde.wall_distance import resolve_wall_distance_torch

            cached_wall_distance = (
                cached_geom.get('wall_distance_torch')
                if wall_distance is None and cached_geom is not None
                else None
            )
            if cached_wall_distance is not None:
                d_wall = cached_wall_distance
                wall_distance_source = 'torch_nearest_wall_segment'
            else:
                d_wall, wall_distance_source = resolve_wall_distance_torch(
                    coords_center=coords_center,
                    coords_vertex=coords_vertex,
                    wall_distance=wall_distance,
                    wall_segment_mask=wall_segment_mask,
                    compute_dtype=compute_dtype,
                )
                if (
                    cached_geom is not None
                    and wall_distance_source == 'torch_nearest_wall_segment'
                ):
                    cached_geom['wall_distance_torch'] = d_wall
            d_wall = d_wall.to(dtype=compute_dtype, device=self.device)
            # 维度处理：确保与fields一致
            if d_wall.ndim == 3 and d_wall.shape[0] == 1:  # (1, H, W)
                d_wall = d_wall.squeeze(0)  # (H, W)

            # 自动裁剪 d_wall 以匹配 fields 的 H 维度（wall_layers 裁剪兼容）
            # fields: (B, C, H, W) 或 (C, H, W)，d_wall: (B, H_full, W) 或 (H_full, W)
            fields_H = rho.shape[-2]  # fields 的 H 维度
            d_wall_H = d_wall.shape[-2]  # d_wall 的 H 维度
            if d_wall_H > fields_H:
                # 裁剪 d_wall 到近壁区 [:fields_H, :]
                if d_wall.ndim == 2:  # (H, W)
                    d_wall = d_wall[:fields_H, :]
                elif d_wall.ndim == 3:  # (B, H, W)
                    d_wall = d_wall[:, :fields_H, :]

        self._residual_profile_mark(profile_timing, "gradients_and_sa_prereq", t_stage0)
        t_stage0 = time.perf_counter() if profile_timing is not None else None
        # 5. 计算有效粘度（与ADflow完全对齐）
        mu_eff = None
        if self.viscosity_source != 'none':
            if flow_conditions is None:
                raise ValueError(
                    f"viscosity_source='{self.viscosity_source}' requires flow_conditions. "
                    "Please provide flow_conditions with Ma, AoA, Re."
                )

            # 提取Ma和Re（支持批量和单样本的向量化处理）
            if isinstance(flow_conditions, dict):
                Ma = flow_conditions['Ma']
                Re = flow_conditions['Re']
            else:
                if flow_conditions_tensor is None:
                    flow_conditions_tensor = flow_conditions.to(dtype=compute_dtype, device=self.device)
                if flow_conditions_tensor.ndim == 1:
                    # 单样本: (3,) -> Ma, AoA, Re
                    Ma = flow_conditions_tensor[0]
                    Re = flow_conditions_tensor[2]
                elif flow_conditions_tensor.ndim == 2:
                    # 批量: (B, 3) -> Ma: (B,), Re: (B,)
                    Ma = flow_conditions_tensor[:, 0]  # (B,)
                    Re = flow_conditions_tensor[:, 2]  # (B,)
                else:
                    raise ValueError(f"Unexpected flow_conditions shape: {flow_conditions_tensor.shape}")

            if self.viscosity_source == 'laminar+SA':
                # 与ADflow完全一致的SA涡粘计算（向量化支持批量）

                # Step 1: 计算自由流参考粘度 muInf（无量纲）
                # ADflow初始化: muInf = (Ma × sqrt(gamma)) / Re
                # 随后 computeLamViscosity 用 Sutherland(T) 生成单元层流粘度场 rlv(T)
                sqrt_gamma = torch.sqrt(
                    torch.tensor(self.gamma, device=self.device, dtype=compute_dtype)
                )
                mu_inf = (Ma * sqrt_gamma) / Re  # 标量或(B,)

                # Step 2: 由无量纲温度 T = p / (RGas * rho) 计算 Sutherland 层流粘度场
                # 在本数据集的无量纲口径下 RGas = 1，因此 T = p / rho。
                # 使用相对自由流形式：
                #   mu_lam = muInf * T^(3/2) * (1 + S') / (T + S')
                # 其中 S' = SSuthDim / TInfDim，且 TInfDim = 300 K（state injection / AeroProblem）
                SSUTH_DIM = 110.55
                TINF_DIM = 300.0
                ssuth = SSUTH_DIM / TINF_DIM

                rho_safe = torch.clamp(rho, min=1e-12)
                T_ratio = torch.clamp(p / rho_safe, min=1e-12)
                suth_factor = torch.pow(T_ratio, 1.5) * (1.0 + ssuth) / (T_ratio + ssuth)

                if isinstance(mu_inf, torch.Tensor) and mu_inf.ndim == 1:
                    mu_inf_expanded = mu_inf[:, None, None]
                else:
                    mu_inf_expanded = mu_inf

                mu_lam = mu_inf_expanded * suth_factor

                # Step 3: 读取SA变量 ν~（无量纲）
                if nu_tilde_field is None:
                    raise ValueError(
                        "viscosity_source='laminar+SA' requires SA nu_tilde field (5th channel). "
                        f"Got fields shape: {fields.shape if not isinstance(fields, dict) else 'dict without nu_tilde'}"
                    )

                nu_tilde = nu_tilde_field  # ν~（无量纲修正涡粘度）, shape: (H, W) or (B, H, W)

                # Step 4: 计算SA涡粘度（ADflow公式）
                # χ = (ρ × ν~) / μ_lam
                chi = (rho * nu_tilde) / (mu_lam + 1e-30)

                # fv1 = χ³ / (χ³ + cv1³)，其中cv1=7.1
                cv1 = 7.1
                chi3 = chi ** 3
                cv1_3 = cv1 ** 3
                fv1 = chi3 / (chi3 + cv1_3)

                # μ_turb = fv1 × ρ × ν~
                mu_turb = fv1 * rho * nu_tilde

                # Step 5: 有效粘度（返回分离的组分以支持壁面边界处理）
                # ADflow: μ_eff = μ_lam + μ_turb
                # mu_turb: (H, W) or (B, H, W), mu_lam: (H, W) or (B, H, W)
                # **方案B修复**：返回字典以便fluxes.py在壁面边界仅使用mu_lam（对齐ADflow BCRoutines.F90:544）
                mu_eff = {
                    'mu_inf': mu_inf,                # 自由流参考粘度（供SA halo构造使用）
                    'mu_lam': mu_lam,               # Sutherland层流粘度场 rlv(T)
                    'mu_turb': mu_turb,              # 湍流粘度
                    'mu_eff': mu_lam + mu_turb      # 总粘度（向后兼容）
                }

            else:
                raise ValueError(f"Invalid viscosity_source: {self.viscosity_source}")

        self._residual_profile_mark(profile_timing, "viscosity", t_stage0)
        t_stage0 = time.perf_counter() if profile_timing is not None else None
        # 6. 计算ss_halo（plan74.md第3点修复：让dss在η方向对边界halo友好）
        # 注意：Ma/AoA和halo值已在前面提取/计算（支持ADFLOW 6-point梯度）
        # 构造包含halo层的熵场 ss_halo (H+2, W)，用于正确计算边界dss
        from surrogate.physics.pde import dissipation as diss_module
        ss_halo = diss_module.compute_ss_with_halo(
            p=p,
            rho=rho,
            gamma=self.gamma,
            halo_wall=halo_wall,
            halo_farfield=halo_farfield
        )
        ss_halo_ext = diss_module.compute_ss_with_two_halos(
            p=p,
            rho=rho,
            gamma=self.gamma,
            halo_wall=halo_wall,
            halo_farfield=halo_farfield,
            basis=self.basis,
        )
        if self.basis == 'entropy':
            rho_safe_sensor = torch.clamp(rho, min=1e-12)
            current_shock_sensor = p / (rho_safe_sensor ** self.gamma)
        else:
            current_shock_sensor = p

        # DEBUG: 打印halo值
        if self.debug:
            print(f"\n[DEBUG torch residual backend] Halo层计算:")
            print(f"  halo_wall shape: {halo_wall.shape}")
            print(f"  halo_wall[0, 0:5] (rho): {halo_wall[0, 0:5].cpu().numpy() if halo_wall.ndim == 2 else halo_wall[0, 0, 0:5].cpu().numpy()}")
            print(f"  halo_farfield shape: {halo_farfield.shape}")

        # 7. 计算所有通量（包含完整η边界面）
        # compute_all_fluxes已经统一生成完整η面通量（H+1, W），支持Euler和N-S
        # **ADflow对齐**：
        #   - 壁面：传递halo_wall实现porK=0（仅压力项）
        #   - 远场：使用Ma/AoA一侧通量（不传halo_farfield），与ADflow保持一致
        # 根据调试环境变量决定是否返回耗散数据
        import os
        return_diss = os.environ.get('SURROGATE_DEBUG_RESIDUAL', '') == '1' or \
                      os.environ.get('SURROGATE_DEBUG_FLUX', '') == '1'
        residual_operator_use_lumped_dissipation = bool(getattr(self, 'residual_operator_use_lumped_dissipation', False))
        residual_operator_lumped_sigma = float(getattr(self, 'residual_operator_lumped_sigma', 6.0))
        residual_operator_use_approximate_viscous_flux = bool(
            getattr(self, 'residual_operator_use_approximate_viscous_flux', False)
        )
        residual_operator_frozen_shock_sensor = getattr(self, 'residual_operator_frozen_shock_sensor', None)
        residual_operator_frozen_ss_halo = getattr(self, 'residual_operator_frozen_ss_halo', None)
        residual_operator_use_dissipation_continuation = bool(
            getattr(self, 'residual_operator_use_dissipation_continuation', False)
        )
        residual_operator_diss_cont_magnitude = float(getattr(self, 'residual_operator_diss_cont_magnitude', 0.0))
        residual_operator_diss_cont_midpoint = float(getattr(self, 'residual_operator_diss_cont_midpoint', 20.0))
        residual_operator_diss_cont_sharpness = float(getattr(self, 'residual_operator_diss_cont_sharpness', 3.0))
        residual_operator_diss_cont_total_r = getattr(self, 'residual_operator_total_r', None)
        residual_operator_diss_cont_total_r0 = getattr(self, 'residual_operator_total_r0', None)
        residual_operator_diss_cont_rfil = float(getattr(self, 'residual_operator_diss_cont_rfil', 1.0))

        self._residual_profile_mark(profile_timing, "shock_sensor_halo", t_stage0)
        t_stage0 = time.perf_counter() if profile_timing is not None else None
        fluxes = self._flux.compute_all_fluxes(
            rho, u, v, p, du_dx, du_dy, dv_dx, dv_dy, face_geom,
            mu_eff=mu_eff, include_viscous=(mu_eff is not None),
            Ma=Ma_val, AoA=AoA_val, gamma=self.gamma,
            dissipation_mode=self.dissipation_mode,
            # Plan86对齐：传入节点梯度（用于粘性通量）
            du_dx_node=du_dx_node,
            du_dy_node=du_dy_node,
            dv_dx_node=dv_dx_node,
            dv_dy_node=dv_dy_node,
            use_nodal_gradients=True,
            vis2=self.vis2,
            vis4=self.vis4,
            dss_max=self.dss_max,
            sslim=self.sslim,
            debug=self.debug,
            halo_wall=halo_wall,
            halo_farfield=halo_farfield,  # 远场使用ADflow bcFarfield halo + 中央通量
            return_dissipation=return_diss,
            basis=self.basis,  # ADflow blockette.F90: RANS用熵基底，Euler用压力基底
            ss_halo=ss_halo,  # plan74.md修复：边界dss计算
            vol=vol,  # 3D谱半径计算：传递体积用于rj_thin计算
            adis=self.adis,  # Jameson耗散各向异性缩放指数（ADflow对齐：0.67）
            acoustic_scale_factor=self.acoustic_scale_factor,  # 声速缩放因子
            # Phase 3: 能量方程支持
            rhoE=rhoE if has_energy_equation else None,
            daa_dx_node=daa_dx_node,
            daa_dy_node=daa_dy_node,
            dp_dx_node=dp_dx_node,
            dp_dy_node=dp_dy_node,
            drho_dx_node=drho_dx_node,
            drho_dy_node=drho_dy_node,
            Pr_laminar=0.72,
            Pr_turbulent=0.9,
            lumped_dissipation=residual_operator_use_lumped_dissipation,
            lumped_sigma=residual_operator_lumped_sigma,
            frozen_shock_sensor=residual_operator_frozen_shock_sensor,
            frozen_ss_halo=residual_operator_frozen_ss_halo,
            use_dissipation_continuation=residual_operator_use_dissipation_continuation,
            diss_cont_magnitude=residual_operator_diss_cont_magnitude,
            diss_cont_midpoint=residual_operator_diss_cont_midpoint,
            diss_cont_sharpness=residual_operator_diss_cont_sharpness,
            diss_cont_total_r=residual_operator_diss_cont_total_r,
            diss_cont_total_r0=residual_operator_diss_cont_total_r0,
            diss_cont_rfil=residual_operator_diss_cont_rfil,
            approximate_viscous_operator=residual_operator_use_approximate_viscous_flux,
            halo_nu_tilde_farfield=halo_nuTilde_ff,
        )

        # ========== DIAGNOSIS: η通量形状检查 ==========
        if self.debug:
            print(f"dissipation_mode：{self.dissipation_mode}")
            H_grid, W_grid = rho.shape[-2], rho.shape[-1]  # 使用rho的形状（确保在作用域内）
            print(f"\n[DIAGNOSIS torch residual backend] η通量计算结果:")
            print(f"  Fc_eta shape: {fluxes['Fc_eta'].shape}, 预期: ({H_grid+1}, {W_grid})")
            print(f"  Fc_eta非零元素数: {(fluxes['Fc_eta'].abs() > 1e-10).sum().item()}/{fluxes['Fc_eta'].numel()}")
            print(f"  Fc_eta[0] (壁面j=0): mean={fluxes['Fc_eta'][0, :].mean():.6e}, max={fluxes['Fc_eta'][0, :].abs().max():.6e}")
            print(f"  Fc_eta[-1] (远场j=H): mean={fluxes['Fc_eta'][-1, :].mean():.6e}, max={fluxes['Fc_eta'][-1, :].abs().max():.6e}")
        # ========== END DIAGNOSIS ==========

        # 7. 装配残差（支持3方程、4方程、5方程模式）
        self._residual_profile_mark(profile_timing, "fluxes", t_stage0)
        t_stage0 = time.perf_counter() if profile_timing is not None else None
        # 7.1 计算SA残差（如果has_sa_equation=True）
        RSA = None
        residual_operator_sa_context = None
        if has_sa_equation and self.viscosity_source == 'laminar+SA':
            # Phase 4: SA湍流方程残差计算
            # 创建SA残差计算器（ADFLOW对齐参数）
            sa_calc = self._get_sa_residual_calculator(
                approx_sa=bool(getattr(self, 'residual_operator_use_approx_sa', False))
            )

            # 准备SA残差计算所需的字段字典
            # 注意：mu_l需要从mu_eff字典中提取
            if isinstance(mu_eff, dict):
                mu_l = mu_eff['mu_lam']
                mu_inf = mu_eff.get('mu_inf', None)
            else:
                # laminar模式（虽然不应该进入这个分支，因为has_sa_equation需要laminar+SA）
                mu_l = mu_eff
                mu_inf = mu_eff

            # ✅ ADFLOW对齐：NS与SA共用同一个 Sutherland 层流粘度场 rlv(T)
            # 此处 mu_l 已经是上面构造好的 rlv(T)，只需继续为 halo 单元补齐同口径的层流运动粘度。
            SSUTH_DIM = 110.55  # ADFLOW默认 SSuthDim (inputParamRoutines.F90:3999)
            TREF_DIM = 300.0    # AeroProblem.T (state_injection.py 固定设置)
            ssuth = SSUTH_DIM / TREF_DIM

            mu_l_sa = mu_l

            # SA粘性项所需的halo运动粘度与halo体积（ADFLOW saViscous）
            if halo_nuTilde_wall is None or halo_nuTilde_ff is None:
                raise ValueError(
                    "SA residual requires halo_nuTilde_wall/halo_nuTilde_ff. "
                    "Ensure halo construction included the nuTilde channel (6-channel halo)."
                )

            # halo 单元粘度仍需从自由流参考粘度 muInf 乘以 halo 温度的 Sutherland 因子得到
            mu_l_for_halo = mu_inf
            if mu_l_for_halo is None:
                raise ValueError("SA residual requires mu_inf in viscosity dict for halo viscosity construction.")
            if isinstance(mu_l_for_halo, torch.Tensor) and mu_l_for_halo.ndim == 3 and mu_l_for_halo.shape[-2:] == (1, 1):
                mu_l_for_halo = mu_l_for_halo[..., 0, 0]  # (batch,)

            halo_nu_l_wall = None
            halo_nu_l_ff = None
            # ✅ ADFLOW对齐：halo单元的层流粘度同样由Sutherland(T_halo)给出（computeLamViscosity(includeHalos)）
            if (halo_rho_wall is not None and halo_rho_ff is not None and
                    halo_p_wall is not None and halo_p_ff is not None):
                rho_wall_T = torch.clamp(halo_rho_wall, min=1e-12)
                rho_ff_T = torch.clamp(halo_rho_ff, min=1e-12)
                T_wall_halo = halo_p_wall / rho_wall_T
                T_ff_halo = halo_p_ff / rho_ff_T
                T_wall_halo = torch.clamp(T_wall_halo, min=1e-12)
                T_ff_halo = torch.clamp(T_ff_halo, min=1e-12)

                suth_wall = torch.pow(T_wall_halo, 1.5) * (1.0 + ssuth) / (T_wall_halo + ssuth)
                suth_ff = torch.pow(T_ff_halo, 1.5) * (1.0 + ssuth) / (T_ff_halo + ssuth)

                if isinstance(mu_l_for_halo, torch.Tensor) and mu_l_for_halo.ndim == 1 and halo_rho_wall.ndim == 2:
                    mu_wall_halo = mu_l_for_halo.unsqueeze(-1) * suth_wall
                    mu_ff_halo = mu_l_for_halo.unsqueeze(-1) * suth_ff
                else:
                    mu_wall_halo = mu_l_for_halo * suth_wall
                    mu_ff_halo = mu_l_for_halo * suth_ff

                halo_nu_l_wall = mu_wall_halo / halo_rho_wall
                halo_nu_l_ff = mu_ff_halo / halo_rho_ff

            fields_sa = {
                'rho': rho,
                'u': u,
                'v': v,
                'nuTilde': nuTilde,
                'mu_l': mu_l_sa,
                'd_wall': d_wall,
                'vort_mag': vort_mag,
                'du_dx': du_dx,
                'du_dy': du_dy,
                'dv_dx': dv_dx,
                'dv_dy': dv_dy,
                'dnu_dx': dnu_dx,
                'dnu_dy': dnu_dy
            }

            geometry_sa = {
                'face_geom': face_geom,
                'vol': vol,
                'contravariant_vel': contravariant_vel,
                # ✅ ADFLOW对齐：turbAdvection与saViscous需要halo(1/2)
                'halo_nuTilde_wall': halo_nuTilde_wall,
                'halo_nuTilde_wall_second': halo_nuTilde_wall_second,
                'halo_nuTilde_ff': halo_nuTilde_ff,
                'halo_nuTilde_ff_second': halo_nuTilde_ff_second,
                'halo_nu_l_wall': halo_nu_l_wall,
                'halo_nu_l_ff': halo_nu_l_ff,
                'halo_vol_wall': halo_vol_wall_sa,
                'halo_vol_ff': halo_vol_ff_sa
            }

            if bool(getattr(self, 'residual_operator_return_sa_context', False)):
                residual_operator_sa_context = {
                    'rho': rho,
                    'u': u,
                    'v': v,
                    'nuTilde': nuTilde,
                    'mu_l': mu_l_sa,
                    'd_wall': d_wall,
                    'du_dx': du_dx,
                    'du_dy': du_dy,
                    'dv_dx': dv_dx,
                    'dv_dy': dv_dy,
                    'face_geom': geometry_sa['face_geom'],
                    'vol': geometry_sa['vol'],
                    'contravariant_vel': geometry_sa['contravariant_vel'],
                    'halo_nuTilde_wall': geometry_sa.get('halo_nuTilde_wall'),
                    'halo_nuTilde_wall_second': geometry_sa.get('halo_nuTilde_wall_second'),
                    'halo_nuTilde_ff': geometry_sa.get('halo_nuTilde_ff'),
                    'halo_nuTilde_ff_second': geometry_sa.get('halo_nuTilde_ff_second'),
                    'halo_nu_l_wall': geometry_sa.get('halo_nu_l_wall'),
                    'halo_nu_l_ff': geometry_sa.get('halo_nu_l_ff'),
                    'halo_vol_wall': geometry_sa.get('halo_vol_wall'),
                    'halo_vol_ff': geometry_sa.get('halo_vol_ff'),
                }

            # 计算SA残差
            RSA = sa_calc.compute_residual(fields_sa, geometry_sa, flow_conditions)

        self._residual_profile_mark(profile_timing, "sa_residual", t_stage0)
        t_stage0 = time.perf_counter() if profile_timing is not None else None
        # 7.2 装配流场残差（连续+动量+能量）
        if has_energy_equation:
            # 4方程或5方程模式：连续+动量x+动量y+能量[+SA]
            Rc, Rmx, Rmy, RE, _ = self._res.assemble_residuals_5eq(
                fluxes, vol=vol, periodic_xi=periodic_xi,
                include_viscous=(mu_eff is not None),
                include_sa=False,  # SA残差已单独计算
                debug=self.debug
            )
            # 合并为4或5通道残差
            if RSA is not None:
                # 5方程完整模式
                residuals_5eq = torch.stack([Rc, Rmx, Rmy, RE, RSA], dim=0)  # (5, H, W) 或 (batch, 5, H, W)
            else:
                # 4方程模式（无SA）
                residuals_5eq = torch.stack([Rc, Rmx, Rmy, RE], dim=0)  # (4, H, W) 或 (batch, 4, H, W)
        else:
            # 3方程模式：连续+动量x+动量y
            Rc, Rmx, Rmy = self._res.assemble_residuals(
                fluxes, vol=vol, periodic_xi=periodic_xi,
                include_viscous=(mu_eff is not None),
                debug=self.debug
            )

        self._residual_profile_mark(profile_timing, "assemble_residuals", t_stage0)
        t_stage0 = time.perf_counter() if profile_timing is not None else None
        # 8. 计算RMS残差范数（与ADflow对齐）
        # 注意：直接使用原始FVM残差，不经rfl激波传感器抑制
        # rfl抑制会导致残差损失训练时模型学习"制造激波"而非"降低残差"
        # 解析权重（支持legacy key: 'momentum'/'wm'）
        from surrogate.physics.residual.weights import ResidualWeights, parse_residual_weights

        default_weights = ResidualWeights()
        wc, wmx, wmy, we, wsa = parse_residual_weights(
            weights,
            wc_default=default_weights.wc,
            wmx_default=default_weights.wmx,
            wmy_default=default_weights.wmy,
            energy_default=1.0,
            turbulence_default=1.0,
        )

        if has_energy_equation:
            # 4或5方程模式：计算能量和SA残差范数
            Rc_norm, Rmx_norm, Rmy_norm = self._res.compute_residual_norm_rms(
                Rc, Rmx, Rmy,
                wall_layers=wall_layers,
                norm_mode=self.residual_norm_mode,
                vol=vol
            )

            # 能量残差范数（与 Rc/Rm* 一致：支持 batch + wall_layers + norm_mode）
            RE_norm = self._res.compute_residual_norm_rms_scalar(
                RE,
                wall_layers=wall_layers,
                norm_mode=self.residual_norm_mode,
                vol=vol,
            )

            # SA残差范数（Phase 4，若启用）
            RSA_norm = None
            if RSA is not None:
                RSA_norm = self._res.compute_residual_norm_rms_scalar(
                    RSA,
                    wall_layers=wall_layers,
                    norm_mode=self.residual_norm_mode,
                    vol=vol,
                )

            # residual score 聚合（4或5方程）
            if RSA is None:
                wsa = 0.0

            residual_score = -(wc * Rc_norm + wmx * Rmx_norm + wmy * Rmy_norm + we * RE_norm)
            if RSA_norm is not None:
                residual_score = residual_score - wsa * RSA_norm
        else:
            # 3方程模式
            Rc_norm, Rmx_norm, Rmy_norm = self._res.compute_residual_norm_rms(
                Rc, Rmx, Rmy,  # 使用原始残差，不经rfl抑制
                wall_layers=wall_layers,
                norm_mode=self.residual_norm_mode,
                vol=vol
            )

            # residual score 聚合（使用独立的 momentum_x 和 momentum_y 权重）
            residual_score = -(wc * Rc_norm + wmx * Rmx_norm + wmy * Rmy_norm)

        self._residual_profile_mark(profile_timing, "norms_residual_score", t_stage0)
        if profile_timing is not None and t_total0 is not None:
            self._residual_profile_mark(profile_timing, "total", t_total0)
        # 10. 返回结果
        if return_components:
            export_dtype = compute_dtype if bool(preserve_residual_dtype) else torch.float32

            def _cast_export_tensor(value):
                if isinstance(value, torch.Tensor):
                    return value.to(export_dtype)
                return value

            result = {
                'residual_score': _cast_export_tensor(residual_score),
                'Rc': _cast_export_tensor(Rc),  # 返回原始FVM残差
                'Rmx': _cast_export_tensor(Rmx),
                'Rmy': _cast_export_tensor(Rmy),
                'Rc_norm': _cast_export_tensor(Rc_norm),
                'Rmx_norm': _cast_export_tensor(Rmx_norm),
                'Rmy_norm': _cast_export_tensor(Rmy_norm),
                'vol': _cast_export_tensor(vol),  # 返回单元体积用于残差归一化分析
                'mu_eff': _cast_export_tensor(mu_eff),
            }
            if wall_distance_source is not None:
                result['wall_distance_source'] = wall_distance_source
                result['_resolved_wall_distance'] = d_wall
            if profile_timing is not None:
                result['residual_operator_profile'] = dict(profile_timing)

            # 能量方程模式：添加RE和RE_norm
            if has_energy_equation:
                result['RE'] = _cast_export_tensor(RE)
                result['RE_norm'] = _cast_export_tensor(RE_norm)

            # SA湍流方程模式：添加RSA和RSA_norm（Phase 4）
            if RSA is not None:
                result['RSA'] = _cast_export_tensor(RSA)
                result['RSA_norm'] = _cast_export_tensor(RSA_norm)
            if residual_operator_sa_context is not None:
                result['residual_operator_sa_context'] = residual_operator_sa_context
            if bool(getattr(self, 'residual_operator_return_shock_sensor', False)):
                result['residual_operator_shock_sensor'] = _cast_export_tensor(current_shock_sensor)
                result['residual_operator_ss_halo'] = _cast_export_tensor(ss_halo_ext)

            return result
        else:
            return residual_score.to(torch.float32) if isinstance(residual_score, torch.Tensor) else residual_score
