"""
残差装配模块 - Phase 4

将对流/黏性通量装配为PDE残差

关键功能：
1. 守恒型散度计算（周期边界支持）
2. 残差装配（Rc, Rmx, Rmy）
3. 权重计算（面积权重、近壁增强）

统一几何：全部使用geometry-based的vol（体积）
"""

import torch
from typing import Dict, Tuple, Optional, Union
import numpy as np


def compute_flux_balance(
    F_xi: torch.Tensor,
    F_eta: torch.Tensor,
    vol: torch.Tensor,
    periodic_xi: bool = True,
    debug: bool = False,
    equation_type: str = 'continuity'
) -> torch.Tensor:
    """
    面通量净和（使用完整η面，per-volume residual）: R = (Δξ(F_ξ) + Δη(F_η)) / V

    Args:
        F_xi: ξ面通量 (batch, H, W) if periodic else (batch, H, W-1)
        F_eta: η面通量 (batch, H+1, W) - **包含边界面j=0和j=H**
        vol: 单元体积 (H, W) 或 (batch, H, W)
        periodic_xi: ξ方向周期性
        equation_type: 方程类型 ('continuity', 'momentum_x', 'momentum_y')，用于区分保存的npz文件

    Returns:
        R: per-volume残差 (batch, H, W)
    """
    # 添加batch维度（如果没有）
    squeeze_output = False
    if F_xi.ndim == 2:
        F_xi = F_xi.unsqueeze(0)
        F_eta = F_eta.unsqueeze(0)
        squeeze_output = True

    # 获取形状信息
    if periodic_xi:
        batch_size, H, W = F_xi.shape
    else:
        batch_size, H, _ = F_xi.shape
        _, _, W = F_eta.shape

    # 验证η面形状
    batch_size_eta, H_plus_1_eta, W_eta = F_eta.shape
    if H_plus_1_eta != H + 1:
        raise ValueError(
            f"F_eta形状错误：期望(batch, H+1, W)={batch_size, H+1, W}，"
            f"实际={F_eta.shape}"
        )

    # ξ方向散度
    if periodic_xi:
        # 周期边界：使用torch.roll
        dF_dxi = F_xi - torch.roll(F_xi, shifts=1, dims=-1)  # (batch, H, W)
    else:
        # 非周期：常规散度
        dF_dxi = F_xi[:, :, 1:] - F_xi[:, :, :-1]  # (batch, H, W-2)

    # η方向使用完整面散度
    # 每个单元j的散度 = F_eta[j+1] - F_eta[j] = F_top - F_bottom
    dF_deta = F_eta[:, 1:, :] - F_eta[:, :-1, :]  # (batch, H, W)

    # 全场残差（所有单元包含完整η面贡献）
    if periodic_xi:
        R_full = dF_dxi + dF_deta  # (batch, H, W)
    else:
        # 非周期：ξ方向需要对齐
        if dF_dxi.shape[-1] == W - 2:
            # dF_dxi: (batch, H, W-2), dF_deta: (batch, H, W)
            # 裁剪η散度到内部
            R_full = torch.zeros((batch_size, H, W), device=F_xi.device, dtype=F_xi.dtype)
            R_full[:, :, 1:-1] = dF_dxi + dF_deta[:, :, 1:-1]
            # ξ方向边界外推
            R_full[:, :, 0] = R_full[:, :, 1]
            R_full[:, :, -1] = R_full[:, :, -2]
        else:
            R_full = dF_dxi + dF_deta

    # ========== DEBUG: 通量散度统计（除体积前） ==========
    if debug:
        print(f"\n[DEBUG residual.py] 通量散度统计（除体积前）:")
        dF_dxi_rms = torch.sqrt(torch.mean(dF_dxi**2))
        dF_deta_rms = torch.sqrt(torch.mean(dF_deta**2))
        R_full_rms = torch.sqrt(torch.mean(R_full**2))
        print(f"  dF_dxi (ξ散度): RMS={dF_dxi_rms:.6e}, min={dF_dxi.min():.6e}, max={dF_dxi.max():.6e}")
        print(f"  dF_deta (η散度): RMS={dF_deta_rms:.6e}, min={dF_deta.min():.6e}, max={dF_deta.max():.6e}")
        print(f"  R_full (总散度): RMS={R_full_rms:.6e}, min={R_full.min():.6e}, max={R_full.max():.6e}")
        print(f"  *** 关键对比 ***")
        print(f"    PyTorch dF_dxi RMS = {dF_dxi_rms:.6e} vs ADflow I-flux dw RMS = 1.18e-02")
        print(f"    PyTorch R_full RMS = {R_full_rms:.6e} vs ADflow total dw RMS = 1.53e-04")
        print(f"    如果PyTorch η-flux正确抵消ξ-flux, R_full应该<<dF_dxi")

        # 深入分析：dF_dxi和dF_deta的相关性
        dF_dxi_flat = dF_dxi.view(-1)
        dF_deta_flat = dF_deta.view(-1)
        # 计算Pearson相关系数
        dF_dxi_mean = dF_dxi_flat.mean()
        dF_deta_mean = dF_deta_flat.mean()
        cov = ((dF_dxi_flat - dF_dxi_mean) * (dF_deta_flat - dF_deta_mean)).mean()
        std_xi = dF_dxi_flat.std()
        std_eta = dF_deta_flat.std()
        corr = cov / (std_xi * std_eta + 1e-12)
        print(f"  *** dF_dxi vs dF_deta 相关性分析 ***")
        print(f"    Pearson相关系数 = {corr:.4f} (理想值: -1.0, 实际应接近-1才能抵消)")
        print(f"    dF_deta RMS = {dF_deta_rms:.6e}")
        # 检查比例关系
        ratio = dF_deta_rms / dF_dxi_rms
        print(f"    dF_deta/dF_dxi RMS比 = {ratio:.4f} (理想值: ~1.0)")
        # 如果完美抵消: R_full = dF_dxi + dF_deta ≈ 0
        # 检查平均误差
        mean_cancel_error = torch.abs(dF_dxi_flat + dF_deta_flat).mean()
        print(f"    平均抵消误差 |dF_dxi + dF_deta| = {mean_cancel_error:.6e}")

        # ===== 特定单元debug：尾缘小体积单元 (对应ADflow i=293..295, k=2) =====
        # PyTorch索引: j=0..1, i=291..295
        print(f"\n[DEBUG residual.py] 特定单元数据（尾缘壁面层，对应ADflow i=293..295, k=2）:")
        print(f"  PyTorch索引: j=0..1, i=291..295")
        # 确保有batch维度
        dF_dxi_3d = dF_dxi if dF_dxi.ndim == 3 else dF_dxi.unsqueeze(0)
        dF_deta_3d = dF_deta if dF_deta.ndim == 3 else dF_deta.unsqueeze(0)
        R_full_3d = R_full if R_full.ndim == 3 else R_full.unsqueeze(0)
        print(f"  dF_dxi[0, 0:2, 291:296]:")
        print(f"    {dF_dxi_3d[0, 0:2, 291:296].cpu().numpy()}")
        print(f"  dF_deta[0, 0:2, 291:296]:")
        print(f"    {dF_deta_3d[0, 0:2, 291:296].cpu().numpy()}")
        print(f"  R_full[0, 0:2, 291:296] (除vol前):")
        print(f"    {R_full_3d[0, 0:2, 291:296].cpu().numpy()}")
    # ========== END DEBUG ==========

    # 除以体积得到per-volume残差（与ADflow对齐）
    if vol.ndim == 2:
        vol = vol.unsqueeze(0)
    # ADflow直接除以体积，不做特殊处理（经确认：residuals.F90:1174行）
    # 移除epsilon以完全对齐ADflow行为
    R_full_before_vol = R_full.clone()  # 保存除vol前的通量散度
    R_full = R_full / vol

    # ========== SURROGATE_DEBUG_RESIDUAL: 保存逐点数据到npz ==========
    import os
    if os.environ.get('SURROGATE_DEBUG_RESIDUAL', '') == '1':
        import numpy as np
        debug_data = {
            'dF_dxi': dF_dxi[0].detach().cpu().numpy(),
            'dF_deta': dF_deta[0].detach().cpu().numpy(),
            'R_full_before_vol': R_full_before_vol[0].detach().cpu().numpy(),
            'vol': vol[0].detach().cpu().numpy(),
            'Rc': R_full[0].detach().cpu().numpy(),
            'H': H,
            'W': W,
            'equation_type': equation_type,
        }
        # 使用 equation_type 区分不同方程的文件
        filename = f'pytorch_residual_{equation_type}_debug.npz'
        np.savez(filename, **debug_data)
        print(f"[DEBUG] Saved residual debug to: {filename}")
        print(f"        Shape: dF_dxi={dF_dxi[0].shape}, vol={vol[0].shape}, Rc={R_full[0].shape}")
    # ========== END SURROGATE_DEBUG_RESIDUAL ==========

    # ========== DEBUG: 残差统计（除体积后） ==========
    if debug:
        print(f"\n[DEBUG residual.py] 残差统计（除体积后）:")
        print(f"  vol: shape={vol.shape}, min={vol.min():.6e}, max={vol.max():.6e}, mean={vol.mean():.6e}")
        print(f"  R_full: shape={R_full.shape}, min={R_full.min():.6e}, max={R_full.max():.6e}")
        print(f"    RMS = {torch.sqrt(torch.mean(R_full**2)):.6e}")
        print(f"  有效体积因子 (flux_div_rms / residual_rms):")
        print(f"    = {torch.sqrt(torch.mean((R_full*vol)**2)):.6e} / {torch.sqrt(torch.mean(R_full**2)):.6e}")
        print(f"    = {(torch.sqrt(torch.mean((R_full*vol)**2)) / torch.sqrt(torch.mean(R_full**2))):.6e}")

        # ===== 空间分布分析：找出哪些区域贡献了大的残差 =====
        R_full_3d = R_full if R_full.ndim == 3 else R_full.unsqueeze(0)
        vol_3d = vol if vol.ndim == 3 else vol.unsqueeze(0)
        B, H_dim, W_dim = R_full_3d.shape

        print(f"\n[DEBUG residual.py] ===== 空间分布分析 =====")

        # 1. 按j层统计RMS
        print(f"  按j层统计RMS（前10层 + 后5层）:")
        layer_rms_list = []
        for j in range(H_dim):
            layer_rms = torch.sqrt(torch.mean(R_full_3d[:, j, :]**2))
            layer_rms_list.append(layer_rms.item())
            if j < 10 or j >= H_dim - 5:
                print(f"    j={j:3d}: RMS={layer_rms:.6e}, vol_mean={vol_3d[:, j, :].mean():.6e}")

        # 找出RMS最大的层
        max_layer_rms = max(layer_rms_list)
        max_layer_j = layer_rms_list.index(max_layer_rms)
        print(f"  最大RMS层: j={max_layer_j}, RMS={max_layer_rms:.6e}")

        # 2. Top-20极端残差位置
        print(f"\n  Top-20极端残差单元位置:")
        abs_R = torch.abs(R_full_3d[0])  # 取第一个batch
        flat_R = abs_R.view(-1)
        top_k = 20
        top_vals, top_indices = torch.topk(flat_R, top_k)
        for rank, (val, idx) in enumerate(zip(top_vals, top_indices)):
            j_idx = (idx // W_dim).item()
            i_idx = (idx % W_dim).item()
            vol_val = vol_3d[0, j_idx, i_idx].item()
            dF_val = (R_full_3d[0, j_idx, i_idx] * vol_3d[0, j_idx, i_idx]).item()
            print(f"    Top-{rank+1:2d}: j={j_idx:3d}, i={i_idx:3d}, |res|={val:.6e}, vol={vol_val:.6e}, dF={dF_val:.6e}")

        # 3. 按i区域统计（前缘i~0或i~145，尾缘i~291）
        print(f"\n  按i区域统计（j=0层，即壁面层）:")
        regions = [
            ("前缘上部", 140, 150),
            ("前缘下部", 0, 10),
            ("尾缘", 285, 292),
        ]
        for name, i_start, i_end in regions:
            region_res = R_full_3d[0, 0, i_start:i_end]
            region_vol = vol_3d[0, 0, i_start:i_end]
            region_rms = torch.sqrt(torch.mean(region_res**2))
            region_vol_mean = region_vol.mean()
            print(f"    {name} (i={i_start}:{i_end}): RMS={region_rms:.6e}, vol_mean={region_vol_mean:.6e}")

        # 4. 验证ADflow统计范围假设：排除j=0层
        dF_full = R_full_3d * vol_3d  # 恢复dF
        dF_inner = dF_full[:, 1:, :]  # 排除j=0（对应ADflow k=2:kl）
        R_inner = R_full_3d[:, 1:, :]  # 排除j=0
        dF_inner_rms = torch.sqrt(torch.mean(dF_inner**2))
        R_inner_rms = torch.sqrt(torch.mean(R_inner**2))
        print(f"\n  排除j=0层后的统计（模拟ADflow k=2:kl统计）:")
        print(f"    dF RMS (j=1:) = {dF_inner_rms:.6e}  (ADflow: 2.73e-04)")
        print(f"    res RMS (j=1:) = {R_inner_rms:.6e}  (ADflow: 2.22e-02)")

        # 5. 找出dF极端值单元（与ADflow范围对比）
        print(f"\n  dF极端值分析（ADflow范围: [-0.002, 0.002]）:")
        dF_abs = torch.abs(dF_full[0])
        flat_dF = dF_abs.view(-1)
        top_k_dF = 10
        top_dF_vals, top_dF_indices = torch.topk(flat_dF, top_k_dF)
        print(f"    Top-{top_k_dF}最大|dF|单元:")
        for rank, (val, idx) in enumerate(zip(top_dF_vals, top_dF_indices)):
            j_idx = (idx // W_dim).item()
            i_idx = (idx % W_dim).item()
            print(f"      Top-{rank+1:2d}: j={j_idx:3d}, i={i_idx:3d}, |dF|={val:.6e}")

        print(f"  ===== 空间分布分析结束 =====")

        # ===== 特定单元debug：尾缘小体积单元（除vol后） =====
        vol_3d = vol if vol.ndim == 3 else vol.unsqueeze(0)
        R_full_3d = R_full if R_full.ndim == 3 else R_full.unsqueeze(0)
        print(f"\n[DEBUG residual.py] 特定单元（除vol后）- 对应ADflow i=293..295, k=2:")
        print(f"  vol[0, 0:2, 291:296]:")
        print(f"    {vol_3d[0, 0:2, 291:296].cpu().numpy()}")
        print(f"  R_full[0, 0:2, 291:296] (除vol后，即残差):")
        print(f"    {R_full_3d[0, 0:2, 291:296].cpu().numpy()}")
        # 逐单元打印（与ADflow格式对齐）
        print(f"\n  逐单元对比（PyTorch j=0 对应 ADflow k=2）:")
        for i_offset in range(5):
            i_py = 291 + i_offset
            i_ad = i_py + 2  # ADflow索引 = PyTorch索引 + 2
            dF_val = (R_full_3d[0, 0, i_py] * vol_3d[0, 0, i_py]).item()
            vol_val = vol_3d[0, 0, i_py].item()
            res_val = R_full_3d[0, 0, i_py].item()
            print(f"    PyTorch(j=0,i={i_py}) / ADflow(k=2,i={i_ad}): dF={dF_val:.6e}, vol={vol_val:.6e}, res={res_val:.6e}")
    # ========== END DEBUG ==========

    # 移除batch维度（如果输入没有）
    if squeeze_output:
        R_full = R_full.squeeze(0)

    return R_full


def assemble_residuals(
    fluxes: Dict[str, torch.Tensor],
    vol: torch.Tensor,
    periodic_xi: bool = True,
    include_viscous: bool = False,
    debug: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    装配PDE残差（连续性 + 动量）- per-volume residual（与ADflow完全对齐）

    统一使用完整η边界面（j=0壁面和j=H远场），适用于Euler和N-S方程。

    Args:
        fluxes: 通量字典（来自fluxes.compute_all_fluxes）
            - 对流（完整η面）：'Fc_xi', 'Fmx_xi', 'Fmy_xi'
            - 对流（完整η面）：'Fc_eta', 'Fmx_eta', 'Fmy_eta' - 所有η通量均为(H+1, W)
            - 黏性（如果include_viscous）：
                'Vmx_xi', 'Vmy_xi'（内部ξ面）
                'Vmx_eta', 'Vmy_eta'（完整η面）
        vol: 单元体积 (H, W) 或 (batch, H, W)
        periodic_xi: ξ方向周期性
        include_viscous: 是否包含黏性项

    Returns:
        (Rc, Rmx, Rmy): 连续性、x动量、y动量残差 (batch, H, W) 或 (H, W)
    """
    # 连续性残差：始终使用完整η面（Euler和黏性模式统一）
    if 'Fc_eta' not in fluxes:
        raise ValueError(
            "assemble_residuals requires 'Fc_eta' (complete eta face flux, shape H+1×W). "
            "Please compute it using compute_inviscid_eta_flux."
        )
    Rc = compute_flux_balance(fluxes['Fc_xi'], fluxes['Fc_eta'], vol, periodic_xi, debug, equation_type='continuity')

    # ========== 保存耗散通量到npz（如果存在） ==========
    import os
    if os.environ.get('SURROGATE_DEBUG_RESIDUAL', '') == '1' and 'D_rho_xi' in fluxes:
        import numpy as np
        # 计算耗散散度（与通量散度相同的方式）
        D_rho_xi = fluxes['D_rho_xi']
        D_rho_eta = fluxes['D_rho_eta']

        if D_rho_xi is not None and D_rho_eta is not None:
            # 确保有batch维度
            if D_rho_xi.ndim == 2:
                D_rho_xi = D_rho_xi.unsqueeze(0)
            if D_rho_eta.ndim == 2:
                D_rho_eta = D_rho_eta.unsqueeze(0)

            batch, H, W = D_rho_xi.shape

            # 计算耗散散度（与compute_flux_balance()定义一致）
            # 注意：此dD_dxi仅用于debug输出，真实残差装配在compute_flux_balance()中
            if periodic_xi:
                # 周期边界：使用roll定义（与真实装配一致）
                dD_dxi = D_rho_xi - torch.roll(D_rho_xi, shifts=1, dims=-1)
            else:
                # 非周期边界：使用差分形式（边界填零）
                D_xi_left = torch.cat([torch.zeros_like(D_rho_xi[..., :, :1]), D_rho_xi], dim=-1)
                D_xi_right = torch.cat([D_rho_xi, torch.zeros_like(D_rho_xi[..., :, :1])], dim=-1)
                dD_dxi = D_xi_right - D_xi_left

            # η方向耗散散度
            dD_deta = D_rho_eta[..., 1:, :] - D_rho_eta[..., :-1, :]  # (batch, H, W)

            # 总耗散散度
            D_total = dD_dxi + dD_deta

            # 保存耗散数据
            dissipation_data = {
                'D_rho_xi': D_rho_xi[0].detach().cpu().numpy(),
                'D_rho_eta': D_rho_eta[0].detach().cpu().numpy(),
                'dD_dxi': dD_dxi[0].detach().cpu().numpy(),
                'dD_deta': dD_deta[0].detach().cpu().numpy(),
                'D_total': D_total[0].detach().cpu().numpy(),
                'H': H,
                'W': W,
            }
            np.savez('pytorch_dissipation_debug.npz', **dissipation_data)
            print(f"[DEBUG] Saved dissipation debug to: pytorch_dissipation_debug.npz")
            print(f"        D_rho_xi shape: {D_rho_xi[0].shape}, D_total shape: {D_total[0].shape}")
    # ========== END 耗散保存 ==========

    # ========== 保存动量通量组件到npz ==========
    import os
    if os.environ.get('SURROGATE_DEBUG_RESIDUAL', '') == '1':
        import numpy as np
        momentum_flux_data = {}

        # ξ方向动量通量组件
        if 'Fmx_xi' in fluxes:
            Fmx_xi_t = fluxes['Fmx_xi']
            if Fmx_xi_t.ndim == 2:
                Fmx_xi_t = Fmx_xi_t.unsqueeze(0)
            momentum_flux_data['Fmx_xi'] = Fmx_xi_t[0].detach().cpu().numpy()

        if 'Fmy_xi' in fluxes:
            Fmy_xi_t = fluxes['Fmy_xi']
            if Fmy_xi_t.ndim == 2:
                Fmy_xi_t = Fmy_xi_t.unsqueeze(0)
            momentum_flux_data['Fmy_xi'] = Fmy_xi_t[0].detach().cpu().numpy()

        # 纯对流通量（如果存在）
        if 'Fmx_conv_xi' in fluxes:
            Fmx_conv_xi_t = fluxes['Fmx_conv_xi']
            if Fmx_conv_xi_t.ndim == 2:
                Fmx_conv_xi_t = Fmx_conv_xi_t.unsqueeze(0)
            momentum_flux_data['Fmx_conv_xi'] = Fmx_conv_xi_t[0].detach().cpu().numpy()

        if 'Fmy_conv_xi' in fluxes:
            Fmy_conv_xi_t = fluxes['Fmy_conv_xi']
            if Fmy_conv_xi_t.ndim == 2:
                Fmy_conv_xi_t = Fmy_conv_xi_t.unsqueeze(0)
            momentum_flux_data['Fmy_conv_xi'] = Fmy_conv_xi_t[0].detach().cpu().numpy()

        # 耗散通量（如果存在）
        if 'D_rhou_xi' in fluxes:
            D_rhou_xi_t = fluxes['D_rhou_xi']
            if D_rhou_xi_t.ndim == 2:
                D_rhou_xi_t = D_rhou_xi_t.unsqueeze(0)
            momentum_flux_data['D_rhou_xi'] = D_rhou_xi_t[0].detach().cpu().numpy()

        if 'D_rhov_xi' in fluxes:
            D_rhov_xi_t = fluxes['D_rhov_xi']
            if D_rhov_xi_t.ndim == 2:
                D_rhov_xi_t = D_rhov_xi_t.unsqueeze(0)
            momentum_flux_data['D_rhov_xi'] = D_rhov_xi_t[0].detach().cpu().numpy()

        # 黏性通量（如果存在）
        if 'Vmx_xi' in fluxes:
            Vmx_xi_t = fluxes['Vmx_xi']
            if Vmx_xi_t.ndim == 2:
                Vmx_xi_t = Vmx_xi_t.unsqueeze(0)
            momentum_flux_data['Vmx_xi'] = Vmx_xi_t[0].detach().cpu().numpy()

        if 'Vmy_xi' in fluxes:
            Vmy_xi_t = fluxes['Vmy_xi']
            if Vmy_xi_t.ndim == 2:
                Vmy_xi_t = Vmy_xi_t.unsqueeze(0)
            momentum_flux_data['Vmy_xi'] = Vmy_xi_t[0].detach().cpu().numpy()

        # η方向动量通量组件
        if 'Fmx_eta' in fluxes:
            Fmx_eta_t = fluxes['Fmx_eta']
            if Fmx_eta_t.ndim == 2:
                Fmx_eta_t = Fmx_eta_t.unsqueeze(0)
            momentum_flux_data['Fmx_eta'] = Fmx_eta_t[0].detach().cpu().numpy()

        if 'Fmy_eta' in fluxes:
            Fmy_eta_t = fluxes['Fmy_eta']
            if Fmy_eta_t.ndim == 2:
                Fmy_eta_t = Fmy_eta_t.unsqueeze(0)
            momentum_flux_data['Fmy_eta'] = Fmy_eta_t[0].detach().cpu().numpy()

        # η方向纯对流通量（如果存在）
        if 'Fmx_conv_eta' in fluxes:
            Fmx_conv_eta_t = fluxes['Fmx_conv_eta']
            if Fmx_conv_eta_t.ndim == 2:
                Fmx_conv_eta_t = Fmx_conv_eta_t.unsqueeze(0)
            momentum_flux_data['Fmx_conv_eta'] = Fmx_conv_eta_t[0].detach().cpu().numpy()

        if 'Fmy_conv_eta' in fluxes:
            Fmy_conv_eta_t = fluxes['Fmy_conv_eta']
            if Fmy_conv_eta_t.ndim == 2:
                Fmy_conv_eta_t = Fmy_conv_eta_t.unsqueeze(0)
            momentum_flux_data['Fmy_conv_eta'] = Fmy_conv_eta_t[0].detach().cpu().numpy()

        # η方向耗散通量（如果存在）
        if 'D_rhou_eta' in fluxes:
            D_rhou_eta_t = fluxes['D_rhou_eta']
            if D_rhou_eta_t.ndim == 2:
                D_rhou_eta_t = D_rhou_eta_t.unsqueeze(0)
            momentum_flux_data['D_rhou_eta'] = D_rhou_eta_t[0].detach().cpu().numpy()

        if 'D_rhov_eta' in fluxes:
            D_rhov_eta_t = fluxes['D_rhov_eta']
            if D_rhov_eta_t.ndim == 2:
                D_rhov_eta_t = D_rhov_eta_t.unsqueeze(0)
            momentum_flux_data['D_rhov_eta'] = D_rhov_eta_t[0].detach().cpu().numpy()

        # η方向黏性通量（如果存在）
        if 'Vmx_eta' in fluxes:
            Vmx_eta_t = fluxes['Vmx_eta']
            if Vmx_eta_t.ndim == 2:
                Vmx_eta_t = Vmx_eta_t.unsqueeze(0)
            momentum_flux_data['Vmx_eta'] = Vmx_eta_t[0].detach().cpu().numpy()

        if 'Vmy_eta' in fluxes:
            Vmy_eta_t = fluxes['Vmy_eta']
            if Vmy_eta_t.ndim == 2:
                Vmy_eta_t = Vmy_eta_t.unsqueeze(0)
            momentum_flux_data['Vmy_eta'] = Vmy_eta_t[0].detach().cpu().numpy()

        if momentum_flux_data:
            np.savez('pytorch_momentum_flux_debug.npz', **momentum_flux_data)
            print(f"[DEBUG] Saved momentum flux components to: pytorch_momentum_flux_debug.npz")
            print(f"        Keys: {list(momentum_flux_data.keys())}")
    # ========== END 动量通量组件保存 ==========

    # 动量残差
    if include_viscous:
        # 使用完整η面黏性通量
        # ξ面：内部面即可
        Fmx_xi_total = fluxes['Fmx_xi'] - fluxes['Vmx_xi']
        Fmy_xi_total = fluxes['Fmy_xi'] - fluxes['Vmy_xi']

        # η面：使用完整面（包含边界）
        if 'Vmx_eta' in fluxes and 'Vmy_eta' in fluxes:
            # 使用完整η面对流通量（真实边界条件）
            if 'Fmx_eta' not in fluxes or 'Fmy_eta' not in fluxes:
                raise ValueError(
                    "assemble_residuals requires 'Fmx_eta' and 'Fmy_eta' "
                    "for complete boundary flux treatment. These should be computed by "
                    "compute_inviscid_eta_flux with proper boundary conditions (wall/farfield)."
                )

            # 直接从fluxes获取完整η面对流通量 (H+1, W)
            Fmx_eta = fluxes['Fmx_eta']
            Fmy_eta = fluxes['Fmy_eta']

            # 总通量 = 对流 - 黏性（均为完整η面，物理一致）
            Fmx_eta_total = Fmx_eta - fluxes['Vmx_eta']
            Fmy_eta_total = Fmy_eta - fluxes['Vmy_eta']

            # 使用完整面装配
            Rmx = compute_flux_balance(Fmx_xi_total, Fmx_eta_total, vol, periodic_xi, debug, equation_type='momentum_x')
            Rmy = compute_flux_balance(Fmy_xi_total, Fmy_eta_total, vol, periodic_xi, debug, equation_type='momentum_y')
        else:
            # 不允许fallback，强制使用完整η面
            raise ValueError(
                "assemble_residuals requires 'Vmx_eta' and 'Vmy_eta' "
                "in fluxes dict. Please compute full eta viscous fluxes first."
            )
    else:
        # Euler模式：也必须使用完整η面对流通量
        if 'Fmx_eta' not in fluxes or 'Fmy_eta' not in fluxes or 'Fc_eta' not in fluxes:
            raise ValueError(
                "assemble_residuals requires 'Fc_eta', 'Fmx_eta', 'Fmy_eta' "
                "for complete boundary flux treatment in all modes (Euler and viscous). "
                "These should be computed by compute_inviscid_eta_flux."
            )

        # 连续性残差也使用完整η面
        Rc = compute_flux_balance(fluxes['Fc_xi'], fluxes['Fc_eta'], vol, periodic_xi, debug, equation_type='continuity')

        # 动量残差使用完整η面
        Rmx = compute_flux_balance(fluxes['Fmx_xi'], fluxes['Fmx_eta'], vol, periodic_xi, debug, equation_type='momentum_x')
        Rmy = compute_flux_balance(fluxes['Fmy_xi'], fluxes['Fmy_eta'], vol, periodic_xi, debug, equation_type='momentum_y')

    return Rc, Rmx, Rmy


def compute_residual_norm_rms_scalar(
    R: torch.Tensor,
    wall_layers: Optional[int] = None,
    norm_mode: str = "standard",
    vol: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute RMS norm for a single residual field.

    Notes:
        - Input residual R is already per-volume (R = divergence / vol), same convention as
          `compute_residual_norm_rms`.
        - Supports batched tensors: (B, H, W) and single sample: (H, W).

    Args:
        R: Residual field, shape (B, H, W) or (H, W)
        wall_layers: Optional j-direction window, only compute over j=0:wall_layers
        norm_mode: 'standard'/'adflow' | 'flux'/'plain' | 'weighted'
        vol: Cell volume field, required for 'flux'/'plain' and 'weighted' modes

    Returns:
        RMS tensor, shape (B,) if batched else scalar tensor
    """
    if wall_layers is not None:
        R = R[..., :wall_layers, :]
        if vol is not None:
            vol = vol[..., :wall_layers, :]

    if vol is not None and vol.ndim == 2 and R.ndim == 3:
        vol = vol.unsqueeze(0)

    def _rms(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            return torch.sqrt(torch.mean(x**2, dim=(1, 2)))
        return torch.sqrt(torch.mean(x**2))

    if norm_mode in ("standard", "adflow"):
        return _rms(R)

    if norm_mode in ("flux", "plain"):
        if vol is None:
            raise ValueError("norm_mode='flux' requires vol to be provided")
        return _rms(R * vol)

    if norm_mode == "weighted":
        if vol is None:
            raise ValueError("norm_mode='weighted' requires vol to be provided")
        return _rms(R * torch.sqrt(vol))

    raise ValueError(
        f"Unknown norm_mode: {norm_mode}. Choose 'standard'/'adflow', 'flux'/'plain', or 'weighted'."
    )


def compute_residual_norm_rms(
    Rc: torch.Tensor,
    Rmx: torch.Tensor,
    Rmy: torch.Tensor,
    wall_layers: Optional[int] = None,
    norm_mode: str = 'standard',
    vol: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算RMS残差范数

    **重要**：输入R已经是per-volume残差（R = 通量散度 / vol），见compute_flux_balance()第147行。

    支持三种模式：
    1. 'standard'/'adflow'：标准残差 RMS = sqrt(mean(R^2))，直接使用per-volume R
    2. 'flux'/'plain'：通量残差 RMS = sqrt(mean((R*vol)^2))，逐点乘vol还原通量散度
    3. 'weighted'：加权残差 RMS = sqrt(mean((R*sqrt(vol))^2))，逐点乘sqrt(vol)

    Args:
        Rc, Rmx, Rmy: 残差场 (batch, H, W) 或 (H, W)，已经是per-volume残差
        wall_layers: 可选的j方向窗口（仅计算j=0:wall_layers区域）
        norm_mode: 'standard'/'adflow' | 'flux'/'plain' | 'weighted'
        vol: 单元体积 (H, W) 或 (batch, H, W)，flux/weighted模式需要

    Returns:
        (Rc_rms, Rmx_rms, Rmy_rms): RMS标量
    """
    # 应用wall_layers窗口
    if wall_layers is not None:
        Rc = Rc[..., :wall_layers, :]
        Rmx = Rmx[..., :wall_layers, :]
        Rmy = Rmy[..., :wall_layers, :]
        if vol is not None:
            vol = vol[..., :wall_layers, :]

    # 确保vol维度匹配
    if vol is not None and vol.ndim == 2 and Rc.ndim == 3:
        vol = vol.unsqueeze(0)

    def _rms(x: torch.Tensor) -> torch.Tensor:
        """计算RMS，支持batch维度"""
        if x.ndim == 3:
            return torch.sqrt(torch.mean(x**2, dim=(1, 2)))
        else:
            return torch.sqrt(torch.mean(x**2))

    if norm_mode in ('standard', 'adflow'):
        # 标准残差：直接使用per-volume R
        # RMS = sqrt(mean(R^2))
        Rc_rms = _rms(Rc)
        Rmx_rms = _rms(Rmx)
        Rmy_rms = _rms(Rmy)

    elif norm_mode in ('flux', 'plain'):
        # 通量残差：还原为原始通量散度（逐点乘vol）
        # RMS = sqrt(mean((R * vol)^2))
        if vol is None:
            raise ValueError("norm_mode='flux' requires vol to be provided")
        Rc_rms = _rms(Rc * vol)
        Rmx_rms = _rms(Rmx * vol)
        Rmy_rms = _rms(Rmy * vol)

    elif norm_mode == 'weighted':
        # 加权残差：乘以sqrt(vol)（逐点）
        # RMS = sqrt(mean((R * sqrt(vol))^2))
        if vol is None:
            raise ValueError("norm_mode='weighted' requires vol to be provided")
        sqrt_vol = torch.sqrt(vol)
        Rc_rms = _rms(Rc * sqrt_vol)
        Rmx_rms = _rms(Rmx * sqrt_vol)
        Rmy_rms = _rms(Rmy * sqrt_vol)

    else:
        raise ValueError(f"Unknown norm_mode: {norm_mode}. Choose 'standard'/'adflow', 'flux'/'plain', or 'weighted'.")

    return Rc_rms, Rmx_rms, Rmy_rms


def assemble_residuals_5eq(
    fluxes: Dict[str, torch.Tensor],
    vol: torch.Tensor,
    periodic_xi: bool = True,
    include_viscous: bool = False,
    include_sa: bool = False,
    debug: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    装配5方程PDE残差（连续性 + 动量x + 动量y + 能量 + SA湍流）

    统一使用完整η边界面（j=0壁面和j=H远场），适用于RANS方程组。

    Args:
        fluxes: 通量字典，包含：
            - 对流通量（ξ方向）：'Fc_xi', 'Fmx_xi', 'Fmy_xi', 'FE_xi'
            - 对流通量（η方向，完整面）：'Fc_eta', 'Fmx_eta', 'Fmy_eta', 'FE_eta'
            - 粘性通量（如果include_viscous）：
                ξ方向：'Vmx_xi', 'Vmy_xi', 'VE_xi'
                η方向：'Vmx_eta', 'Vmy_eta', 'VE_eta'
            - SA残差（如果include_sa）：'R_SA'（已经是per-volume残差）
        vol: 单元体积 (H, W) 或 (batch, H, W)
        periodic_xi: ξ方向周期性
        include_viscous: 是否包含粘性项
        include_sa: 是否包含SA湍流方程
        debug: 是否输出调试信息

    Returns:
        (Rc, Rmx, Rmy, RE, RSA): 5个方程残差 (batch, H, W) 或 (H, W)
            - Rc: 连续性方程残差
            - Rmx: x动量方程残差
            - Rmy: y动量方程残差
            - RE: 能量方程残差
            - RSA: SA湍流方程残差（如果include_sa=False，返回None）

    注意：
        - 能量方程通量必须在fluxes中提供（FE_xi, FE_eta）
        - SA残差直接从fluxes['R_SA']获取（不需要通量装配）
    """
    # ========== 前3个方程：连续性 + 动量 ==========
    Rc, Rmx, Rmy = assemble_residuals(
        fluxes, vol, periodic_xi, include_viscous, debug
    )

    # ========== 第4个方程：能量 ==========

    # 验证能量通量存在
    if 'FE_xi' not in fluxes or 'FE_eta' not in fluxes:
        raise ValueError(
            "assemble_residuals_5eq requires 'FE_xi' and 'FE_eta' "
            "for energy equation residual. These should be computed by "
            "inviscid_central_flux with rhoE parameter."
        )

    if include_viscous:
        # 能量粘性通量
        if 'VE_xi' not in fluxes or 'VE_eta' not in fluxes:
            raise ValueError(
                "assemble_residuals_5eq with include_viscous=True requires "
                "'VE_xi' and 'VE_eta' for energy viscous flux."
            )

        # 总能量通量 = 对流 - 粘性
        FE_xi_total = fluxes['FE_xi'] - fluxes['VE_xi']
        FE_eta_total = fluxes['FE_eta'] - fluxes['VE_eta']

        RE = compute_flux_balance(
            FE_xi_total, FE_eta_total, vol, periodic_xi, debug, equation_type='energy'
        )
    else:
        # 仅对流通量
        RE = compute_flux_balance(
            fluxes['FE_xi'], fluxes['FE_eta'], vol, periodic_xi, debug, equation_type='energy'
        )

    # ========== 第5个方程：SA湍流（如果启用） ==========

    if include_sa:
        # SA残差直接从fluxes获取（已经是per-volume residual）
        if 'R_SA' not in fluxes:
            raise ValueError(
                "assemble_residuals_5eq with include_sa=True requires "
                "'R_SA' in fluxes dict. This should be computed by SA residual calculator."
            )
        RSA = fluxes['R_SA']
    else:
        RSA = None

    return Rc, Rmx, Rmy, RE, RSA
