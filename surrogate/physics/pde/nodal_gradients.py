"""
ADFLOW节点梯度计算（体积法）- 与ADflow allNodalGradients完全对齐

实现ADFLOW blockette.F90:5350-5634的2D伪3D退化版本

原理：
    节点梯度 = (1/V_node) * Σ_faces [ubar * S⃗_face]

其中：
    - V_node: 节点控制体体积（周围4个单元体积之和）
    - ubar: 面上的值（相邻2单元平均，从3D的4单元退化）
    - S⃗_face: 面积向量（相邻4面求和，从3D的8面退化）

ADflow 3D → 2D 伪3D退化规则（thin axis=j）：
    - 3D中4单元平均 → 2D中2单元平均
    - 3D中8面求和 → 2D中4面求和
    - 3D中8个体积 → 2D中4个体积

索引映射（本算例 thin axis=j，CGNS维度 ni=305, nj=2, nk=85）：
    - ADFLOW k方向（5354-5434）→ PyTorch η面贡献（径向/非周期）
      * ubar沿i(W)平均: 0.5*(phi[k,i] + phi[k,i+1])
      * 使用sj面积向量（η-face）
      * 散布到η方向(H)两个节点，节点列n=i+1固定
    - ADFLOW j方向（5440-5516）→ thin方向（退化，不贡献x/y梯度）
    - ADFLOW i方向（5522-5598）→ PyTorch ξ面贡献（切向/周期）
      * ubar沿k(H)平均: 0.5*(phi_hat[m,i] + phi_hat[m+1,i])
      * 使用si面积向量（ξ-face）
      * 散布到ξ方向(W)两个节点，节点行m固定
"""

import torch
from typing import Tuple, Optional


_DEBUG_COMPONENT_CALL_COUNTER = 0


def compute_nodal_gradients_volume_method(
    phi: torch.Tensor,
    si_x: torch.Tensor,
    si_y: torch.Tensor,
    sj_x: torch.Tensor,
    sj_y: torch.Tensor,
    volumes: torch.Tensor,
    halo_eta_bottom: Optional[torch.Tensor] = None,
    halo_eta_top: Optional[torch.Tensor] = None,
    periodic_xi: bool = True,
    # Plan91: 新增 halo 几何参数
    halo_vol_wall: Optional[torch.Tensor] = None,
    halo_vol_ff: Optional[torch.Tensor] = None,
    si_x_halo_wall: Optional[torch.Tensor] = None,
    si_y_halo_wall: Optional[torch.Tensor] = None,
    si_x_halo_ff: Optional[torch.Tensor] = None,
    si_y_halo_ff: Optional[torch.Tensor] = None,
    # Plan96: 新增完整η面向量参数
    sj_x_hat: Optional[torch.Tensor] = None,
    sj_y_hat: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算节点梯度（与ADflow allNodalGradients对齐）

    参考: ADFLOW blockette.F90:5350-5634

    Args:
        phi: 标量场 (batch, H, W) 或 (H, W)
        si_x: ξ方向面积向量x分量 (batch, H, W) 或 (H, W)
        si_y: ξ方向面积向量y分量 (batch, H, W) 或 (H, W)
        sj_x: η方向面积向量x分量 (batch, H+1, W) 或 (H+1, W)
        sj_y: η方向面积向量y分量 (batch, H+1, W) 或 (H+1, W)
        volumes: 单元体积 (batch, H, W) 或 (H, W)
        halo_eta_bottom: 壁面halo值 (batch, W) 或 (W,) - 必须提供！
        halo_eta_top: 远场halo值 (batch, W) 或 (W,) - 必须提供！
        periodic_xi: ξ方向是否周期性（O-grid为True）

    Returns:
        dphi_dx_node: 节点x梯度 (batch, H+1, W+1) 或 (H+1, W+1)
        dphi_dy_node: 节点y梯度 (batch, H+1, W+1) 或 (H+1, W+1)

    网格拓扑（2D O-grid）：
        - 单元: (H, W) = (84, 292)
        - 节点: (H+1, W+1) = (85, 293)
        - η方向: j=0..H，j=0是壁面，j=H是远场
        - ξ方向: i=0..W-1，周期（i=W等于i=0）

    ADflow对齐说明：
        - 节点(m, n)在η面的贡献：来自η面m（下方）和η面m+1（上方）
        - 节点(m, n)在ξ面的贡献：来自ξ面n-1（左侧）和ξ面n（右侧）
    """
    # 验证halo必须提供
    if halo_eta_bottom is None or halo_eta_top is None:
        raise ValueError(
            "compute_nodal_gradients_volume_method requires halo_eta_bottom and halo_eta_top. "
            "These must be provided from the boundary condition module."
        )

    # P2修复：强制转换为float64以提高近壁数值精度
    # 近壁小体积 + 大量相减在float32下容易导致数值误差
    original_dtype = phi.dtype
    phi = phi.to(torch.float64)
    si_x = si_x.to(torch.float64)
    si_y = si_y.to(torch.float64)
    sj_x = sj_x.to(torch.float64)
    sj_y = sj_y.to(torch.float64)
    volumes = volumes.to(torch.float64)
    halo_eta_bottom = halo_eta_bottom.to(torch.float64)
    halo_eta_top = halo_eta_top.to(torch.float64)
    if halo_vol_wall is not None:
        halo_vol_wall = halo_vol_wall.to(torch.float64)
    if halo_vol_ff is not None:
        halo_vol_ff = halo_vol_ff.to(torch.float64)
    if si_x_halo_wall is not None:
        si_x_halo_wall = si_x_halo_wall.to(torch.float64)
    if si_y_halo_wall is not None:
        si_y_halo_wall = si_y_halo_wall.to(torch.float64)
    if si_x_halo_ff is not None:
        si_x_halo_ff = si_x_halo_ff.to(torch.float64)
    if si_y_halo_ff is not None:
        si_y_halo_ff = si_y_halo_ff.to(torch.float64)
    if sj_x_hat is not None:
        sj_x_hat = sj_x_hat.to(torch.float64)
    if sj_y_hat is not None:
        sj_y_hat = sj_y_hat.to(torch.float64)

    # 处理batch维度
    squeeze_output = False
    if phi.ndim == 2:
        phi = phi.unsqueeze(0)
        si_x = si_x.unsqueeze(0)
        si_y = si_y.unsqueeze(0)
        sj_x = sj_x.unsqueeze(0)
        sj_y = sj_y.unsqueeze(0)
        volumes = volumes.unsqueeze(0)
        halo_eta_bottom = halo_eta_bottom.unsqueeze(0) if halo_eta_bottom.ndim == 1 else halo_eta_bottom
        halo_eta_top = halo_eta_top.unsqueeze(0) if halo_eta_top.ndim == 1 else halo_eta_top
        squeeze_output = True

    batch, H, W = phi.shape
    device, dtype = phi.device, phi.dtype

    # 步骤1: 构造halo扩展的phi（对应ADflow的double halo）
    # phi_padded: (batch, H+2, W)
    # 索引: 0=halo_bottom, 1..H=物理单元, H+1=halo_top
    phi_padded = torch.zeros(batch, H + 2, W, device=device, dtype=dtype)
    phi_padded[:, 0, :] = halo_eta_bottom  # 壁面halo
    phi_padded[:, 1:H + 1, :] = phi         # 物理单元
    phi_padded[:, H + 1, :] = halo_eta_top  # 远场halo

    import os
    debug_components_file = os.environ.get("SURROGATE_DEBUG_NODAL_COMPONENTS_FILE", "").strip()
    debug_components = len(debug_components_file) > 0

    # 步骤2: 初始化节点梯度（对应ADFLOW line 5350-5352）
    dphi_dx_node = torch.zeros(batch, H + 1, W + 1, device=device, dtype=dtype)
    dphi_dy_node = torch.zeros(batch, H + 1, W + 1, device=device, dtype=dtype)

    # 步骤3: ξ面贡献（对应ADFLOW i-part，line 5522-5598）
    # 沿k(H)方向平均，散布到i(W)方向两节点
    # 需要phi_padded（带halo）
    # Plan91 方案D: 传入 halo 面向量
    dphi_dx_node, dphi_dy_node = _accumulate_xi_face_contributions(
        phi_padded, si_x, si_y, dphi_dx_node, dphi_dy_node, periodic_xi,
        si_x_halo_wall=si_x_halo_wall,
        si_y_halo_wall=si_y_halo_wall,
        si_x_halo_ff=si_x_halo_ff,
        si_y_halo_ff=si_y_halo_ff
    )
    if debug_components:
        xi_num_x = dphi_dx_node.detach().clone()
        xi_num_y = dphi_dy_node.detach().clone()

    # 步骤4: η面贡献（对应ADFLOW k-part，line 5354-5434）
    # Plan96修复: 使用phi_padded（含halo）和sj_x_hat/sj_y_hat（含halo面）
    # 沿i(W)方向平均，散布到k(H)方向两节点
    if sj_x_hat is None or sj_y_hat is None:
        raise ValueError(
            "compute_nodal_gradients_volume_method requires sj_x_hat and sj_y_hat (Plan96). "
            "These must be computed using compute_halo_eta_face_vectors."
        )

    # 处理sj_x_hat/sj_y_hat的batch维度
    if sj_x_hat.ndim == 2:
        sj_x_hat_batched = sj_x_hat.unsqueeze(0).expand(batch, -1, -1)
        sj_y_hat_batched = sj_y_hat.unsqueeze(0).expand(batch, -1, -1)
    else:
        sj_x_hat_batched = sj_x_hat
        sj_y_hat_batched = sj_y_hat

    phi_ip1 = torch.roll(phi_padded, -1, dims=2)
    eta_ubar = 0.5 * (phi_padded + phi_ip1)
    sj_x_ip1 = torch.roll(sj_x_hat_batched, -1, dims=2)
    sj_y_ip1 = torch.roll(sj_y_hat_batched, -1, dims=2)
    eta_sx = (
        sj_x_hat_batched[:, :-1, :] + sj_x_hat_batched[:, 1:, :]
        + sj_x_ip1[:, :-1, :] + sj_x_ip1[:, 1:, :]
    )
    eta_sy = (
        sj_y_hat_batched[:, :-1, :] + sj_y_hat_batched[:, 1:, :]
        + sj_y_ip1[:, :-1, :] + sj_y_ip1[:, 1:, :]
    )
    eta_contrib_x = eta_ubar * eta_sx
    eta_contrib_y = eta_ubar * eta_sy

    dphi_dx_node, dphi_dy_node = _accumulate_eta_face_contributions(
        phi_padded, sj_x_hat_batched, sj_y_hat_batched, dphi_dx_node, dphi_dy_node, periodic_xi
    )
    if debug_components:
        total_num_x_prediv = dphi_dx_node.detach().clone()
        total_num_y_prediv = dphi_dy_node.detach().clone()
        eta_num_x = total_num_x_prediv - xi_num_x
        eta_num_y = total_num_y_prediv - xi_num_y

    # 步骤5: 除以节点体积（对应ADFLOW line 5600-5633）
    # Plan91 方案C: 使用真实 halo 体积
    node_volumes = _compute_node_volumes(
        volumes, periodic_xi,
        halo_vol_wall=halo_vol_wall,
        halo_vol_ff=halo_vol_ff
    )
    if debug_components:
        node_volumes_prediv = node_volumes.detach().clone()
    dphi_dx_node = dphi_dx_node / node_volumes
    dphi_dy_node = dphi_dy_node / node_volumes
    if debug_components:
        total_grad_x_preseam = dphi_dx_node.detach().clone()
        total_grad_y_preseam = dphi_dy_node.detach().clone()
        xi_grad_x_preseam = xi_num_x / node_volumes_prediv
        xi_grad_y_preseam = xi_num_y / node_volumes_prediv
        eta_grad_x_preseam = eta_num_x / node_volumes_prediv
        eta_grad_y_preseam = eta_num_y / node_volumes_prediv

    # 步骤6: 周期seam闭合（确保node列0和W一致）
    # Plan91 方案E: 改为求和合并（不是平均）
    if periodic_xi:
        # 求和合并而非平均（与ADFlow的累加逻辑一致）
        sum_x = dphi_dx_node[:, :, 0] + dphi_dx_node[:, :, W]
        sum_y = dphi_dy_node[:, :, 0] + dphi_dy_node[:, :, W]
        dphi_dx_node[:, :, 0] = sum_x
        dphi_dx_node[:, :, W] = sum_x
        dphi_dy_node[:, :, 0] = sum_y
        dphi_dy_node[:, :, W] = sum_y
    if debug_components:
        total_grad_x_postseam = dphi_dx_node.detach().clone()
        total_grad_y_postseam = dphi_dy_node.detach().clone()

    # 移除batch维度
    if squeeze_output:
        dphi_dx_node = dphi_dx_node.squeeze(0)
        dphi_dy_node = dphi_dy_node.squeeze(0)

    # P2修复：转回原始dtype
    dphi_dx_node = dphi_dx_node.to(original_dtype)
    dphi_dy_node = dphi_dy_node.to(original_dtype)

    # ========== DEBUG: 输出节点梯度 ==========
    if os.environ.get('SURROGATE_DEBUG_GRADIENT') == '1':
        import numpy as np
        # 保存节点梯度到npz文件（追加模式）
        debug_file = 'pytorch_nodal_gradient_debug.npz'

        # 获取现有数据（如果文件存在）
        existing_data = {}
        if os.path.exists(debug_file):
            with np.load(debug_file) as data:
                existing_data = {k: data[k] for k in data.files}

        # 转换为numpy
        grad_x = dphi_dx_node.detach().cpu().numpy()
        grad_y = dphi_dy_node.detach().cpu().numpy()

        # 如果有batch维度，取第一个
        if grad_x.ndim == 3:
            grad_x = grad_x[0]
            grad_y = grad_y[0]

        # 注意：这里的phi是标量场，调用者会多次调用此函数计算u和v的梯度
        # 使用不同的key来区分（通过phi的统计特征来猜测）
        phi_np = phi.detach().cpu().numpy()
        if phi_np.ndim == 3:
            phi_np = phi_np[0]
        phi_mean = phi_np.mean()

        # 简单启发式：u速度通常有较大均值，v速度均值接近0
        if abs(phi_mean) > 0.1:  # 可能是u
            existing_data['nodal_ux'] = grad_x
            existing_data['nodal_uy'] = grad_y
        else:  # 可能是v
            existing_data['nodal_vx'] = grad_x
            existing_data['nodal_vy'] = grad_y

        np.savez(debug_file, **existing_data)
        print(f"  [DEBUG] Saved nodal gradients to {debug_file}, phi_mean={phi_mean:.4f}")

    if debug_components:
        import numpy as np
        global _DEBUG_COMPONENT_CALL_COUNTER

        def _to_numpy(arr: torch.Tensor) -> np.ndarray:
            return np.asarray(arr.detach().cpu().numpy(), dtype=np.float64)

        def _to_numpy_slice(arr: torch.Tensor, batch_idx: int) -> np.ndarray:
            if arr.ndim == 0:
                return np.asarray(arr.detach().cpu().numpy(), dtype=np.float64)
            return np.asarray(arr[batch_idx].detach().cpu().numpy(), dtype=np.float64)

        existing_data: dict[str, np.ndarray] = {}
        if os.path.exists(debug_components_file):
            with np.load(debug_components_file, allow_pickle=False) as data:
                existing_data = {str(k): np.asarray(data[k]) for k in data.files}

        phi_np = phi.detach().cpu().numpy()
        phi_means = np.asarray(phi_np.reshape(phi_np.shape[0], -1).mean(axis=1), dtype=np.float64)
        call_idx = int(_DEBUG_COMPONENT_CALL_COUNTER)
        _DEBUG_COMPONENT_CALL_COUNTER = call_idx + 1
        existing_data[f"call{call_idx}_phi_means"] = phi_means
        tensor_map = {
            "phi_padded": phi_padded,
            "eta_ubar": eta_ubar,
            "eta_Sx": eta_sx,
            "eta_Sy": eta_sy,
            "eta_contrib_x": eta_contrib_x,
            "eta_contrib_y": eta_contrib_y,
            "xi_num_x_prediv": xi_num_x,
            "xi_num_y_prediv": xi_num_y,
            "eta_num_x_prediv": eta_num_x,
            "eta_num_y_prediv": eta_num_y,
            "total_num_x_prediv": total_num_x_prediv,
            "total_num_y_prediv": total_num_y_prediv,
            "node_volumes": node_volumes_prediv,
            "xi_grad_x_preseam": xi_grad_x_preseam,
            "xi_grad_y_preseam": xi_grad_y_preseam,
            "eta_grad_x_preseam": eta_grad_x_preseam,
            "eta_grad_y_preseam": eta_grad_y_preseam,
            "total_grad_x_preseam": total_grad_x_preseam,
            "total_grad_y_preseam": total_grad_y_preseam,
            "total_grad_x_postseam": total_grad_x_postseam,
            "total_grad_y_postseam": total_grad_y_postseam,
        }
        for name, tensor in tensor_map.items():
            existing_data[f"call{call_idx}_{name}"] = _to_numpy(tensor)

        if call_idx == 0 and int(phi.shape[0]) == 2:
            for alias, batch_idx in (("u", 0), ("v", 1)):
                for name, tensor in tensor_map.items():
                    existing_data[f"{alias}_{name}"] = _to_numpy_slice(tensor, batch_idx)
        np.savez(debug_components_file, **existing_data)

    return dphi_dx_node, dphi_dy_node


def _accumulate_xi_face_contributions(
    phi_padded: torch.Tensor,
    si_x: torch.Tensor,
    si_y: torch.Tensor,
    dphi_dx_node: torch.Tensor,
    dphi_dy_node: torch.Tensor,
    periodic_xi: bool,
    # Plan91 方案D: halo 面向量参数
    si_x_halo_wall: Optional[torch.Tensor] = None,
    si_y_halo_wall: Optional[torch.Tensor] = None,
    si_x_halo_ff: Optional[torch.Tensor] = None,
    si_y_halo_ff: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    累积ξ面（周期）对节点梯度的贡献 - 对应ADFLOW i-part

    参考: blockette.F90:5522-5598

    关键公式（plan88.md第93-124行）:
    1. ubar = 0.5 * (phi_hat[m, i] + phi_hat[m+1, i])  # 沿k(H)平均，需要halo
    2. S = 4个ξ-face面积向量: si_hat[m,fL] + si_hat[m+1,fL] + si_hat[m,fR] + si_hat[m+1,fR]
       其中 fL = (i-1) mod W, fR = i
    3. 散布: Gx[:, m, i] += ubar*S, Gx[:, m, i+1] -= ubar*S
       节点行m固定，只在i(W)方向散布！

    Args:
        phi_padded: (batch, H+2, W) halo扩展的单元值
        si_x, si_y: (batch, H, W) ξ面面积向量
        dphi_dx_node, dphi_dy_node: (batch, H+1, W+1) 累积中
        periodic_xi: 周期性标志

    Returns:
        更新后的 (dphi_dx_node, dphi_dy_node)
    """
    batch, H_plus_2, W = phi_padded.shape
    H = H_plus_2 - 2

    # ======== ADFLOW i-part: 沿k方向平均，散布到i方向两节点 ========

    # 1. 扩展si到 cell-centered width W 以匹配 phi_padded。
    # periodic_xi=True: si 已经是 W 个周期面。
    # periodic_xi=False: si 只有 W-1 个内部面，这里将其放到 [:W-1]，
    # 右端缺失边界面保持为 0。外层 halo 窗口会丢弃受其影响的边界单元。
    si_x_hat = torch.zeros(batch, H + 2, W, device=si_x.device, dtype=si_x.dtype)
    si_y_hat = torch.zeros(batch, H + 2, W, device=si_x.device, dtype=si_x.dtype)
    xi_face_width = int(si_x.shape[-1])
    if periodic_xi:
        if xi_face_width != W:
            raise ValueError(
                f"Periodic xi expects W faces, got si_x width={xi_face_width}, W={W}"
            )
        si_x_hat[:, 1:H+1, :] = si_x
        si_y_hat[:, 1:H+1, :] = si_y
    else:
        if xi_face_width != W - 1:
            raise ValueError(
                "Non-periodic xi expects W-1 internal faces, "
                f"got si_x width={xi_face_width}, W={W}"
            )
        si_x_hat[:, 1:H+1, :W-1] = si_x
        si_y_hat[:, 1:H+1, :W-1] = si_y

    # 首行(halo_wall) - 必须提供真实 halo 面向量
    if si_x_halo_wall is None or si_y_halo_wall is None:
        raise ValueError(
            "si_x_halo_wall and si_y_halo_wall must be provided. "
            "No fallback allowed per Plan91."
        )
    if si_x_halo_wall.ndim == 1:
        if periodic_xi:
            si_x_hat[:, 0, :] = si_x_halo_wall.unsqueeze(0).expand(batch, -1)
            si_y_hat[:, 0, :] = si_y_halo_wall.unsqueeze(0).expand(batch, -1)
        else:
            si_x_hat[:, 0, :W-1] = si_x_halo_wall.unsqueeze(0).expand(batch, -1)
            si_y_hat[:, 0, :W-1] = si_y_halo_wall.unsqueeze(0).expand(batch, -1)
    else:
        if periodic_xi:
            si_x_hat[:, 0, :] = si_x_halo_wall
            si_y_hat[:, 0, :] = si_y_halo_wall
        else:
            si_x_hat[:, 0, :W-1] = si_x_halo_wall
            si_y_hat[:, 0, :W-1] = si_y_halo_wall

    # 末行(halo_ff) - 必须提供真实 halo 面向量
    if si_x_halo_ff is None or si_y_halo_ff is None:
        raise ValueError(
            "si_x_halo_ff and si_y_halo_ff must be provided. "
            "No fallback allowed per Plan91."
        )
    if si_x_halo_ff.ndim == 1:
        if periodic_xi:
            si_x_hat[:, H+1, :] = si_x_halo_ff.unsqueeze(0).expand(batch, -1)
            si_y_hat[:, H+1, :] = si_y_halo_ff.unsqueeze(0).expand(batch, -1)
        else:
            si_x_hat[:, H+1, :W-1] = si_x_halo_ff.unsqueeze(0).expand(batch, -1)
            si_y_hat[:, H+1, :W-1] = si_y_halo_ff.unsqueeze(0).expand(batch, -1)
    else:
        if periodic_xi:
            si_x_hat[:, H+1, :] = si_x_halo_ff
            si_y_hat[:, H+1, :] = si_y_halo_ff
        else:
            si_x_hat[:, H+1, :W-1] = si_x_halo_ff
            si_y_hat[:, H+1, :W-1] = si_y_halo_ff

    # 2. ubar（沿k/H方向2单元平均）- 形状(batch, H+1, W)
    # ubar[m, i] = 0.5 * (phi_hat[m, i] + phi_hat[m+1, i])
    ubar = 0.5 * (phi_padded[:, :-1, :] + phi_padded[:, 1:, :])  # (batch, H+1, W)

    # 3. 面积向量求和（4个ξ-face）
    # 左face fL = (i-1) mod W，右face fR = i
    si_x_fL = torch.roll(si_x_hat, 1, dims=2)  # si_hat[:, :, i-1]
    si_y_fL = torch.roll(si_y_hat, 1, dims=2)

    # S = si_hat[m, fL] + si_hat[m+1, fL] + si_hat[m, fR] + si_hat[m+1, fR]
    Sx = (si_x_hat[:, :-1, :] + si_x_hat[:, 1:, :] +
          si_x_fL[:, :-1, :] + si_x_fL[:, 1:, :])  # (batch, H+1, W)
    Sy = (si_y_hat[:, :-1, :] + si_y_hat[:, 1:, :] +
          si_y_fL[:, :-1, :] + si_y_fL[:, 1:, :])

    # 4. 贡献值
    contrib_x = ubar * Sx  # (batch, H+1, W)
    contrib_y = ubar * Sy

    # 5. 散布到ξ方向两个节点（节点行固定！）
    # ADFLOW i-part: i-1节点 += ubar*S, i节点 -= ubar*S
    # 节点列i接收contrib[i]的正贡献
    dphi_dx_node[:, :H+1, :W] += contrib_x
    dphi_dy_node[:, :H+1, :W] += contrib_y

    # 节点列i+1接收contrib[i]的负贡献
    if periodic_xi:
        # 节点列1..W接收负贡献
        dphi_dx_node[:, :H+1, 1:W+1] -= contrib_x
        dphi_dy_node[:, :H+1, 1:W+1] -= contrib_y
    else:
        dphi_dx_node[:, :H+1, 1:W] -= contrib_x[:, :, :-1]
        dphi_dy_node[:, :H+1, 1:W] -= contrib_y[:, :, :-1]

    return dphi_dx_node, dphi_dy_node


def _accumulate_eta_face_contributions(
    phi_padded: torch.Tensor,
    sj_x_hat: torch.Tensor,
    sj_y_hat: torch.Tensor,
    dphi_dx_node: torch.Tensor,
    dphi_dy_node: torch.Tensor,
    periodic_xi: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    累积η面（非周期）对节点梯度的贡献 - 对应ADFLOW k-part

    Plan96修复: 包含halo cell贡献，与ADflow allNodalGradients完全对齐

    参考: blockette.F90:5354-5434

    ADflow循环范围: do k = 1, ke（包含halo cell）
    散布规则（if-guard）:
        - if (k > 1): ux(i,j,k-1) += ubar*sx  # 下方节点
        - if (k < ke): ux(i,j,k) -= ubar*sx   # 上方节点

    关键公式:
    1. ubar = 0.5 * (phi_padded[k, i] + phi_padded[k, i+1])  # 沿i(W)平均
    2. S = 4个η-face面积向量: sj_hat[k,i] + sj_hat[k+1,i] + sj_hat[k,i+1] + sj_hat[k+1,i+1]
    3. 散布（含if-guard）:
       - if k > 0: Gx[:, k-1, n] += ubar*S
       - if k < H+1: Gx[:, k, n] -= ubar*S
       其中 n = i+1（节点列比单元列偏移1）

    Args:
        phi_padded: (batch, H+2, W) halo扩展的单元中心值
            - [0]: wall halo, [1:H+1]: 物理单元, [H+1]: farfield halo
        sj_x_hat, sj_y_hat: (batch, H+3, W) 完整η面面积向量
            - [0]: wall外侧halo面, [1:H+2]: 物理η面, [H+2]: farfield外侧halo面
        dphi_dx_node, dphi_dy_node: (batch, H+1, W+1) 累积中
        periodic_xi: 周期性标志

    Returns:
        更新后的 (dphi_dx_node, dphi_dy_node)
    """
    batch, H_plus_2, W = phi_padded.shape
    H = H_plus_2 - 2  # 物理单元数

    # ======== ADFLOW k-part: 沿i方向平均，散布到k方向两节点 ========
    # 循环 k=0..H+1（对应ADflow k=1..ke，包含两个halo cell）

    # 1. ubar（沿i/W方向2单元平均）- 对所有单元（含halo）
    # ubar[k, i] = 0.5 * (phi_padded[k, i] + phi_padded[k, i+1])
    phi_ip1 = torch.roll(phi_padded, -1, dims=2)  # phi[:, :, i+1]（周期）
    ubar = 0.5 * (phi_padded + phi_ip1)  # (batch, H+2, W)

    # 2. 面积向量求和（4个η-face）
    # S = sj_hat[k,i] + sj_hat[k+1,i] + sj_hat[k,i+1] + sj_hat[k+1,i+1]
    # sj_hat形状是(batch, H+3, W)
    sj_x_ip1 = torch.roll(sj_x_hat, -1, dims=2)
    sj_y_ip1 = torch.roll(sj_y_hat, -1, dims=2)

    # 对于每个单元(k, i)，计算4个η-face求和
    # k从0到H+1（含halo），对应的面索引是k和k+1
    Sx = (sj_x_hat[:, :-1, :] + sj_x_hat[:, 1:, :] +
          sj_x_ip1[:, :-1, :] + sj_x_ip1[:, 1:, :])  # (batch, H+2, W)
    Sy = (sj_y_hat[:, :-1, :] + sj_y_hat[:, 1:, :] +
          sj_y_ip1[:, :-1, :] + sj_y_ip1[:, 1:, :])

    # 3. 贡献值
    contrib_x = ubar * Sx  # (batch, H+2, W)
    contrib_y = ubar * Sy

    # 4. 散布到η方向两个节点（节点列固定！）
    # ADflow k-part散布规则（Plan96核心修复）:
    #   - if (k > 1): ux(i,j,k-1) += ubar*sx  (k-1节点接收正贡献)
    #   - if (k < ke): ux(i,j,k) -= ubar*sx   (k节点接收负贡献)
    #
    # 在我们的0-based索引中:
    #   - 单元k=0（wall halo）: 只影响节点行0（正贡献），不影响节点行-1（不存在）
    #   - 单元k=1..H（物理单元）: 影响节点行k-1（正）和节点行k（负）
    #   - 单元k=H+1（farfield halo）: 只影响节点行H（负贡献），不影响节点行H+1（不存在）
    #
    # 节点列 n = i+1，所以 contrib[k, i] 散布到节点列 i+1

    # 正贡献: 节点行k-1（k从1到H+1，对应节点行0到H）
    # contrib[k=1..H+1, :] -> node[0..H, :]
    dphi_dx_node[:, :H+1, 1:W+1] += contrib_x[:, 1:, :]
    dphi_dy_node[:, :H+1, 1:W+1] += contrib_y[:, 1:, :]

    # 负贡献: 节点行k（k从0到H，对应节点行0到H）
    # contrib[k=0..H, :] -> node[0..H, :]
    dphi_dx_node[:, :H+1, 1:W+1] -= contrib_x[:, :H+1, :]
    dphi_dy_node[:, :H+1, 1:W+1] -= contrib_y[:, :H+1, :]

    return dphi_dx_node, dphi_dy_node


def _compute_node_volumes(
    volumes: torch.Tensor,
    periodic_xi: bool,
    # Plan91 方案C: halo 体积参数
    halo_vol_wall: Optional[torch.Tensor] = None,
    halo_vol_ff: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    计算节点控制体体积 - 与ADFLOW同构（向量化版本）

    参考: blockette.F90:5608-5611

    对每个节点(m, n)，体积 = 4个相邻单元体积之和：
    V_node[m, n] = vol[m, iL] + vol[m, iR] + vol[m+1, iL] + vol[m+1, iR]
    其中 iL = (n-1) mod W, iR = n mod W

    关键：与散布拓扑同构
    - η面贡献写入节点列1..W
    - ξ面贡献写入节点列0..W
    - 节点列0和W由周期seam闭合强制一致

    Plan91 方案C: 支持传入真实 halo 单元体积

    Args:
        volumes: 单元体积 (batch, H, W)
        periodic_xi: ξ方向周期性
        halo_vol_wall: 壁面 halo 层体积 (W,) 或 (batch, W) - 可选
        halo_vol_ff: 远场 halo 层体积 (W,) 或 (batch, W) - 可选

    Returns:
        node_volumes: 节点体积 (batch, H+1, W+1)
    """
    batch, H, W = volumes.shape
    device, dtype = volumes.device, volumes.dtype

    # 扩展volumes到(batch, H+2, W)以匹配散布索引
    vol_hat = torch.zeros(batch, H + 2, W, device=device, dtype=dtype)
    vol_hat[:, 1:H+1, :] = volumes

    # Plan91 方案C: 必须提供真实 halo 体积
    if halo_vol_wall is None:
        raise ValueError(
            "halo_vol_wall must be provided. No fallback allowed per Plan91."
        )
    if halo_vol_wall.ndim == 1:
        vol_hat[:, 0, :] = halo_vol_wall.unsqueeze(0).expand(batch, -1)
    else:
        vol_hat[:, 0, :] = halo_vol_wall

    if halo_vol_ff is None:
        raise ValueError(
            "halo_vol_ff must be provided. No fallback allowed per Plan91."
        )
    if halo_vol_ff.ndim == 1:
        vol_hat[:, H+1, :] = halo_vol_ff.unsqueeze(0).expand(batch, -1)
    else:
        vol_hat[:, H+1, :] = halo_vol_ff

    # 周期左移：vol_iL = vol[:, :, (i-1) mod W]
    vol_iL = torch.roll(vol_hat, 1, dims=2)

    # 4单元求和：[m,iL] + [m,iR] + [m+1,iL] + [m+1,iR]
    # 这里 iR = 当前列, iL = 当前列-1 (mod W)
    V_sum = (vol_hat[:, :-1, :] + vol_iL[:, :-1, :] +
             vol_hat[:, 1:, :] + vol_iL[:, 1:, :])  # (batch, H+1, W)

    # CRITICAL FIX (Plan98): ADflow边界节点体积**包含halo单元**！
    # 对应ADflow allNodalGradients (flowUtils.F90:2024-2027)：
    #   oneOverV = 1 / (vol(i,j,k) + vol(i+1,j,k) + vol(i,j+1,k) + vol(i+1,j+1,k)
    #                  + vol(i,j,k+1) + vol(i+1,j,k+1) + vol(i,j+1,k+1) + vol(i+1,j+1,k+1))
    # 在2D伪3D中（thin axis=j）：
    #   - Wall节点(m=0): V_sum[:, 0, :] = vol_hat[:, 0, :] + vol_iL[:, 0, :]
    #                                    + vol_hat[:, 1, :] + vol_iL[:, 1, :]
    #                   = halo_vol_wall(i) + halo_vol_wall(i-1) + vol[0](i) + vol[0](i-1)
    #                   → 包含halo + 第0层物理单元 ✓
    #   - Farfield节点(m=H): V_sum[:, H, :] = vol_hat[:, H, :] + vol_iL[:, H, :]
    #                                        + vol_hat[:, H+1, :] + vol_iL[:, H+1, :]
    #                        = vol[H-1](i) + vol[H-1](i-1) + halo_vol_ff(i) + halo_vol_ff(i-1)
    #                        → 第H-1层物理单元 + 包含halo ✓
    #
    # V_sum的默认计算（无需特判）已经正确！
    # 旧的特判代码（已删除）：
    #   V_sum[:, 0, :] = vol_hat[:, 1, :] + vol_iL[:, 1, :]  # 错误：不含halo
    #   V_sum[:, -1, :] = vol_hat[:, H, :] + vol_iL[:, H, :]  # 错误：不含halo

    # 填充节点体积数组
    node_vols = torch.zeros(batch, H + 1, W + 1, device=device, dtype=dtype)

    # Plan93 方案J: 使用0-based索引（与dphi_dx_node一致）
    # V_sum[:, :, i] = 节点列i周围的4单元体积和（roll已处理周期性）
    # 所以 node_vols[:, :, i] = V_sum[:, :, i] = 节点列i的体积
    node_vols[:, :, :W] = V_sum

    # Plan93 方案J: 周期闭合使用简单复制（不是求和）
    # V_sum[:, :, 0] 已经通过roll正确包含了周期wrapped的单元
    # 节点W是节点0的周期副本，直接复制
    if periodic_xi:
        node_vols[:, :, W] = node_vols[:, :, 0]

    # 避免除零
    node_vols = torch.clamp(node_vols, min=1e-30)

    return node_vols
