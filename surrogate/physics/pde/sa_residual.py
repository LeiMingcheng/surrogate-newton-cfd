"""
Spalart-Allmaras湍流模型残差计算模块 - ADFLOW完全对齐版本

实现与ADFLOW逐点一致的SA湍流方程残差计算，所有组件严格遵循ADFLOW算法。

组件对齐状态：
=============
✅ 源项（saSource）：完全对齐
   - fv1, fv2, fw, ft2等修正函数
   - strain/vorticity生成项基底
   - 所有中间变量公式与sa.F90:88-342完全一致

✅ 对流项（turbAdvection）：完全对齐
   - ADFLOW算法：1阶迎风 + 两个独立非线性修正（kappa=-1）
   - 与简化MUSCL的区别：使用两个独立minmod而非单一limiter
   - 正向/反向流动的差分方向正确处理
   - 参考：turbUtils.F90:825-978

✅ 粘性项（saViscous）：完全对齐
   - 三对角系数形式（非通量散度形式）
   - 度量张量 ttm, ttp 计算
   - cb2梯度修正项：cnud = -cb2 * nuTilde * cb3Inv
   - 非负性保证：c1m = max(cdm + cam, 0)
   - 参考：sa.F90:344-673

✅ SA常数：完全对齐 paramTurb.F90:13
   - cv1=7.1, cb1=0.1355, cb2=0.622, sigma=2/3
   - kappa=0.41, cw2=0.3, cw3=2.0, ct3=1.2, ct4=0.5
   - cb3Inv = 1.5 (= 1 / (2/3))

✅ 残差Scaling（saResScale）：正确
   - 返回 -R_raw（per-volume形式）
   - 与ADFLOW dw = -vol * scratch一致

边界条件：
=========
- 壁面（j=0）：nuTilde = 0（Dirichlet BC）
- 远场（j=H-1）：外推
- 周期（ξ方向）：roll连接首尾

参考ADFLOW源码：
================
- sa.F90:16 (sa_block) - 主SA残差计算入口
- sa.F90:88-342 (saSource) - 源项
- sa.F90:344-673 (saViscous) - 粘性项
- sa.F90:675-711 (saResScale) - 残差scaling
- turbUtils.F90:825-978 (turbAdvection) - 对流项
- turbUtils.F90:939-963 (forward flow corrections) - 正向流修正
- turbUtils.F90:1027-1051 (backward flow corrections) - 反向流修正
- paramTurb.F90 - SA模型常数
"""

import torch
from typing import Dict, Tuple, Optional


class SAResidualCalculator:
    """
    Spalart-Allmaras湍流模型残差计算器（ADFLOW完全对齐）

    参考：sa.F90:16 (sa_block)

    SA残差组成：
        R_SA = R_source + R_advection + R_viscous

    其中：
        - R_source: 生成项 - 破坏项
        - R_advection: 对流项（二阶MUSCL迎风）
        - R_viscous: 粘性/扩散项（包含cb2修正）
    """

    def __init__(
        self,
        gamma: float = 1.4,
        # SA常数（paramTurb.F90:13）
        cv1: float = 7.1,
        cb1: float = 0.1355,
        cb2: float = 0.622,
        sigma: float = 2.0/3.0,
        kappa: float = 0.41,
        cw2: float = 0.3,
        cw3: float = 2.0,
        ct3: float = 1.2,
        ct4: float = 0.5,
        # 数值参数
        xminn: float = 1e-10,  # nuTilde最小值（sa.F90:118）
        use_ft2: bool = True,  # ft2修正（ADFLOW默认启用，pyADflow.py:5680）
        use_rotation: bool = False,  # 旋转/曲率修正
        prod_mode: str = 'strain',  # 生成项基底（ADFLOW默认strain，inputParamRoutines.F90:3971）
        # ADFLOW对齐参数
        order_turb: str = 'first',  # 对流项离散阶次：'first'(默认,ADFLOW) 或 'second'
        approx_sa: bool = False,  # approxSA开关（ADFLOW默认False，sa.F90:294）
        device: str = 'cuda'
    ):
        """
        初始化SA残差计算器

        Args:
            gamma: 比热比
            cv1-ct4: SA模型常数（与ADFLOW完全对齐）
            xminn: nuTilde最小值（避免除零）
            use_ft2: 是否使用ft2修正函数（ADFLOW默认True）
            use_rotation: 是否使用旋转/曲率修正
            prod_mode: 生成项基底 'strain'(默认) 或 'vorticity'
            order_turb: 对流项离散阶次 'first'(默认,ADFLOW) 或 'second'
            approx_sa: approxSA开关，True时term1=0（ADFLOW默认False）
            device: 计算设备
        """
        self.gamma = gamma
        self.device = device

        # SA常数
        self.cv1 = cv1
        self.cb1 = cb1
        self.cb2 = cb2
        self.sigma = sigma
        self.kappa = kappa
        self.cw2 = cw2
        self.cw3 = cw3
        self.ct3 = ct3
        self.ct4 = ct4
        self.xminn = xminn
        self.use_ft2 = use_ft2
        self.use_rotation = use_rotation
        self.prod_mode = prod_mode
        self.order_turb = order_turb
        self.approx_sa = approx_sa

        # 派生常数（sa.F90:121-124）
        self.cw1 = cb1 / (kappa**2) + (1.0 + cb2) / sigma
        self.cv13 = cv1**3
        self.cw36 = cw3**6
        self.kar2Inv = 1.0 / (kappa**2)
        # ✅ ADFLOW paramTurb.F90:16: rsaCb3 = 2/3, cb3Inv = 1/rsaCb3 = 1.5
        rsaCb3 = 2.0 / 3.0
        self.cb3Inv = 1.0 / rsaCb3  # = 1.5 (ADFLOW标准)

    def compute_source_term(
        self,
        rho: torch.Tensor,
        nuTilde: torch.Tensor,
        mu_l: torch.Tensor,
        d_wall: torch.Tensor,
        du_dx: torch.Tensor,
        du_dy: torch.Tensor,
        dv_dx: torch.Tensor,
        dv_dy: torch.Tensor,
        vort_mag: Optional[torch.Tensor] = None  # 可选，向后兼容
    ) -> torch.Tensor:
        """
        SA源项（production - destruction）- ADFLOW精确对齐版本

        参考：sa.F90:88-342 (saSource subroutine)

        ADFlow源项公式（sa.F90:294-302）：
            term1 = cb1 * (1 - ft2) * ss                    # 生成项
            term2 = dist2Inv * (kar2Inv*cb1*((1-ft2)*fv2 + ft2) - cw1*fw)
            source = (term1 + term2 * nuTilde) * nuTilde    # 最终源项

        关键对齐点：
        1. 生成项基底ss默认使用strain（ADFLOW默认），支持vorticity
        2. ft2处理：生成项乘以(1-ft2)，term2使用((1-ft2)*fv2+ft2)组合
        3. fw公式无eps（因为cw3^6=64>0保证分母非零）

        Args:
            rho: 密度 (H, W) 或 (batch, H, W)
            nuTilde: SA湍流变量
            mu_l: 层流粘度
            d_wall: 壁面距离
            du_dx, du_dy, dv_dx, dv_dy: 速度梯度
            vort_mag: 涡量模（可选，向后兼容）

        Returns:
            source: SA源项 (H, W) 或 (batch, H, W)
        """
        from .sa_utils import compute_production_term

        # ADFLOW对齐：saSource 不对 nuTilde 做 xminn clamp（xminn 仅用于 sst 下限）

        # 1. 计算生成项基底ss（strain或vorticity，sa.F90:196-237）
        ss = compute_production_term(du_dx, du_dy, dv_dx, dv_dy, mode=self.prod_mode)

        # 2. Chi参数（sa.F90:245-247）
        # ADFLOW无eps：nu = rlv(i,j,k) / w(i,j,k,irho), chi = w(i,j,k,itu1) / nu
        nu = mu_l / rho
        chi = nuTilde / nu
        chi2 = chi ** 2
        chi3 = chi ** 3

        # 3. fv1函数（sa.F90:250）
        fv1 = chi3 / (chi3 + self.cv13)

        # 4. fv2函数（sa.F90:251）
        fv2 = 1.0 - chi / (1.0 + chi * fv1)

        # 5. ft2函数（sa.F90:257-261）
        if self.use_ft2:
            ft2 = self.ct3 * torch.exp(-self.ct4 * chi2)
        else:
            ft2 = torch.zeros_like(chi)

        # 6. 壁面距离相关计算
        d_wall_clamp = torch.clamp(d_wall, min=1e-14)
        dist2Inv = 1.0 / (d_wall_clamp ** 2)

        # 7. 修正生成项sst（sa.F90:266）
        sst = ss + nuTilde * fv2 * self.kar2Inv * dist2Inv
        sst = torch.clamp(sst, min=self.xminn)  # sa.F90:278

        # 8. r参数（sa.F90:284-285）
        rr = nuTilde * self.kar2Inv * dist2Inv / sst
        rr = torch.clamp(rr, max=10.0)

        # 9. g函数（sa.F90:286）
        gg = rr + self.cw2 * (rr ** 6 - rr)
        gg6 = gg ** 6

        # 10. fw函数（sa.F90:288-289）- 无eps
        # ADFLOW: termFw = ((one + cw36) / (gg6 + cw36))**sixth
        # 因为cw36 = 64 > 0，分母始终非零，无需eps
        termFw = ((1.0 + self.cw36) / (gg6 + self.cw36)) ** (1.0 / 6.0)
        fw = gg * termFw

        # 11. 源项（ADFlow精确公式 sa.F90:294-302）
        # approxSA开关：当approxSA=True时，term1=0（sa.F90:294-298）
        if self.approx_sa:
            term1 = torch.zeros_like(ss)
        else:
            # term1 = cb1 * (1-ft2) * ss  （关键：生成项乘以(1-ft2)）
            term1 = self.cb1 * (1.0 - ft2) * ss

        # term2 = dist2Inv * (kar2Inv*cb1*((1-ft2)*fv2 + ft2) - cw1*fw)
        # 关键：使用((1-ft2)*fv2 + ft2)组合，而非简单的ft2
        term2 = dist2Inv * (
            self.kar2Inv * self.cb1 * ((1.0 - ft2) * fv2 + ft2)
            - self.cw1 * fw
        )

        # 最终源项：source = (term1 + term2 * nuTilde) * nuTilde
        source = (term1 + term2 * nuTilde) * nuTilde

        return source

    def compute_advection_term(
        self,
        rho: torch.Tensor,
        nuTilde: torch.Tensor,
        contravariant_vel: Dict[str, torch.Tensor],  # {'uu_xi', 'uu_eta'}
        face_geom: Dict[str, torch.Tensor],
        vol: torch.Tensor,
        second_order: bool = True,
        limiter: str = 'minmod',
        # ✅ ADFLOW对齐：η方向需要两层halo（turb2ndHalo）
        halo_nuTilde_wall: Optional[torch.Tensor] = None,
        halo_nuTilde_wall_second: Optional[torch.Tensor] = None,
        halo_nuTilde_ff: Optional[torch.Tensor] = None,
        halo_nuTilde_ff_second: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        SA对流项 - 调用ADFLOW对齐版本

        已弃用简化MUSCL实现，现在使用compute_advection_term_adflow
        """
        return self.compute_advection_term_adflow(
            nuTilde=nuTilde,
            uu_xi=contravariant_vel['uu_xi'],
            uu_eta=contravariant_vel['uu_eta'],
            second_order=second_order,
            halo_nuTilde_wall=halo_nuTilde_wall,
            halo_nuTilde_wall_second=halo_nuTilde_wall_second,
            halo_nuTilde_ff=halo_nuTilde_ff,
            halo_nuTilde_ff_second=halo_nuTilde_ff_second
        )

    def compute_advection_term_adflow(
        self,
        nuTilde: torch.Tensor,
        uu_xi: torch.Tensor,
        uu_eta: torch.Tensor,
        second_order: bool = True,
        halo_nuTilde_wall: Optional[torch.Tensor] = None,
        halo_nuTilde_wall_second: Optional[torch.Tensor] = None,
        halo_nuTilde_ff: Optional[torch.Tensor] = None,
        halo_nuTilde_ff_second: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        SA对流项 - 严格遵循ADFLOW turbAdvection算法

        参考：turbUtils.F90:825-978 (turbAdvection)

        ADFLOW算法（kappa=-1完全迎风）：
        1. 基础一阶迎风：dwtk = dwt
        2. 第一个非线性修正（前向）：if (dwt*dwtp1>0): dwtk += 0.5*minmod(dwt, dwtp1)
        3. 第二个非线性修正（后向）：if (dwt*dwtm1>0): dwtk -= 0.5*minmod(dwt, dwtm1)
        4. 残差更新：R -= uu * dwtk

        与简化MUSCL的关键区别：
        - 使用两个独立的minmod修正而非单一limiter
        - 直接输出导数形式而非通量形式
        - 正向/反向流动的差分方向相反

        Args:
            nuTilde: SA变量 (batch, H, W) 或 (H, W)
            uu_xi: ξ方向逆变速度/vol (batch, H, W)，单元中心值
            uu_eta: η方向逆变速度/vol (batch, H, W)，单元中心值（ADFLOW定义）
            second_order: 是否使用二阶格式

        Returns:
            R_adv: 对流残差 (batch, H, W) 或 (H, W)
        """
        # 确保batch维度
        if nuTilde.ndim == 2:
            nuTilde = nuTilde.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # 确保uu_xi和uu_eta有正确的batch维度（避免重复unsqueeze）
        if uu_xi.ndim == 2:
            uu_xi = uu_xi.unsqueeze(0)
        if uu_eta.ndim == 2:
            uu_eta = uu_eta.unsqueeze(0)

        batch, H, W = nuTilde.shape
        R_adv = torch.zeros_like(nuTilde)

        # ========== ξ方向（周期边界） ==========
        # ADFLOW turbAdvection逐cell处理，我们使用向量化实现
        R_adv_xi = self._advection_direction_adflow(
            nuTilde, uu_xi, direction='xi', periodic=True, second_order=second_order
        )
        R_adv += R_adv_xi

        # ========== η方向（壁面-远场边界） ==========
        R_adv_eta = self._advection_direction_adflow(
            nuTilde,
            uu_eta,
            direction='eta',
            periodic=False,
            second_order=second_order,
            halo_phi_wall=halo_nuTilde_wall,
            halo_phi_wall_second=halo_nuTilde_wall_second,
            halo_phi_farfield=halo_nuTilde_ff,
            halo_phi_farfield_second=halo_nuTilde_ff_second
        )
        R_adv += R_adv_eta

        if squeeze_output:
            R_adv = R_adv.squeeze(0)

        return R_adv

    def _advection_direction_adflow(
        self,
        phi: torch.Tensor,
        uu: torch.Tensor,
        direction: str,
        periodic: bool,
        second_order: bool,
        halo_phi_wall: Optional[torch.Tensor] = None,
        halo_phi_wall_second: Optional[torch.Tensor] = None,
        halo_phi_farfield: Optional[torch.Tensor] = None,
        halo_phi_farfield_second: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        单方向对流项（ADFLOW turbAdvection算法）

        参考：turbUtils.F90:939-963（正向）, 1027-1051（反向）

        算法（正向流uu>0）：
            dwt   = phi[k] - phi[k-1]      # 迎风差分（基础）
            dwtm1 = phi[k-1] - phi[k-2]    # 后向差分
            dwtp1 = phi[k+1] - phi[k]      # 前向差分

            dwtk = dwt                      # 基础一阶
            if (dwt * dwtp1 > 0):          # 前向修正
                dwtk += 0.5 * minmod(dwt, dwtp1)
            if (dwt * dwtm1 > 0):          # 后向修正
                dwtk -= 0.5 * minmod(dwt, dwtm1)

            R[k] -= uu * dwtk

        Args:
            phi: 标量场 (batch, H, W)
            uu: 逆变速度/vol (batch, H, W)
            direction: 'xi' 或 'eta'
            periodic: 是否周期边界
            second_order: 是否二阶

        Returns:
            R: 对流残差贡献 (batch, H, W)
        """
        batch, H, W = phi.shape
        R = torch.zeros_like(phi)

        if direction == 'xi':
            # ξ方向（周期）
            # 索引：phi[..., i] 对应第i列

            # 差分（周期边界使用roll）
            # dwt[i] = phi[i] - phi[i-1]（当前-左邻居）
            dwt = phi - torch.roll(phi, 1, dims=-1)
            # dwtm1[i] = phi[i-1] - phi[i-2]
            dwtm1 = torch.roll(phi, 1, dims=-1) - torch.roll(phi, 2, dims=-1)
            # dwtp1[i] = phi[i+1] - phi[i]
            dwtp1 = torch.roll(phi, -1, dims=-1) - phi
            # dwtp2[i] = phi[i+2] - phi[i+1]（用于反向流二阶修正）
            dwtp2 = torch.roll(phi, -2, dims=-1) - torch.roll(phi, -1, dims=-1)

            if second_order:
                dwtk = self._compute_dwtk_adflow(dwt, dwtm1, dwtp1, uu, dwtp2=dwtp2)
            else:
                # 一阶迎风
                dwtk = torch.where(uu > 0, dwt, dwtp1)

            # 残差更新：R -= uu * dwtk
            R -= uu * dwtk

        elif direction == 'eta':
            # η方向（非周期）：ADFLOW turbAdvection需要两层halo（turb2ndHalo）
            if uu.shape != phi.shape:
                raise ValueError(f"uu shape {tuple(uu.shape)} must match phi shape {tuple(phi.shape)} for eta-advection.")

            # 默认halo（保持向后兼容，但不保证与ADFLOW远场入流一致）
            # 壁面：ADFLOW bmt=1,bvt=0 → halo = -phi[0]
            if halo_phi_wall is None:
                halo_phi_wall = -phi[:, 0, :]
            if halo_phi_wall_second is None:
                halo_phi_wall_second = halo_phi_wall

            # 远场：若未提供，默认零梯度外推
            if halo_phi_farfield is None:
                halo_phi_farfield = phi[:, -1, :]
            if halo_phi_farfield_second is None:
                halo_phi_farfield_second = halo_phi_farfield

            # 规范halo维度到(batch, W)
            def _as_batch_w(x: torch.Tensor) -> torch.Tensor:
                if x.ndim == 1:  # (W,)
                    return x.unsqueeze(0).expand(batch, -1)
                if x.ndim == 2:  # (batch, W)
                    if x.shape[0] == 1 and batch > 1:
                        x = x.expand(batch, -1)
                    if x.shape[0] != batch:
                        raise ValueError(f"halo must have batch dim {batch}, got {tuple(x.shape)}")
                    return x
                raise ValueError(f"Invalid halo shape: {tuple(x.shape)}; expected (W,) or (batch, W).")

            h2_wall = _as_batch_w(halo_phi_wall_second)
            h1_wall = _as_batch_w(halo_phi_wall)
            h1_ff = _as_batch_w(halo_phi_farfield)
            h2_ff = _as_batch_w(halo_phi_farfield_second)

            # 组装两层halo扩展数组：
            #   phi_hat[:,0]   = second halo at wall  (k=0)
            #   phi_hat[:,1]   = first  halo at wall  (k=1)
            #   phi_hat[:,2..] = physical cells       (k=2..H+1)
            #   phi_hat[:,H+2] = first  halo at far   (k=ke)
            #   phi_hat[:,H+3] = second halo at far   (k=kb)
            phi_hat = torch.cat(
                [
                    h2_wall.unsqueeze(-2),
                    h1_wall.unsqueeze(-2),
                    phi,
                    h1_ff.unsqueeze(-2),
                    h2_ff.unsqueeze(-2),
                ],
                dim=-2,
            )  # (batch, H+4, W)

            # 对应ADFLOW循环 k=2..kl（物理单元）：向量化构造差分
            phi_km2 = phi_hat[:, 0:H, :]
            phi_km1 = phi_hat[:, 1:H+1, :]
            phi_k = phi_hat[:, 2:H+2, :]
            phi_kp1 = phi_hat[:, 3:H+3, :]
            phi_kp2 = phi_hat[:, 4:H+4, :]

            dwtm1 = phi_km1 - phi_km2
            dwt = phi_k - phi_km1
            dwtp1 = phi_kp1 - phi_k
            dwtp2 = phi_kp2 - phi_kp1

            if second_order:
                dwtk = self._compute_dwtk_adflow(dwt, dwtm1, dwtp1, uu, dwtp2=dwtp2)
            else:
                dwtk = torch.where(uu > 0, dwt, dwtp1)

            R -= uu * dwtk

        return R

    def _compute_dwtk_adflow(
        self,
        dwt: torch.Tensor,
        dwtm1: torch.Tensor,
        dwtp1: torch.Tensor,
        uu: torch.Tensor,
        dwtp2: torch.Tensor = None
    ) -> torch.Tensor:
        """
        计算ADFLOW二阶迎风导数（turbUtils.F90:947-963, 1027-1051）

        正向流（uu > 0）(turbUtils.F90:939-963)：
            dwtm1 = w(k-1) - w(k-2)
            dwt   = w(k) - w(k-1)
            dwtp1 = w(k+1) - w(k)

            dwtk = dwt
            if (dwt * dwtp1 > 0):
                dwtk += 0.5 * minmod(dwt, dwtp1)  # 前向修正
            if (dwt * dwtm1 > 0):
                dwtk -= 0.5 * minmod(dwt, dwtm1)  # 后向修正

        反向流（uu <= 0）(turbUtils.F90:1027-1051)：
            dwtm1' = w(k) - w(k-1)     = dwt
            dwt'   = w(k+1) - w(k)     = dwtp1
            dwtp1' = w(k+2) - w(k+1)   = dwtp2  ← 关键：需要额外数据！

            dwtk = dwt'
            if (dwt' * dwtp1' > 0):
                dwtk -= 0.5 * minmod(dwt', dwtp1')  # 前向修正（符号相反）
            if (dwt' * dwtm1' > 0):
                dwtk += 0.5 * minmod(dwt', dwtm1')  # 后向修正（符号相反）

        Args:
            dwt: 迎风差分 phi[k] - phi[k-1]
            dwtm1: 后向差分 phi[k-1] - phi[k-2]
            dwtp1: 前向差分 phi[k+1] - phi[k]
            uu: 法向速度
            dwtp2: 更前向差分 phi[k+2] - phi[k+1]，用于反向流（可选）

        Returns:
            dwtk: 修正后的导数
        """
        # ========== 正向流 (uu > 0) ==========
        dwtk_pos = dwt.clone()

        # 第一个非线性修正（前向）
        # if (dwt * dwtp1 > 0): dwtk += 0.5 * minmod(dwt, dwtp1)
        same_sign_forward = (dwt * dwtp1 > 0)
        minmod_forward = torch.where(
            torch.abs(dwt) < torch.abs(dwtp1),
            dwt,
            dwtp1
        )
        dwtk_pos = dwtk_pos + 0.5 * minmod_forward * same_sign_forward.float()

        # 第二个非线性修正（后向）
        # if (dwt * dwtm1 > 0): dwtk -= 0.5 * minmod(dwt, dwtm1)
        same_sign_backward = (dwt * dwtm1 > 0)
        minmod_backward = torch.where(
            torch.abs(dwt) < torch.abs(dwtm1),
            dwt,
            dwtm1
        )
        dwtk_pos = dwtk_pos - 0.5 * minmod_backward * same_sign_backward.float()

        # ========== 反向流 (uu <= 0) ==========
        # ADFLOW turbUtils.F90:1027-1051
        # 反向流使用相反方向的差分
        # dwt'   = dwtp1 = w(k+1) - w(k)
        # dwtm1' = dwt   = w(k) - w(k-1)
        # dwtp1' = dwtp2 = w(k+2) - w(k+1)

        dwt_neg = dwtp1
        dwtm1_neg = dwt
        # dwtp1_neg = dwtp2 （如果提供）
        if dwtp2 is None:
            dwtp1_neg = torch.zeros_like(dwtp1)
        else:
            dwtp1_neg = dwtp2

        dwtk_neg = dwt_neg.clone()

        # 第一个非线性修正（前向，符号相反）
        # if (dwt' * dwtp1' > 0): dwtk -= 0.5 * minmod(dwt', dwtp1')
        same_sign_neg_forward = (dwt_neg * dwtp1_neg > 0)
        minmod_neg_forward = torch.where(
            torch.abs(dwt_neg) < torch.abs(dwtp1_neg),
            dwt_neg,
            dwtp1_neg
        )
        dwtk_neg = dwtk_neg - 0.5 * minmod_neg_forward * same_sign_neg_forward.float()

        # 第二个非线性修正（后向，符号相反）
        # if (dwt' * dwtm1' > 0): dwtk += 0.5 * minmod(dwt', dwtm1')
        same_sign_neg_backward = (dwt_neg * dwtm1_neg > 0)
        minmod_neg_backward = torch.where(
            torch.abs(dwt_neg) < torch.abs(dwtm1_neg),
            dwt_neg,
            dwtm1_neg
        )
        dwtk_neg = dwtk_neg + 0.5 * minmod_neg_backward * same_sign_neg_backward.float()

        # 根据流向选择
        dwtk = torch.where(uu > 0, dwtk_pos, dwtk_neg)

        return dwtk

    def _compute_muscl_flux(
        self,
        phi: torch.Tensor,
        uu: torch.Tensor,
        limiter: str = 'minmod'
    ) -> torch.Tensor:
        """
        MUSCL二阶重构通量（turbUtils.F90:893-1071）

        算法：
        1. 计算梯度：grad_i = (phi[i+1] - phi[i-1]) / 2
        2. Limiter：grad_limited = limiter(grad_i, grad_i-1)
        3. 重构：
           - phi_L = phi[i] + 0.5*grad_limited[i]*dx
           - phi_R = phi[i+1] - 0.5*grad_limited[i+1]*dx
        4. 迎风：flux = uu>0 ? uu*phi_L : uu*phi_R

        Args:
            phi: 守恒变量 (H, W) 或 (batch, H, W)
            uu: 逆变速度
            limiter: 限制器类型

        Returns:
            flux: 对流通量
        """
        # 计算梯度（中心差分）
        # 注意：需要处理周期边界
        if phi.ndim == 2:
            # (H, W)
            grad = (torch.roll(phi, -1, dims=-1) - torch.roll(phi, 1, dims=-1)) / 2.0
        else:
            # (batch, H, W)
            grad = (torch.roll(phi, -1, dims=-1) - torch.roll(phi, 1, dims=-1)) / 2.0

        # Limiter（minmod型）
        if limiter == 'minmod':
            grad_left = torch.roll(grad, 1, dims=-1)
            grad_limited = self._minmod_limiter(grad, grad_left)
        elif limiter == 'vanLeer':
            grad_limited = self._van_leer_limiter(grad)
        elif limiter == 'none':
            grad_limited = grad
        else:
            raise ValueError(f"Unknown limiter: {limiter}")

        # 重构到面
        phi_L = phi + 0.5 * grad_limited
        phi_R = torch.roll(phi - 0.5 * grad_limited, -1, dims=-1)

        # 迎风选择
        flux = torch.where(uu > 0, uu * phi_L, uu * phi_R)

        return flux

    def _compute_first_order_flux(
        self,
        phi: torch.Tensor,
        uu: torch.Tensor
    ) -> torch.Tensor:
        """
        一阶迎风通量（turbUtils.F90:840-891）

        Args:
            phi: 守恒变量
            uu: 逆变速度

        Returns:
            flux: 一阶对流通量
        """
        # 左右状态
        phi_L = phi
        phi_R = torch.roll(phi, -1, dims=-1)

        # 迎风选择
        flux = torch.where(uu > 0, uu * phi_L, uu * phi_R)

        return flux

    def _minmod_limiter(
        self,
        a: torch.Tensor,
        b: torch.Tensor
    ) -> torch.Tensor:
        """
        Minmod限制器（turbUtils.F90:966-974）

        minmod(a, b) = sign(a) * max(0, min(|a|, sign(a)*b))

        Args:
            a, b: 梯度值

        Returns:
            limited: 限制后的梯度
        """
        sign_a = torch.sign(a)
        abs_a = torch.abs(a)
        abs_b = torch.abs(b)

        # 同号条件
        same_sign = (a * b > 0).float()

        # minmod公式
        limited = sign_a * torch.clamp(
            torch.min(abs_a, abs_b),
            min=0.0
        ) * same_sign

        return limited

    def _van_leer_limiter(
        self,
        grad: torch.Tensor
    ) -> torch.Tensor:
        """
        Van Leer限制器

        Args:
            grad: 梯度值

        Returns:
            limited: 限制后的梯度
        """
        # Van Leer: (|grad| + grad) / (1 + |grad|)
        abs_grad = torch.abs(grad)
        limited = (abs_grad + grad) / (1.0 + abs_grad + 1e-14)

        return limited

    def _compute_divergence(
        self,
        flux: torch.Tensor,
        direction: str,
        periodic: bool
    ) -> torch.Tensor:
        """
        计算通量散度

        Args:
            flux: 面通量
            direction: 'xi' 或 'eta'
            periodic: 是否周期边界

        Returns:
            div: 通量散度
        """
        if direction == 'xi':
            # ξ方向散度：flux[i+1] - flux[i]
            if periodic:
                # 周期边界
                flux_right = torch.roll(flux, -1, dims=-1)
                div = flux_right - flux
            else:
                # 非周期
                div = flux[..., 1:] - flux[..., :-1]
        elif direction == 'eta':
            # η方向散度：flux[j+1] - flux[j]
            div = flux[..., 1:, :] - flux[..., :-1, :]
        else:
            raise ValueError(f"Invalid direction: {direction}")

        return div

    def compute_viscous_term(
        self,
        rho: torch.Tensor,
        nuTilde: torch.Tensor,
        mu_l: torch.Tensor,
        dnu_dx: torch.Tensor,
        dnu_dy: torch.Tensor,
        face_geom: Dict[str, torch.Tensor],
        vol: torch.Tensor,
        halo_nuTilde_wall: torch.Tensor = None,
        halo_nu_l_wall: torch.Tensor = None,
        halo_vol_wall: torch.Tensor = None,
        halo_nuTilde_ff: torch.Tensor = None,
        halo_nu_l_ff: torch.Tensor = None,
        halo_vol_ff: torch.Tensor = None
    ) -> torch.Tensor:
        """
        SA粘性/扩散项 - 调用ADFLOW对齐版本

        已弃用简化通量散度实现，现在使用compute_viscous_term_adflow

        默认halo值（如果不显式传递）：
        - halo_nuTilde_wall = -nuTilde[0]（反射BC）
        - halo_vol_wall = vol[0]
        - halo_nu_l_wall = nu_l[0]
        - 远场halo使用常值外推
        """
        # 提取层流运动粘度 nu_l = mu_l / rho（ADFLOW无epsilon）
        nu_l = mu_l / rho

        return self.compute_viscous_term_adflow(
            nuTilde=nuTilde,
            nu_l=nu_l,
            face_geom=face_geom,
            vol=vol,
            halo_nuTilde_wall=halo_nuTilde_wall,
            halo_nu_l_wall=halo_nu_l_wall,
            halo_vol_wall=halo_vol_wall,
            halo_nuTilde_ff=halo_nuTilde_ff,
            halo_nu_l_ff=halo_nu_l_ff,
            halo_vol_ff=halo_vol_ff
        )

    def compute_viscous_term_adflow(
        self,
        nuTilde: torch.Tensor,
        nu_l: torch.Tensor,
        face_geom: Dict[str, torch.Tensor],
        vol: torch.Tensor,
        halo_nuTilde_wall: torch.Tensor = None,
        halo_nu_l_wall: torch.Tensor = None,
        halo_vol_wall: torch.Tensor = None,
        halo_nuTilde_ff: torch.Tensor = None,
        halo_nu_l_ff: torch.Tensor = None,
        halo_vol_ff: torch.Tensor = None
    ) -> torch.Tensor:
        """
        SA粘性项 - 严格遵循ADFLOW saViscous算法

        参考：sa.F90:344-673 (saViscous)

        ADFLOW算法：
        1. 计算度量张量 ttm, ttp ~ 1/Δξ²
        2. 梯度修正系数（必须为负）：cnud = -cb2 * nuTilde * cb3Inv
        3. 面平均值：nutm, nutp, num, nup
        4. 扩散系数：cdm = (num + (1+cb2)*nutm) * ttm * cb3Inv
        5. 非负性保证：c1m = max(cdm + cam, 0)
        6. 三对角更新：R += c1m*phi[k-1] - c10*phi[k] + c1p*phi[k+1]

        关键公式（sa.F90:418-444）：
            cnud = -rsaCb2 * nuTilde(k) * cb3Inv
            cdm = (num + (1 + rsaCb2) * nutm) * ttm * cb3Inv
            c1m = max(cdm + ttm*cnud, 0)
            scratch += c1m * nuTilde(k-1) - c10 * nuTilde(k) + c1p * nuTilde(k+1)

        Args:
            nuTilde: SA变量 (batch, H, W) 或 (H, W)
            nu_l: 层流运动粘度 = mu_l / rho (batch, H, W)
            face_geom: 面几何信息（需要面法向量和面积）
            vol: 单元体积 (batch, H, W)
            halo_nuTilde_wall: 壁面halo nuTilde (batch, W)
            halo_nu_l_wall: 壁面halo nu_l (batch, W)
            halo_vol_wall: 壁面halo体积 (batch, W)
            halo_nuTilde_ff: 远场halo nuTilde (batch, W)
            halo_nu_l_ff: 远场halo nu_l (batch, W)
            halo_vol_ff: 远场halo体积 (batch, W)

        Returns:
            R_visc: 粘性残差 (batch, H, W) 或 (H, W)
        """
        # 确保batch维度
        if nuTilde.ndim == 2:
            nuTilde = nuTilde.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # 确保nu_l和vol有正确的batch维度（避免重复unsqueeze）
        if nu_l.ndim == 2:
            nu_l = nu_l.unsqueeze(0)
        if vol.ndim == 2:
            vol = vol.unsqueeze(0)

        batch, H, W = nuTilde.shape
        R_visc = torch.zeros_like(nuTilde)

        # ADFLOW常数
        # cb3Inv = 1.5 (已在__init__中定义)
        # cb2 = 0.622 (已在__init__中定义)

        # ========== η方向（壁面-远场，主要边界层方向） ==========
        R_visc_eta = self._viscous_direction_adflow(
            nuTilde, nu_l, vol, face_geom, direction='eta',
            halo_nuTilde_wall=halo_nuTilde_wall,
            halo_nu_l_wall=halo_nu_l_wall,
            halo_vol_wall=halo_vol_wall,
            halo_nuTilde_ff=halo_nuTilde_ff,
            halo_nu_l_ff=halo_nu_l_ff,
            halo_vol_ff=halo_vol_ff
        )
        R_visc += R_visc_eta

        # ========== ξ方向（周期，沿翼型表面） ==========
        # xi方向是周期边界，不需要halo参数
        R_visc_xi = self._viscous_direction_adflow(
            nuTilde, nu_l, vol, face_geom, direction='xi'
        )
        R_visc += R_visc_xi

        if squeeze_output:
            R_visc = R_visc.squeeze(0)

        return R_visc

    def _viscous_direction_adflow(
        self,
        nuTilde: torch.Tensor,
        nu_l: torch.Tensor,
        vol: torch.Tensor,
        face_geom: Dict[str, torch.Tensor],
        direction: str,
        halo_nuTilde_wall: torch.Tensor = None,
        halo_nu_l_wall: torch.Tensor = None,
        halo_vol_wall: torch.Tensor = None,
        halo_nuTilde_ff: torch.Tensor = None,
        halo_nu_l_ff: torch.Tensor = None,
        halo_vol_ff: torch.Tensor = None
    ) -> torch.Tensor:
        """
        单方向粘性项（ADFLOW saViscous算法）

        参考：sa.F90:476-573（η方向）, sa.F90:574-672（ξ方向）

        算法步骤：
        1. 计算度量张量 ttm, ttp
        2. 计算梯度修正系数 cnud, cam, cap
        3. 计算面平均值 nutm, nutp, num, nup
        4. 计算扩散系数 cdm, cdp
        5. 非负性保证 c1m, c1p, c10
        6. 残差更新

        Args:
            nuTilde: SA变量 (batch, H, W)
            nu_l: 层流运动粘度 (batch, H, W)
            vol: 单元体积 (batch, H, W)
            face_geom: 面几何（需要法向量和面积）
            direction: 'xi' 或 'eta'
            halo_nuTilde_wall: 壁面halo nuTilde (batch, W)，ADFLOW = -nuTilde[0]
            halo_nu_l_wall: 壁面halo nu_l (batch, W)
            halo_vol_wall: 壁面halo体积 (batch, W)，ADFLOW = vol[0]
            halo_nuTilde_ff: 远场halo nuTilde (batch, W)
            halo_nu_l_ff: 远场halo nu_l (batch, W)
            halo_vol_ff: 远场halo体积 (batch, W)

        Returns:
            R: 粘性残差贡献 (batch, H, W)
        """
        batch, H, W = nuTilde.shape
        R = torch.zeros_like(nuTilde)

        if direction == 'eta':
            # η方向（j方向，壁面-远场）
            # 面法向量：A_eta = (A_x_eta, A_y_eta)，面积 = sqrt(A_x^2 + A_y^2)

            # 从face_geom提取面法向量
            # 假设face_geom包含：A_x_eta (H+1, W), A_y_eta (H+1, W)
            if 'A_x_eta' in face_geom and 'A_y_eta' in face_geom:
                A_x = face_geom['A_x_eta']  # (H+1, W) 或 (batch, H+1, W)
                A_y = face_geom['A_y_eta']
                if A_x.ndim == 2:
                    A_x = A_x.unsqueeze(0).expand(batch, -1, -1)
                    A_y = A_y.unsqueeze(0).expand(batch, -1, -1)
            else:
                # 简化：假设均匀网格，使用默认度量张量
                return self._viscous_direction_uniform(nuTilde, nu_l, vol, direction)

            # ========== ADFLOW 对齐：使用扩展数组统一处理边界 ==========
            # 构造包含 halo 的扩展数组 (batch, H+2, W)
            # 索引映射：halo_wall -> 0, physical[0..H-1] -> 1..H, halo_ff -> H+1

            # 设置默认 halo 值（如果未提供）
            if halo_nuTilde_wall is None:
                # ADFLOW默认：壁面 halo = -nuTilde[0]（反射BC）
                halo_nuTilde_wall = -nuTilde[:, 0, :]
            if halo_nu_l_wall is None:
                halo_nu_l_wall = nu_l[:, 0, :]
            if halo_vol_wall is None:
                # ADFLOW默认：halo体积 = 物理层体积
                halo_vol_wall = vol[:, 0, :]
            if halo_nuTilde_ff is None:
                # 远场 halo = 常值外推
                halo_nuTilde_ff = nuTilde[:, -1, :]
            if halo_nu_l_ff is None:
                halo_nu_l_ff = nu_l[:, -1, :]
            if halo_vol_ff is None:
                halo_vol_ff = vol[:, -1, :]

            # 构造扩展数组
            nuTilde_hat = torch.cat(
                [
                    halo_nuTilde_wall.unsqueeze(-2),
                    nuTilde,
                    halo_nuTilde_ff.unsqueeze(-2),
                ],
                dim=-2,
            )
            nu_l_hat = torch.cat(
                [
                    halo_nu_l_wall.unsqueeze(-2),
                    nu_l,
                    halo_nu_l_ff.unsqueeze(-2),
                ],
                dim=-2,
            )
            vol_hat = torch.cat(
                [
                    halo_vol_wall.unsqueeze(-2),
                    vol,
                    halo_vol_ff.unsqueeze(-2),
                ],
                dim=-2,
            )

            # 向量化计算所有物理单元 j_phys=0..H-1，对应扩展数组 j_hat=1..H
            vol_c = vol_hat[:, 1:H+1, :]
            vol_m = vol_hat[:, 0:H, :]
            vol_p = vol_hat[:, 2:H+2, :]

            # 度量张量计算 (sa.F90:492-507)
            # ADFLOW无eps：voli = one / vol(i, j, k)
            voli = 1.0 / vol_c
            volmi = 2.0 / (vol_c + vol_m)
            volpi = 2.0 / (vol_c + vol_p)

            # 面法向量 (sa.F90:496-501)
            A_x_m = A_x[:, 0:H, :]
            A_y_m = A_y[:, 0:H, :]
            A_x_p = A_x[:, 1:H+1, :]
            A_y_p = A_y[:, 1:H+1, :]

            xm = A_x_m * volmi
            ym = A_y_m * volmi
            xp = A_x_p * volpi
            yp = A_y_p * volpi

            # 平均法向量 (sa.F90:503-505)
            xa = 0.5 * (A_x_p + A_x_m) * voli
            ya = 0.5 * (A_y_p + A_y_m) * voli

            # 度量张量 (sa.F90:506-507)
            ttm = xm * xa + ym * ya
            ttp = xp * xa + yp * ya

            # 梯度修正系数 (sa.F90:523-525)
            nuTilde_c = nuTilde_hat[:, 1:H+1, :]
            nuTilde_m = nuTilde_hat[:, 0:H, :]
            nuTilde_p = nuTilde_hat[:, 2:H+2, :]
            cnud = -self.cb2 * nuTilde_c * self.cb3Inv
            cam = ttm * cnud
            cap = ttp * cnud

            # 面平均值 (sa.F90:527-531)
            nu_l_c = nu_l_hat[:, 1:H+1, :]
            nu_l_m = nu_l_hat[:, 0:H, :]
            nu_l_p = nu_l_hat[:, 2:H+2, :]
            nutm = 0.5 * (nuTilde_m + nuTilde_c)
            num = 0.5 * (nu_l_m + nu_l_c)
            nutp = 0.5 * (nuTilde_p + nuTilde_c)
            nup = 0.5 * (nu_l_p + nu_l_c)

            # 扩散系数 (sa.F90:532-533)
            cdm = (num + (1.0 + self.cb2) * nutm) * ttm * self.cb3Inv
            cdp = (nup + (1.0 + self.cb2) * nutp) * ttp * self.cb3Inv

            # 非负性保证 (sa.F90:535-537)
            c1m = torch.clamp(cdm + cam, min=0.0)
            c1p = torch.clamp(cdp + cap, min=0.0)
            c10 = c1m + c1p

            # 残差更新 (sa.F90:542-543)
            R += c1m * nuTilde_m - c10 * nuTilde_c + c1p * nuTilde_p

        elif direction == 'xi':
            # ξ方向（i方向）
            if 'A_x_xi' in face_geom and 'A_y_xi' in face_geom:
                A_x = face_geom['A_x_xi']  # 周期: (H, W)；非周期: (H, W-1)
                A_y = face_geom['A_y_xi']
                if A_x.ndim == 2:
                    A_x = A_x.unsqueeze(0).expand(batch, -1, -1)
                    A_y = A_y.unsqueeze(0).expand(batch, -1, -1)
            else:
                return self._viscous_direction_uniform(nuTilde, nu_l, vol, direction)

            periodic_xi = bool(face_geom.get('periodic_xi', True))
            if periodic_xi:
                if int(A_x.shape[-1]) != W:
                    raise ValueError(
                        f"Periodic xi expects W faces, got A_x width={int(A_x.shape[-1])}, W={W}"
                    )
                A_x_eff = A_x
                A_y_eff = A_y
                nuTilde_m = torch.roll(nuTilde, 1, dims=-1)  # i-1
                nuTilde_p = torch.roll(nuTilde, -1, dims=-1)  # i+1
                nu_l_m = torch.roll(nu_l, 1, dims=-1)
                nu_l_p = torch.roll(nu_l, -1, dims=-1)
                vol_m = torch.roll(vol, 1, dims=-1)
                vol_p = torch.roll(vol, -1, dims=-1)
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
                nuTilde_m = torch.cat([nuTilde[..., :1], nuTilde[..., :-1]], dim=-1)
                nuTilde_p = torch.cat([nuTilde[..., 1:], nuTilde[..., -1:]], dim=-1)
                nu_l_m = torch.cat([nu_l[..., :1], nu_l[..., :-1]], dim=-1)
                nu_l_p = torch.cat([nu_l[..., 1:], nu_l[..., -1:]], dim=-1)
                vol_m = torch.cat([vol[..., :1], vol[..., :-1]], dim=-1)
                vol_p = torch.cat([vol[..., 1:], vol[..., -1:]], dim=-1)

            # 度量张量（ADFLOW无eps）
            voli = 1.0 / vol
            volmi = 2.0 / (vol + vol_m)
            volpi = 2.0 / (vol + vol_p)

            # 面法向量（周期网格：A_x_xi / A_y_xi 与其他模块一致，存的是“当前单元右侧面”）
            # 因此对单元 i 而言：
            #   - 左侧面 si(i-1) 需要 roll(+1)
            #   - 右侧面 si(i)   就是当前 A_x / A_y
            A_x_m = torch.roll(A_x_eff, 1, dims=-1)
            A_y_m = torch.roll(A_y_eff, 1, dims=-1)
            A_x_p = A_x_eff
            A_y_p = A_y_eff

            xm = A_x_m * volmi
            ym = A_y_m * volmi
            xp = A_x_p * volpi
            yp = A_y_p * volpi

            xa = 0.5 * (A_x_p + A_x_m) * voli
            ya = 0.5 * (A_y_p + A_y_m) * voli

            ttm = xm * xa + ym * ya
            ttp = xp * xa + yp * ya

            # 梯度修正
            cnud = -self.cb2 * nuTilde * self.cb3Inv
            cam = ttm * cnud
            cap = ttp * cnud

            # 面平均值
            nutm = 0.5 * (nuTilde_m + nuTilde)
            nutp = 0.5 * (nuTilde_p + nuTilde)
            num = 0.5 * (nu_l_m + nu_l)
            nup = 0.5 * (nu_l_p + nu_l)

            # 扩散系数
            cdm = (num + (1.0 + self.cb2) * nutm) * ttm * self.cb3Inv
            cdp = (nup + (1.0 + self.cb2) * nutp) * ttp * self.cb3Inv

            # 非负性保证
            c1m = torch.clamp(cdm + cam, min=0.0)
            c1p = torch.clamp(cdp + cap, min=0.0)
            c10 = c1m + c1p

            # 残差更新
            R += c1m * nuTilde_m - c10 * nuTilde + c1p * nuTilde_p

        return R

    def _viscous_direction_uniform(
        self,
        nuTilde: torch.Tensor,
        nu_l: torch.Tensor,
        vol: torch.Tensor,
        direction: str
    ) -> torch.Tensor:
        """
        均匀网格简化版粘性项

        当缺少详细面几何时使用，假设网格正交均匀。
        度量张量简化为：ttm ≈ ttp ≈ 1/Δ²

        Args:
            nuTilde: SA变量 (batch, H, W)
            nu_l: 层流运动粘度 (batch, H, W)
            vol: 单元体积 (batch, H, W)
            direction: 'xi' 或 'eta'

        Returns:
            R: 粘性残差贡献 (batch, H, W)
        """
        batch, H, W = nuTilde.shape
        R = torch.zeros_like(nuTilde)

        # 简化度量张量：ttm ≈ ttp ≈ 1.0（归一化）
        # 实际上需要根据vol估计网格尺度

        if direction == 'eta':
            # η方向（非周期）
            for j in range(H):
                # 度量张量简化
                if j > 0 and j < H - 1:
                    # 内部：使用体积估计
                    delta_m = torch.sqrt(vol[:, j, :])
                    delta_p = torch.sqrt(vol[:, j, :])
                    ttm = 1.0 / (delta_m ** 2 + 1e-14)
                    ttp = 1.0 / (delta_p ** 2 + 1e-14)
                else:
                    ttm = 1.0 / (vol[:, j, :] + 1e-14)
                    ttp = 1.0 / (vol[:, j, :] + 1e-14)

                # 梯度修正
                cnud = -self.cb2 * nuTilde[:, j, :] * self.cb3Inv
                cam = ttm * cnud
                cap = ttp * cnud

                # 面平均值
                if j > 0:
                    nutm = 0.5 * (nuTilde[:, j-1, :] + nuTilde[:, j, :])
                    num = 0.5 * (nu_l[:, j-1, :] + nu_l[:, j, :])
                else:
                    nutm = 0.5 * nuTilde[:, j, :]
                    num = nu_l[:, j, :]

                if j < H - 1:
                    nutp = 0.5 * (nuTilde[:, j+1, :] + nuTilde[:, j, :])
                    nup = 0.5 * (nu_l[:, j+1, :] + nu_l[:, j, :])
                else:
                    nutp = nuTilde[:, j, :]
                    nup = nu_l[:, j, :]

                # 扩散系数
                cdm = (num + (1.0 + self.cb2) * nutm) * ttm * self.cb3Inv
                cdp = (nup + (1.0 + self.cb2) * nutp) * ttp * self.cb3Inv

                # 非负性保证
                c1m = torch.clamp(cdm + cam, min=0.0)
                c1p = torch.clamp(cdp + cap, min=0.0)
                c10 = c1m + c1p

                # 残差更新
                if j > 0:
                    nuTilde_m = nuTilde[:, j-1, :]
                else:
                    nuTilde_m = torch.zeros_like(nuTilde[:, j, :])

                if j < H - 1:
                    nuTilde_p = nuTilde[:, j+1, :]
                else:
                    nuTilde_p = nuTilde[:, j, :]

                R[:, j, :] += c1m * nuTilde_m - c10 * nuTilde[:, j, :] + c1p * nuTilde_p

        elif direction == 'xi':
            # ξ方向（周期）
            nuTilde_m = torch.roll(nuTilde, 1, dims=-1)
            nuTilde_p = torch.roll(nuTilde, -1, dims=-1)
            nu_l_m = torch.roll(nu_l, 1, dims=-1)
            nu_l_p = torch.roll(nu_l, -1, dims=-1)

            # 简化度量张量
            delta = torch.sqrt(vol)
            ttm = 1.0 / (delta ** 2 + 1e-14)
            ttp = ttm

            # 梯度修正
            cnud = -self.cb2 * nuTilde * self.cb3Inv
            cam = ttm * cnud
            cap = ttp * cnud

            # 面平均值
            nutm = 0.5 * (nuTilde_m + nuTilde)
            nutp = 0.5 * (nuTilde_p + nuTilde)
            num = 0.5 * (nu_l_m + nu_l)
            nup = 0.5 * (nu_l_p + nu_l)

            # 扩散系数
            cdm = (num + (1.0 + self.cb2) * nutm) * ttm * self.cb3Inv
            cdp = (nup + (1.0 + self.cb2) * nutp) * ttp * self.cb3Inv

            # 非负性保证
            c1m = torch.clamp(cdm + cam, min=0.0)
            c1p = torch.clamp(cdp + cap, min=0.0)
            c10 = c1m + c1p

            # 残差更新
            R += c1m * nuTilde_m - c10 * nuTilde + c1p * nuTilde_p

        return R

    def _compute_diffusion_flux(
        self,
        mu_eff_sa: torch.Tensor,
        grad_nu: torch.Tensor,
        face_geom: Dict[str, torch.Tensor],
        direction: str
    ) -> torch.Tensor:
        """
        计算扩散通量

        Args:
            mu_eff_sa: 有效扩散系数
            grad_nu: nuTilde梯度
            face_geom: 面几何
            direction: 'xi' 或 'eta'

        Returns:
            flux: 扩散通量
        """
        # 简化实现：中心差分
        if direction == 'xi':
            # μ平均到面
            mu_face = 0.5 * (mu_eff_sa[..., :-1] + mu_eff_sa[..., 1:])
            # 梯度平均到面
            grad_face = 0.5 * (grad_nu[..., :-1] + grad_nu[..., 1:])
            # 面面积
            A_x = face_geom['A_x_xi']
            A_y = face_geom['A_y_xi']
        else:  # eta
            mu_face = 0.5 * (mu_eff_sa[..., :-1, :] + mu_eff_sa[..., 1:, :])
            grad_face = 0.5 * (grad_nu[..., :-1, :] + grad_nu[..., 1:, :])
            A_x = face_geom['A_x_eta']
            A_y = face_geom['A_y_eta']

        # 扩散通量：-μ * grad(nu) * A
        # 简化：假设grad沿法向
        flux = -mu_face * grad_face * torch.sqrt(A_x**2 + A_y**2)

        return flux

    def compute_residual(
        self,
        fields: Dict[str, torch.Tensor],
        geometry: Dict[str, torch.Tensor],
        flow_conditions: Dict[str, float]
    ) -> torch.Tensor:
        """
        完整SA残差装配 - ADFLOW对齐版本

        参考：
        - sa.F90:16 (sa_block) - 主SA残差计算入口
        - sa.F90:88-342 (saSource) - 源项
        - turbUtils.F90:825-978 (turbAdvection) - 对流项
        - sa.F90:344-673 (saViscous) - 粘性项
        - sa.F90:675-711 (saResScale) - 残差scaling

        组件对齐状态：
        ✅ 源项：完全对齐（fv1, fv2, fw, ft2等修正函数）
        ✅ 对流项：ADFLOW turbAdvection算法（1阶迎风 + 两个非线性修正）
        ✅ 粘性项：ADFLOW saViscous算法（三对角系数 + cb2梯度修正 + 非负性保证）
        ✅ 常数：完全对齐paramTurb.F90
        ✅ Scaling：-R_raw（per-volume形式）

        边界条件：
        - 壁面（j=0）：nuTilde = 0（Dirichlet BC）
        - 远场（j=H-1）：外推
        - 周期（ξ方向）：roll连接首尾

        Args:
            fields: 流场变量字典
                - 'rho': 密度 (H, W) 或 (batch, H, W)
                - 'nuTilde': SA湍流变量
                - 'mu_l': 层流粘度
                - 'd_wall': 壁面距离
                - 'du_dx', 'du_dy', 'dv_dx', 'dv_dy': 速度梯度
                - 'dnu_dx', 'dnu_dy': nuTilde梯度（cell-center）
                - 'vort_mag': 涡量模（可选，向后兼容）
            geometry: 几何信息字典
                - 'face_geom': 面几何（A_x_xi, A_y_xi, A_x_eta, A_y_eta）
                - 'vol': 单元体积
                - 'contravariant_vel': 逆变速度 {'uu_xi', 'uu_eta'}
            flow_conditions: 流动条件 {'Ma', 'Re', ...}

        Returns:
            R_SA: SA方程残差 (H, W) 或 (batch, H, W)，per-volume形式
        """
        # 源项（saSource，sa.F90:88-342）
        # 完全对齐ADFLOW，使用速度梯度计算strain/vorticity
        R_source = self.compute_source_term(
            rho=fields['rho'],
            nuTilde=fields['nuTilde'],
            mu_l=fields['mu_l'],
            d_wall=fields['d_wall'],
            du_dx=fields['du_dx'],
            du_dy=fields['du_dy'],
            dv_dx=fields['dv_dx'],
            dv_dy=fields['dv_dy'],
            vort_mag=fields.get('vort_mag')  # 可选，向后兼容
        )

        # 对流项（turbAdvection，turbUtils.F90:825-978）
        # ADFLOW默认orderTurb=firstOrder（inputParamRoutines.F90:3793）
        second_order = (self.order_turb == 'second')
        R_adv = self.compute_advection_term(
            fields['rho'], fields['nuTilde'],
            geometry['contravariant_vel'],
            geometry['face_geom'], geometry['vol'],
            second_order=second_order,
            limiter='minmod',
            halo_nuTilde_wall=geometry.get('halo_nuTilde_wall'),
            halo_nuTilde_wall_second=geometry.get('halo_nuTilde_wall_second'),
            halo_nuTilde_ff=geometry.get('halo_nuTilde_ff'),
            halo_nuTilde_ff_second=geometry.get('halo_nuTilde_ff_second')
        )

        # 粘性项（saViscous，sa.F90:344-673）
        # ADFLOW算法：三对角系数形式 + cb2梯度修正 + max()非负性保证
        R_visc = self.compute_viscous_term(
            fields['rho'], fields['nuTilde'], fields['mu_l'],
            fields['dnu_dx'], fields['dnu_dy'],
            geometry['face_geom'], geometry['vol'],
            halo_nuTilde_wall=geometry.get('halo_nuTilde_wall'),
            halo_nu_l_wall=geometry.get('halo_nu_l_wall'),
            halo_vol_wall=geometry.get('halo_vol_wall'),
            halo_nuTilde_ff=geometry.get('halo_nuTilde_ff'),
            halo_nu_l_ff=geometry.get('halo_nu_l_ff'),
            halo_vol_ff=geometry.get('halo_vol_ff')
        )

        # 总残差组装
        # scratch = source + advection + viscous
        R_SA_raw = R_source + R_adv + R_visc

        # 残差scaling（saResScale，sa.F90:675-711）
        # ADFLOW输出：dw = -volRef * scratch * iblank
        # 我们返回per-volume形式：dw/volRef = -scratch = -R_SA_raw
        # 符号翻转以与流场残差口径一致（ADFLOW约定）
        R_SA = -R_SA_raw

        return R_SA
