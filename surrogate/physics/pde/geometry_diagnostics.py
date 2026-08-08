"""
几何诊断工具 - ADflow corner-based方法验证

检测内容：
1. GCL闭合性验证（核心）
2. 几何质量统计（面积、体积分布等）
3. Jacobian质量检查（退化单元检测）
"""

import torch
import numpy as np
from typing import Dict
from pathlib import Path


def compute_cell_closure_residual_adflow(
    face_geom: Dict[str, torch.Tensor],
    periodic_xi: bool = True,
    use_full_faces: bool = True
) -> torch.Tensor:
    """
    GCL诊断：验证控制体面积向量封闭性（基于ADflow corner-based方法）

    核心修复：使用完整η面几何实现真正的GCL闭合

    对于四边形控制体，面积向量应满足：
    S_ξ(i+1/2,j) - S_ξ(i-1/2,j) + S_η(i,j+1/2) - S_η(i,j-1/2) = 0

    Args:
        face_geom: 面几何字典，来自 compute_face_area_vectors_full()
        periodic_xi: ξ方向周期性
        use_full_faces: 是否使用完整面几何（推荐True）

    Returns:
        closure_residual: 单元闭合残差范数 (H, W)
    """
    # 提取x,y分量
    A_x_xi = face_geom['A_x_xi']  # (H, W) 或 (H, W-1)
    A_y_xi = face_geom['A_y_xi']

    # 关键修复：使用完整η面几何 (H+1, W) 而不是内部面 (H-1, W)
    if use_full_faces and 'A_x_eta_full' in face_geom:
        # 使用完整面几何：包含边界面j=0和j=H
        A_x_eta = face_geom['A_x_eta_full']  # (H+1, W)
        A_y_eta = face_geom['A_y_eta_full']
    else:
        # 回退到内部面几何（向后兼容）
        A_x_eta = face_geom['A_x_eta']  # (H-1, W)
        A_y_eta = face_geom['A_y_eta']

    if periodic_xi:
        # 周期边界：ξ面有W个，η面有H+1个（完整）或H-1个（内部）
        # ξ方向：使用torch.roll确保单元i的差分为 S(i+1/2) - S(i-1/2)
        # 这与η方向 S(j+1/2) - S(j-1/2) 配准在同一单元上（plan35.md关键修复）
        delta_S_xi_x = A_x_xi - torch.roll(A_x_xi, 1, dims=-1)  # (H, W)
        delta_S_xi_y = A_y_xi - torch.roll(A_y_xi, 1, dims=-1)  # (H, W)
    else:
        # 非周期：ξ面有W-1个，η面有H+1个（完整）或H-1个（内部）
        # ξ方向：需要构造(H, W)的差分
        delta_S_xi_x = torch.zeros(A_x_eta.shape[0] - 1, A_x_eta.shape[1],
                                  device=A_x_eta.device, dtype=A_x_eta.dtype)
        delta_S_xi_y = torch.zeros(A_y_eta.shape[0] - 1, A_y_eta.shape[1],
                                  device=A_y_eta.device, dtype=A_y_eta.dtype)

        # 内部单元的ξ方向差分
        delta_S_xi_x[:-1, :] = A_x_xi[:, 1:] - A_x_xi[:, :-1]
        delta_S_xi_y[:-1, :] = A_y_xi[:, 1:] - A_y_xi[:, :-1]

    # 关键修复：η方向使用完整面几何差分
    if use_full_faces and 'A_x_eta_full' in face_geom:
        # 完整η面：(H+1, W) → (H, W) 差分，所有单元都有完整的面贡献
        delta_S_eta_x = A_x_eta[1:] - A_x_eta[:-1]  # (H, W)
        delta_S_eta_y = A_y_eta[1:] - A_y_eta[:-1]  # (H, W)
    else:
        # 内部η面：(H-1, W) → 构造(H, W)的差分，边界为零（导致GCL不闭合）
        delta_S_eta_x = torch.zeros(A_x_eta.shape[0] + 1, A_x_eta.shape[1],
                                  device=A_x_eta.device, dtype=A_x_eta.dtype)
        delta_S_eta_y = torch.zeros(A_y_eta.shape[0] + 1, A_y_eta.shape[1],
                                  device=A_y_eta.device, dtype=A_y_eta.dtype)

        # 内部单元的η方向差分
        delta_S_eta_x[1:-1, :] = A_x_eta[1:, :] - A_x_eta[:-1, :]
        delta_S_eta_y[1:-1, :] = A_y_eta[1:, :] - A_y_eta[:-1, :]

    # GCL闭合残差：ΔS_ξ + ΔS_η
    closure_x = delta_S_xi_x + delta_S_eta_x
    closure_y = delta_S_xi_y + delta_S_eta_y

    # 计算残差范数
    closure_residual = torch.sqrt(closure_x**2 + closure_y**2)

    return closure_residual


class GeometryDiagnostics:
    """几何诊断器"""

    def __init__(self, device='cpu'):
        self.device = device
        self.diagnostics = {}

    def diagnose_all(
        self,
        grid: Dict[str, torch.Tensor],
        periodic_xi: bool = True,
        enforce_periodic_seam: bool = False  # review4.md新增：可选的强制seam闭合
    ) -> Dict:
        """
        运行几何质量诊断（使用完整面几何修复GCL问题）

        检查内容：
        1. GCL闭合性验证（核心修复：使用完整面几何）
        2. 几何质量统计（面积、体积分布等）
        3. Jacobian质量检查（退化单元检测）

        Args:
            grid: grid dict - {'center': (2,H,W), 'vertex': (2,H+1,W+1)}
            periodic_xi: ξ方向周期性
            enforce_periodic_seam: 是否强制seam节点严格闭合（仅用于诊断，不修改原数据）

        Returns:
            diagnostics: 诊断结果字典
        """
        # 提取坐标并转换为float64（review3.md核心修复：全链路float64精度）
        coords_center = grid['center'].to(torch.float64)  # (2, H, W)
        coords_vertex = grid['vertex'].to(torch.float64)  # (2, H+1, W+1)

        # review4.md新增：可选的强制seam节点闭合
        if periodic_xi and enforce_periodic_seam:
            coords_vertex = coords_vertex.clone()  # 避免修改原数据
            seam_avg_x = 0.5 * (coords_vertex[0, :, 0] + coords_vertex[0, :, -1])
            seam_avg_y = 0.5 * (coords_vertex[1, :, 0] + coords_vertex[1, :, -1])
            coords_vertex[0, :, 0] = seam_avg_x
            coords_vertex[0, :, -1] = seam_avg_x
            coords_vertex[1, :, 0] = seam_avg_y
            coords_vertex[1, :, -1] = seam_avg_y
            print(f"[GeometryDiagnostics] 已强制 seam 节点严格闭合（仅用于诊断）")

        # 诊断输出：确认dtype
        print(f"[GeometryDiagnostics] 输入坐标dtype: center={coords_center.dtype}, vertex={coords_vertex.dtype}")

        # 计算完整ADflow几何（review3.md核心修复：调整顺序，先获取块级sign）
        from . import geometry as geom_module

        # 1. 先计算体积，获取块级手性符号（review3.md关键修复）
        vol, sign = geom_module.compute_cell_volume_adflow(
            coords_vertex, periodic_xi
        )

        # 2. 将sign传递给面几何函数，使用块级fact统一方向
        face_geom_full = geom_module.compute_face_area_vectors_full(
            coords_center, coords_vertex, periodic_xi, sign=sign
        )
        # 使用face_geom_full作为唯一几何（已统一）
        face_geom_legacy = face_geom_full  # 统一使用完整几何

        # 1. GCL闭合性诊断（核心修复：使用完整面几何）
        closure_residual = compute_cell_closure_residual_adflow(
            face_geom_full, periodic_xi, use_full_faces=True
        )

        # 计算分方向闭合残差用于诊断（plan35.md建议）
        A_x_xi = face_geom_full['A_x_xi']
        A_y_xi = face_geom_full['A_y_xi']
        A_x_eta = face_geom_full['A_x_eta_full']
        A_y_eta = face_geom_full['A_y_eta_full']

        # ξ向差分（使用roll配准）
        delta_S_xi_x = A_x_xi - torch.roll(A_x_xi, 1, dims=-1)
        delta_S_xi_y = A_y_xi - torch.roll(A_y_xi, 1, dims=-1)
        delta_S_xi_norm = torch.sqrt(delta_S_xi_x**2 + delta_S_xi_y**2)

        # η向差分（完整面）
        delta_S_eta_x = A_x_eta[1:] - A_x_eta[:-1]
        delta_S_eta_y = A_y_eta[1:] - A_y_eta[:-1]
        delta_S_eta_norm = torch.sqrt(delta_S_eta_x**2 + delta_S_eta_y**2)

        # review4.md增强：内部单元 vs 全域闭合统计
        # 内部单元：j=1..H-2（排除边界层j=0和j=H-1）
        H = closure_residual.shape[0]
        interior_mask = torch.zeros_like(closure_residual, dtype=torch.bool)
        interior_mask[1:H-1, :] = True
        closure_interior = closure_residual[interior_mask]

        gcl_stats = {
            'closure_residual': {
                'mean': closure_residual.mean().item(),
                'max': closure_residual.max().item(),
                'median': closure_residual.median().item(),
                'p95': np.percentile(closure_residual.cpu().numpy(), 95),
                'p99': np.percentile(closure_residual.cpu().numpy(), 99)
            },
            # review4.md增强：内部单元统计（更能反映计算单元质量）
            'closure_interior': {
                'mean': closure_interior.mean().item(),
                'max': closure_interior.max().item(),
                'median': closure_interior.median().item()
            },
            'gcl_satisfied': closure_residual.max().item() < 1e-6,
            'gcl_excellent': closure_residual.max().item() < 1e-8,
            # 新增：分方向闭合残差（plan35.md诊断增强）
            'directional_closure': {
                'delta_S_xi_max': delta_S_xi_norm.max().item(),
                'delta_S_eta_max': delta_S_eta_norm.max().item()
            }
        }

        # 诊断输出：问题单元位置分析
        problem_mask = closure_residual > 1e-4
        n_problem = problem_mask.sum().item()
        if n_problem > 0:
            problem_indices = torch.nonzero(problem_mask, as_tuple=False)
            print(f"\n[GeometryDiagnostics] 发现{n_problem}个GCL残差>1e-4的问题单元")
            if n_problem <= 10:
                print(f"  问题单元位置(j,i): {problem_indices.tolist()}")
            else:
                print(f"  前10个问题单元(j,i): {problem_indices[:10].tolist()}")
            # 分析位置分布
            j_coords = problem_indices[:, 0]
            H = closure_residual.shape[0]
            print(f"  j坐标范围: [{j_coords.min().item()}, {j_coords.max().item()}] (总共H={H})")
            n_boundary = ((j_coords == 0) | (j_coords == H-1)).sum().item()
            print(f"  边界附近单元(j=0或j=H-1): {n_boundary}个 ({100*n_boundary/n_problem:.1f}%)")
        else:
            print(f"\n[GeometryDiagnostics] ✅ 无GCL残差>1e-4的问题单元")

        # review4.md增强：内部 vs 全域闭合统计
        print(f"\n[GeometryDiagnostics] GCL闭合统计（review4.md诊断增强）:")
        print(f"  全域闭合残差: mean={gcl_stats['closure_residual']['mean']:.3e}, max={gcl_stats['closure_residual']['max']:.3e}")
        print(f"  内部闭合残差(j=1..H-2): mean={gcl_stats['closure_interior']['mean']:.3e}, max={gcl_stats['closure_interior']['max']:.3e}")
        print(f"  说明：内部单元统计更能反映计算单元质量（ADflow也以compute cells为准）")

        # plan41.md增强：分方向残差详细统计
        delta_S_xi_x = delta_S_xi_x.double()
        delta_S_xi_y = delta_S_xi_y.double()
        delta_S_eta_x = delta_S_eta_x.double()
        delta_S_eta_y = delta_S_eta_y.double()

        delta_S_xi_x_max = torch.max(torch.abs(delta_S_xi_x)).item()
        delta_S_xi_y_max = torch.max(torch.abs(delta_S_xi_y)).item()
        delta_S_xi_max = max(delta_S_xi_x_max, delta_S_xi_y_max)
        delta_S_xi_mean = torch.mean(torch.sqrt(delta_S_xi_x**2 + delta_S_xi_y**2)).item()

        delta_S_eta_x_max = torch.max(torch.abs(delta_S_eta_x)).item()
        delta_S_eta_y_max = torch.max(torch.abs(delta_S_eta_y)).item()
        delta_S_eta_max = max(delta_S_eta_x_max, delta_S_eta_y_max)
        delta_S_eta_mean = torch.mean(torch.sqrt(delta_S_eta_x**2 + delta_S_eta_y**2)).item()

        # 诊断输出：关键几何量统计
        print(f"\n[GeometryDiagnostics] 几何量统计:")
        print(f"  |A_x_xi|: max={face_geom_full['A_x_xi'].abs().max():.3e}")
        print(f"  |A_x_eta_full|: max={face_geom_full['A_x_eta_full'].abs().max():.3e}")
        print(f"  |ΔS_xi|: max={delta_S_xi_max:.3e}, mean={delta_S_xi_mean:.3e} (分量: x={delta_S_xi_x_max:.3e}, y={delta_S_xi_y_max:.3e})")
        print(f"  |ΔS_eta|: max={delta_S_eta_max:.3e}, mean={delta_S_eta_mean:.3e} (分量: x={delta_S_eta_x_max:.3e}, y={delta_S_eta_y_max:.3e})")

        # review4.md修正：Seam自检改为顶点周期闭合检查
        print(f"\n[GeometryDiagnostics] Seam周期闭合自检:")
        # 1. 顶点坐标检查
        max_diff_vertex_x = torch.max(torch.abs(coords_vertex[0, :, 0] - coords_vertex[0, :, -1])).item()
        max_diff_vertex_y = torch.max(torch.abs(coords_vertex[1, :, 0] - coords_vertex[1, :, -1])).item()
        print(f"  顶点周期闭合: max|x[:,0]-x[:,-1]| = {max_diff_vertex_x:.6e}, max|y[:,0]-y[:,-1]| = {max_diff_vertex_y:.6e}")

        # 2. plan41.md增强：ξ面积向量seam闭合检查
        if 'seam_diff_max' in face_geom_full and periodic_xi:
            seam_diff = face_geom_full['seam_diff_max']
            print(f"  ξ面积向量Seam差异: max = {seam_diff:.6e} (目标<1e-10)")
            if seam_diff < 1e-10:
                print(f"  ✅ ξ面积向量seam完全闭合")
            elif seam_diff < 1e-6:
                print(f"  ⚠️  ξ面积向量seam接近闭合但未达最优")
            else:
                print(f"  ❌ ξ面积向量seam闭合不足，plan41.md修复可能未生效")

        if max_diff_vertex_x > 1e-10 or max_diff_vertex_y > 1e-10:
            print(f"  ⚠️  警告：顶点周期闭合不严格，可能影响GCL精度")
            print(f"  建议：使用 enforce_periodic_seam=True 强制闭合")

        # review3.md建议：边界η面检查
        H = A_x_eta.shape[0] - 1
        max_eta_boundary_0 = torch.max(torch.abs(A_x_eta[0, :])).item()
        max_eta_boundary_H = torch.max(torch.abs(A_x_eta[H, :])).item()
        print(f"\n[GeometryDiagnostics] 边界η面检查:")
        print(f"  边界η面: max|A_x_eta[0,:]| = {max_eta_boundary_0:.6e}")
        print(f"  边界η面: max|A_x_eta[{H},:]| = {max_eta_boundary_H:.6e}")

        # η面全局取向判定：内部面法向与分隔法向的点积
        if 'A_x_eta_full' in face_geom_full:
            A_x_eta_full = face_geom_full['A_x_eta_full'].double()
            A_y_eta_full = face_geom_full['A_y_eta_full'].double()

            # 内部面单位法向
            norm_eta = torch.sqrt(A_x_eta_full**2 + A_y_eta_full**2 + 1e-12)
            n_eta_x = A_x_eta_full / (norm_eta + 1e-12)
            n_eta_y = A_y_eta_full / (norm_eta + 1e-12)

            # 分隔法向（使用中心坐标差分）
            dx_sep_eta = coords_center[0, 1:, :] - coords_center[0, :-1, :]
            dy_sep_eta = coords_center[1, 1:, :] - coords_center[1, :-1, :]
            d_sep_eta = torch.sqrt(dx_sep_eta**2 + dy_sep_eta**2 + 1e-12)
            ssx_eta = dx_sep_eta / (d_sep_eta + 1e-12)
            ssy_eta = dy_sep_eta / (d_sep_eta + 1e-12)

            if ssx_eta.numel() > 0:
                dot_eta = n_eta_x[1:-1, :] * ssx_eta + n_eta_y[1:-1, :] * ssy_eta
                dot_eta_mean = dot_eta.mean().item()
                dot_eta_min = dot_eta.min().item()
                dot_eta_max = dot_eta.max().item()
                print(f"  η面方向点积: mean={dot_eta_mean:.3f}, min={dot_eta_min:.3f}, max={dot_eta_max:.3f}")

        # review3.md核心 + plan41.md增强：块级手性与翻转统计
        print(f"\n[GeometryDiagnostics] 块级手性统计:")
        print(f"  块级符号sign = {sign:.1f}")

        # plan41.md增强：计算翻转比例并警告
        flipped_xi_count = face_geom_full.get('flipped_xi_count', 0)
        flipped_eta_count = face_geom_full.get('flipped_eta_full_count', 0)

        H_grid = face_geom_full['A_x_xi'].shape[1]  # ξ面: (W, H)
        W_grid = face_geom_full['A_x_xi'].shape[0]
        total_xi = W_grid * H_grid
        total_eta = (H_grid + 1) * W_grid

        flipped_xi_ratio = (flipped_xi_count / total_xi * 100) if total_xi > 0 else 0
        flipped_eta_ratio = (flipped_eta_count / total_eta * 100) if total_eta > 0 else 0

        print(f"  ξ面翻转: {flipped_xi_count}/{total_xi} ({flipped_xi_ratio:.1f}%) (目标<5%)")
        print(f"  η面翻转: {flipped_eta_count}/{total_eta} ({flipped_eta_ratio:.1f}%) (目标<5%)")

        if flipped_xi_ratio > 10.0:
            print(f"  ⚠️  警告：ξ面翻转比例过高({flipped_xi_ratio:.1f}%)，块级sign可能有误")
        if flipped_eta_ratio > 10.0:
            print(f"  ⚠️  警告：η面翻转比例过高({flipped_eta_ratio:.1f}%)，块级sign可能有误")
        print()

        # 2. 几何质量统计（使用传统几何向后兼容）
        geometry_quality = {
            'face_length_xi': {
                'min': face_geom_legacy['s_area_xi'].min().item(),
                'max': face_geom_legacy['s_area_xi'].max().item(),
                'mean': face_geom_legacy['s_area_xi'].mean().item()
            },
            'face_length_eta': {
                'min': face_geom_legacy['s_area_eta'].min().item(),
                'max': face_geom_legacy['s_area_eta'].max().item(),
                'mean': face_geom_legacy['s_area_eta'].mean().item()
            },
            'cell_volume': {
                'min': vol.min().item(),
                'max': vol.max().item(),
                'mean': vol.mean().item()
            },
            'aspect_ratio': {
                'mean': (face_geom_legacy['s_area_xi'].mean() /
                         face_geom_legacy['s_area_eta'].mean()).item()
            },
            # 新增：完整面几何信息
            'full_eta_faces': {
                'dims': face_geom_full.get('eta_dims', 'unknown'),
                'enabled': 'A_x_eta_full' in face_geom_full
            },
            # 新增：方向翻转统计（plan35.md诊断增强）
            'direction_alignment': {
                'flipped_xi_count': face_geom_full.get('flipped_xi_count', 0),
                'flipped_eta_count': face_geom_full.get('flipped_eta_full_count', 0),
                'xi_total': A_x_xi.numel(),
                'eta_total': A_x_eta.numel()
            }
        }

        # 3. Jacobian质量（检测退化单元）
        jacobian_quality = {
            'min': vol.min().item(),
            'negative_cells': (vol < 0).sum().item(),
            'near_zero_cells': (vol.abs() < 1e-12).sum().item()
        }

        # 存储到实例变量（修复：确保self.diagnostics正确设置）
        self.diagnostics = {
            'gcl': gcl_stats,
            'geometry_quality': geometry_quality,
            'jacobian_quality': jacobian_quality
        }

        return self.diagnostics

    def print_summary(self):
        """打印诊断摘要"""
        if not self.diagnostics:
            print("[GeometryDiagnostics] 请先运行diagnose_all()")
            return

        print("\n" + "="*70)
        print("几何一致性诊断报告")
        print("="*70)

        # GCL诊断结果
        if 'gcl' in self.diagnostics:
            gcl = self.diagnostics['gcl']
            print("\n1. GCL离散闭合性诊断:")
            closure = gcl['closure_residual']
            print(f"   全域闭合残差: mean={closure['mean']:.2e}, max={closure['max']:.2e}")
            print(f"   p95/p99: {closure['p95']:.2e} / {closure['p99']:.2e}")

            # review4.md增强：内部单元统计
            if 'closure_interior' in gcl:
                interior = gcl['closure_interior']
                print(f"   内部闭合残差(j=1..H-2): mean={interior['mean']:.2e}, max={interior['max']:.2e}")

            # 新增：分方向闭合残差（plan35.md诊断增强）
            if 'directional_closure' in gcl:
                dir_closure = gcl['directional_closure']
                print(f"   分方向残差: |ΔSξ|_max={dir_closure['delta_S_xi_max']:.2e}, "
                      f"|ΔSη|_max={dir_closure['delta_S_eta_max']:.2e}")

            if gcl['gcl_excellent']:
                print("   结论: ✅ GCL优秀 (机器精度)")
            elif gcl['gcl_satisfied']:
                print("   结论: ✅ GCL良好 (float32精度)")
            else:
                print("   结论: ❌ GCL不满足 (离散破坏)")

        # 几何修复状态
        if 'geometry_quality' in self.diagnostics:
            geo_info = self.diagnostics['geometry_quality']
            print("\n1.5. 几何修复状态:")
            if 'full_eta_faces' in geo_info:
                full_info = geo_info['full_eta_faces']
                if full_info.get('enabled', False):
                    print(f"   完整η面几何: ✅ 启用")
                    print(f"   η面维度: {full_info.get('dims', 'unknown')}")
                    print("   状态: ✅ 边界面已包含，GCL应可闭合")
                else:
                    print("   完整η面几何: ❌ 未启用")
                    print("   状态: ⚠️  仅使用内部面，边界GCL可能不闭合")

            # 新增：方向翻转统计（plan35.md诊断增强）
            if 'direction_alignment' in geo_info:
                align_info = geo_info['direction_alignment']
                xi_flip = align_info['flipped_xi_count']
                eta_flip = align_info['flipped_eta_count']
                xi_total = align_info['xi_total']
                eta_total = align_info['eta_total']
                print(f"\n   方向对齐统计:")
                print(f"   ξ面翻转: {xi_flip}/{xi_total} ({100*xi_flip/xi_total:.1f}%)")
                print(f"   η面翻转: {eta_flip}/{eta_total} ({100*eta_flip/eta_total:.1f}%)")

        # 几何质量
        if 'geometry_quality' in self.diagnostics:
            quality = self.diagnostics['geometry_quality']
            print("\n2. 几何质量统计:")

            face_xi = quality['face_length_xi']
            face_eta = quality['face_length_eta']
            vol_stats = quality['cell_volume']

            print(f"   面长度(ξ): min={face_xi['min']:.3e}, max={face_xi['max']:.3e}, mean={face_xi['mean']:.3e}")
            print(f"   面长度(η): min={face_eta['min']:.3e}, max={face_eta['max']:.3e}, mean={face_eta['mean']:.3e}")
            print(f"   体积: min={vol_stats['min']:.3e}, max={vol_stats['max']:.3e}, mean={vol_stats['mean']:.3e}")
            print(f"   长宽比: mean={quality['aspect_ratio']['mean']:.2f}")

        # Jacobian质量
        if 'jacobian_quality' in self.diagnostics:
            jac = self.diagnostics['jacobian_quality']
            print(f"\n3. Jacobian质量:")
            print(f"   最小体积: {jac['min']:.2e}")
            print(f"   负体积单元: {jac['negative_cells']}")
            print(f"   接近零体积: {jac['near_zero_cells']}")

        print("="*70)

    def save_report(self, output_path: str):
        """保存诊断报告为JSON"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        import json
        with open(output_path, 'w') as f:
            json.dump(self.diagnostics, f, indent=2)

        print(f"[GeometryDiagnostics] 报告已保存: {output_path}")
