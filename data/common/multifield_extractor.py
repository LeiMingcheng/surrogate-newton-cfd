"""
Multi-field Data Extractor for UniFoil
扩展现有CGNS读取功能，批量提取多物理场数据用于RL训练

提取字段: Density(ρ), Velocity(u,v), Pressure(p), TurbulentSANuTilde(ν~)
网格尺寸: 自动推断或指定 - 完整O-grid为 84×304 (84径向 × 304周向单元)
          节点坐标: 85×305 (保持周期闭合)
输出格式: .npz文件 {fields, coords_center, coords_vertex, flow_conditions, geometry_params}

支持Turb数据自动转换: CoefPressure + Mach → Pressure + Density
"""

import os
import sys
import numpy as np
import pyvista as pv
from typing import Dict, Tuple, Optional, List
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# 导入Turb转换器
try:
    from data.common.turb_field_converter import convert_turb_to_standard
    TURB_CONVERTER_AVAILABLE = True
except ImportError:
    TURB_CONVERTER_AVAILABLE = False
    warnings.warn("Turb转换器不可用，将无法处理Turb数据")


class MultiFieldExtractor:
    """
    多物理场数据提取器

    Attributes:
        image_shape: 目标网格形状 (n_rings, n_pts_per_ring)，None时自动从CGNS推断
                    完整O-grid应为 (84, 304) - 84径向 × 304周向单元
        field_names: 需要提取的物理场名称列表
        block_index: CGNS文件中存储数据的block索引（默认3）
    """

    def __init__(
        self,
        image_shape: Optional[Tuple[int, int]] = None,
        field_names: Optional[List[str]] = None,
        block_index: int = 3
    ):
        self.image_shape = image_shape  # None表示自动推断
        self.block_index = block_index

        # 默认提取的物理场（按顺序：ρ, u, v, p, ν~）
        if field_names is None:
            self.field_names = ['Density', 'VelocityX', 'VelocityY', 'Pressure', 'TurbulentSANuTilde']
        else:
            self.field_names = field_names

        # 如果提供了image_shape，初始化尺寸参数；否则在提取时动态确定
        if self.image_shape is not None:
            self.n_rings = image_shape[0]
            self.n_pts_per_ring = image_shape[1]
            self.total_pts = self.n_rings * self.n_pts_per_ring
        else:
            self.n_rings = None
            self.n_pts_per_ring = None
            self.total_pts = None

    @staticmethod
    def _infer_farfield_aoa_deg(
        vx: np.ndarray,
        vy: np.ndarray,
        n_outer_rings: int = 5,
    ) -> float:
        """Infer AoA (deg) from farfield velocity direction."""
        if vx.ndim != 2 or vy.ndim != 2:
            raise ValueError(f"Expected 2D velocity planes, got vx={vx.shape}, vy={vy.shape}")

        n_outer_rings = int(max(1, min(n_outer_rings, vx.shape[0])))
        vx_ff = float(np.mean(vx[-n_outer_rings:, :]))
        vy_ff = float(np.mean(vy[-n_outer_rings:, :]))
        return float(np.degrees(np.arctan2(vy_ff, vx_ff)))

    def _select_target_block(
        self,
        all_blocks: List["pv.DataSet"],
    ) -> Tuple[Optional["pv.DataSet"], bool]:
        """
        Select the best block that contains required fields.

        Returns:
            (target_block, is_turb_data)
        """
        required_scalars_transi = {"Pressure", "Density"}
        required_scalars_turb = {"CoefPressure", "Mach"}

        candidates_transi: List[pv.DataSet] = []
        candidates_turb: List[pv.DataSet] = []

        for blk in all_blocks:
            if not hasattr(blk, "cell_data"):
                continue
            cell_names = set(blk.cell_data.keys())
            has_velocity = ("Velocity" in cell_names) or (
                "VelocityX" in cell_names and "VelocityY" in cell_names
            )

            if required_scalars_transi.issubset(cell_names) and has_velocity:
                candidates_transi.append(blk)
            if required_scalars_turb.issubset(cell_names) and has_velocity:
                candidates_turb.append(blk)

        def priority(blk: pv.DataSet) -> Tuple[bool, bool, int]:
            """
            Prefer blocks matching requested image_shape, then larger blocks.

            In UniFoil Turb CGNS, boundary zones can contain the same arrays but with
            dimensions like (293, 2, 1); we need the main volume block (e.g., (293, 85, 1)).
            """
            n_cells = int(getattr(blk, "n_cells", 0))
            match_cells = False
            match_vertices = False

            if self.image_shape is not None and hasattr(blk, "dimensions"):
                try:
                    dims = tuple(int(d) for d in blk.dimensions)
                    if len(dims) == 3:
                        H, W = self.image_shape
                        cell_dims = tuple(max(d - 1, 1) for d in dims)
                        thin_axis = dims.index(2) if 2 in dims else int(np.argmin(dims))
                        plane_cell_dims = [cell_dims[i] for i in range(3) if i != thin_axis]
                        plane_node_dims = [dims[i] for i in range(3) if i != thin_axis]

                        match_cells = (H in plane_cell_dims) and (W in plane_cell_dims)
                        match_vertices = (H + 1 in plane_node_dims) and (W + 1 in plane_node_dims)
                except Exception:
                    pass

            return match_cells, match_vertices, n_cells

        if candidates_transi:
            return max(candidates_transi, key=priority), False
        if candidates_turb:
            return max(candidates_turb, key=priority), True

        return None, False

    def extract_from_cgns(
        self,
        cgns_path: str,
        geometry_params: Optional[np.ndarray] = None,
        flow_conditions: Optional[Dict[str, float]] = None
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        从单个CGNS文件提取多物理场数据

        Args:
            cgns_path: CGNS文件路径
            geometry_params: 几何参数（14维CST系数等）
            flow_conditions: 流动条件 {'Ma': 0.7, 'AoA': 2.0, 'Re': 5e6}

        Returns:
            数据字典包含:
                - fields: (C, H, W) 多通道物理场
                - coords_center: (2, H, W) 单元中心坐标 (x, y)
                - coords_vertex: (2, H+1, W+1) 节点坐标 (x, y)
                - geometry_params: (14,) 几何参数
                - flow_conditions: (3,) [Ma, AoA, Re]
            失败时返回None
        """
        if not os.path.exists(cgns_path):
            print(f"文件不存在: {cgns_path}")
            return None

        try:
            # 读取CGNS文件
            reader = pv.CGNSReader(cgns_path)
            reader.load_boundary_patch = False
            cgns_data = reader.read()

            # 提取所有blocks
            all_blocks = []
            self._extract_blocks_recursive(cgns_data, all_blocks)

            # 优先自动选择包含所需字段的block，其次回退到默认索引或第一个可用block
            target_block, is_turb_data = self._select_target_block(all_blocks)

            if target_block is None:
                # 宽松回退：优先使用默认索引，否则使用第一个可用 block
                if len(all_blocks) >= self.block_index + 1:
                    target_block = all_blocks[self.block_index]
                elif len(all_blocks) > 0:
                    target_block = all_blocks[0]
                else:
                    print(f"CGNS文件无可用block: {cgns_path}")
                    return None

            # 提取多物理场数据
            fields_dict = {}

            # 提取格心坐标（用于流场数据）
            cell_centers = target_block.cell_centers()
            coords_center = cell_centers.points

            # 提取节点坐标（用于几何离散）
            coords_vertex_all = target_block.points
            # 获取3D网格维度
            ni, nj, nk = target_block.dimensions

            # 获取3D网格维度（用于字段重排）
            ni, nj, nk = target_block.dimensions
            dims = (int(ni), int(nj), int(nk))

            # 识别挤出轴（薄层方向）
            if 2 in dims:
                thin_axis = dims.index(2)
            else:
                thin_axis = int(np.argmin(dims))

            # 计算单元中心的维度 (节点维度减1，防止出现0)
            cell_dims = tuple(max(d - 1, 1) for d in dims)

            # 检测是否为Turb数据并进行转换
            if is_turb_data and TURB_CONVERTER_AVAILABLE:
                # Turb数据：提取CoefPressure, Mach, Velocity
                fields_dict_raw = {}
                for field_name in ['CoefPressure', 'Mach', 'VelocityX', 'VelocityY']:
                    if field_name == 'VelocityX' or field_name == 'VelocityY':
                        field_data = self._extract_velocity_component(target_block, field_name)
                    else:
                        if field_name not in target_block.cell_data:
                            print(f"Turb字段 {field_name} 不存在于 {cgns_path}")
                            return None
                        field_data = target_block.cell_data[field_name]

                    # 重排为3D网格，然后提取2D平面
                    field_3d = field_data.reshape(cell_dims, order='F')
                    field_plane = np.take(field_3d, indices=0, axis=thin_axis)
                    fields_dict_raw[field_name] = field_plane

                # Turb: 组装/补全流动条件（用于Cp→p/ρ转换 & 写入H5）
                # - Ma∞: 优先读取CGNS block 的 field_data['Mach']（通常为来流Mach）
                # - AoA : 后续会从 farfield 速度方向推断并覆盖
                # - Re  : CGNS中通常缺失，使用传入值或默认
                flow_conditions = dict(flow_conditions or {})

                # Reynolds fallback
                if not np.isfinite(flow_conditions.get("Re", np.nan)):
                    flow_conditions["Re"] = 3e6

                # Mach fallback
                ma_meta = None
                try:
                    if hasattr(target_block, "field_data") and "Mach" in target_block.field_data:
                        ma_meta = float(np.asarray(target_block.field_data["Mach"]).ravel()[0])
                except Exception:
                    ma_meta = None

                if ma_meta is not None and np.isfinite(ma_meta) and ma_meta > 0.0:
                    flow_conditions["Ma"] = ma_meta
                elif not np.isfinite(flow_conditions.get("Ma", np.nan)):
                    # Fallback: use farfield Mach from the Mach plane if Ma not provided
                    mach_plane = np.asarray(fields_dict_raw.get("Mach"))
                    if mach_plane.ndim == 2 and mach_plane.size > 0:
                        radial_axis = int(np.argmin(mach_plane.shape))
                        n_outer = min(5, mach_plane.shape[radial_axis])
                        if radial_axis == 0:
                            mach_ff = float(np.mean(mach_plane[-n_outer:, :]))
                        else:
                            mach_ff = float(np.mean(mach_plane[:, -n_outer:]))
                        if np.isfinite(mach_ff) and mach_ff > 0.0:
                            flow_conditions["Ma"] = mach_ff
                    flow_conditions.setdefault("Ma", 0.75)

                # 进行物理转换
                try:
                    density, vx, vy, pressure = convert_turb_to_standard(
                        fields_dict_raw, flow_conditions
                    )
                    fields_dict['Density'] = density
                    fields_dict['VelocityX'] = vx
                    fields_dict['VelocityY'] = vy
                    fields_dict['Pressure'] = pressure
                except Exception as e:
                    print(f"Turb数据转换失败 {cgns_path}: {str(e)}")
                    return None
            else:
                # Transi数据：直接提取标准字段
                for field_name in self.field_names:
                    if field_name == 'VelocityX' or field_name == 'VelocityY':
                        # Velocity vector components extraction
                        field_data = self._extract_velocity_component(
                            target_block, field_name
                        )
                    else:
                        # Scalar field extraction (Density, Pressure, TurbulentSANuTilde)
                        if field_name in target_block.cell_data:
                            field_data = target_block.cell_data[field_name]
                        else:
                            print(f"Field {field_name} not found in {cgns_path}")
                            return None

                    # Reshape to 3D grid and extract 2D plane
                    field_3d = field_data.reshape(cell_dims, order='F')
                    field_plane = np.take(field_3d, indices=0, axis=thin_axis)
                    fields_dict[field_name] = field_plane

            # 验证节点坐标数量
            if np.prod(dims) != coords_vertex_all.shape[0]:
                print(f"节点坐标数与网格维度不匹配: dims={dims}, points={coords_vertex_all.shape[0]} ({cgns_path})")
                return None

            # 验证格心坐标数量
            if np.prod(cell_dims) != coords_center.shape[0]:
                print(
                    f"格心坐标数与推断的单元维度不匹配: cell_dims={cell_dims}, points={coords_center.shape[0]} ({cgns_path})"
                )
                return None

            try:
                centers_reshaped = coords_center.reshape(cell_dims + (3,), order='F')
                vertices_reshaped = coords_vertex_all.reshape(dims + (3,), order='F')
            except ValueError as err:
                print(f"坐标重排失败 {cgns_path}: {err}")
                return None

            # 在挤出轴上取 index=0，得到二维平面
            center_plane = np.take(centers_reshaped, indices=0, axis=thin_axis)
            vertex_plane = np.take(vertices_reshaped, indices=0, axis=thin_axis)

            # 从字段平面推断实际的 H 和 W（如果未指定image_shape）
            # field_plane的shape应该是 (H, W) 或 (W, H)，需要识别哪个是径向
            field_plane_shape = list(fields_dict.values())[0].shape

            # 根据field_plane确定H和W
            if field_plane_shape[0] < field_plane_shape[1]:
                # 较小的是径向（H），较大的是周向（W）
                H_inferred = field_plane_shape[0]
                W_inferred = field_plane_shape[1]
            else:
                # 需要交换
                H_inferred = field_plane_shape[1]
                W_inferred = field_plane_shape[0]
                # 同时需要转置所有field
                fields_dict = {k: v.T for k, v in fields_dict.items()}

            # 如果self.image_shape为None，使用推断值并更新实例属性
            if self.image_shape is None:
                self.image_shape = (H_inferred, W_inferred)
                self.n_rings = H_inferred
                self.n_pts_per_ring = W_inferred
                self.total_pts = H_inferred * W_inferred
                H, W = H_inferred, W_inferred
            else:
                H, W = self.image_shape

            # Turb: infer effective AoA from farfield velocity direction (deg)
            if is_turb_data and flow_conditions is not None:
                try:
                    aoa_deg = self._infer_farfield_aoa_deg(
                        fields_dict["VelocityX"],
                        fields_dict["VelocityY"],
                        n_outer_rings=5,
                    )
                    flow_conditions = dict(flow_conditions)
                    flow_conditions["AoA"] = aoa_deg
                except Exception:
                    pass

            # 将格心平面整理为 (H, W)
            center_shape = center_plane.shape[:2]
            if H in center_shape:
                radial_axis = center_shape.index(H)
            else:
                print(
                    f"格心平面未包含目标径向长度: plane={center_shape}, H={H} ({cgns_path})"
                )
                return None

            if radial_axis != 0:
                center_plane = center_plane.swapaxes(0, radial_axis)

            # 将节点平面整理为 (H+1, W+1) - 完整周向应为305列
            vertex_shape = vertex_plane.shape[:2]
            target_radial = H + 1
            target_tangential = W + 1  # 对于W=304，这将是305

            if target_radial in vertex_shape:
                radial_axis_v = vertex_shape.index(target_radial)
            else:
                print(
                    f"节点平面未包含目标径向长度: plane={vertex_shape}, H+1={target_radial} ({cgns_path})"
                )
                return None

            if radial_axis_v != 0:
                vertex_plane = vertex_plane.swapaxes(0, radial_axis_v)

            n_center_cols = center_plane.shape[1]
            n_vertex_cols = vertex_plane.shape[1]

            # 对于完整O-grid，期望n_center_cols >= 304, n_vertex_cols >= 305
            if n_center_cols < W:
                print(
                    f"格心切向列数不足: center={n_center_cols}, 目标 W={W} ({cgns_path})"
                )
                return None

            if n_vertex_cols < target_tangential:
                print(
                    f"节点切向列数不足: vertex={n_vertex_cols}, 目标 W+1={target_tangential} ({cgns_path})"
                )
                return None

            # 提取完整的周向长度（不使用模运算，直接取0:W和0:W+1）
            # 对于完整网格，应该恰好是304和305
            if n_center_cols >= W:
                ordered_center = center_plane[:, :W, :]

            if n_vertex_cols >= target_tangential:
                ordered_vertex = vertex_plane[:, :target_tangential, :]

            # 验证周期闭合：检查首尾节点差异
            periodic_diff = np.abs(ordered_vertex[:, -1, :2] - ordered_vertex[:, 0, :2])
            periodic_max_diff = float(periodic_diff.max())

            if periodic_max_diff < 1e-12:
                # 完美周期闭合
                pass
            elif periodic_max_diff < 1e-6:
                # 几乎闭合，给出警告并强制闭合
                print(
                    f"警告: 周期面差异较小但需要强制闭合 (max_diff={periodic_max_diff:.2e}): {cgns_path}"
                )
                ordered_vertex[:, -1, :2] = ordered_vertex[:, 0, :2]
            else:
                # 差异过大，这可能表示数据有问题或者周向长度不正确
                print(
                    f"错误: 周期面不闭合 (max_diff={periodic_max_diff:.2e}): {cgns_path}"
                )
                print(f"  首节点[:5]: {ordered_vertex[:, 0, :2]}")
                print(f"  尾节点[:5]: {ordered_vertex[:, -1, :2]}")
                print(f"  提示: 周向长度可能不正确，当前W={W}, target_tangential={target_tangential}")
                return None

            x_center = ordered_center[:, :, 0]
            y_center = ordered_center[:, :, 1]
            coords_center_array = np.stack([x_center, y_center], axis=0)  # (2, H, W)

            x_vertex = ordered_vertex[:, :, 0]
            y_vertex = ordered_vertex[:, :, 1]

            coords_vertex_array = np.stack([x_vertex, y_vertex], axis=0)  # (2, H+1, W+1)

            # 基本范围校验，确保节点覆盖格心
            if not (x_vertex.min() <= x_center.min() <= x_center.max() <= x_vertex.max()):
                print(f"警告: 节点X范围未完全覆盖格心X范围: {cgns_path}")
            if not (y_vertex.min() <= y_center.min() <= y_center.max() <= y_vertex.max()):
                print(f"警告: 节点Y范围未完全覆盖格心Y范围: {cgns_path}")

            # Stack all fields into unified tensor (C, H, W)
            fields_list = [fields_dict[name] for name in self.field_names]
            fields_array = np.stack(fields_list, axis=0)  # (C, H, W) - e.g., (5, 84, 304)

            # Organize output with dual coordinate systems
            # 坐标保持原始float64精度（由下游H5ShardBuilder的dtype_coords参数决定最终存储精度）
            # 物理场保持float32（足够数值计算精度且节省存储）
            result = {
                'fields': fields_array,              # (C, H, W) - e.g., (5, 84, 304)
                'coords_center': coords_center_array,                    # (2, H, W) - 保持float64
                'coords_vertex': coords_vertex_array,                    # (2, H+1, W+1) - 保持float64
                'field_names': self.field_names
            }

            # 添加几何参数（如果提供）
            if geometry_params is not None:
                result['geometry_params'] = geometry_params.astype(np.float32)

            # 添加流动条件（如果提供）
            if flow_conditions is not None:
                flow_array = np.array([
                    flow_conditions.get('Ma', 0.0),
                    flow_conditions.get('AoA', 0.0),
                    flow_conditions.get('Re', 0.0)
                ], dtype=np.float32)
                result['flow_conditions'] = flow_array

            return result

        except Exception as e:
            print(f"处理文件失败 {cgns_path}: {str(e)}")
            return None

    def _extract_velocity_component(
        self,
        block: pv.DataSet,
        component_name: str
    ) -> np.ndarray:
        """
        从Velocity矢量场提取分量

        Args:
            block: PyVista数据块
            component_name: 'VelocityX' 或 'VelocityY'

        Returns:
            速度分量数组
        """
        if 'Velocity' in block.cell_data:
            velocity = block.cell_data['Velocity']
            if component_name == 'VelocityX':
                return velocity[:, 0]
            elif component_name == 'VelocityY':
                return velocity[:, 1]
            else:
                raise ValueError(f"未知的速度分量: {component_name}")
        else:
            # 如果Velocity不存在，尝试直接读取分量
            if component_name in block.cell_data:
                return block.cell_data[component_name]
            else:
                raise ValueError(f"无法找到速度场或其分量: {component_name}")

    def _extract_blocks_recursive(
        self,
        obj: pv.MultiBlock,
        blocks_list: List[pv.DataSet]
    ):
        """递归提取所有数据块"""
        if isinstance(obj, pv.MultiBlock):
            for sub in obj:
                self._extract_blocks_recursive(sub, blocks_list)
        elif isinstance(obj, pv.DataSet):
            blocks_list.append(obj)

    def batch_extract(
        self,
        cgns_files: List[str],
        output_dir: str,
        geometry_data_file: Optional[str] = None,
        flow_conditions_file: Optional[str] = None,
        n_workers: int = 4,
        overwrite: bool = False
    ) -> int:
        """
        批量提取多个CGNS文件

        Args:
            cgns_files: CGNS文件路径列表
            output_dir: 输出目录
            geometry_data_file: 几何参数文件路径（.dat格式）
            flow_conditions_file: 流动条件文件路径（.csv格式）
            n_workers: 并行worker数量
            overwrite: 是否覆盖已存在的文件

        Returns:
            成功处理的文件数量
        """
        os.makedirs(output_dir, exist_ok=True)

        # 加载几何参数（如果提供）
        geometry_dict = None
        if geometry_data_file and os.path.exists(geometry_data_file):
            geometry_data = np.loadtxt(geometry_data_file)
            # 假设每行对应一个翼型，前14列为几何参数
            geometry_dict = {
                i: geometry_data[i, :14] for i in range(len(geometry_data))
            }
            print(f"加载几何参数: {len(geometry_dict)} 个样本")

        # 加载流动条件（如果提供）
        flow_dict = None
        if flow_conditions_file and os.path.exists(flow_conditions_file):
            flow_data = np.loadtxt(flow_conditions_file, skiprows=1, delimiter=',')
            # 假设列为: airfoil(1-based), case(1-based), Ma, AoA, Re
            # 统一转换为0-based以匹配文件名中的 airfoil_{id} (id为1-based但内部减1) 和 case_{id} (0-based)
            flow_dict = {}
            for row in flow_data:
                try:
                    airfoil_0b = int(row[0]) - 1
                    case_0b = int(row[1]) - 1
                    idx = airfoil_0b * 100 + case_0b
                    flow_dict[idx] = {
                        'Ma': float(row[2]),
                        'AoA': float(row[3]),
                        'Re': float(row[4])
                    }
                except Exception:
                    continue
            print(f"加载流动条件: {len(flow_dict)} 个样本 (索引已转换为0-based)")

        # 准备任务列表
        tasks = []
        for cgns_path in cgns_files:
            # 生成输出文件名
            base_name = Path(cgns_path).stem
            output_path = os.path.join(output_dir, f"{base_name}.npz")

            # 检查是否跳过
            if not overwrite and os.path.exists(output_path):
                continue

            # 解析文件名提取airfoil_id和case_id（根据UniFoil命名规则）
            # 例如: airfoil_1_G2_A_L0_case_0_000_surf_turb.cgns
            airfoil_id = self._parse_airfoil_id(base_name)
            case_id = self._parse_case_id(base_name)

            # 获取对应的几何参数和流动条件
            geom_params = geometry_dict.get(airfoil_id) if geometry_dict else None
            flow_conds = flow_dict.get(airfoil_id * 100 + case_id) if flow_dict else None

            tasks.append((
                cgns_path, output_path, geom_params, flow_conds
            ))

        if len(tasks) == 0:
            print("没有需要处理的文件")
            return 0

        print(f"开始批量提取: {len(tasks)} 个文件")

        # 并行处理
        if n_workers > 1:
            with Pool(min(n_workers, cpu_count())) as pool:
                results = list(tqdm(
                    pool.imap(self._process_single_wrapper, tasks),
                    total=len(tasks),
                    desc="提取进度"
                ))
        else:
            results = [
                self._process_single_wrapper(task)
                for task in tqdm(tasks, desc="提取进度")
            ]

        success_count = sum(results)
        print(f"完成批量提取: {success_count}/{len(tasks)} 个文件成功")

        return success_count

    def _process_single_wrapper(self, task: Tuple) -> bool:
        """单个文件处理的包装函数（用于多进程）"""
        cgns_path, output_path, geom_params, flow_conds = task

        result = self.extract_from_cgns(cgns_path, geom_params, flow_conds)

        if result is not None:
            try:
                np.savez_compressed(output_path, **result)
                return True
            except Exception as e:
                print(f"保存失败 {output_path}: {str(e)}")
                return False
        return False

    def _parse_airfoil_id(self, filename: str) -> int:
        """从文件名解析airfoil_id"""
        try:
            # airfoil_1_G2_A_L0_case_0_000_surf_turb
            parts = filename.split('_')
            return int(parts[1]) - 1  # 转为0-based索引
        except:
            return 0

    def _parse_case_id(self, filename: str) -> int:
        """从文件名解析case_id"""
        try:
            # airfoil_1_G2_A_L0_case_0_000_surf_turb
            parts = filename.split('_')
            case_idx = parts.index('case')
            return int(parts[case_idx + 1])
        except:
            return 0


def load_multifield_data(npz_path: str) -> Dict[str, np.ndarray]:
    """
    加载保存的多物理场数据

    Args:
        npz_path: .npz文件路径

    Returns:
        数据字典
    """
    data = np.load(npz_path, allow_pickle=True)
    return {key: data[key] for key in data.files}
