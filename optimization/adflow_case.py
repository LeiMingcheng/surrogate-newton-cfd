#!/usr/bin/env python
"""运行单个翼型的 ADflow CFD 计算。"""
import os
import sys
import json
import argparse
import tempfile
import shutil
import time
import numpy as np
from pathlib import Path
import warnings
import re
warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.common.flow_conditions import (
    REFERENCE_CHORD,
    REFERENCE_P_INF,
    REFERENCE_T_INF,
    coupled_reynolds_from_mach,
)
from surrogate.utils.runtime_paths import resolve_runtime_dir


def _configure_runtime_tmpdir():
    candidates = []
    override = os.environ.get('CFD_RUNTIME_TMPDIR')
    if override:
        candidates.append(Path(override))
    candidates.append(resolve_runtime_dir(None, default_subdir='cfd_runtime'))

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / f'.tmp_probe_{os.getpid()}'
            probe.write_text('ok')
            probe.unlink()
            runtime_tmpdir = str(candidate)
            os.environ['CFD_RUNTIME_TMPDIR'] = runtime_tmpdir
            os.environ['TMPDIR'] = runtime_tmpdir
            os.environ['TMP'] = runtime_tmpdir
            os.environ['TEMP'] = runtime_tmpdir
            tempfile.tempdir = runtime_tmpdir
            return runtime_tmpdir
        except Exception:
            continue
    raise RuntimeError(
        'No writable CFD runtime tmpdir available under the project runtime root. '
        'Set CFD_RUNTIME_TMPDIR to a writable directory.'
    )


_RUNTIME_TMPDIR = _configure_runtime_tmpdir()

# ADflow imports
try:
    from mpi4py import MPI
    from baseclasses import AeroProblem
    from adflow import ADFLOW
    HAS_ADFLOW = True
except ImportError as e:
    print(f"警告: ADflow 导入失败: {e}")
    HAS_ADFLOW = False


def _json_scalar(value):
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _optional_float(value):
    if value is None:
        return None
    return float(value)


def _parse_optional_bool(value):
    if value is None or isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if lowered in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _build_reference_state_inputs(
    *,
    mach,
    aoa,
    reynolds,
    run_name,
    eval_funcs,
    reference_state_mode,
    temperature,
    pressure,
    reynolds_length,
    area_ref,
    chord_ref,
    x_ref,
    y_ref,
    z_ref,
):
    mode = str(reference_state_mode)
    if mode not in {"dataset_unified", "match_reynolds"}:
        raise ValueError(f"Unsupported reference_state_mode: {mode}")

    temperature_value = REFERENCE_T_INF if temperature is None else float(temperature)
    area_ref_value = 1.0 if area_ref is None else float(area_ref)
    chord_ref_value = 1.0 if chord_ref is None else float(chord_ref)
    x_ref_value = 0.0 if x_ref is None else float(x_ref)
    y_ref_value = 0.0 if y_ref is None else float(y_ref)
    z_ref_value = 0.0 if z_ref is None else float(z_ref)

    ap_kwargs = {
        "name": run_name,
        "mach": float(mach),
        "alpha": float(aoa),
        "T": temperature_value,
        "areaRef": area_ref_value,
        "chordRef": chord_ref_value,
        "xRef": x_ref_value,
        "yRef": y_ref_value,
        "zRef": z_ref_value,
        "evalFuncs": list(eval_funcs),
    }
    state_meta = {
        "mode": mode,
        "requested_reynolds": float(reynolds),
        "input_temperature": temperature_value,
        "input_pressure": None,
        "input_reynolds_length": None,
        "area_ref": area_ref_value,
        "chord_ref": chord_ref_value,
        "x_ref": x_ref_value,
        "y_ref": y_ref_value,
        "z_ref": z_ref_value,
        "requested_vs_applied_re_rel_error": None,
    }

    if mode == "dataset_unified":
        pressure_value = REFERENCE_P_INF if pressure is None else float(pressure)
        applied_reynolds = coupled_reynolds_from_mach(
            mach,
            chord=REFERENCE_CHORD,
            temperature=temperature_value,
            pressure=pressure_value,
        )
        ap_kwargs["P"] = pressure_value
        state_meta["input_pressure"] = pressure_value
        state_meta["applied_reynolds"] = float(applied_reynolds)
        state_meta["applied_reynolds_length"] = float(REFERENCE_CHORD)
        state_meta["requested_vs_applied_re_rel_error"] = abs(float(reynolds) - applied_reynolds) / max(
            abs(applied_reynolds),
            1.0,
        )
        state_meta["ignored_inputs"] = {
            "reynolds": True,
            "reynolds_length": reynolds_length is not None,
        }
        return ap_kwargs, state_meta

    reynolds_length_value = REFERENCE_CHORD if reynolds_length is None else float(reynolds_length)
    ap_kwargs["reynolds"] = float(reynolds)
    ap_kwargs["reynoldsLength"] = reynolds_length_value
    state_meta["input_reynolds_length"] = reynolds_length_value
    state_meta["ignored_inputs"] = {
        "pressure": pressure is not None,
    }
    return ap_kwargs, state_meta


def _summarize_cl_solve_result(cl_solve_result):
    if not isinstance(cl_solve_result, dict):
        return None

    summary = {'method': 'adflow_solveCL'}
    for key in ('converged', 'iterations', 'l2convergence', 'alpha', 'cl', 'clstar', 'error', 'clalpha', 'time'):
        if key in cl_solve_result:
            summary[key] = _json_scalar(cl_solve_result.get(key))
    return summary


def _make_safe_convergence_history_getter(CFDSolver):
    def _safe_get_convergence_history(workUnitTime=None):
        comm = CFDSolver.comm
        if comm.rank == 0:
            history = {}
            try:
                converge_array = CFDSolver._trimHistoryData(CFDSolver.adflow.monitor.convarray)
                solver_array = CFDSolver._trimHistoryData(CFDSolver.adflow.monitor.solverdataarray)
                if solver_array is not None and getattr(solver_array, "shape", (0,))[0] > 0:
                    history["total minor iters"] = np.array(solver_array[:, 0], dtype=int)
                    if solver_array.shape[1] > 1:
                        history["CFL"] = np.array(solver_array[:, 1], dtype=float)
                    if solver_array.shape[1] > 2:
                        history["step"] = np.array(solver_array[:, 2], dtype=float)
                    if solver_array.shape[1] > 3:
                        history["linear res"] = np.array(solver_array[:, 3], dtype=float)
                    if solver_array.shape[1] > 4:
                        history["wall time"] = np.array(solver_array[:, 4], dtype=float)
                        if workUnitTime is not None:
                            history["work units"] = history["wall time"] * float(comm.size) / float(workUnitTime)
                if converge_array is not None and getattr(converge_array, "shape", (0,))[0] > 0:
                    history["monitor_values"] = np.array(converge_array, copy=True)
                    history["monitor_count"] = int(converge_array.shape[1]) if converge_array.ndim == 2 else None
            except Exception as exc:
                history = {"history_error": str(exc)}
        else:
            history = {}
        return comm.bcast(history, root=0)

    return _safe_get_convergence_history


def get_adflow_options(
    mode: str = 'turbulent',
    max_iterations: int = 6000,
    *,
    l2_convergence: float = 1e-8,
    print_iterations: bool = False,
    turbulence_model: str = 'SA',
) -> dict:
    """
    获取 ADflow 配置选项

    Args:
        mode: 当前仅支持 'turbulent'
        max_iterations: 最大迭代步数
    """
    # 标准 RANS-SA 湍流模型配置
    # 基于 UniFoil 原始 turb_sim_run/run_one_airfoil.py
    # ADflow 默认使用 SA (Spalart-Allmaras) 湍流模型，无需额外参数

    # ========== RANS-SA 求解器配置 ==========
    # 参考：原始 UniFoil turb_sim_run/run_one_airfoil.py + review3 审查意见
    options = {
        # 物理模型
        'equationType': 'RANS',
        'turbulenceModel': str(turbulence_model),

        # 时间推进
        'CFL': 1.5,
        'CFLCoarse': 1.25,
        'MGCycle': 'sg',  # Single grid

        # ANK 求解器
        'useANKSolver': True,
        'ANKNSubiterTurb': 3,        # 默认1太少，增大帮助跨声速湍流收敛
        'nsubiterturb': 10,          # NK 阶段湍流子迭代
        'anksecondordswitchtol': 1e-3,
        'ankstepfactor': 0.5,
        'ankmaxiter': 40,

        # NK 求解器
        # NKSwitchTol: 原始 UniFoil 用 1e-6；review3 建议 1e-6~1e-5
        # 设太严(1e-8)会导致 NK 永远不被激活，跨声速卡在 ANK 平台
        'useNKSolver': True,
        'nkswitchtol': 1e-5,
        'nkadpc': True,
        'nkinnerpreconits': 2,
        'nkjacobianlag': 3,
        'nkouterpreconits': 3,
        'nkpcilufill': 2,
        'nksubspacesize': 100,

        # 耗散延续（跨声速建议开启，帮助激波位置稳定）
        'useDissContinuation': True,

        # 收敛标准
        # ADflow 的 L2Convergence 判据是 totalR/totalR0（相对收敛）
        'L2Convergence': float(l2_convergence),
        'L2ConvergenceCoarse': 1e-2,
        'nCycles': max_iterations,

        # 输出配置 - 减少输出
        'monitorvariables': ['resrho', 'cl', 'cd', 'cmz'],
        'surfacevariables': ['cp', 'cf', 'mach'],
        'writeSurfaceSolution': False,
        'writeVolumeSolution': False,
        'outputsurfacefamily': 'wall',
        'outputDirectory': '.',

        # 数值方法
        'smoother': 'DADI',
        'MGStartLevel': -1,

        # 其他设置
        'printIterations': bool(print_iterations),
        'printTiming': False,
        'setMonitor': True,
    }

    if mode != 'turbulent':
        raise ValueError(f"Unsupported mode: {mode}")

    return options


def get_adflow_options_v2(
    mode: str = 'turbulent',
    max_iterations: int = 6000,
    *,
    l2_convergence: float = 1e-10,
    print_iterations: bool = False,
    turbulence_model: str = 'SA',
) -> dict:
    """
    Option2: review2 跨声速加强配置
    - CFL 1.0（更保守）
    - ANKNSubiterTurb=5, nSubiterTurb=25（更多湍流子迭代）
    - NKSwitchTol=1e-5（更早切 NK）
    - useDissContinuation=True + 拉长耗散延续
    - L2Convergence=1e-10（避免力系数未稳定就早停）
    """
    options = {
        'equationType': 'RANS',
        'turbulenceModel': str(turbulence_model),

        'CFL': 1.0,
        'CFLCoarse': 1.25,
        'MGCycle': 'sg',

        'useANKSolver': True,
        'ANKNSubiterTurb': 5,
        'anksecondordswitchtol': 1e-3,
        'ankstepfactor': 0.5,
        'ankmaxiter': 40,

        'useNKSolver': True,
        'nkswitchtol': 1e-5,
        'nkadpc': True,
        'nkinnerpreconits': 2,
        'nkjacobianlag': 3,
        'nkouterpreconits': 3,
        'nkpcilufill': 2,
        'nksubspacesize': 100,
        'nsubiterturb': 25,

        'useDissContinuation': True,
        'dissContMagnitude': 2.0,
        'dissContMidpoint': 20.0,
        'dissContSharpness': 3.0,

        'L2Convergence': float(l2_convergence),
        'L2ConvergenceCoarse': 1e-2,
        'nCycles': max_iterations,

        'monitorvariables': ['resrho', 'cl', 'cd', 'cmz'],
        'surfacevariables': ['cp', 'cf', 'mach'],
        'writeSurfaceSolution': False,
        'writeVolumeSolution': False,
        'outputsurfacefamily': 'wall',
        'outputDirectory': '.',

        'smoother': 'DADI',
        'MGStartLevel': -1,

        'printIterations': bool(print_iterations),
        'printTiming': False,
        'setMonitor': True,
    }

    if mode != 'turbulent':
        raise ValueError(f"Unsupported mode: {mode}")

    return options


def get_adflow_options_v3(
    mode: str = 'turbulent',
    max_iterations: int = 6000,
    *,
    l2_convergence: float = 1e-10,
    print_iterations: bool = False,
    turbulence_model: str = 'SA',
) -> dict:
    """
    Option3: v1 基础上只改 L2Convergence=1e-10 + nCycles=6000
    测试假设：v2 的主要改善来自更严的收敛门槛，而非耗散延续拉长
    """
    options = get_adflow_options(
        mode=mode,
        max_iterations=max_iterations,
        l2_convergence=l2_convergence,
        print_iterations=print_iterations,
        turbulence_model=turbulence_model,
    )
    return options


def get_adflow_options_v4(
    mode: str = 'turbulent',
    max_iterations: int = 6000,
    *,
    l2_convergence: float = 1e-10,
    print_iterations: bool = False,
    turbulence_model: str = 'SA',
) -> dict:
    """
    Option4: pure pseudo-time stepping 专用配置
    - `MGCycle=3w`：启用真实三层 multigrid，而不是 `sg` 单网格
    - `MGStartLevel=1`：跳过 RANS 下易失稳的 full-multigrid coarse-start
    - `CFL=3.0, CFLCoarse=1.5`：贴近 ADFLOW DADI 文档建议的实用区间
    - `nSubiter=3`：每个 smoothing step 做更多 DADI 子迭代
    - `resAveraging=always`：给 pure pseudo 增强阻尼
    - 保留 v2 的耗散延续与更严格收敛门槛
    """
    options = get_adflow_options_v2(
        mode=mode,
        max_iterations=max_iterations,
        l2_convergence=l2_convergence,
        print_iterations=print_iterations,
        turbulence_model=turbulence_model,
    )
    options.update(
        {
            'CFL': 3.0,
            'CFLCoarse': 1.5,
            'MGCycle': '3w',
            'MGStartLevel': 1,
            'nSubiter': 3,
            'resAveraging': 'always',
        }
    )
    return options


def get_adflow_options_v5(
    mode: str = 'turbulent',
    max_iterations: int = 6000,
    *,
    l2_convergence: float = 1e-10,
    print_iterations: bool = False,
    turbulence_model: str = 'SA',
) -> dict:
    """
    Option5: pure pseudo-time stepping 稳健版配置
    - `MGCycle=3w`：保持真实三层 multigrid pseudo
    - `smoother=Runge-Kutta`：按 ADFLOW 文档，较 DADI 更稳健
    - `CFL=1.0, CFLCoarse=0.75`：回到文档建议的保守 CFL 区间
    - `nSubiter=1`：减少每次非线性迭代中的 pseudo 子步激进程度
    - `resAveraging=always`：保持 residual smoothing 阻尼
    - 保留 v2 的耗散延续与严格收敛门槛
    """
    options = get_adflow_options_v2(
        mode=mode,
        max_iterations=max_iterations,
        l2_convergence=l2_convergence,
        print_iterations=print_iterations,
        turbulence_model=turbulence_model,
    )
    options.update(
        {
            'CFL': 1.0,
            'CFLCoarse': 0.75,
            'MGCycle': '3w',
            'MGStartLevel': 1,
            'smoother': 'Runge-Kutta',
            'nSubiter': 1,
            'resAveraging': 'always',
        }
    )
    return options


def get_adflow_options_v6(
    mode: str = 'turbulent',
    max_iterations: int = 6000,
    *,
    l2_convergence: float = 1e-10,
    print_iterations: bool = False,
    turbulence_model: str = 'SA',
) -> dict:
    """
    Option6: pure pseudo-time stepping 加速试验版配置
    - 保持 `3w + Runge-Kutta` 的真实 multigrid pseudo 路径
    - `CFL=1.5, CFLCoarse=1.0`：较 v5 略激进，但仍低于先前失稳区间
    - `nsubiterturb=8`：显著降低 v2 继承来的 25 次湍流子迭代成本
    - `resAveraging=alternate`：保留阻尼，但减少每步平滑开销
    """
    options = get_adflow_options_v5(
        mode=mode,
        max_iterations=max_iterations,
        l2_convergence=l2_convergence,
        print_iterations=print_iterations,
        turbulence_model=turbulence_model,
    )
    options.update(
        {
            'CFL': 1.5,
            'CFLCoarse': 1.0,
            'nsubiterturb': 8,
            'resAveraging': 'alternate',
        }
    )
    return options


def _extract_solver_options_summary(options: dict) -> dict:
    keys = (
        'CFL',
        'CFLCoarse',
        'MGCycle',
        'MGStartLevel',
        'ANKNSubiterTurb',
        'nSubiter',
        'nsubiterturb',
        'resAveraging',
        'ANKSwitchTol',
        'nkswitchtol',
        'anksecondordswitchtol',
        'useDissContinuation',
        'dissContMagnitude',
        'dissContMidpoint',
        'dissContSharpness',
        'vis2',
        'vis4',
        'L2Convergence',
        'nCycles',
        'equationType',
        'turbulenceModel',
    )
    return {key: _json_scalar(options.get(key)) for key in keys if key in options}


def run_adflow_cfd(
    cgns_file,
    output_dir,
    mach,
    aoa,
    reynolds,
    mode: str = 'turbulent',
    max_iterations: int = 6000,
    *,
    l2_convergence: float = 1e-8,
    print_iterations: bool = False,
    turbulence_model: str = 'SA',
    options_version: int = 1,
    reference_state_mode: str = 'dataset_unified',
    temperature: float = None,
    pressure: float = None,
    reynolds_length: float = None,
    area_ref: float = None,
    chord_ref: float = None,
    x_ref: float = None,
    y_ref: float = None,
    z_ref: float = None,
    cl_target: float = None,
    cl_tolerance: float = 1e-3,
    cl_solve_max_iter: int = None,
    ank_second_ord_switch_tol: float = None,
    ank_switch_tol: float = None,
    cfl: float = None,
    nk_switch_tol: float = None,
    ank_n_subiter_turb: int = None,
    n_subiter_turb: int = None,
    use_diss_continuation: bool = None,
    diss_cont_magnitude: float = None,
    diss_cont_midpoint: float = None,
    diss_cont_sharpness: float = None,
    vis2: float = None,
    vis4: float = None,
):
    """
    运行 ADflow CFD 计算

    Args:
        cgns_file: CGNS 网格文件路径
        output_dir: 输出目录
        mach: 马赫数
        aoa: 攻角 (度)
        reynolds: 雷诺数
        mode: 当前仅支持 'turbulent'
        max_iterations: 最大迭代步数

    Returns:
        dict: 包含 converged, iterations, forces 等信息
    """
    if not HAS_ADFLOW:
        raise ImportError("ADflow 未安装或导入失败")

    # 创建输出目录，并记录启动目录。
    # 某些 ADFLOW/CGNS 组合会忽略 outputDirectory，将 vol/surf CGNS 落到启动目录。
    os.makedirs(output_dir, exist_ok=True)
    launch_cwd = Path.cwd().resolve()

    comm = MPI.COMM_WORLD

    # 获取 ADflow 配置
    options_func = {
        1: get_adflow_options,
        2: get_adflow_options_v2,
        3: get_adflow_options_v3,
        4: get_adflow_options_v4,
        5: get_adflow_options_v5,
        6: get_adflow_options_v6,
    }[options_version]
    adflow_options = options_func(
        mode=mode,
        max_iterations=max_iterations,
        l2_convergence=l2_convergence,
        print_iterations=print_iterations,
        turbulence_model=turbulence_model,
    )
    if cfl is not None:
        adflow_options['CFL'] = float(cfl)
    if ank_switch_tol is not None:
        adflow_options['ANKSwitchTol'] = float(ank_switch_tol)
    if nk_switch_tol is not None:
        adflow_options['nkswitchtol'] = float(nk_switch_tol)
    if ank_n_subiter_turb is not None:
        adflow_options['ANKNSubiterTurb'] = int(ank_n_subiter_turb)
    if n_subiter_turb is not None:
        adflow_options['nsubiterturb'] = int(n_subiter_turb)
    if ank_second_ord_switch_tol is not None:
        adflow_options['anksecondordswitchtol'] = float(ank_second_ord_switch_tol)
    if use_diss_continuation is not None:
        adflow_options['useDissContinuation'] = bool(use_diss_continuation)
    if diss_cont_magnitude is not None:
        adflow_options['dissContMagnitude'] = float(diss_cont_magnitude)
    if diss_cont_midpoint is not None:
        adflow_options['dissContMidpoint'] = float(diss_cont_midpoint)
    if diss_cont_sharpness is not None:
        adflow_options['dissContSharpness'] = float(diss_cont_sharpness)
    if vis2 is not None:
        adflow_options['vis2'] = float(vis2)
    if vis4 is not None:
        adflow_options['vis4'] = float(vis4)
    adflow_options['gridFile'] = str(cgns_file)
    adflow_options['outputDirectory'] = str(output_dir)
    solver_options_summary = _extract_solver_options_summary(adflow_options)

    # 初始化 ADflow
    if comm.rank == 0:
        print(f"\n初始化 ADflow (RANS-SA 湍流模型)...")
    CFDSolver = None
    try:
        CFDSolver = ADFLOW(options=adflow_options, comm=MPI.COMM_WORLD)
    except Exception as e:
        if comm.rank == 0:
            print(f"ADflow 初始化失败: {e}")
        raise

    # 生成本次case的唯一名称（用于输出文件和func key）
    # 期望格式: airfoil_{id}_G2_A_L0_case_{case_id}
    mesh_base = Path(cgns_file).stem  # e.g., airfoil_1_G2_A_L0
    case_id = 0
    try:
        out_name = Path(output_dir).name
        m = re.search(r"_case_(\d+)$", out_name)
        if m:
            case_id = int(m.group(1))
    except Exception:
        case_id = 0

    run_name = f"{mesh_base}_case_{case_id}"
    eval_funcs = ['cl', 'cd', 'cmz', 'cdp', 'cdv']
    ap_kwargs, state_meta = _build_reference_state_inputs(
        mach=mach,
        aoa=aoa,
        reynolds=reynolds,
        run_name=run_name,
        eval_funcs=eval_funcs,
        reference_state_mode=reference_state_mode,
        temperature=temperature,
        pressure=pressure,
        reynolds_length=reynolds_length,
        area_ref=area_ref,
        chord_ref=chord_ref,
        x_ref=x_ref,
        y_ref=y_ref,
        z_ref=z_ref,
    )
    ap = AeroProblem(**ap_kwargs)
    applied_reynolds = float(state_meta.get('applied_reynolds', ap.reynolds))
    state_meta['applied_reynolds'] = applied_reynolds
    state_meta['applied_temperature'] = float(ap.T)
    state_meta['applied_pressure'] = float(ap.P)
    state_meta['applied_reynolds_length'] = float(
        state_meta.get('applied_reynolds_length', getattr(ap, 'reynoldsLength', REFERENCE_CHORD))
    )

    if comm.rank == 0:
        print(f"  Ma={mach:.3f}, AoA={aoa:.2f}°, requested Re={float(reynolds):.2e}")
        print(f"  reference_state_mode={state_meta['mode']}")
        if ank_second_ord_switch_tol is not None:
            print(f"  override: ANKSecondOrdSwitchTol={float(ank_second_ord_switch_tol):.3e}")
        print(f"  solver options: {solver_options_summary}")
        print(
            f"  applied state: T={state_meta['applied_temperature']:.3f} K, "
            f"P={state_meta['applied_pressure']:.3f} Pa, Re={state_meta['applied_reynolds']:.2e}"
        )
        if state_meta['mode'] == 'dataset_unified':
            if state_meta.get('requested_vs_applied_re_rel_error') is not None and state_meta['requested_vs_applied_re_rel_error'] > 1e-6:
                print("  注意: dataset_unified 模式按 Mach/T/P 统一参考状态求解，传入的 Re 不直接作为 ADflow 输入")
        else:
            print(
                f"  match_reynolds: reynoldsLength={state_meta['applied_reynolds_length']:.6f}, "
                f"xRef={state_meta['x_ref']:.6f}"
            )

    # 运行 CFD 求解 / 定升力求解
    cl_solve_result = None
    if comm.rank == 0:
        if cl_target is None:
            print(f"\n开始 CFD 求解 (nCycles={max_iterations}; ADflow 使用 approxTotalIts 计数)...")
        else:
            solvecl_max_iter = int(max_iterations if cl_solve_max_iter is None else cl_solve_max_iter)
            print(
                f"\n开始 ADflow solveCL 定升力求解 (CL*={float(cl_target):.6f}, alpha0={float(aoa):.3f}°, "
                f"solveCL maxIter={solvecl_max_iter}, nCycles={max_iterations})..."
            )
    original_get_convergence_history = None
    try:
        if cl_target is None:
            CFDSolver(ap)
        else:
            solvecl_max_iter = int(max_iterations if cl_solve_max_iter is None else cl_solve_max_iter)
            original_get_convergence_history = CFDSolver.getConvergenceHistory
            CFDSolver.getConvergenceHistory = _make_safe_convergence_history_getter(CFDSolver)
            cl_solve_result = CFDSolver.solveCL(
                ap,
                float(cl_target),
                alpha0=float(aoa),
                tol=float(cl_tolerance),
                maxIter=solvecl_max_iter,
                relaxCLa=0.9,
                relaxAlpha=1.0,
                stopOnStall=True,
            )
    except Exception as e:
        if comm.rank == 0:
            print(f"CFD 求解失败: {e}")
        return {
            'converged': False,
            'error': str(e),
            'mode': mode
        }

    finally:
        if original_get_convergence_history is not None:
            CFDSolver.getConvergenceHistory = original_get_convergence_history

    final_aoa = float(ap.alpha)

    # 提取力系数
    funcs = {}
    CFDSolver.evalFunctions(ap, funcs)

    # 获取收敛信息
    # ADflow 的 L2Convergence 判据是 totalR/totalR0（相对收敛）
    l2_target = float(adflow_options.get('L2Convergence', 1e-8))

    # 收敛判据：避免使用 getConvergenceHistory()
    # 当前 ADflow 版本在 getConvergenceHistory() 中可能触发 UnicodeDecodeError（Fortran 字符串解码）。
    # 这里直接读取 monitor.convarray，取最后一次迭代的 resrho（monitorVariables 的第一个变量）。
    final_residual = None
    if comm.rank == 0:
        try:
            conv = CFDSolver.adflow.monitor.convarray
            conv_trim = CFDSolver._trimHistoryData(conv)
            if conv_trim is not None and getattr(conv_trim, "size", 0) > 0:
                final_residual = float(abs(conv_trim[-1, 0]))
        except Exception:
            final_residual = None

    final_residual = comm.bcast(final_residual, root=0)
    if final_residual is None:
        final_residual = 1e9

    # Compute totalR-based convergence ratio using ADFLOW's official iteration module.
    # This must match pyADflow.solveCL/checkConvergence(), which uses:
    #   iteration.totalrfinal / iteration.totalr0
    # Do not recompute from monitor.convarray because monitor column layout is not
    # guaranteed to expose totalR in a fixed position.
    l2_ratio = None
    total_r0 = None
    total_rfinal = None
    n_outer_iter = None
    try:
        iteration_module = CFDSolver.adflow.iteration
        total_r0_raw = getattr(iteration_module, "totalr0", None)
        total_rfinal_raw = getattr(iteration_module, "totalrfinal", None)
        if total_r0_raw is not None:
            total_r0 = float(total_r0_raw)
        if total_rfinal_raw is not None:
            total_rfinal = float(total_rfinal_raw)
        if total_r0 is not None and total_r0 > 0.0 and total_rfinal is not None:
            l2_ratio = float(total_rfinal / total_r0)

        conv = CFDSolver.adflow.monitor.convarray
        conv_trim = CFDSolver._trimHistoryData(conv)
        if conv_trim is not None and getattr(conv_trim, "shape", (0,))[0] > 0:
            n_outer_iter = int(conv_trim.shape[0] - 1)
    except Exception:
        l2_ratio = None
        total_r0 = None
        total_rfinal = None
        n_outer_iter = None

    # Broadcast metrics
    l2_ratio = comm.bcast(l2_ratio, root=0)
    total_r0 = comm.bcast(total_r0, root=0)
    total_rfinal = comm.bcast(total_rfinal, root=0)
    n_outer_iter = comm.bcast(n_outer_iter, root=0)

    # ADflow internal counters / flags (useful for diagnosing "early stop")
    approx_total_its = None
    iter_tot = None
    routine_failed = None
    fatal_fail = None
    if comm.rank == 0:
        try:
            # ADflow counts work using approxTotalIts and checks it against nCycles.
            approx_total_its = int(getattr(CFDSolver.adflow.iteration, "approxtotalits"))
        except Exception:
            approx_total_its = None
        try:
            iter_tot = int(getattr(CFDSolver.adflow.iteration, "itertot"))
        except Exception:
            iter_tot = None
        try:
            routine_failed = bool(getattr(CFDSolver.adflow.killsignals, "routinefailed"))
        except Exception:
            routine_failed = None
        try:
            fatal_fail = bool(getattr(CFDSolver.adflow.killsignals, "fatalfail"))
        except Exception:
            fatal_fail = None

    approx_total_its = comm.bcast(approx_total_its, root=0)
    iter_tot = comm.bcast(iter_tot, root=0)
    routine_failed = comm.bcast(routine_failed, root=0)
    fatal_fail = comm.bcast(fatal_fail, root=0)

    # Verification acceptance rule:
    # 1) strict CFD convergence uses the user-specified L2Convergence target;
    # 2) if max_cycles is reached without strict convergence, allow a relaxed
    #    acceptance only when L2 <= accept_l2 (default 1e-3) and force coefficients are stable.
    residual_converged = None
    if l2_ratio is not None:
        residual_converged = bool(l2_ratio <= l2_target)
    else:
        residual_converged = bool(final_residual < 1e-5)

    relaxed_l2_target = 1e-3
    try:
        relaxed_l2_target = float(os.environ.get('CFD_ACCEPT_L2', '1e-3'))
    except Exception:
        relaxed_l2_target = 1e-3
    if relaxed_l2_target <= 0:
        relaxed_l2_target = 1e-3

    reached_max_cycles = bool(approx_total_its is not None and int(approx_total_its) >= int(max_iterations))
    relaxed_l2_acceptable = bool(
        (not residual_converged)
        and reached_max_cycles
        and l2_ratio is not None
        and float(l2_ratio) <= float(relaxed_l2_target)
    )

    converged = bool(residual_converged)

    # 力系数稳定性检测（review3 审查意见）
    # 检查最近 N 步的 CL/CD 变化幅度，判断力系数是否已稳定
    force_stability = None
    if comm.rank == 0:
        try:
            conv = CFDSolver.adflow.monitor.convarray
            conv_trim = CFDSolver._trimHistoryData(conv)
            if conv_trim is not None and conv_trim.shape[0] >= 3:
                if conv_trim.ndim == 3:
                    cl_hist = conv_trim[:, 0, 1]
                    cd_hist = conv_trim[:, 0, 2]
                else:
                    cl_hist = conv_trim[:, 1]
                    cd_hist = conv_trim[:, 2]

                hist_len = int(conv_trim.shape[0])
                try:
                    window_ratio = float(os.environ.get('CFD_FORCE_STABILITY_WINDOW_RATIO', '0.02'))
                except Exception:
                    window_ratio = 0.02
                if not np.isfinite(window_ratio):
                    window_ratio = 0.02
                window_ratio = min(max(window_ratio, 0.01), 0.03)

                n_check = int(np.ceil(hist_len * window_ratio))
                n_check = max(3, min(hist_len, n_check))
                cl_tail = cl_hist[-n_check:]
                cd_tail = cd_hist[-n_check:]
                delta_cl = float(np.max(cl_tail) - np.min(cl_tail))
                delta_cd = float(np.max(cd_tail) - np.min(cd_tail))
                cl_mean = float(np.mean(np.abs(cl_tail)))
                cd_mean = float(np.mean(np.abs(cd_tail)))
                try:
                    force_rel_tol = float(os.environ.get('CFD_FORCE_STABILITY_REL_TOL', '3e-2'))
                except Exception:
                    force_rel_tol = 3e-2
                if not np.isfinite(force_rel_tol) or force_rel_tol <= 0:
                    force_rel_tol = 3e-2

                # 相对稳定性准则: 默认要求 ΔCL/|CL| 和 ΔCD/|CD| < 3e-2 (3%)
                cl_stable = delta_cl < max(cl_mean * force_rel_tol, 1e-6)
                cd_stable = delta_cd < max(cd_mean * force_rel_tol, 1e-7)
                force_stability = {
                    'stable': bool(cl_stable and cd_stable),
                    'delta_cl': delta_cl,
                    'delta_cd': delta_cd,
                    'rel_delta_cl': delta_cl / cl_mean if cl_mean > 1e-10 else float('inf'),
                    'rel_delta_cd': delta_cd / cd_mean if cd_mean > 1e-10 else float('inf'),
                    'relative_tolerance': force_rel_tol,
                    'n_check': n_check,
                    'history_len': hist_len,
                    'window_ratio': window_ratio,
                    'window_fraction_used': float(n_check) / float(hist_len) if hist_len > 0 else None,
                }
        except Exception:
            force_stability = None
    force_stability = comm.bcast(force_stability, root=0)
    force_stable = bool(isinstance(force_stability, dict) and force_stability.get('stable') is True)

    target_cl_converged = None
    if cl_target is not None:
        try:
            target_cl_converged = bool(
                abs(float(funcs[f"{ap.name}_cl"] - float(cl_target))) <= float(cl_tolerance)
            )
        except Exception:
            target_cl_converged = False

    solvecl_converged = None
    solvecl_iterations = None
    if isinstance(cl_solve_result, dict):
        solvecl_converged = bool(cl_solve_result.get("converged", False))
        try:
            if cl_solve_result.get("iterations") is not None:
                solvecl_iterations = int(cl_solve_result.get("iterations"))
        except Exception:
            solvecl_iterations = None

    # Final verification criterion for this CFD run:
    #   - fixed-AoA: strict/relaxed CFD acceptance and force stable
    #   - solveCL: strict/relaxed CFD acceptance and force stable and target CL hit
    accepted_converged = bool((residual_converged or relaxed_l2_acceptable) and force_stable)
    if cl_target is not None:
        accepted_converged = bool(accepted_converged and target_cl_converged)
    converged = bool(accepted_converged)

    stop_reason = None
    if fatal_fail:
        stop_reason = "fatal_fail"
    elif cl_target is not None and not bool(target_cl_converged):
        if solvecl_iterations is not None and cl_solve_max_iter is not None and int(solvecl_iterations) >= int(cl_solve_max_iter):
            stop_reason = "max_aoa_iter"
        else:
            stop_reason = "target_cl_not_converged"
    elif residual_converged and not force_stable:
        stop_reason = "force_unstable"
    elif residual_converged and force_stable:
        stop_reason = "converged"
    elif relaxed_l2_acceptable and force_stable:
        stop_reason = "relaxed_l2_accepted"
    elif relaxed_l2_acceptable and not force_stable:
        stop_reason = "force_unstable"
    elif reached_max_cycles:
        stop_reason = "max_cycles"
    elif routine_failed:
        stop_reason = "routine_failed"
    else:
        stop_reason = "unknown"

    # Iteration counters:
    minor_iters_total = None
    if comm.rank == 0:
        try:
            sol = CFDSolver.adflow.monitor.solverdataarray
            sol_trim = CFDSolver._trimHistoryData(sol)
            if sol_trim is not None and getattr(sol_trim, "shape", (0,))[0] > 0:
                if sol_trim.ndim == 3:
                    minor_iters_total = int(sol_trim[-1, 0, 0])
                else:
                    minor_iters_total = int(sol_trim[-1, 0])
        except Exception:
            minor_iters_total = None
    minor_iters_total = comm.bcast(minor_iters_total, root=0)
    iterations = int(n_outer_iter) if n_outer_iter is not None else int(CFDSolver.getOption('nCycles'))
    if minor_iters_total is not None:
        # In some builds, solverdataarray is not populated unless iteration printing is enabled.
        # Guard against reporting a bogus 0.
        if minor_iters_total <= 0 or minor_iters_total < iterations:
            minor_iters_total = None

    # 写入 CGNS 文件（统一命名到 run_name，同时保留case信息）
    # 注意：pyADFLOW 的 writeSolution() 受 writeVolumeSolution/writeSurfaceSolution 逻辑开关控制。
    # 由于我们默认关闭自动写入，这里显式调用低层写文件接口，确保生成 vol/surf CGNS。
    vol_file = Path(output_dir) / f"{run_name}_000_vol.cgns"
    surf_file = Path(output_dir) / f"{run_name}_000_surf.cgns"
    vol_write_target = vol_file.name
    surf_write_target = surf_file.name
    if comm.rank == 0:
        print(f"\n写入 CGNS 文件...")
    try:
        # Use relative filenames inside output_dir. Some ADflow/CGNS builds mishandle
        # long absolute targets and silently emit truncated names.
        CFDSolver.writeVolumeSolutionFile(vol_write_target, writeGrid=True)
        CFDSolver.writeSurfaceSolutionFile(surf_write_target)
    except Exception as e:
        if comm.rank == 0:
            print(f"写入 CGNS 文件失败: {e}")
        return {
            'converged': False,
            'error': f"writeSolution failed: {e}",
            'mode': mode,
        }
    # Ensure all ranks complete I/O before rank0 inspects files / writes forces.json
    if comm.rank == 0:
        print("writeSolution completed; entering post-write barrier...", flush=True)
    comm.barrier()
    if comm.rank == 0:
        print("post-write barrier passed; finalizing output files...", flush=True)

    def _finalize_output_file(target_path: Path) -> Path | None:
        deadline = time.perf_counter() + 10.0
        candidate_dirs = []
        for candidate_dir in (
            target_path.parent.resolve(),
            launch_cwd,
            Path.cwd().resolve(),
        ):
            if candidate_dir not in candidate_dirs:
                candidate_dirs.append(candidate_dir)

        while True:
            if target_path.exists():
                return target_path

            for candidate_dir in candidate_dirs:
                candidate = candidate_dir / target_path.name
                if not candidate.exists():
                    continue
                if candidate.resolve() == target_path.resolve():
                    return target_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(candidate), str(target_path))
                return target_path

            if time.perf_counter() >= deadline:
                return None
            time.sleep(0.1)

    finalized_vol_file = None
    finalized_surf_file = None
    if comm.rank == 0:
        finalized_vol_file = _finalize_output_file(vol_file)
        finalized_surf_file = _finalize_output_file(surf_file)
    finalized_vol_file = comm.bcast(str(finalized_vol_file) if finalized_vol_file is not None else None, root=0)
    finalized_surf_file = comm.bcast(str(finalized_surf_file) if finalized_surf_file is not None else None, root=0)
    finalized_vol_file = Path(finalized_vol_file) if finalized_vol_file is not None else None
    finalized_surf_file = Path(finalized_surf_file) if finalized_surf_file is not None else None
    if comm.rank == 0:
        print(
            f"finalized output files: volume={finalized_vol_file}, surface={finalized_surf_file}",
            flush=True,
        )

    # 结果字典（确保所有值都是 JSON 可序列化的）
    # 注意：MPI模式下，只允许 rank0 写 forces.json；这里将结果广播给所有 rank 以便一致退出。
    result = None
    if comm.rank == 0:
        reference_state = {
            'mode': str(state_meta['mode']),
            'requested_reynolds': float(state_meta['requested_reynolds']),
            'applied_reynolds': float(state_meta['applied_reynolds']),
            'requested_vs_applied_re_rel_error': _optional_float(state_meta.get('requested_vs_applied_re_rel_error')),
            'temperature': float(state_meta['applied_temperature']),
            'pressure': float(state_meta['applied_pressure']),
            'reynolds_length': _optional_float(state_meta.get('applied_reynolds_length')),
            'area_ref': float(state_meta['area_ref']),
            'chord_ref': float(state_meta['chord_ref']),
            'x_ref': float(state_meta['x_ref']),
            'y_ref': float(state_meta['y_ref']),
            'z_ref': float(state_meta['z_ref']),
            'ignored_inputs': state_meta.get('ignored_inputs', {}),
        }
        result = {
            'converged': converged,
            'residual_converged': bool(residual_converged),
            'relaxed_l2_acceptable': bool(relaxed_l2_acceptable),
            'accepted_l2_target': float(relaxed_l2_target),
            'force_stable': bool(force_stable),
            'stop_reason': str(stop_reason) if stop_reason is not None else None,
            'iterations': int(iterations),
            'minor_iterations': int(minor_iters_total) if minor_iters_total is not None else None,
            'approx_total_iterations': int(approx_total_its) if approx_total_its is not None else None,
            'adflow_itertot': int(iter_tot) if iter_tot is not None else None,
            'routine_failed': bool(routine_failed) if routine_failed is not None else None,
            'fatal_fail': bool(fatal_fail) if fatal_fail is not None else None,
            'nCycles': int(max_iterations),
            'final_residual': float(final_residual),
            'final_total_residual': float(total_rfinal) if total_rfinal is not None else None,
            'l2_ratio': float(l2_ratio) if l2_ratio is not None else None,
            'l2_target': float(l2_target),
            'mode': str(mode),
            'turbulence_model': str(adflow_options.get('turbulenceModel', turbulence_model)),
            'options_version': int(options_version),
            'solver_options': solver_options_summary,
            'force_coefficients': {
                'cl': float(funcs[f"{ap.name}_cl"]),
                'cd': float(funcs[f"{ap.name}_cd"]),
                'cmz': float(funcs[f"{ap.name}_cmz"]),
                'cdp': float(funcs.get(f"{ap.name}_cdp", 0.0)),
                'cdv': float(funcs.get(f"{ap.name}_cdv", 0.0)),
            },
            'flow_conditions': {
                'Mach': float(mach),
                'AoA': float(final_aoa),
                'Reynolds': float(state_meta['applied_reynolds']),
                'RequestedReynolds': float(state_meta['requested_reynolds']),
                'Temperature': float(state_meta['applied_temperature']),
                'Pressure': float(state_meta['applied_pressure']),
                'reference_state_mode': str(state_meta['mode']),
            },
            'reference_state': reference_state,
            'cl_target': float(cl_target) if cl_target is not None else None,
            'cl_tolerance': float(cl_tolerance) if cl_target is not None else None,
            'cl_error': float(funcs[f"{ap.name}_cl"] - float(cl_target)) if cl_target is not None else None,
            'target_cl_converged': target_cl_converged,
            'cl_solve': _summarize_cl_solve_result(cl_solve_result),
            'force_stability': force_stability,
            'output_files': {
                'volume': str(finalized_vol_file) if finalized_vol_file is not None else None,
                'surface': str(finalized_surf_file) if finalized_surf_file is not None else None,
            }
        }

    # Broadcast result to all ranks so they have identical exit status
    if comm.rank == 0:
        print("broadcasting final result...", flush=True)
    result = comm.bcast(result, root=0)
    if comm.rank == 0:
        print("final result broadcast completed.", flush=True)

    if comm.rank == 0:
        print(f"\n✓ CFD 计算完成")
        print(f"  收敛状态: {'✓ 收敛' if result['converged'] else '✗ 未收敛'}")
        print(f"  外迭代步数: {result['iterations']}")
        if result.get("minor_iterations") is not None:
            print(f"  子迭代总数: {result['minor_iterations']}")
        if result.get("approx_total_iterations") is not None:
            print(f"  approxTotalIts: {result['approx_total_iterations']} (nCycles={result.get('nCycles')})")
        if result.get("stop_reason"):
            print(f"  停止原因: {result['stop_reason']}")
        if result.get("l2_ratio") is not None:
            print(f"  L2 ratio: {result['l2_ratio']:.2e} (target {result['l2_target']:.2e})")
        print(f"  Res rho : {result['final_residual']:.2e}")
        print(f"  AoA = {result['flow_conditions']['AoA']:.4f}°")
        print(f"  Re = {result['flow_conditions']['Reynolds']:.4e} ({result['flow_conditions']['reference_state_mode']})")
        print(f"  Cl = {result['force_coefficients']['cl']:.4f}")
        print(f"  Cd = {result['force_coefficients']['cd']:.5f}")
        print(f"  L/D = {result['force_coefficients']['cl'] / max(result['force_coefficients']['cd'], 1e-8):.2f}")
        if result.get('cl_target') is not None:
            print(
                f"  CL target = {result['cl_target']:.4f}, error = {result['cl_error']:.2e}, "
                f"target_hit = {bool(result['target_cl_converged'])}"
            )
            if isinstance(result.get('cl_solve'), dict):
                print(
                    f"  solveCL: strict_converged = {bool(result['cl_solve'].get('converged', False))}, "
                    f"iterations = {result['cl_solve'].get('iterations')}"
                )
        if result.get('force_stability'):
            fs = result['force_stability']
            print(f"  力系数稳定性: {'✓' if fs['stable'] else '✗'} (ΔCL={fs['delta_cl']:.2e}, ΔCD={fs['delta_cd']:.2e}, last {fs['n_check']} iters)")

    # 显式释放 ADflow 资源
    try:
        if CFDSolver is not None:
            if comm.rank == 0:
                print("starting CFDSolver cleanup...", flush=True)
            del CFDSolver
            import gc
            gc.collect()
            if comm.rank == 0:
                print("CFDSolver cleanup completed.", flush=True)
    except:
        pass

    return result


def main():
    parser = argparse.ArgumentParser(description='运行单个翼型的 ADflow CFD 计算')
    parser.add_argument('--cgns', type=str, required=True, help='CGNS 网格文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--mach', type=float, required=True, help='马赫数')
    parser.add_argument('--aoa', type=float, required=True, help='攻角 (度)')
    parser.add_argument('--reynolds', type=float, required=True, help='雷诺数')
    parser.add_argument('--mode', type=str, default='turbulent',
                       choices=['turbulent'],
                       help='CFD 模式')
    parser.add_argument('--max-iter', type=int, default=6000, help='最大迭代步数')
    parser.add_argument(
        '--l2conv',
        type=float,
        default=1e-8,
        help='ADflow 收敛阈值（totalR/totalR0）；值越小越严格，可能需要更多迭代',
    )
    parser.add_argument(
        '--print-iterations',
        action='store_true',
        help='启用 ADflow 迭代信息打印（用于调试；会显著增大日志）',
    )
    parser.add_argument(
        '--turb-model',
        type=str,
        default='SA',
        help="ADflow turbulenceModel (e.g., 'SA', 'Menter SST', 'k-omega Wilcox')",
    )
    parser.add_argument(
        '--reference-state-mode',
        type=str,
        default='dataset_unified',
        choices=['dataset_unified', 'match_reynolds'],
        help='参考状态模式: dataset_unified=当前数据集统一 Mach-Re 参考状态, match_reynolds=显式匹配给定 Re',
    )
    parser.add_argument('--temperature', type=float, default=None, help='参考温度 K')
    parser.add_argument('--pressure', type=float, default=None, help='参考静压 Pa；仅 dataset_unified 模式直接使用')
    parser.add_argument('--reynolds-length', type=float, default=None, help='雷诺数参考长度；仅 match_reynolds 模式直接使用')
    parser.add_argument('--area-ref', type=float, default=None, help='参考面积')
    parser.add_argument('--chord-ref', type=float, default=None, help='参考弦长')
    parser.add_argument('--x-ref', type=float, default=None, help='力矩参考点 x')
    parser.add_argument('--y-ref', type=float, default=None, help='力矩参考点 y')
    parser.add_argument('--z-ref', type=float, default=None, help='力矩参考点 z')
    parser.add_argument('--force-json', type=str, default=None,
                       help='力系数 JSON 输出文件')
    parser.add_argument('--options-version', type=int, default=1, choices=[1, 2, 3, 4, 5, 6],
                       help='求解器参数版本: 1=默认, 2=跨声速加强(review2), 3=v1+L2=1e-10, 4=pure-pseudo(DADI), 5=pure-pseudo稳健版, 6=pure-pseudo加速试验版')
    parser.add_argument('--cl-target', type=float, default=None, help='目标升力系数；提供后启用 ADflow solveCL')
    parser.add_argument('--cl-tol', type=float, default=1e-3, help='solveCL 的 CL 收敛容差')
    parser.add_argument('--cl-solve-max-iter', type=int, default=None, help='ADflow solveCL 最大外层迭代次数；默认回退到 --max-iter')
    parser.add_argument('--cfl', type=float, default=None, help='显式覆盖 CFL')
    parser.add_argument('--ank-switch-tol', type=float, default=None, help='显式覆盖 ANKSwitchTol')
    parser.add_argument('--nk-switch-tol', type=float, default=None, help='显式覆盖 NKSwitchTol')
    parser.add_argument('--ank-nsubiterturb', type=int, default=None, help='显式覆盖 ANKNSubiterTurb')
    parser.add_argument('--nsubiterturb', type=int, default=None, help='显式覆盖 nSubiterTurb')
    parser.add_argument(
        '--use-diss-continuation',
        type=_parse_optional_bool,
        default=None,
        help='显式覆盖 useDissContinuation (true/false)',
    )
    parser.add_argument('--diss-cont-magnitude', type=float, default=None, help='显式覆盖 dissContMagnitude')
    parser.add_argument('--diss-cont-midpoint', type=float, default=None, help='显式覆盖 dissContMidpoint')
    parser.add_argument('--diss-cont-sharpness', type=float, default=None, help='显式覆盖 dissContSharpness')
    parser.add_argument('--vis2', type=float, default=None, help='显式覆盖二阶人工粘性系数 vis2')
    parser.add_argument('--vis4', type=float, default=None, help='显式覆盖四阶人工粘性系数 vis4')
    parser.add_argument(
        '--ank-second-ord-switch-tol',
        type=float,
        default=None,
        help='显式覆盖 ANKSecondOrdSwitchTol',
    )

    args = parser.parse_args()

    # 验证输入文件
    cgns_path = Path(args.cgns).expanduser().resolve()
    if not cgns_path.exists():
        raise FileNotFoundError(f"CGNS 文件不存在: {cgns_path}")
    output_dir = str(Path(args.output).expanduser().resolve())

    # 运行 CFD
    result = run_adflow_cfd(
        cgns_file=cgns_path,
        output_dir=output_dir,
        mach=args.mach,
        aoa=args.aoa,
        reynolds=args.reynolds,
        mode=args.mode,
        max_iterations=args.max_iter,
        l2_convergence=args.l2conv,
        print_iterations=args.print_iterations,
        turbulence_model=args.turb_model,
        options_version=args.options_version,
        reference_state_mode=args.reference_state_mode,
        temperature=args.temperature,
        pressure=args.pressure,
        reynolds_length=args.reynolds_length,
        area_ref=args.area_ref,
        chord_ref=args.chord_ref,
        x_ref=args.x_ref,
        y_ref=args.y_ref,
        z_ref=args.z_ref,
        cl_target=args.cl_target,
        cl_tolerance=args.cl_tol,
        cl_solve_max_iter=args.cl_solve_max_iter,
        ank_second_ord_switch_tol=args.ank_second_ord_switch_tol,
        ank_switch_tol=args.ank_switch_tol,
        cfl=args.cfl,
        nk_switch_tol=args.nk_switch_tol,
        ank_n_subiter_turb=args.ank_nsubiterturb,
        n_subiter_turb=args.nsubiterturb,
        use_diss_continuation=args.use_diss_continuation,
        diss_cont_magnitude=args.diss_cont_magnitude,
        diss_cont_midpoint=args.diss_cont_midpoint,
        diss_cont_sharpness=args.diss_cont_sharpness,
        vis2=args.vis2,
        vis4=args.vis4,
    )

    # 保存力系数到 JSON
    if args.force_json:
        force_json_path = Path(args.force_json).expanduser().resolve()
        force_json_path.parent.mkdir(parents=True, exist_ok=True)
        result['cgns_file'] = str(cgns_path)

        # MPI模式下，仅允许 rank0 写入，避免多进程并发写文件导致损坏
        comm = MPI.COMM_WORLD
        if comm.rank == 0:
            with open(force_json_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n力系数已保存到: {force_json_path}")
        comm.barrier()

    # 返回退出码
    sys.exit(0 if result['converged'] else 1)


if __name__ == '__main__':
    main()
