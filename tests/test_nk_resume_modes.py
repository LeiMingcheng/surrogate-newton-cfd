import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from NK_resume import (
    NKWorkPlan,
    ResumeMode,
    SolverPreset,
    build_resume_case,
    create_pipeline,
    finalonly_plan,
    resume_case_from_payload,
    resume_plan_from_dict,
)
from NK_resume.solver.adflow import ADflowBackend
from NK_resume.solver.adflow_options import ADflowOptionRequest, build_adflow_options
from deployment.run import _resume_work_plan
from optimization.config import (
    NKConfig,
    OptimizationConfig,
    OptimizerConfig,
    TaskConfig,
)
from optimization.evaluators import SurrogateNKEvaluator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE = PROJECT_ROOT / "optimization" / "baselines" / "rae2822"


def test_terminal_resume_defaults_to_ank_nk() -> None:
    plan = finalonly_plan("fsb")
    work = plan.final_stage.work

    assert work.resume_mode == ResumeMode.ANK_NK
    assert work.solver_preset == SolverPreset.PROD
    assert work.fixed_cycles == 2000
    assert work.time_limit_s == 10.0
    assert work.nk_switch_tolerance == 1.0e-4
    assert work.to_dict()["resume_mode"] == "ank_nk"


def test_repeated_nk_is_an_explicit_cumulative_direct_nk_schedule() -> None:
    work = NKWorkPlan.repeated_nk((6, 8, 10), threshold=1.0e-8)
    loaded = resume_plan_from_dict(finalonly_plan("fsb", work=work).to_dict())

    assert work.resume_mode == ResumeMode.REPEATED_NK
    assert work.solver_preset == SolverPreset.NK
    assert work.adaptive_schedule.cumulative_cycles == (6, 8, 10)
    assert work.to_dict()["resume_mode"] == "repeated_nk"
    assert loaded.final_stage.work.resume_mode == ResumeMode.REPEATED_NK


def test_fixed_lift_context_survives_payload_round_trip(tmp_path: Path) -> None:
    cgns_root = tmp_path / "cgns"
    cgns_root.mkdir()
    field = np.ones((5, 2, 3), dtype=np.float64)
    case = build_resume_case(
        case_id="solvecl",
        predictor_kind="fsb",
        cgns_basename="solvecl.cgns",
        cgns_root=cgns_root,
        prediction_field=field,
        coords_center=np.zeros((2, 2, 3), dtype=np.float64),
        coords_vertex=np.zeros((2, 3, 4), dtype=np.float64),
        flow_conditions=(0.73, 2.1, 20.0e6),
        fixed_lift={
            "target_cl": 0.8,
            "cl_tolerance": 0.004,
            "max_aoa_solves": 3,
        },
    )
    exported = create_pipeline().export_cases(
        [case],
        finalonly_plan("fsb"),
        output_dir=tmp_path / "export",
    )

    loaded = resume_case_from_payload(exported.payload_paths[0])

    assert loaded.solver_context.fixed_lift.target_cl == 0.8
    assert loaded.solver_context.fixed_lift.cl_tolerance == 0.004
    assert loaded.solver_context.fixed_lift.max_aoa_solves == 3
    assert loaded.solver_context.fixed_lift.total_time_limit_s == 30.0


def test_production_options_enable_ank_to_nk_switching() -> None:
    options = build_adflow_options(
        ADflowOptionRequest(
            cgns_path="mesh.cgns",
            output_dir="solver-output",
            options_version=2,
            cycles=10000,
            solver_preset="prod",
        )
    )

    assert options["useANKSolver"] is True
    assert options["useNKSolver"] is True
    assert options["NKSwitchTol"] == 1.0e-4


class _ResidualSolver:
    def __init__(self) -> None:
        self.calls = 0
        self.rootChangedOptions = {}
        self.adflow = SimpleNamespace(
            iteration=SimpleNamespace(itertot=0, approxtotalits=0.0),
            nksolver=SimpleNamespace(nk_iter=0),
        )

    def __call__(self, aero_problem, **kwargs) -> None:
        del aero_problem, kwargs
        self.calls += 1
        self.adflow.iteration.itertot = self.calls
        self.adflow.iteration.approxtotalits = float(self.calls * 3)

    def getResNorms(self):
        start = 100.0 / (10 ** (self.calls - 1))
        return (1000.0, start, start / 10.0)


def test_ank_nk_work_ceiling_is_one_solver_call() -> None:
    solver = _ResidualSolver()
    _, calls = ADflowBackend()._execute_budget(
        solver=solver,
        aero_problem=object(),
        delta_cycles=10000,
        comm=None,
        single_call=True,
    )

    assert solver.calls == 1
    assert calls[0]["cycle"] == 10000


class _RootComm:
    rank = 0

    @staticmethod
    def bcast(value, root=0):
        del root
        return value


class _SolveCLSolver:
    def __init__(self) -> None:
        self.comm = _RootComm()
        self.adflow = SimpleNamespace(
            iteration=SimpleNamespace(itertot=0, approxtotalits=0.0),
            nksolver=SimpleNamespace(nk_iter=0),
            monitor=SimpleNamespace(solverdataarray=np.zeros((4, 5))),
        )
        self.original_getter = lambda **_kwargs: {"original": True}
        self.getConvergenceHistory = self.original_getter
        self.kwargs = {}
        self.options = {}

    def setOption(self, name, value):
        self.options[name] = value

    def _trimHistoryData(self, values):
        return values[: self.adflow.iteration.itertot + 1]

    def getResNorms(self):
        return (1000.0, 10.0, 1.0)

    def solveCL(self, aero_problem, target_cl, **kwargs):
        self.kwargs = {"target_cl": target_cl, **kwargs}
        history = []
        for itertot, work in ((2, 5.0), (3, 8.0)):
            self.adflow.iteration.itertot = itertot
            self.adflow.iteration.approxtotalits = work
            history.append(self.getConvergenceHistory())
        aero_problem.alpha = 2.25
        return {"converged": True, "alpha": 2.25, "cl": target_cl, "history": history}


def test_fixed_lift_ank_nk_uses_native_solvecl_limits() -> None:
    solver = _SolveCLSolver()
    aero_problem = SimpleNamespace(alpha=2.0)
    fixed_lift = SimpleNamespace(
        target_cl=0.8,
        cl_tolerance=0.01,
        max_aoa_solves=3,
        cl_alpha_guess=0.1,
        delta_alpha=0.5,
        total_time_limit_s=30.0,
    )

    _, calls, summary = ADflowBackend()._execute_fixed_lift(
        solver=solver,
        aero_problem=aero_problem,
        fixed_lift=fixed_lift,
    )

    assert solver.getConvergenceHistory is solver.original_getter
    assert solver.kwargs["tol"] == 0.01
    assert solver.kwargs["maxIter"] == 3
    assert solver.kwargs["autoReset"] is False
    assert solver.options["timeLimit"] == -1.0
    assert [call["approx_total_its"] for call in calls] == [5.0, 8.0]
    assert summary["api"] == "ADFLOW.solveCL"
    assert summary["flow_solve_calls"] == 2


def test_optimization_nk_defaults_match_terminal_resume_contract() -> None:
    config = NKConfig()

    assert config.resume_mode == "ank_nk"
    assert config.max_work_per_flow_solve == 1000
    assert config.max_aoa_solves == 5
    assert config.total_time_limit_s == 30.0
    assert config.nk_switch_tolerance == 1.0e-4
    assert config.repeated_nk_cycles == (6, 8, 10)


def test_deployment_defaults_to_ank_nk_and_can_select_repeated_nk() -> None:
    config = {
        "resume_mode": "ank_nk",
        "max_work": 2000,
        "time_limit_s": 10.0,
        "nk_switch_tolerance": 1.0e-4,
        "repeated_nk_cycles": [1, 2, 3],
        "residual_ratio": 1.0e-8,
    }

    default_work = _resume_work_plan(config)
    repeated_work = _resume_work_plan(config, "repeated_nk")

    assert default_work.resume_mode == ResumeMode.ANK_NK
    assert default_work.fixed_cycles == 2000
    assert default_work.time_limit_s == 10.0
    assert default_work.nk_switch_tolerance == 1.0e-4
    assert repeated_work.resume_mode == ResumeMode.REPEATED_NK
    assert repeated_work.adaptive_schedule.cumulative_cycles == (1, 2, 3)


def test_optimization_defaults_to_native_solvecl_with_shared_limits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = OptimizationConfig(
        mode="surrogate_nk",
        output_dir=str(tmp_path / "run"),
        baseline_dir=str(BASELINE),
        task=TaskConfig(mach=(0.71,), target_cl=0.8),
        nk=NKConfig(
            selection="all",
            pool_count=1,
            ranks_per_case=1,
            cl_tolerance=2.5e-3,
            max_work_per_flow_solve=4321,
            max_aoa_solves=4,
            total_time_limit_s=27.0,
            nk_switch_tolerance=2.0e-5,
        ),
        optimizer=OptimizerConfig(n_proc=1),
    )
    prepared = SimpleNamespace(
        geometry=np.zeros(27),
        coords=np.zeros((4, 2, 3)),
        coords_vertex=np.zeros((2, 3, 4)),
    )
    mesh_path = tmp_path / "mesh.cgns"
    mesh_path.write_text("mesh", encoding="utf-8")
    monkeypatch.setattr(
        "optimization.evaluators._prepare_authority_mesh",
        lambda *_args: (prepared, mesh_path),
    )

    class FakeClient:
        def ping(self):
            return {"protocol_version": 1, "model_key": "fsb_dit"}

        def request(self, payload):
            assert payload["target_cl"] == 0.8
            return {"aoa": [2.0], "fields": np.zeros((1, 5, 2, 3))}

    monkeypatch.setattr("optimization.evaluators._client", lambda _config: FakeClient())

    class FakePipeline:
        def export_cases(self, cases, plan, *, output_dir):
            self.cases = tuple(cases)
            self.plan = plan
            path = Path(output_dir) / "manifest.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
            return SimpleNamespace(manifest_path=str(path))

        def project_manifest_warm_pools(self, _manifest_path, **_kwargs):
            path = tmp_path / "native_solvecl_result.json"
            path.write_text(
                json.dumps(
                    {
                        "stages": [
                            {
                                "metrics": {
                                    "force_coefficients": {
                                        "cl": 0.8,
                                        "cd": 0.01,
                                        "cm": -0.05,
                                    },
                                    "fixed_lift": {
                                        "final_alpha": 2.1,
                                        "flow_solve_calls": 2,
                                        "target_cl_converged": True,
                                    },
                                    "solver_work": {
                                        "termination": "converged",
                                        "verified_l2_ratio": 1.0e-8,
                                    },
                                },
                                "output_paths": {},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            return SimpleNamespace(result_paths=(str(path),))

    pipeline = FakePipeline()
    monkeypatch.setattr("optimization.evaluators.create_pipeline", lambda: pipeline)

    result = SurrogateNKEvaluator(config).evaluate(object(), tmp_path)

    fixed_lift = pipeline.cases[0].solver_context.fixed_lift
    assert fixed_lift.cl_tolerance == config.nk.cl_tolerance
    assert fixed_lift.max_aoa_solves == 4
    assert fixed_lift.total_time_limit_s == 27.0
    assert pipeline.plan.final_stage.work.resume_mode == ResumeMode.ANK_NK
    assert pipeline.plan.final_stage.work.max_work == 4321
    assert pipeline.plan.final_stage.work.time_limit_s == 27.0
    assert pipeline.plan.final_stage.work.nk_switch_tolerance == 2.0e-5
    assert result.converged
    assert result.points[0].aoa == pytest.approx(2.1)
