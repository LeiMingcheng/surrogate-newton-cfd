"""The only AeroOpt adapter used by surrogate, CFD, and surrogate+NK runs."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import replace
import json
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any

import numpy as np

from NK_resume import ResidentWarmPoolController
from optimization.config import (
    OptimizationConfig,
    load_optimization_config,
    write_optimization_config,
)
from optimization.evaluators import SurrogateEvaluator, SurrogateNKEvaluator
from optimization.initial_population import generate_initial_population
from optimization.objective import evaluate_workdir, output_vector


from AeroOpt.basic import Database, Problem
from AeroOpt.optimize.basic import OptBasic, OptFunc_Multiprocess
from AeroOpt.optimize.stochastic import DiffEvo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = PROJECT_ROOT / "optimization" / "templates"


def _de(name: str, config: OptimizationConfig) -> Any:
    de = DiffEvo(name=name)
    de.expert_setting(
        warning=True,
        trace_evolution=True,
        nPopSize=config.optimizer.n_pop,
        nGen=config.optimizer.n_gen,
        iMMethod=0,
        iCMethod=5,
        MRate_u=0.8,
        MRate_l=0.2,
        CRate_u=0.8,
        CRate_l=0.2,
        gm_std=1.0e-2,
        pow_poly=20.0,
        pow_sbx=20.0,
        ng_rate=3,
        d_perturb=1.0e-2,
        ratio_elite_modify_neighbor=0.1,
        ratio_parent_modify_neighbor=0.2,
        ratio_parent_modify=0.2,
        ratio_perturbation=0.1,
        r_modify=0.05,
        ratio_pop=0.3,
        restrict_dx=[0.02, 1.0],
        n_ref=5,
        d_ref=0.1,
        r_reserve=0.3,
        potential_cri=0.99,
        r_mature_cri=0.99,
    )
    return de


class ResumeOptBasic(OptBasic):
    """OptBasic resume stage with globally continuous generation logs."""

    def __init__(self, *args: Any, generation_offset: int, **kwargs: Any) -> None:
        self.generation_offset = int(generation_offset)
        super().__init__(*args, **kwargs)

    @property
    def global_generation(self) -> int:
        return self.generation_offset + int(self.iteration)

    def evolution(self) -> None:
        self.DE.iGen = self.global_generation
        super().evolution()

    def resume(self, previous_db: Any, evolution: Any, save_elite: bool = True) -> None:
        generations = {
            int(individual.ID): int(individual.gen)
            for individual in previous_db.indis
        }
        super().resume(previous_db, evolution, save_elite=save_elite)
        for database_name in ("total", "valid", "elite", "parent"):
            database = getattr(self, database_name)
            for individual in database.indis:
                individual.gen = generations[int(individual.ID)]

    def termination(self) -> bool:
        terminated = self.global_generation >= int(self.DE.nGen)
        if self.log_type >= 0:
            if terminated:
                self.log()
            elif self.log_type > 0 and self.iteration % self.log_type == 0:
                self.log()
        return terminated

    def evaluation(self, pop: Any, *args: Any, **kwargs: Any) -> None:
        super().evaluation(pop, *args, **kwargs)
        if self.iteration <= 0:
            return
        global_generation = self.global_generation
        for individual in pop.indis:
            if individual.source2int == 1:
                continue
            individual.gen = global_generation
            if individual.ID in self.total.idList:
                self.total.indis[self.total.ID2index(individual.ID)].gen = global_generation

    def log(self) -> None:
        self.total.output(fname=self.fname_total)
        if self.iteration <= 0:
            self.elite.sort_order(sort_type=4)
            self.elite.output(fname=self.fname_elite)
            return
        generation = self.global_generation
        self.parent.log_pop(fname=self.pop_log, append=True, gen=generation)
        self.parent.output(fname=self.pop_process, append=True, gen=generation)
        self.elite.sort_order(sort_type=4)
        self.elite.output(fname=self.elite_process, append=True, gen=generation)
        self.elite.output(fname=self.fname_elite)


class TopKNKHook:
    """Replace screened top-k results with the same framework's NK evaluator."""

    def __init__(self, run_dir: Path, config: OptimizationConfig) -> None:
        self.run_dir = run_dir
        self.config = config
        self.executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="optimization-nk")
            if config.nk.coupling == "async"
            else None
        )
        self.pending: dict[int, Future[dict[str, float]]] = {}
        self.applied = 0

    def _refine(self, sample_id: int) -> dict[str, float]:
        return evaluate_workdir(
            self.run_dir / "Calculation" / str(sample_id),
            self.config,
            refinement=True,
        )

    @staticmethod
    def _set_y(individual: Any, outputs: dict[str, float]) -> None:
        individual.y = output_vector(outputs, names=individual.problem.name_out)
        individual.calc_ctr()

    def _apply_id(self, optimizer: Any, sample_id: int, outputs: dict[str, float]) -> None:
        for name in ("total", "parent", "elite", "offspring"):
            database = getattr(optimizer, name, None)
            if database is None or sample_id not in getattr(database, "idList", []):
                continue
            self._set_y(database.indis[database.ID2index(sample_id)], outputs)
        self.applied += 1

    def pre_evaluate(self, _population: Any, optimizer: Any) -> None:
        if self.executor is None:
            return
        completed = [sample_id for sample_id, future in self.pending.items() if future.done()]
        for sample_id in completed:
            self._apply_id(optimizer, sample_id, self.pending.pop(sample_id).result())

    def post_evaluate(self, population: Any, optimizer: Any = None) -> None:
        del optimizer

        def rank_key(individual: Any) -> tuple[bool, float, float]:
            _, violation = individual.problem.calc_ctr(individual.x, individual.y)
            return violation > 0.0, float(violation), float(individual.y[0])

        ranked = sorted(population.indis, key=rank_key)
        feasible = [individual for individual in ranked if rank_key(individual)[1] <= 0.0]
        selected = feasible[: min(self.config.nk.top_k, len(feasible))]
        if self.executor is None:
            for individual in selected:
                self._set_y(individual, self._refine(int(individual.ID)))
                self.applied += 1
            return
        for individual in selected:
            sample_id = int(individual.ID)
            self.pending[sample_id] = self.executor.submit(self._refine, sample_id)

    def finalize(self, optimizer: Any) -> dict[str, int]:
        if self.executor is not None:
            for sample_id, future in list(self.pending.items()):
                self._apply_id(optimizer, sample_id, future.result())
            self.pending.clear()
            self.executor.shutdown(wait=True)
        return {"applied": self.applied}


def _persist_refined_state(optimizer: Any) -> None:
    """Persist NK-corrected databases without appending a duplicate generation."""

    optimizer.select_valid_elite()
    optimizer.total.output(fname=optimizer.fname_total)
    optimizer.elite.output(fname=optimizer.fname_elite)


class DirectSurrogateOptFunc:
    """Evaluate surrogate candidates concurrently without per-candidate Python startup."""

    def __init__(
        self,
        problem: Any,
        config: OptimizationConfig,
    ) -> None:
        self.prob = problem
        self.config = config
        self.parallel = True
        worker_count = int(config.optimizer.n_proc)
        self.mesh_executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("spawn"),
        )
        self.evaluation_executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="aeroopt-surrogate",
        )
        self.evaluator = SurrogateEvaluator(
            config,
            mesh_executor=self.mesh_executor,
        )

    def _evaluate_one(
        self,
        x: np.ndarray,
        *,
        sample_name: str,
    ) -> np.ndarray:
        workdir = Path("Calculation") / str(sample_name)
        workdir.mkdir(parents=True, exist_ok=True)
        self.prob.write_data(str(workdir / "input.txt"), x)
        outputs = evaluate_workdir(
            workdir,
            self.config,
            evaluator=self.evaluator,
        )
        return output_vector(outputs, names=self.prob.name_out)

    def evaluate(
        self,
        xs: np.ndarray,
        name_list: list[str] | None = None,
        return_succeed: bool = False,
        **_kwargs: Any,
    ) -> Any:
        values = np.asarray(xs, dtype=np.float64)
        if values.size == 0:
            ys = np.empty((0, self.prob.n_out), dtype=np.float64)
            return (ys, []) if return_succeed else ys
        if values.ndim == 1:
            values = values[None, :]
        if name_list is None or len(name_list) != len(values):
            raise ValueError("Direct surrogate evaluation requires one sample name per row")
        ys = np.zeros((len(values), self.prob.n_out), dtype=np.float64)
        futures = {
            self.evaluation_executor.submit(
                self._evaluate_one,
                values[index],
                sample_name=str(name),
            ): index
            for index, name in enumerate(name_list)
        }
        for future, index in ((future, futures[future]) for future in futures):
            ys[index] = future.result()
        succeeds = [True] * len(values)
        if return_succeed:
            return ys, succeeds
        return ys

    def close(self) -> None:
        self.evaluation_executor.shutdown(wait=True)
        self.mesh_executor.shutdown(wait=True)


class DirectSurrogateNKOptFunc:
    """Recover candidates sequentially through resident parallel NK pools."""

    def __init__(
        self,
        problem: Any,
        config: OptimizationConfig,
    ) -> None:
        self.prob = problem
        self.config = config
        self.parallel = True
        self.resident_pool = ResidentWarmPoolController(
            ranks_per_case=config.nk.ranks_per_case,
            pool_count=config.nk.pool_count,
            mpi_launcher=config.nk.mpi_launcher,
            mpi_omp_threads=config.nk.mpi_omp_threads,
            output_dir=(
                Path(config.output_dir)
                / "nk_resident_runtime"
                / f"controller_{os.getpid()}"
            ),
            ready_timeout_sec=config.nk.timeout_s,
            submit_timeout_sec=config.nk.timeout_s,
            request_wait_timeout_sec=config.nk.timeout_s,
        )
        self.evaluator = SurrogateNKEvaluator(
            config,
            resident_pool=self.resident_pool,
        )

    def _evaluate_one(
        self,
        x: np.ndarray,
        *,
        sample_name: str,
    ) -> np.ndarray:
        workdir = Path("Calculation") / str(sample_name)
        workdir.mkdir(parents=True, exist_ok=True)
        self.prob.write_data(str(workdir / "input.txt"), x)
        outputs = evaluate_workdir(
            workdir,
            self.config,
            evaluator=self.evaluator,
        )
        return output_vector(outputs, names=self.prob.name_out)

    def evaluate(
        self,
        xs: np.ndarray,
        name_list: list[str] | None = None,
        return_succeed: bool = False,
        **_kwargs: Any,
    ) -> Any:
        values = np.asarray(xs, dtype=np.float64)
        if values.size == 0:
            ys = np.empty((0, self.prob.n_out), dtype=np.float64)
            return (ys, []) if return_succeed else ys
        if values.ndim == 1:
            values = values[None, :]
        if name_list is None or len(name_list) != len(values):
            raise ValueError(
                "Direct surrogate-NK evaluation requires one sample name per row"
            )
        ys = np.zeros((len(values), self.prob.n_out), dtype=np.float64)
        for index, name in enumerate(name_list):
            ys[index] = self._evaluate_one(
                values[index],
                sample_name=str(name),
            )
        succeeds = [True] * len(values)
        if return_succeed:
            return ys, succeeds
        return ys

    def close(self) -> None:
        self.resident_pool.close()


def _optimization_function(
    problem: Any,
    config: OptimizationConfig,
) -> Any:
    if config.mode == "surrogate":
        return DirectSurrogateOptFunc(problem, config)
    if config.mode == "surrogate_nk" and config.nk.selection == "all":
        return DirectSurrogateNKOptFunc(problem, config)
    return OptFunc_Multiprocess(problem, n_proc=config.optimizer.n_proc)


def _prepare_run_directory(config: OptimizationConfig) -> tuple[Path, Path]:
    run_dir = Path(config.output_dir).resolve()
    occupied = [
        path.name
        for path in (run_dir / "Calculation", run_dir / "total-db.dat")
        if path.exists()
    ]
    if occupied:
        raise FileExistsError(
            f"Optimization output directory already contains run state {occupied}: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    settings_name = (
        "Settings_enhanced.txt" if config.use_enhanced_cst else "Settings_classic.txt"
    )
    settings = (TEMPLATE_ROOT / settings_name).read_text(encoding="utf-8")
    lower_aoa, upper_aoa = config.task.aoa_bounds
    settings = settings.replace(
        "\tAoAmax   -  5.0",
        f"\tAoAmax   -  {float(upper_aoa)}",
    ).replace(
        "\t1.0  -   AoAmin",
        f"\t{float(lower_aoa)}  -   AoAmin",
    )
    (run_dir / "Settings.txt").write_text(settings, encoding="utf-8")
    runfiles = run_dir / "Runfiles"
    runfiles.mkdir(exist_ok=True)
    shutil.copy2(TEMPLATE_ROOT / "Runfiles" / "runfoil.py", runfiles / "runfoil.py")
    shutil.copy2(TEMPLATE_ROOT / "Runfiles" / "run.sh", runfiles / "run.sh")
    config_path = write_optimization_config(config, run_dir / "optimization_config.json")
    return run_dir, config_path


def run_optimization(config: OptimizationConfig) -> Path:
    """Run one complete AeroOpt experiment with a selected evaluator."""

    run_dir, config_path = _prepare_run_directory(config)
    os.environ["SURROGATE_NEWTON_OPT_CONFIG"] = str(config_path)
    random.seed(config.optimizer.seed)
    np.random.seed(config.optimizer.seed)
    if config.optimizer.initial_population:
        initial = np.load(config.optimizer.initial_population)
    else:
        initial = generate_initial_population(
            config.optimizer.initial_population_size,
            baseline_dir=config.baseline_dir,
            use_enhanced=config.use_enhanced_cst,
            tail=config.geometry.tail,
        )
    np.save(run_dir / "initial_population.npy", initial)
    manifest = {
        "schema_version": "optimization_run_v1",
        "mode": config.mode,
        "config": str(config_path),
        "initial_population_shape": list(initial.shape),
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    previous_cwd = Path.cwd()
    function = None
    os.chdir(run_dir)
    try:
        problem = Problem("Foils-opt1", fname="Settings.txt")
        problem.var_cridis = 1.0e-10
        function = _optimization_function(problem, config)
        hook = None
        if config.mode == "surrogate_nk" and config.nk.selection == "topk":
            hook = TopKNKHook(run_dir, config)
        optimizer = OptBasic(
            _de(problem.name, config),
            function,
            info_opt=True,
            show_results=True,
            log_type=1,
            i_tune=0,
            pre_evaluate=None if hook is None else hook.pre_evaluate,
            post_evaluate=None if hook is None else hook.post_evaluate,
            phi_cri=0.2,
        )
        optimizer.main(previous_db=None, xs_usr=initial, strategy="DoE")
        if hook is not None:
            final = hook.finalize(optimizer)
            if final["applied"]:
                _persist_refined_state(optimizer)
            (run_dir / "nk_selection_summary.json").write_text(
                json.dumps(final, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    finally:
        if function is not None and hasattr(function, "close"):
            function.close()
        os.chdir(previous_cwd)
    return run_dir


def _max_calculation_id(run_dir: Path) -> int:
    calculation = run_dir / "Calculation"
    return max(
        (
            int(path.name)
            for path in calculation.iterdir()
            if path.is_dir() and path.name.isdigit()
        ),
        default=0,
    )


def _load_resume_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": "optimization_resume_v1", "stages": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "optimization_resume_v1":
        raise ValueError(f"Unsupported optimization resume manifest: {path}")
    return payload


def _write_resume_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _restore_generation_tags(
    database: Any,
    *,
    initial_population_size: int,
    population_size: int,
) -> int:
    restored = 0
    for individual in database.indis:
        sample_id = int(individual.ID)
        generation = (
            0
            if sample_id <= int(initial_population_size)
            else (sample_id - int(initial_population_size) - 1)
            // int(population_size)
            + 1
        )
        if int(individual.gen) != generation:
            individual.gen = generation
            restored += 1
    return restored


def _resume_post_evaluate(hook: TopKNKHook):
    def post_evaluate(population: Any, optimizer: Any = None) -> None:
        if optimizer is not None and int(optimizer.iteration) == 0:
            return
        hook.post_evaluate(population, optimizer)

    return post_evaluate


def resume_optimization(
    run_dir: str | Path,
    *,
    target_n_gen: int | None = None,
) -> Path:
    """Resume one unified run from its persisted total database."""

    run_path = Path(run_dir).resolve()
    config_path = run_path / "optimization_config.json"
    total_path = run_path / "total-db.dat"
    if not config_path.is_file() or not total_path.is_file():
        raise FileNotFoundError(
            f"Unified resume requires optimization_config.json and total-db.dat in {run_path}"
        )
    config = load_optimization_config(config_path)
    target = config.optimizer.n_gen if target_n_gen is None else int(target_n_gen)
    if target < 0:
        raise ValueError("target_n_gen must be non-negative")

    previous_cwd = Path.cwd()
    function = None
    os.chdir(run_path)
    manifest_path = run_path / "optimization_resume_manifest.json"
    manifest = _load_resume_manifest(manifest_path)
    stage: dict[str, Any] | None = None
    try:
        problem = Problem("Foils-opt1", fname="Settings.txt")
        problem.var_cridis = 1.0e-10
        previous = Database(db_type="all")
        previous.read(fname=str(total_path), problem_db=problem, check_dup=False, info=True)
        restored_generation_tags = _restore_generation_tags(
            previous,
            initial_population_size=config.optimizer.initial_population_size,
            population_size=config.optimizer.n_pop,
        )
        if restored_generation_tags:
            previous.output(fname=str(total_path))
            manifest.setdefault("recovery_events", []).append(
                {
                    "event": "restored_generation_tags",
                    "restored_samples": restored_generation_tags,
                    "recovered_unix_s": time.time(),
                }
            )
            _write_resume_manifest(manifest_path, manifest)
            print(
                "[INFO] Restored persisted generation tags: "
                f"samples={restored_generation_tags}"
            )
        completed_samples = int(previous.size) - int(config.optimizer.initial_population_size)
        if completed_samples < 0 or completed_samples % int(config.optimizer.n_pop) != 0:
            raise RuntimeError(
                "Persisted total-db.dat does not align with the configured initial population "
                f"and population size: total={previous.size}, "
                f"initial={config.optimizer.initial_population_size}, n_pop={config.optimizer.n_pop}"
            )
        completed = completed_samples // int(config.optimizer.n_pop)
        remaining = target - completed
        if remaining <= 0:
            return run_path

        max_total_id = int(previous.IDmax)
        max_calculation_id = _max_calculation_id(run_path)
        discarded_partial_ids = list(range(max_total_id + 1, max_calculation_id + 1))
        stage_index = len(manifest["stages"]) + 1
        resume_log = f"optimization_resume_{stage_index:02d}.log"
        stage = {
            "stage": stage_index,
            "status": "running",
            "started_unix_s": time.time(),
            "completed_generations_before": completed,
            "target_generations": target,
            "additional_generations": remaining,
            "max_total_id_before": max_total_id,
            "max_calculation_id_before": max_calculation_id,
            "discarded_partial_ids": discarded_partial_ids,
            "log": resume_log,
        }
        manifest["run_dir"] = str(run_path)
        manifest["stages"].append(stage)
        _write_resume_manifest(manifest_path, manifest)

        os.environ["SURROGATE_NEWTON_OPT_CONFIG"] = str(config_path)
        random.seed(config.optimizer.seed + completed)
        np.random.seed(config.optimizer.seed + completed)
        stage_config = replace(
            config,
            optimizer=replace(config.optimizer, n_gen=target),
        )
        function = _optimization_function(problem, config)
        hook = None
        if config.mode == "surrogate_nk" and config.nk.selection == "topk":
            hook = TopKNKHook(run_path, config)
        optimizer = ResumeOptBasic(
            _de(problem.name, stage_config),
            function,
            generation_offset=completed,
            info_opt=True,
            show_results=True,
            log_type=1,
            i_tune=0,
            pre_evaluate=None if hook is None else hook.pre_evaluate,
            post_evaluate=None if hook is None else _resume_post_evaluate(hook),
            phi_cri=0.2,
            fname_log=resume_log,
        )
        optimizer.main(previous_db=previous, xs_usr=None, strategy="DoE")
        if hook is not None:
            final = hook.finalize(optimizer)
            if final["applied"]:
                _persist_refined_state(optimizer)
            (run_path / f"nk_selection_resume_{stage_index:02d}.json").write_text(
                json.dumps(final, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        completed_samples_after = (
            int(optimizer.total.size) - int(config.optimizer.initial_population_size)
        )
        if (
            completed_samples_after < 0
            or completed_samples_after % int(config.optimizer.n_pop) != 0
        ):
            raise RuntimeError(
                "Resumed total database is not generation-aligned: "
                f"total={optimizer.total.size}, "
                f"initial={config.optimizer.initial_population_size}, "
                f"n_pop={config.optimizer.n_pop}"
            )
        completed_after = completed_samples_after // int(config.optimizer.n_pop)
        if completed_after != target:
            raise RuntimeError(
                "Optimization resume did not reach its generation target: "
                f"actual={completed_after}, target={target}"
            )
        stage["status"] = "completed"
        stage["finished_unix_s"] = time.time()
        stage["completed_generations_after"] = completed_after
        stage["max_total_id_after"] = int(optimizer.total.IDmax)
        _write_resume_manifest(manifest_path, manifest)
    except KeyboardInterrupt:
        if stage is not None:
            stage["status"] = "interrupted"
            stage["finished_unix_s"] = time.time()
            _write_resume_manifest(manifest_path, manifest)
        raise
    except BaseException as exc:
        if stage is not None:
            stage["status"] = "failed"
            stage["finished_unix_s"] = time.time()
            stage["error"] = f"{type(exc).__name__}: {exc}"
            _write_resume_manifest(manifest_path, manifest)
        raise
    finally:
        if function is not None and hasattr(function, "close"):
            function.close()
        os.chdir(previous_cwd)
    return run_path


__all__ = [
    "DirectSurrogateOptFunc",
    "DirectSurrogateNKOptFunc",
    "ResumeOptBasic",
    "TopKNKHook",
    "resume_optimization",
    "run_optimization",
]
