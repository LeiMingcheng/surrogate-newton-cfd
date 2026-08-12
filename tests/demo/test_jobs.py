from __future__ import annotations

import sqlite3
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from typing import Any

from demo.jobs import JobNotFoundError, JobScheduler, JobStateError, QueueCapacityError

GEOMETRY27 = [0.0] * 27


class FakeEngine:
    def __init__(self, *, delay: float = 0.01, gate: threading.Event | None = None) -> None:
        self.delay = delay
        self.gate = gate
        self.started = threading.Event()
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls: list[str] = []

    def _execute(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append(action)
        self.started.set()
        try:
            if self.gate is not None:
                if not self.gate.wait(timeout=5):
                    raise TimeoutError("test gate timed out")
            else:
                time.sleep(self.delay)
            if payload.get("name") == "fail":
                raise RuntimeError("synthetic engine failure")
            return {"action": action, "payload": payload}
        finally:
            with self.lock:
                self.active -= 1

    def prepare_mesh(self, **payload: Any) -> dict[str, Any]:
        return self._execute("mesh", payload)

    def predict(self, **payload: Any) -> dict[str, Any]:
        return self._execute("predict", payload)

    def recover(self, case_id: str, **payload: Any) -> dict[str, Any]:
        return self._execute("recover", {"case_id": case_id, **payload})

    def reference(self, case_id: str, **payload: Any) -> dict[str, Any]:
        return self._execute("reference", {"case_id": case_id, **payload})


class JobSchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="demo-jobs-")
        self.root = Path(self.temp.name) / "scheduler"
        self.schedulers: list[JobScheduler] = []

    def tearDown(self) -> None:
        for scheduler in self.schedulers:
            scheduler.close(timeout=5)
        self.temp.cleanup()

    def scheduler(self, engine: FakeEngine, **options: Any) -> JobScheduler:
        scheduler = JobScheduler(engine, runtime_root=self.root, **options)
        self.schedulers.append(scheduler)
        return scheduler

    def wait_for_state(
        self,
        scheduler: JobScheduler,
        job_id: str,
        expected: set[str] | None = None,
        timeout: float = 8.0,
    ) -> dict[str, Any]:
        expected = expected or {"succeeded", "failed", "cancelled", "expired"}
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            summary = scheduler.get(job_id)
            if summary["state"] in expected:
                return summary
            time.sleep(0.01)
        self.fail(f"Job {job_id} did not reach {sorted(expected)}")

    def test_sixteen_concurrent_clients_never_run_two_engine_calls(self) -> None:
        engine = FakeEngine(delay=0.02)
        scheduler = self.scheduler(
            engine,
            max_pending_jobs=32,
            max_pending_jobs_per_client=2,
            cleanup_interval_sec=1,
        )

        def submit(index: int) -> dict[str, Any]:
            return scheduler.submit(
                "mesh",
                {"geometry27": GEOMETRY27, "name": f"case-{index}"},
                client_id=f"client-{index}",
            )

        with ThreadPoolExecutor(max_workers=16) as executor:
            jobs = list(executor.map(submit, range(16)))
        summaries = [self.wait_for_state(scheduler, job["job_id"]) for job in jobs]

        self.assertTrue(all(item["state"] == "succeeded" for item in summaries))
        self.assertEqual(engine.max_active, 1)
        self.assertEqual(len(engine.calls), 16)
        self.assertEqual(scheduler.stats()["concurrency_limit"], 1)

    def test_two_workers_run_distinct_resources_concurrently(self) -> None:
        engine = FakeEngine(delay=0.08)
        scheduler = self.scheduler(
            [engine, engine],
            max_pending_jobs=8,
            max_pending_jobs_per_client=2,
            cleanup_interval_sec=1,
        )
        jobs = [
            scheduler.submit(
                "mesh",
                {"geometry27": [float(index)] + [0.0] * 26, "name": f"case-{index}"},
                client_id=f"client-{index}",
            )
            for index in range(2)
        ]
        summaries = [self.wait_for_state(scheduler, job["job_id"]) for job in jobs]
        self.assertTrue(all(item["state"] == "succeeded" for item in summaries))
        self.assertEqual(engine.max_active, 2)
        self.assertEqual(scheduler.stats()["concurrency_limit"], 2)

    def test_same_case_never_runs_nk_and_cold_start_concurrently(self) -> None:
        engine = FakeEngine(delay=0.04)
        scheduler = self.scheduler([engine, engine], cleanup_interval_sec=1)
        jobs = [
            scheduler.submit(
                "recover",
                {"case_id": "case_shared", "cycles": 2},
                client_id="nk-client",
            ),
            scheduler.submit(
                "reference",
                {"case_id": "case_shared", "max_cycles": 25},
                client_id="cold-client",
            ),
        ]
        summaries = [self.wait_for_state(scheduler, job["job_id"]) for job in jobs]
        self.assertTrue(all(item["state"] == "succeeded" for item in summaries))
        self.assertEqual(engine.max_active, 1)

    def test_nk_priority_uses_three_to_one_anti_starvation_policy(self) -> None:
        engine = FakeEngine(delay=0.005)
        scheduler = self.scheduler(
            engine,
            autostart=False,
            nk_burst_limit=3,
            cleanup_interval_sec=1,
        )
        jobs = [
            scheduler.submit(
                "reference",
                {"case_id": "case_cold", "max_cycles": 25},
                client_id="cold-client",
            )
        ]
        jobs.extend(
            scheduler.submit(
                "recover",
                {"case_id": f"case_nk_{index}", "cycles": 2},
                client_id=f"nk-client-{index}",
            )
            for index in range(4)
        )
        scheduler.start()
        for job in jobs:
            self.assertEqual(self.wait_for_state(scheduler, job["job_id"])["state"], "succeeded")
        self.assertEqual(
            engine.calls,
            ["recover", "recover", "recover", "reference", "recover"],
        )

    def test_queue_capacity_is_atomic_under_a_running_job(self) -> None:
        gate = threading.Event()
        engine = FakeEngine(gate=gate)
        scheduler = self.scheduler(
            engine,
            max_pending_jobs=2,
            max_pending_jobs_per_client=3,
            cleanup_interval_sec=1,
        )
        first = scheduler.submit(
            "mesh", {"geometry27": GEOMETRY27, "name": "first"}, client_id="one"
        )
        self.assertTrue(engine.started.wait(timeout=2))
        second = scheduler.submit(
            "mesh", {"geometry27": GEOMETRY27, "name": "second"}, client_id="two"
        )
        with self.assertRaisesRegex(QueueCapacityError, "queue is full"):
            scheduler.submit(
                "mesh", {"geometry27": GEOMETRY27, "name": "third"}, client_id="three"
            )
        self.assertEqual(scheduler.get(second["job_id"])["queue_position"], 1)
        gate.set()
        self.assertEqual(self.wait_for_state(scheduler, first["job_id"])["state"], "succeeded")
        self.assertEqual(self.wait_for_state(scheduler, second["job_id"])["state"], "succeeded")

    def test_queued_and_running_cancellation_are_safe(self) -> None:
        gate = threading.Event()
        engine = FakeEngine(gate=gate)
        scheduler = self.scheduler(engine, cleanup_interval_sec=1)
        running = scheduler.submit(
            "mesh", {"geometry27": GEOMETRY27, "name": "running"}, client_id="one"
        )
        self.assertTrue(engine.started.wait(timeout=2))
        queued = scheduler.submit(
            "mesh", {"geometry27": GEOMETRY27, "name": "queued"}, client_id="two"
        )
        cancelled_queued = scheduler.cancel(queued["job_id"])
        self.assertEqual(cancelled_queued["state"], "cancelled")
        cancel_requested = scheduler.cancel(running["job_id"])
        self.assertEqual(cancel_requested["state"], "running")
        self.assertTrue(cancel_requested["cancel_requested"])
        gate.set()
        self.assertEqual(
            self.wait_for_state(scheduler, running["job_id"])["state"], "cancelled"
        )
        with self.assertRaises(JobStateError):
            scheduler.result(running["job_id"])

    def test_failure_does_not_stall_the_next_job(self) -> None:
        engine = FakeEngine()
        scheduler = self.scheduler(engine, cleanup_interval_sec=1)
        failed = scheduler.submit(
            "mesh", {"geometry27": GEOMETRY27, "name": "fail"}, client_id="one"
        )
        succeeding = scheduler.submit(
            "mesh", {"geometry27": GEOMETRY27, "name": "next"}, client_id="two"
        )
        self.assertEqual(self.wait_for_state(scheduler, failed["job_id"])["state"], "failed")
        self.assertEqual(
            self.wait_for_state(scheduler, succeeding["job_id"])["state"], "succeeded"
        )
        result = scheduler.result(succeeding["job_id"])
        self.assertEqual(result["payload"]["name"], "next")

    def test_action_timeout_marks_job_failed_and_continues(self) -> None:
        engine = FakeEngine(delay=0.04)
        scheduler = self.scheduler(
            engine,
            action_timeouts={
                "mesh": 0.01,
                "predict": 1,
                "recover": 1,
                "reference": 1,
            },
            cleanup_interval_sec=1,
        )
        timed_out = scheduler.submit(
            "mesh", {"geometry27": GEOMETRY27, "name": "slow"}, client_id="one"
        )
        next_job = scheduler.submit(
            "predict",
            {"geometry27": GEOMETRY27, "name": "next", "mach": 0.74, "aoa": 1.0},
            client_id="two",
        )
        timeout_summary = self.wait_for_state(scheduler, timed_out["job_id"])
        self.assertEqual(timeout_summary["state"], "failed")
        self.assertIn("time limit", timeout_summary["error"])
        self.assertEqual(self.wait_for_state(scheduler, next_job["job_id"])["state"], "succeeded")

    def test_restart_marks_running_job_failed_and_preserves_queued_job(self) -> None:
        engine = FakeEngine()
        first_scheduler = JobScheduler(engine, runtime_root=self.root, autostart=False)
        running = first_scheduler.submit(
            "mesh", {"geometry27": GEOMETRY27, "name": "interrupted"}, client_id="one"
        )
        queued = first_scheduler.submit(
            "mesh", {"geometry27": GEOMETRY27, "name": "queued"}, client_id="two"
        )
        with closing(sqlite3.connect(first_scheduler.db_path)) as connection:
            with connection:
                connection.execute(
                    "UPDATE jobs SET state = 'running' WHERE job_id = ?",
                    (running["job_id"],),
                )
        first_scheduler.close()

        restarted = self.scheduler(engine, autostart=False)
        interrupted = restarted.get(running["job_id"])
        self.assertEqual(interrupted["state"], "failed")
        self.assertIn("restarted", interrupted["error"])
        self.assertEqual(restarted.get(queued["job_id"])["state"], "queued")

    def test_result_ttl_expires_state_and_exact_result_directory(self) -> None:
        engine = FakeEngine()
        scheduler = self.scheduler(
            engine,
            result_ttl_sec=0.05,
            cleanup_interval_sec=0.01,
        )
        job = scheduler.submit(
            "mesh", {"geometry27": GEOMETRY27, "name": "ttl"}, client_id="one"
        )
        self.assertEqual(self.wait_for_state(scheduler, job["job_id"])["state"], "succeeded")
        result_directory = scheduler.job_root / job["job_id"]
        self.assertTrue(result_directory.is_dir())
        expired = self.wait_for_state(
            scheduler, job["job_id"], expected={"expired"}, timeout=2
        )
        self.assertEqual(expired["state"], "expired")
        self.assertFalse(result_directory.exists())

    def test_public_payload_limits_are_enforced_before_enqueue(self) -> None:
        scheduler = self.scheduler(FakeEngine(), autostart=False)
        invalid_payloads = [
            ("mesh", {"geometry27": [0.0] * 26}),
            ("predict", {"geometry27": GEOMETRY27, "mach": 1.2, "aoa": 0}),
            ("recover", {"case_id": "../case", "cycles": 6}),
            ("reference", {"case_id": "case_ok", "max_cycles": 24}),
            ("reference", {"case_id": "case_ok", "max_cycles": 3001}),
        ]
        for action, payload in invalid_payloads:
            with self.subTest(action=action), self.assertRaises(ValueError):
                scheduler.submit(action, payload, client_id="client")

    def test_job_and_case_ownership_hide_other_sessions(self) -> None:
        engine = FakeEngine()
        case_root = self.root.parent / "cases"
        (case_root / "case_owned").mkdir(parents=True)
        scheduler = self.scheduler(
            engine,
            autostart=False,
            case_root=case_root,
            enforce_case_ownership=True,
        )
        scheduler.register_case("case_owned", "owner")
        job = scheduler.submit(
            "recover", {"case_id": "case_owned", "cycles": 2}, client_id="owner"
        )
        self.assertEqual(scheduler.get(job["job_id"], client_id="owner")["state"], "queued")
        with self.assertRaises(JobNotFoundError):
            scheduler.get(job["job_id"], client_id="other")
        with self.assertRaises(FileNotFoundError):
            scheduler.authorize_case("case_owned", "other")
        with self.assertRaises(FileNotFoundError):
            scheduler.submit(
                "reference",
                {"case_id": "case_owned", "max_cycles": 25},
                client_id="other",
            )

    def test_case_ttl_removes_only_registered_case_directory(self) -> None:
        engine = FakeEngine()
        case_root = self.root.parent / "cases"
        owned = case_root / "case_expiring"
        unrelated = case_root / "keep-this"
        owned.mkdir(parents=True)
        unrelated.mkdir(parents=True)
        scheduler = self.scheduler(
            engine,
            autostart=False,
            case_root=case_root,
            case_ttl_sec=0.01,
        )
        scheduler.register_case("case_expiring", "owner")
        time.sleep(0.02)
        self.assertEqual(scheduler.cleanup_expired(), 1)
        self.assertFalse(owned.exists())
        self.assertTrue(unrelated.exists())

    def test_geometry_ttl_removes_only_exact_unreferenced_cache(self) -> None:
        engine = FakeEngine()
        case_root = self.root.parent / "cases"
        mesh_root = self.root.parent / "meshes"
        solver_root = self.root.parent / "solver_prepare"
        geometry_key = "a" * 12
        geometry_id = "b" * 16
        mesh_root.mkdir(parents=True)
        (mesh_root / f"{geometry_key}.demo.json").write_text("{}")
        (mesh_root / f"{geometry_id}.cgns").write_bytes(b"fixture")
        (mesh_root / f"{geometry_id}.cgns.lock").write_bytes(b"")
        (mesh_root / "keep.txt").write_text("keep")
        (solver_root / geometry_key).mkdir(parents=True)
        scheduler = self.scheduler(
            engine,
            autostart=False,
            case_root=case_root,
            mesh_root=mesh_root,
            solver_prepare_root=solver_root,
            case_ttl_sec=0.01,
        )
        now = time.time()
        with scheduler._connection() as connection:
            scheduler._register_geometry_in_connection(
                geometry_key, geometry_id, connection, now=now
            )
        time.sleep(0.02)
        self.assertEqual(scheduler.cleanup_expired(), 1)
        self.assertFalse((mesh_root / f"{geometry_key}.demo.json").exists())
        self.assertFalse((mesh_root / f"{geometry_id}.cgns").exists())
        self.assertFalse((mesh_root / f"{geometry_id}.cgns.lock").exists())
        self.assertFalse((solver_root / geometry_key).exists())
        self.assertTrue((mesh_root / "keep.txt").exists())

    def test_watchdog_escalates_stuck_native_call(self) -> None:
        gate = threading.Event()
        escalated = threading.Event()
        scheduler = self.scheduler(
            FakeEngine(gate=gate),
            action_timeouts={
                "mesh": 0.05,
                "predict": 1,
                "recover": 1,
                "reference": 1,
            },
            hard_timeout_handler=lambda _event: escalated.set(),
            cleanup_interval_sec=1,
        )
        scheduler.submit(
            "mesh", {"geometry27": GEOMETRY27, "name": "stuck"}, client_id="owner"
        )
        self.assertTrue(escalated.wait(timeout=2))
        gate.set()


if __name__ == "__main__":
    unittest.main()
