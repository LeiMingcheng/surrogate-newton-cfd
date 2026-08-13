import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from NK_resume.solver.warm_pool import ResidentWarmPoolController


class _FakeProcess:
    _next_pid = 1000

    def __init__(self) -> None:
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.returncode = None

    def poll(self):
        return self.returncode


class ResidentPoolRecoveryTests(unittest.TestCase):
    def test_resident_pool_rebuilds_after_worker_exit(self) -> None:
        processes: list[_FakeProcess] = []

        def launch(*_args, **_kwargs):
            process = _FakeProcess()
            processes.append(process)
            return process

        def mark_ready(entries, _timeout):
            for entry in entries:
                Path(entry["ready_file"]).write_text("{}", encoding="utf-8")

        with tempfile.TemporaryDirectory() as directory:
            controller = ResidentWarmPoolController(
                ranks_per_case=2,
                pool_count=1,
                mpi_launcher="mpirun",
                output_dir=Path(directory) / "resident",
                request_wait_timeout_sec=0.0,
            )
            with (
                patch(
                    "NK_resume.solver.warm_pool.resolve_mpi_launcher",
                    return_value=["mpirun"],
                ),
                patch(
                    "NK_resume.solver.warm_pool.inject_mpi_runtime_env_args",
                    side_effect=lambda command, _env: command,
                ),
                patch(
                    "NK_resume.solver.warm_pool.python_executable",
                    return_value="python",
                ),
                patch(
                    "NK_resume.solver.warm_pool.subprocess.Popen",
                    side_effect=launch,
                ),
                patch(
                    "NK_resume.solver.warm_pool._wait_for_ready_files",
                    side_effect=mark_ready,
                ),
                patch("NK_resume.solver.warm_pool._terminate_process_group"),
            ):
                controller.start()
                self.assertTrue(controller.status()["healthy"])

                processes[0].returncode = 1
                status = controller.recover_if_idle()

        self.assertEqual(len(processes), 2)
        self.assertTrue(status["healthy"])
        self.assertEqual(status["launch_count"], 2)
        self.assertEqual(status["restart_count"], 1)
        self.assertFalse(status["poisoned"])
        self.assertEqual(status["last_exit"]["workers"][0]["returncode"], 1)
        self.assertIsNone(status["last_recovery_error"])

    def test_failed_background_recovery_is_reported_without_breaking_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = ResidentWarmPoolController(
                ranks_per_case=2,
                pool_count=1,
                output_dir=Path(directory) / "resident",
            )
            controller._launch_count = 1
            with patch.object(
                controller,
                "_launch",
                side_effect=RuntimeError("relaunch failed"),
            ):
                status = controller.recover_if_idle()

        self.assertFalse(status["healthy"])
        self.assertEqual(
            status["last_recovery_error"],
            "RuntimeError: relaunch failed",
        )


if __name__ == "__main__":
    unittest.main()
