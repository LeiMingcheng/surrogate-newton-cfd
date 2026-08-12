from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

from demo.compute import DemoEngine
from demo.jobs import JobScheduler
from demo.server import DemoRequestHandler

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = REPO_ROOT / "demo" / "static" / "example-airfoil.txt"


class DemoApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        temp_parent = REPO_ROOT / "tmp"
        temp_parent.mkdir(exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(prefix="demo-api-", dir=temp_parent)
        cls.temp_root = Path(cls.temp.name)
        cls.asset_root = cls.temp_root / "uiuc"
        coordinate_root = cls.asset_root / "coordinates"
        coordinate_root.mkdir(parents=True)
        example = EXAMPLE_PATH.read_text(encoding="utf-8")
        (coordinate_root / "fixture.dat").write_text(example, encoding="utf-8")
        catalog = {
            "source": {"name": "Synthetic UIUC test fixture", "url": "https://example.invalid"},
            "filter": {"accepted_count": 1, "rejected_count": 0},
            "airfoils": [
                {
                    "key": "uiuc:fixture.dat",
                    "name": "FIXTURE",
                    "filename": "fixture.dat",
                    "description": "Synthetic test coordinate",
                    "coordinate_path": "coordinates/fixture.dat",
                    "coordinate_url": "https://example.invalid/fixture.dat",
                }
            ],
        }
        (cls.asset_root / "catalog.json").write_text(
            json.dumps(catalog), encoding="utf-8"
        )
        cls.engine = DemoEngine(
            runtime_root=cls.temp_root / "runtime",
            airfoil_library_root=cls.asset_root,
            surrogate_port=9,
        )
        cls.scheduler = JobScheduler(
            cls.engine,
            runtime_root=cls.temp_root / "scheduler",
            autostart=False,
        )
        handler = type(
            "TestDemoRequestHandler",
            (DemoRequestHandler,),
            {"engine": cls.engine, "scheduler": cls.scheduler},
        )
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.server.daemon_threads = True
        cls.port = int(cls.server.server_address[1])
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls.scheduler.close()
        cls.engine.close()
        cls.temp.cleanup()

    def request(
        self,
        method: str,
        path: str,
        payload: object | None = None,
        *,
        raw_body: str | None = None,
    ) -> tuple[int, dict[str, object]]:
        connection = HTTPConnection("127.0.0.1", self.port, timeout=10)
        body = raw_body if raw_body is not None else (
            None if payload is None else json.dumps(payload)
        )
        headers = {} if body is None else {"Content-Type": "application/json"}
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        result = json.loads(response.read().decode("utf-8"))
        connection.close()
        return response.status, result

    def request_bytes(self, path: str) -> tuple[int, str, bytes]:
        connection = HTTPConnection("127.0.0.1", self.port, timeout=10)
        connection.request("GET", path)
        response = connection.getresponse()
        body = response.read()
        content_type = response.getheader("Content-Type") or ""
        connection.close()
        return response.status, content_type, body

    def test_status_and_presets_work_with_surrogate_offline(self) -> None:
        status_code, status = self.request("GET", "/api/status")
        self.assertEqual(status_code, 200)
        self.assertFalse(status["surrogate_online"])
        self.assertTrue(status["airfoil_library_available"])
        self.assertEqual(status["scheduler"]["concurrency_limit"], 1)
        presets_code, presets = self.request("GET", "/api/presets")
        self.assertEqual(presets_code, 200)
        self.assertEqual([item["name"] for item in presets["presets"]], ["RAE2822", "OAT15A"])

    def test_geometry_import_and_projection(self) -> None:
        content = EXAMPLE_PATH.read_text(encoding="utf-8")
        import_code, imported = self.request(
            "POST",
            "/api/geometry/import",
            {"filename": "airfoil.csv", "content": content},
        )
        self.assertEqual(import_code, 200)
        self.assertEqual(len(imported["geometry27"]), 27)
        project_code, projected = self.request(
            "POST",
            "/api/geometry/project",
            {
                "name": "Projected fixture",
                "x": imported["x"],
                "upper": imported["upper"],
                "lower": imported["lower"],
            },
        )
        self.assertEqual(project_code, 200)
        self.assertEqual(len(projected["geometry27"]), 27)

    def test_uiuc_catalog_and_airfoil(self) -> None:
        catalog_code, catalog = self.request("GET", "/api/uiuc/catalog")
        self.assertEqual(catalog_code, 200)
        self.assertEqual(catalog["count"], 1)
        airfoil_code, airfoil = self.request("GET", "/api/uiuc/airfoil/fixture.dat")
        self.assertEqual(airfoil_code, 200)
        self.assertEqual(airfoil["geometry"]["name"], "FIXTURE")

    def test_missing_external_library_has_explicit_error(self) -> None:
        engine = DemoEngine(
            runtime_root=self.temp_root / "missing-runtime",
            airfoil_library_root=self.temp_root / "missing-assets",
            surrogate_port=9,
        )
        try:
            with self.assertRaisesRegex(FileNotFoundError, "not mounted"):
                engine.uiuc_catalog()
        finally:
            engine.close()

    def test_invalid_json_routes_cases_and_paths_return_4xx(self) -> None:
        requests = [
            ("POST", "/api/geometry/import", "{not json"),
            ("POST", "/api/unknown", "{}"),
            ("GET", "/api/cases/case_missing", None),
            ("GET", "/api/uiuc/airfoil/%2e%2e%2fsecret.dat", None),
            ("GET", "/api/uiuc/airfoil/fixture.py", None),
        ]
        for method, path, body in requests:
            with self.subTest(path=path):
                status, response = self.request(method, path, raw_body=body)
                self.assertGreaterEqual(status, 400)
                self.assertLess(status, 500)
                self.assertFalse(response["ok"])
                self.assertNotIn("/root/", str(response))
                self.assertNotIn("Traceback", str(response))

    def test_upload_extension_is_checked(self) -> None:
        status, response = self.request(
            "POST",
            "/api/geometry/import",
            {"filename": "payload.json", "content": EXAMPLE_PATH.read_text()},
        )
        self.assertEqual(status, 400)
        self.assertIn(".dat", response["error"])

    def test_job_submit_query_cancel_and_result_state(self) -> None:
        submit_code, submitted = self.request(
            "POST",
            "/api/jobs",
            {
                "action": "mesh",
                "payload": {"geometry27": [0.0] * 27, "name": "Queued fixture"},
            },
        )
        self.assertEqual(submit_code, 202)
        job_id = str(submitted["job"]["job_id"])
        self.assertEqual(submitted["job"]["state"], "queued")
        query_code, queried = self.request("GET", f"/api/jobs/{job_id}")
        self.assertEqual(query_code, 200)
        self.assertEqual(queried["job"]["queue_position"], 1)
        cancel_code, cancelled = self.request("DELETE", f"/api/jobs/{job_id}")
        self.assertEqual(cancel_code, 200)
        self.assertEqual(cancelled["job"]["state"], "cancelled")
        result_code, result = self.request("GET", f"/api/jobs/{job_id}/result")
        self.assertEqual(result_code, 409)
        self.assertFalse(result["ok"])

    def test_legacy_heavy_route_also_enqueues(self) -> None:
        submit_code, submitted = self.request(
            "POST",
            "/api/mesh",
            {"geometry27": [0.0] * 27, "name": "Legacy fixture"},
        )
        self.assertEqual(submit_code, 202)
        self.assertEqual(submitted["job"]["action"], "mesh")
        self.assertEqual(submitted["job"]["state"], "queued")
        self.request("DELETE", f"/api/jobs/{submitted['job']['job_id']}")

    def test_job_payload_validation_happens_before_enqueue(self) -> None:
        status, response = self.request(
            "POST",
            "/api/jobs",
            {"action": "predict", "payload": {"geometry27": [0.0] * 27, "mach": 2}},
        )
        self.assertEqual(status, 400)
        self.assertFalse(response["ok"])

    def test_static_routes_return_self_contained_assets(self) -> None:
        routes = (
            "/",
            "/demo",
            "/app.js",
            "/styles.css",
            "/project.css",
            "/example-airfoil.dat",
            "/assets/workflow.svg",
            "/assets/benchmark.svg",
            "/assets/recovery.svg",
            "/assets/paper-workflow.jpg",
            "/assets/paper-ood-benchmark.jpg",
            "/assets/paper-recovery.jpg",
            "/assets/paper-optimization.png",
            "/assets/figures/fig1_workflow.png",
            "/assets/figures/fig2_a_dataset_geometries.png",
            "/assets/figures/fig2_b_geometry_ood_score_distribution.png",
            "/assets/figures/fig3_a_recovery_comparison.png",
            "/assets/figures/fig3_c_ellipse_pressure_recovery.png",
            "/assets/figures/fig5_a_optimization_curves_2x2_nk.png",
            "/assets/figures/fig6_a_geometry_ood.png",
            "/assets/figures/fig6_c_surface_cp_recovery.png",
            "/assets/figures/fig6_d_pressure_flow_recovery.png",
            "/assets/figure-sources/fig1_workflow.pdf",
            "/assets/figure-sources/fig2_a_dataset_geometries.pdf",
            "/assets/figure-sources/fig2_b_geometry_ood_score_distribution.pdf",
            "/assets/figure-sources/fig3_a_recovery_comparison.pdf",
            "/assets/figure-sources/fig3_c_ellipse_pressure_recovery.pdf",
            "/assets/figure-sources/fig5_a_optimization_curves_2x2_nk.pdf",
            "/assets/figure-sources/fig6_a_geometry_ood.pdf",
            "/assets/figure-sources/fig6_c_surface_cp_recovery.pdf",
            "/assets/figure-sources/fig6_d_pressure_flow_recovery.pdf",
        )
        for route in routes:
            with self.subTest(route=route):
                status, content_type, body = self.request_bytes(route)
                self.assertEqual(status, 200)
                self.assertTrue(content_type)
                self.assertTrue(body)


class ConcentratedAccessApiTests(unittest.TestCase):
    class Engine:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.active = 0
            self.max_active = 0

        def prepare_mesh(self, **payload: object) -> dict[str, object]:
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.02)
                return {"name": payload["name"], "queued_test": True}
            finally:
                with self.lock:
                    self.active -= 1

    def setUp(self) -> None:
        temp_parent = REPO_ROOT / "tmp"
        temp_parent.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="demo-load-api-", dir=temp_parent)
        self.engine = self.Engine()
        self.scheduler = JobScheduler(
            self.engine,
            runtime_root=Path(self.temp.name) / "scheduler",
            max_pending_jobs=32,
            max_pending_jobs_per_client=2,
            cleanup_interval_sec=1,
        )
        handler = type(
            "LoadTestDemoRequestHandler",
            (DemoRequestHandler,),
            {"engine": self.engine, "scheduler": self.scheduler},
        )
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.server.daemon_threads = True
        self.port = int(self.server.server_address[1])
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.scheduler.close(timeout=5)
        self.temp.cleanup()

    def request(
        self,
        method: str,
        path: str,
        payload: object | None = None,
        *,
        forwarded_for: str | None = None,
    ) -> tuple[int, dict[str, object]]:
        connection = HTTPConnection("127.0.0.1", self.port, timeout=10)
        body = None if payload is None else json.dumps(payload)
        headers = {} if body is None else {"Content-Type": "application/json"}
        if forwarded_for:
            headers["X-Forwarded-For"] = forwarded_for
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        result = json.loads(response.read().decode("utf-8"))
        connection.close()
        return response.status, result

    def test_twelve_simultaneous_http_clients_are_serialized(self) -> None:
        def submit(index: int) -> str:
            status, response = self.request(
                "POST",
                "/api/jobs",
                {
                    "action": "mesh",
                    "payload": {"geometry27": [0.0] * 27, "name": f"visitor-{index}"},
                },
                forwarded_for=f"198.51.100.{index + 1}",
            )
            self.assertEqual(status, 202)
            return str(response["job"]["job_id"])

        with ThreadPoolExecutor(max_workers=12) as executor:
            job_ids = list(executor.map(submit, range(12)))

        deadline = time.monotonic() + 8
        pending = set(job_ids)
        while pending and time.monotonic() < deadline:
            for job_id in list(pending):
                status, response = self.request("GET", f"/api/jobs/{job_id}")
                self.assertEqual(status, 200)
                if response["job"]["state"] == "succeeded":
                    pending.remove(job_id)
            time.sleep(0.01)

        self.assertFalse(pending)
        self.assertEqual(self.engine.max_active, 1)


if __name__ == "__main__":
    unittest.main()
