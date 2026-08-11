from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

from demo.compute import DemoEngine
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
        handler = type(
            "TestDemoRequestHandler",
            (DemoRequestHandler,),
            {"engine": cls.engine},
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

    def test_static_routes_return_self_contained_assets(self) -> None:
        routes = (
            "/",
            "/demo",
            "/app.js",
            "/styles.css",
            "/example-airfoil.dat",
            "/assets/workflow.svg",
            "/assets/benchmark.svg",
            "/assets/recovery.svg",
            "/assets/paper-workflow.jpg",
            "/assets/paper-ood-benchmark.jpg",
            "/assets/paper-recovery.jpg",
            "/assets/paper-optimization.png",
        )
        for route in routes:
            with self.subTest(route=route):
                status, content_type, body = self.request_bytes(route)
                self.assertEqual(status, 200)
                self.assertTrue(content_type)
                self.assertTrue(body)


if __name__ == "__main__":
    unittest.main()
