from __future__ import annotations

import re
import shutil
import subprocess
import unittest
from pathlib import Path

from demo.compute import _default_runtime_root

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_ROOT = REPO_ROOT / "demo"


class DemoStaticSecurityTests(unittest.TestCase):
    def test_demo_presents_recovery_as_nk_without_server_budget_details(self) -> None:
        html = (DEMO_ROOT / "static" / "demo.html").read_text(encoding="utf-8")
        script = (DEMO_ROOT / "static" / "app.js").read_text(encoding="utf-8")
        public_copy = html + script
        self.assertIn("Surrogate + NK convergence target", html)
        self.assertIn("10^-p · fixed compute budget", html)
        self.assertNotIn('class="target-value"', html)
        self.assertIn('class="horizontal-setting custom-setting"', html)
        self.assertNotIn("compact-control", html)
        self.assertNotIn("nk-setting", html)
        self.assertIn("fixed compute budget", public_copy)
        self.assertNotIn("ANK", public_copy)
        self.assertNotIn("work ceiling", public_copy)
        self.assertNotIn("wall-time limit", public_copy)
        self.assertNotIn("ADFLOW work / ceiling", public_copy)
        self.assertIn("MAX_HANDLE_COUNT = 16", script)
        self.assertIn("state.handleCount = DEFAULT_HANDLE_COUNT", script)

    def test_demo_flow_controls_match_the_public_operating_range(self) -> None:
        html = (DEMO_ROOT / "static" / "demo.html").read_text(encoding="utf-8")
        self.assertRegex(
            html,
            r'id="mach-range"[^>]+min="0\.4"[^>]+max="0\.8"',
        )
        self.assertRegex(
            html,
            r'id="aoa-range"[^>]+min="-2"[^>]+max="6"',
        )

    def test_public_demo_has_no_private_runtime_paths_or_credentials(self) -> None:
        forbidden = (
            "/root/",
            "PIR-DM",
            "giao-dm",
            "build_libraries",
            "BEGIN OPENSSH PRIVATE KEY",
            "BEGIN RSA PRIVATE KEY",
            "proxy_password",
            "server_info.md",
        )
        text_files = [
            path
            for path in DEMO_ROOT.rglob("*")
            if path.is_file()
            and path.suffix.lower()
            in {".py", ".md", ".html", ".css", ".js", ".txt", ".svg"}
        ]
        for path in text_files:
            content = path.read_text(encoding="utf-8")
            for marker in forbidden:
                with self.subTest(path=path, marker=marker):
                    self.assertNotIn(marker, content)

    def test_static_html_references_existing_packaged_files(self) -> None:
        route_to_file = {
            "/styles.css": DEMO_ROOT / "static" / "styles.css",
            "/project.css": DEMO_ROOT / "static" / "project.css",
            "/app.js": DEMO_ROOT / "static" / "app.js",
            "/example-airfoil.dat": DEMO_ROOT / "static" / "example-airfoil.txt",
            "/assets/workflow.svg": DEMO_ROOT / "static" / "assets" / "workflow.svg",
            "/assets/benchmark.svg": DEMO_ROOT / "static" / "assets" / "benchmark.svg",
            "/assets/recovery.svg": DEMO_ROOT / "static" / "assets" / "recovery.svg",
            "/assets/paper-workflow.jpg": DEMO_ROOT / "static" / "assets" / "paper-workflow.jpg",
            "/assets/paper-ood-benchmark.jpg": (
                DEMO_ROOT / "static" / "assets" / "paper-ood-benchmark.jpg"
            ),
            "/assets/paper-recovery.jpg": DEMO_ROOT / "static" / "assets" / "paper-recovery.jpg",
            "/assets/paper-optimization.png": (
                DEMO_ROOT / "static" / "assets" / "paper-optimization.png"
            ),
            **{
                f"/assets/figures/{name}.png": (
                    DEMO_ROOT / "static" / "assets" / "figures" / f"{name}.png"
                )
                for name in (
                    "fig1_workflow",
                    "fig2_a_dataset_geometries",
                    "fig2_b_geometry_ood_score_distribution",
                    "fig3_a_recovery_comparison",
                    "fig3_c_ellipse_pressure_recovery",
                    "fig5_a_optimization_curves_2x2_nk",
                    "fig6_a_geometry_ood",
                    "fig6_c_surface_cp_recovery",
                    "fig6_d_pressure_flow_recovery",
                )
            },
            **{
                f"/assets/figure-sources/{name}.pdf": (
                    DEMO_ROOT / "static" / "assets" / "figure-sources" / f"{name}.pdf"
                )
                for name in (
                    "fig1_workflow",
                    "fig2_a_dataset_geometries",
                    "fig2_b_geometry_ood_score_distribution",
                    "fig3_a_recovery_comparison",
                    "fig3_c_ellipse_pressure_recovery",
                    "fig5_a_optimization_curves_2x2_nk",
                    "fig6_a_geometry_ood",
                    "fig6_c_surface_cp_recovery",
                    "fig6_d_pressure_flow_recovery",
                )
            },
        }
        html_files = (DEMO_ROOT / "static" / "index.html", DEMO_ROOT / "static" / "demo.html")
        for html_path in html_files:
            content = html_path.read_text(encoding="utf-8")
            local_references = re.findall(r"(?:src|href)=\"(/[^\"#]+)\"", content)
            for route in local_references:
                route = route.split("?", 1)[0]
                if route in {"/", "/demo"}:
                    continue
                with self.subTest(html=html_path.name, route=route):
                    self.assertIn(route, route_to_file)
                    self.assertTrue(route_to_file[route].is_file())

    def test_demo_tree_has_no_generated_runtime_artifacts(self) -> None:
        forbidden_names = {"__pycache__", "server_info.md"}
        forbidden_suffixes = {".pyc", ".cgns", ".npz", ".pt", ".log"}
        if shutil.which("git"):
            result = subprocess.run(
                [
                    "git",
                    "ls-files",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                    "demo",
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            paths = (REPO_ROOT / relative for relative in result.stdout.splitlines())
        else:
            paths = (
                path
                for path in DEMO_ROOT.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            )
        for path in paths:
            with self.subTest(path=path):
                self.assertNotIn(path.name, forbidden_names)
                self.assertNotIn(path.suffix.lower(), forbidden_suffixes)

    def test_default_runtime_is_external_and_container_remains_non_root(self) -> None:
        self.assertFalse(_default_runtime_root().resolve().is_relative_to(REPO_ROOT.resolve()))
        dockerfile = (REPO_ROOT / "deployment" / "container" / "Dockerfile").read_text(
            encoding="utf-8"
        )
        self.assertIn("USER surrogate", dockerfile)
        self.assertIn("chown -R surrogate:surrogate /runtime", dockerfile)

    def test_package_and_console_entry_include_demo(self) -> None:
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('surrogate-newton-demo = "demo.server:main"', pyproject)
        self.assertIn('"demo*"', pyproject)
        self.assertIn('"static/assets/*.svg"', pyproject)
        self.assertIn('"static/assets/*.jpg"', pyproject)
        self.assertIn('"static/assets/*.png"', pyproject)

    def test_demo_discloses_transport_and_shared_compute_limits(self) -> None:
        html = (DEMO_ROOT / "static" / "demo.html").read_text(encoding="utf-8")
        self.assertIn("Public research preview over HTTP", html)
        self.assertIn("Compute resources are limited", html)
        self.assertIn("requests may queue", html)


if __name__ == "__main__":
    unittest.main()
