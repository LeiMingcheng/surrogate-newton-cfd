"""Static contracts for the reproducible container build."""

from __future__ import annotations

from pathlib import Path
import unittest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTAINER_ROOT = REPO_ROOT / "deployment/container"


class ContainerContractTests(unittest.TestCase):
    def test_runtime_dependencies_are_locked(self) -> None:
        environment = yaml.safe_load(
            (CONTAINER_ROOT / "environment.runtime.yml").read_text(encoding="utf-8")
        )
        pip_dependencies = next(
            item["pip"] for item in environment["dependencies"] if isinstance(item, dict)
        )
        expected = {
            "tensorboard==2.16.2",
            "pyvista==0.46.4",
            "vtk==9.5.2",
            "absl-py==2.1.0",
            "grpcio==1.62.1",
            "Markdown==3.7",
            "MarkupSafe==3.0.2",
            "protobuf==4.24.4",
            "six==1.16.0",
            "tensorboard-data-server==0.7.2",
            "Werkzeug==3.1.3",
            "certifi==2025.10.5",
            "charset-normalizer==3.4.4",
            "idna==3.11",
            "platformdirs==4.5.0",
            "pooch==1.8.2",
            "requests==2.32.5",
            "scooby==0.11.0",
            "urllib3==2.5.0",
        }
        self.assertTrue(expected.issubset(set(pip_dependencies)))
        self.assertFalse(any(item.startswith("cgnsutilities") for item in pip_dependencies))
        self.assertFalse(any(item.startswith("torch") for item in pip_dependencies))

    def test_build_compiler_contract(self) -> None:
        environment = yaml.safe_load(
            (CONTAINER_ROOT / "environment.build.yml").read_text(encoding="utf-8")
        )
        dependencies = set(environment["dependencies"])
        self.assertNotIn("cmake=3.31.6", dependencies)
        self.assertIn("make=4.4.1", dependencies)
        self.assertIn("gcc_linux-64=15.1.0", dependencies)
        build_script = (REPO_ROOT / "scripts/build_solver_stack.sh").read_text(encoding="utf-8")
        self.assertIn("x86_64-conda-linux-gnu-cc", build_script)
        self.assertIn("x86_64-conda-linux-gnu-gfortran", build_script)
        self.assertIn("Fortran name mangling convention: LOWERCASE_", build_script)
        for name in ("pyhyp", "adflow"):
            config = (CONTAINER_ROOT / f"config/{name}.config.mk.in").read_text(
                encoding="utf-8"
            )
            self.assertIn("@ENV_PREFIX@/bin/mpicc", config)
            self.assertIn("@ENV_PREFIX@/bin/mpifort", config)

    def test_offline_and_runtime_image_contract(self) -> None:
        dockerfile = (CONTAINER_ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertFalse(dockerfile.startswith("# syntax="))
        for context in ("solver-bundles", "miniforge-installer", "python-wheels-cu128"):
            self.assertIn(f"from={context}", dockerfile)
        for package in (
            "libegl1",
            "libgl1",
            "libsm6",
            "libx11-6",
            "libxext6",
            "libxkbcommon0",
            "libxrender1",
            "libxt6",
        ):
            self.assertIn(package, dockerfile)
        self.assertIn("SURROGATE_NEWTON_RUNTIME_ROOT=/runtime/tmp", dockerfile)
        self.assertIn("CFD_RUNTIME_TMPDIR=/runtime/tmp/cfd", dockerfile)
        self.assertIn("USER surrogate", dockerfile)
        self.assertIn("python -m pip check", dockerfile)
        records = {}
        for line in (CONTAINER_ROOT / "offline-inputs.lock").read_text(
            encoding="utf-8"
        ).splitlines():
            if line and not line.startswith("#"):
                name, filename, size_bytes, sha256 = line.split("\t")
                records[name] = (filename, int(size_bytes), sha256)
        self.assertEqual(
            records["miniforge"],
            (
                "Miniforge3-26.3.2-2-Linux-x86_64.sh",
                106038245,
                "42260ffe3830fb953d5eee1bbb32229ff06aa7c3833c1ed7a9a0420a95685d94",
            ),
        )
        self.assertEqual(
            records["torch"],
            (
                "torch-2.8.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl",
                889148606,
                "0c96999d15cf1f13dd7c913e0b21a9a355538e6cfc10861a17158320292f5954",
            ),
        )

    def test_compiled_verification_contract(self) -> None:
        verifier = (REPO_ROOT / "scripts/verify_solver_stack.sh").read_text(encoding="utf-8")
        for marker in (
            "libcgns_utils.so",
            "import cgnsutilities",
            "import tensorboard",
            "import pyvista",
            "import vtk",
            "ldd",
            "not found",
            "-m pip check",
        ):
            self.assertIn(marker, verifier)


if __name__ == "__main__":
    unittest.main()
