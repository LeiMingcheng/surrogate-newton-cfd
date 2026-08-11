from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from demo.compute import (
    MAX_UPLOAD_BYTES,
    _fit_coordinate_text_to_cst,
    _geometry_payload,
    _split_coordinate_text,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = REPO_ROOT / "demo" / "static" / "example-airfoil.txt"


def _zones(x: np.ndarray, upper: np.ndarray, lower: np.ndarray, *, comma: bool = False) -> str:
    separator = "," if comma else " "
    upper_rows = "\n".join(f"{a:.8f}{separator}{b:.8f}" for a, b in zip(x, upper))
    lower_rows = "\n".join(f"{a:.8f}{separator}{b:.8f}" for a, b in zip(x, lower))
    return f"zone upper\n{upper_rows}\nzone lower\n{lower_rows}\n"


class CoordinateGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.example = EXAMPLE_PATH.read_text(encoding="utf-8")

    def test_example_two_zone_import_projects_to_27_parameters(self) -> None:
        geometry, fit_mse = _fit_coordinate_text_to_cst(self.example)
        payload = _geometry_payload("example", geometry, fit_mse=fit_mse)
        self.assertEqual(len(payload["geometry27"]), 27)
        self.assertTrue(np.isfinite(payload["metrics"]["fit_mse"]))

    def test_closed_contour_import(self) -> None:
        upper, lower = _split_coordinate_text(self.example)
        contour = np.vstack([upper[::-1], lower[1:]])
        content = "Example closed contour\n" + "\n".join(
            f"{x:.8f} {y:.8f}" for x, y in contour
        )
        geometry, _ = _fit_coordinate_text_to_cst(content)
        self.assertEqual(geometry.shape, (27,))

    def test_comma_separated_import(self) -> None:
        x = np.linspace(0.0, 1.0, 41)
        shape = np.sin(np.pi * x)
        geometry, _ = _fit_coordinate_text_to_cst(
            _zones(x, 0.06 * shape, -0.04 * shape, comma=True)
        )
        self.assertEqual(geometry.shape, (27,))

    def test_empty_and_non_numeric_inputs_fail(self) -> None:
        for content in ("", "airfoil\nnot coordinate data\n"):
            with self.subTest(content=content):
                with self.assertRaises(ValueError):
                    _split_coordinate_text(content)

    def test_nan_and_inf_fail(self) -> None:
        for invalid in ("nan", "inf", "-inf"):
            content = self.example.replace("0.010000  0.015000", f"0.010000  {invalid}")
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "finite"):
                    _split_coordinate_text(content)

    def test_oversized_upload_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "2 MB"):
            _split_coordinate_text("0 0\n" + "x" * MAX_UPLOAD_BYTES)

    def test_surface_crossing_and_thickness_limit_fail(self) -> None:
        x = np.linspace(0.0, 1.0, 61)
        shape = np.sin(np.pi * x)
        invalid_geometries = (
            _zones(x, 0.03 * shape, 0.04 * shape),
            _zones(x, 0.30 * shape, -0.30 * shape),
        )
        for content in invalid_geometries:
            with self.subTest():
                geometry, fit_mse = _fit_coordinate_text_to_cst(content)
                with self.assertRaises(ValueError):
                    _geometry_payload("invalid", geometry, fit_mse=fit_mse)

    def test_illegal_surface_sequence_fails(self) -> None:
        lines = self.example.splitlines()
        first = lines.index("zone upper") + 1
        lines[first + 4], lines[first + 14] = lines[first + 14], lines[first + 4]
        with self.assertRaisesRegex(ValueError, "monotonic"):
            _split_coordinate_text("\n".join(lines))


if __name__ == "__main__":
    unittest.main()
