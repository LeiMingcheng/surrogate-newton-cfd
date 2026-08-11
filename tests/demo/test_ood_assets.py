from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from demo import build_ood_assets
from demo.ood import OodGeometryIndex

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_fixture(root: Path, *, count: int = 6) -> None:
    root.mkdir(parents=True)
    keys = np.asarray([f"train:{index}" for index in range(count)])
    cst26 = np.zeros((count, 26), dtype=np.float64)
    for index in range(count):
        cst26[index, 0] = 0.05 + 0.01 * index
        cst26[index, 13] = -0.04 - 0.008 * index
    np.savez(root / "cst26_coefficients.npz", geometry_key=keys, cst26=cst26)
    with (root / "ood_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "geometry_key", "ood_k5"])
        writer.writeheader()
        for index, key in enumerate(keys):
            writer.writerow(
                {"split": "train", "geometry_key": key, "ood_k5": 0.001 * (index + 1)}
            )


class OodAssetTests(unittest.TestCase):
    def test_index_loads_validated_assets_and_scores_geometry(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ood-index-", dir=REPO_ROOT.parent) as temp:
            root = Path(temp) / "source"
            _write_fixture(root)
            index = OodGeometryIndex.from_asset_root(root)
            result = index.score(np.zeros(27, dtype=np.float64))
            self.assertEqual(index.geometry_count, 6)
            self.assertEqual(index.training_count, 6)
            self.assertGreaterEqual(result["percentile"], 0.0)
            self.assertLessEqual(result["percentile"], 1.0)
            self.assertGreaterEqual(result["distance_k5"], 0.0)

    def test_index_rejects_too_few_training_rows(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ood-invalid-", dir=REPO_ROOT.parent) as temp:
            root = Path(temp) / "source"
            _write_fixture(root, count=4)
            with self.assertRaisesRegex(ValueError, "at least five"):
                OodGeometryIndex.from_asset_root(root)

    def test_builder_creates_external_manifest_and_checksums(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ood-builder-", dir=REPO_ROOT.parent) as temp:
            source = Path(temp) / "source"
            output = Path(temp) / "demo-assets-ood"
            _write_fixture(source)
            with patch.object(
                sys,
                "argv",
                [
                    "build_ood_assets",
                    "--source-root",
                    str(source),
                    "--output-root",
                    str(output),
                    "--source-revision",
                    "fixture-revision",
                ],
            ):
                self.assertEqual(build_ood_assets.main(), 0)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["index"]["geometry_count"], 6)
            self.assertEqual(manifest["index"]["training_count"], 6)
            self.assertEqual(manifest["source"]["revision"], "fixture-revision")
            for line in (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
                expected, relative = line.split("  ", 1)
                actual = hashlib.sha256((output / relative).read_bytes()).hexdigest()
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
