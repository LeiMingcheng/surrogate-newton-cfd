from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from demo import build_uiuc_library

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = REPO_ROOT / "demo" / "static" / "example-airfoil.txt"


class UiucAssetBuilderTests(unittest.TestCase):
    def test_builder_creates_external_manifest_and_checksums(self) -> None:
        archive_buffer = BytesIO()
        with ZipFile(archive_buffer, "w") as archive:
            archive.writestr("fixture.dat", EXAMPLE_PATH.read_bytes())
        catalog_html = (
            '<a href="coord/fixture.dat">fixture.dat</a>'
            "Synthetic test airfoil<br>"
        ).encode("iso-8859-1")

        with tempfile.TemporaryDirectory(
            prefix="uiuc-builder-test-", dir=REPO_ROOT.parent
        ) as temporary:
            output_root = Path(temporary) / "demo-assets-uiuc"

            def download(url: str) -> bytes:
                if url == build_uiuc_library.UIUC_CATALOG_URL:
                    return catalog_html
                if url == build_uiuc_library.UIUC_ARCHIVE_URL:
                    return archive_buffer.getvalue()
                raise AssertionError(url)

            with (
                patch.object(build_uiuc_library, "_download", side_effect=download),
                patch.object(
                    sys,
                    "argv",
                    ["build_uiuc_library", "--output-root", str(output_root)],
                ),
            ):
                self.assertEqual(build_uiuc_library.main(), 0)

            manifest_path = output_root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["coordinate_count"], 1)
            self.assertEqual(manifest["library_root"], "uiuc")
            self.assertTrue((output_root / "uiuc" / "coordinates" / "fixture.dat").is_file())
            for line in (output_root / "SHA256SUMS").read_text().splitlines():
                expected, relative = line.split("  ", 1)
                actual = hashlib.sha256((output_root / relative).read_bytes()).hexdigest()
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
