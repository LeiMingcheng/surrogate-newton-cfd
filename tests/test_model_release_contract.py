"""Contracts for the immutable public inference-model release."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "model-manifest.json"
DOWNLOADER = REPO_ROOT / "scripts/download_checkpoint.sh"


class ModelReleaseContractTests(unittest.TestCase):
    def test_manifest_pins_hugging_face_commit_and_inference_checkpoint(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        hosting = manifest["hosting"]
        revision = hosting["revision"]

        self.assertEqual(hosting["provider"], "huggingface")
        self.assertEqual(
            hosting["repository"],
            "xzztj/surrogate-newton-cfd-fsb-dit-airfoil-inference",
        )
        self.assertRegex(revision, r"^[0-9a-f]{40}$")
        self.assertEqual(
            hosting["base_url"],
            f"https://huggingface.co/{hosting['repository']}/resolve/{revision}",
        )
        self.assertEqual(
            manifest["release"],
            "model-2608.04400-inference-v1",
        )

        model = manifest["model"]
        self.assertEqual(model["checkpoint_type"], "inference_only")
        self.assertEqual(model["weights"], "ema")
        self.assertFalse(model["resume_training"])
        self.assertTrue(model["filename"].endswith("-inference.pt"))
        self.assertEqual(len(model["sha256"]), 64)
        self.assertGreater(model["size_bytes"], 0)
        self.assertLess(model["size_bytes"], 500_000_000)
        self.assertEqual(
            model["sha256"],
            "4291eb87bd9771e5619dc9745ee4268b2c1ca47d17c3c4e692f0fa5f1876447c",
        )
        self.assertEqual(model["size_bytes"], 362_194_924)
        self.assertEqual(
            model["source_training_checkpoint_sha256"],
            "0b8be8a31cc972fb817f46369c1d39efd39b703f307ba28eb85be388c7b2d942",
        )
        config_path = REPO_ROOT / model["config"]
        self.assertEqual(
            hashlib.sha256(config_path.read_bytes()).hexdigest(),
            model["config_sha256"],
        )

    def test_downloader_uses_manifest_url_and_verifies_release_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_string:
            temp_dir = Path(temp_dir_string)
            fixtures = temp_dir / "fixtures"
            fixtures.mkdir()
            artifact_contents = {
                "test-inference.pt": b"verified inference checkpoint\n",
                "test-stats.json": b'{"version": "test"}\n',
            }
            for filename, content in artifact_contents.items():
                (fixtures / filename).write_bytes(content)

            manifest = {
                "hosting": {"base_url": ""},
                "model": self._artifact_record("test-inference.pt", artifact_contents),
                "normalization_statistics": self._artifact_record(
                    "test-stats.json", artifact_contents
                ),
            }
            manifest["hosting"]["base_url"] = (
                "https://huggingface.co/test/repo/resolve/"
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            )
            manifest_path = temp_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            fake_bin = temp_dir / "bin"
            fake_bin.mkdir()
            fake_curl = fake_bin / "curl"
            fake_curl.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    output=
                    url=
                    while (($#)); do
                        case "$1" in
                            --output)
                                output=$2
                                shift 2
                                ;;
                            --retry|--continue-at)
                                shift 2
                                ;;
                            --fail|--location)
                                shift
                                ;;
                            *)
                                url=$1
                                shift
                                ;;
                        esac
                    done
                    filename=${url##*/}
                    printf '%s\\n' "$url" >>"$CURL_LOG"
                    cp "$FIXTURE_DIR/$filename" "$output"
                    """
                ),
                encoding="utf-8",
            )
            fake_curl.chmod(0o755)

            output_dir = temp_dir / "artifacts"
            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{fake_bin}:{environment['PATH']}",
                    "CURL_LOG": str(temp_dir / "curl.log"),
                    "FIXTURE_DIR": str(fixtures),
                    "SURROGATE_NEWTON_MODEL_MANIFEST": str(manifest_path),
                    "SURROGATE_NEWTON_PYTHON": os.environ.get(
                        "SURROGATE_NEWTON_PYTHON", "python3"
                    ),
                }
            )
            result = subprocess.run(
                ["bash", str(DOWNLOADER), str(output_dir)],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for filename, content in artifact_contents.items():
                self.assertEqual((output_dir / filename).read_bytes(), content)

            base_url = manifest["hosting"]["base_url"]
            self.assertEqual(
                (temp_dir / "curl.log").read_text(encoding="utf-8").splitlines(),
                [f"{base_url}/{filename}" for filename in artifact_contents],
            )

            second_result = subprocess.run(
                ["bash", str(DOWNLOADER), str(output_dir)],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(second_result.returncode, 0, second_result.stderr)
            self.assertEqual(second_result.stdout.count("Already verified:"), 2)
            self.assertEqual(
                len((temp_dir / "curl.log").read_text(encoding="utf-8").splitlines()),
                2,
            )

    def test_downloader_rejects_non_https_manifest_url(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_string:
            temp_dir = Path(temp_dir_string)
            manifest = {
                "hosting": {"base_url": "http://example.invalid/model"},
                "model": {
                    "filename": "model.pt",
                    "sha256": "0" * 64,
                    "size_bytes": 1,
                },
                "normalization_statistics": {
                    "filename": "stats.json",
                    "sha256": "0" * 64,
                    "size_bytes": 1,
                },
            }
            manifest_path = temp_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            environment = os.environ.copy()
            environment["SURROGATE_NEWTON_MODEL_MANIFEST"] = str(manifest_path)
            environment["SURROGATE_NEWTON_PYTHON"] = os.environ.get(
                "SURROGATE_NEWTON_PYTHON", "python3"
            )
            result = subprocess.run(
                ["bash", str(DOWNLOADER), str(temp_dir / "artifacts")],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must be an HTTPS URL", result.stderr)

    @staticmethod
    def _artifact_record(filename: str, contents: dict[str, bytes]) -> dict[str, object]:
        content = contents[filename]
        return {
            "filename": filename,
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }


if __name__ == "__main__":
    unittest.main()
