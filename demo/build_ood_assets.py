"""Validate and freeze the offline geometry-distance inputs as an external bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from demo.ood import OOD_ASSET_FILENAMES, OodGeometryIndex


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        required=True,
        help="Directory containing cst26_coefficients.npz and ood_scores.csv.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="New external bundle directory that will receive validated assets.",
    )
    parser.add_argument(
        "--source-name",
        default="offline OOD geometry analysis",
        help="Human-readable provenance label stored in manifest.json.",
    )
    parser.add_argument(
        "--source-revision",
        default="unrecorded",
        help="Optional source revision or experiment identifier.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        raise ValueError("--output-root must be an absolute path outside the source tree")
    source_tree = Path(__file__).resolve().parents[1]
    if output_root == source_tree or output_root.is_relative_to(source_tree):
        raise ValueError("The OOD analysis assets must remain outside the public source tree")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite the existing bundle: {output_root}")

    index = OodGeometryIndex.from_asset_root(source_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ood-build-", dir=output_root.parent) as temp:
        staging = Path(temp) / "bundle"
        staging.mkdir()
        copied_files: list[dict[str, int | str]] = []
        for filename in OOD_ASSET_FILENAMES:
            source = source_root / filename
            target = staging / filename
            shutil.copyfile(source, target)
            copied_files.append(
                {
                    "path": filename,
                    "size_bytes": target.stat().st_size,
                    "sha256": _sha256(target),
                }
            )

        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "name": output_root.name,
                    "distribution_strategy": "external-bundle",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source": {
                        "name": str(args.source_name),
                        "revision": str(args.source_revision),
                    },
                    "index": index.summary(),
                    "files": copied_files,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        checksum_paths = [manifest_path, *(staging / name for name in OOD_ASSET_FILENAMES)]
        (staging / "SHA256SUMS").write_text(
            "\n".join(f"{_sha256(path)}  {path.name}" for path in checksum_paths) + "\n",
            encoding="utf-8",
        )
        staging.rename(output_root)

    print(
        f"Stored {index.geometry_count} geometries and {index.training_count} "
        f"training scores under {output_root}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
