"""Build a CST-screened UIUC airfoil library outside the source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

from demo.compute import (
    FIXED_TRAILING_EDGE_THICKNESS,
    UIUC_CATALOG_URL,
    UIUC_COORDINATE_ROOT,
    UIUC_CST_FIT_MSE_LIMIT,
    UIUC_SITE_URL,
    _fit_coordinate_text_to_cst,
    _geometry_payload,
    _parse_uiuc_catalog,
)

UIUC_ARCHIVE_URL = (
    "https://m-selig.ae.illinois.edu/ads/archives/coord_seligFmt.zip"
)
USER_AGENT = "Surrogate-Newton CFD academic airfoil demo"


def _download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30.0) as response:
        return response.read()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        required=True,
        help="New external bundle directory that will receive uiuc/ and checksums.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        raise ValueError("--output-root must be an absolute path outside the source tree.")
    source_root = Path(__file__).resolve().parents[1]
    if output_root == source_root or output_root.is_relative_to(source_root):
        raise ValueError("The complete UIUC library must remain outside the public source tree.")
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite the existing library: {output_root}"
        )

    official_catalog = _parse_uiuc_catalog(
        _download(UIUC_CATALOG_URL).decode("iso-8859-1")
    )
    descriptions = {
        entry["filename"].lower(): entry["description"]
        for entry in official_catalog
    }
    archive = ZipFile(BytesIO(_download(UIUC_ARCHIVE_URL)))
    members = {
        Path(member).name.lower(): member
        for member in archive.namelist()
        if member.lower().endswith(".dat")
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".uiuc-build-", dir=output_root.parent) as temp:
        staging_bundle = Path(temp) / "bundle"
        staging_root = staging_bundle / "uiuc"
        coordinate_root = staging_root / "coordinates"
        coordinate_root.mkdir(parents=True)
        accepted: list[dict[str, object]] = []
        rejected: list[dict[str, object]] = []

        for filename, member in sorted(members.items()):
            raw = archive.read(member)
            content = raw.decode("iso-8859-1")
            try:
                geometry27, fit_mse = _fit_coordinate_text_to_cst(content)
                geometry = _geometry_payload(filename, geometry27, fit_mse=fit_mse)
            except ValueError as exc:
                rejected.append({"filename": filename, "reason": str(exc)})
                continue
            if fit_mse > UIUC_CST_FIT_MSE_LIMIT:
                rejected.append(
                    {
                        "filename": filename,
                        "reason": "CST reconstruction MSE exceeds the library limit.",
                        "fit_mse": fit_mse,
                    }
                )
                continue

            (coordinate_root / filename).write_bytes(raw)
            accepted.append(
                {
                    "key": f"uiuc:{filename}",
                    "name": Path(filename).stem.upper(),
                    "filename": filename,
                    "description": descriptions.get(filename, Path(filename).stem),
                    "coordinate_path": f"coordinates/{filename}",
                    "coordinate_url": f"{UIUC_COORDINATE_ROOT}/{filename}",
                    "fit_mse": fit_mse,
                    "max_thickness": geometry["metrics"]["max_thickness"],
                }
            )

        generated_at = datetime.now(timezone.utc).isoformat()
        source = {
            "name": "UIUC Airfoil Data Site",
            "url": UIUC_SITE_URL,
            "catalog_url": UIUC_CATALOG_URL,
            "archive_url": UIUC_ARCHIVE_URL,
            "retrieved_at": generated_at,
            "archive_coordinate_count": len(members),
        }
        filtering = {
            "cst_parameter_count": 26,
            "fixed_trailing_edge_thickness": FIXED_TRAILING_EDGE_THICKNESS,
            "cst_fit_mse_max": UIUC_CST_FIT_MSE_LIMIT,
            "max_thickness": 0.35,
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
        }
        (staging_root / "catalog.json").write_text(
            json.dumps(
                {"source": source, "filter": filtering, "airfoils": accepted},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (staging_root / "rejected.json").write_text(
            json.dumps(
                {"source": source, "filter": filtering, "airfoils": rejected},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        files = []
        for path in sorted(item for item in staging_root.rglob("*") if item.is_file()):
            files.append(
                {
                    "path": path.relative_to(staging_bundle).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        manifest_path = staging_bundle / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "name": output_root.name,
                    "distribution_strategy": "external-bundle",
                    "library_root": "uiuc",
                    "coordinate_count": len(accepted),
                    "source": source,
                    "filter": filtering,
                    "files": files,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        checksum_paths = [
            manifest_path,
            *sorted(item for item in staging_root.rglob("*") if item.is_file()),
        ]
        (staging_bundle / "SHA256SUMS").write_text(
            "\n".join(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
                f"{path.relative_to(staging_bundle).as_posix()}"
                for path in checksum_paths
            )
            + "\n",
            encoding="utf-8",
        )
        staging_bundle.rename(output_root)

    print(
        f"Stored {len(accepted)} UIUC airfoils and rejected {len(rejected)} "
        f"under {output_root / 'uiuc'}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
