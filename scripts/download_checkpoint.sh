#!/usr/bin/env bash
set -euo pipefail

artifact_dir=${1:-artifacts}
base_url=${SURROGATE_NEWTON_ASSET_BASE_URL:-https://github.com/LeiMingcheng/surrogate-newton-cfd/releases/download/model-2608.04400}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
manifest=${SURROGATE_NEWTON_MODEL_MANIFEST:-"$repo_root/model-manifest.json"}
python_bin=${SURROGATE_NEWTON_PYTHON:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}
python_bin=${python_bin:-python3}

mkdir -p "$artifact_dir"

download_and_verify() {
    filename=$1
    expected_sha256=$2
    expected_size=$3
    target="$artifact_dir/$filename"
    partial="$target.part"

    if [[ -f "$target" ]]; then
        actual_size=$(stat --format='%s' "$target")
        if [[ "$actual_size" == "$expected_size" ]] \
            && printf '%s  %s\n' "$expected_sha256" "$target" | sha256sum --check --status -; then
            printf 'Already verified: %s\n' "$target"
            return
        fi
        printf 'ERROR: existing final artifact failed verification: %s\n' "$target" >&2
        exit 1
    fi

    printf 'Downloading %s...\n' "$filename"
    curl \
        --fail \
        --location \
        --retry 3 \
        --continue-at - \
        --output "$partial" \
        "${base_url%/}/$filename"

    actual_size=$(stat --format='%s' "$partial")
    if [[ "$actual_size" != "$expected_size" ]] \
        || ! printf '%s  %s\n' "$expected_sha256" "$partial" | sha256sum --check --status -; then
        rm -f "$partial"
        printf 'ERROR: downloaded artifact failed size or SHA-256 verification: %s\n' \
            "$filename" >&2
        exit 1
    fi

    mv "$partial" "$target"
    printf 'Verified: %s\n' "$target"
}

artifact_output=$(
    "$python_bin" - "$manifest" <<'PY'
from pathlib import Path
import json
import sys

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for key in ("model", "normalization_statistics"):
    item = manifest[key]
    filename = item["filename"]
    if Path(filename).name != filename:
        raise SystemExit(f"unsafe artifact filename: {filename}")
    print(f"{filename}\t{item['sha256']}\t{item['size_bytes']}")
PY
)
mapfile -t artifacts <<<"$artifact_output"

for artifact in "${artifacts[@]}"; do
    IFS=$'\t' read -r filename sha256 size_bytes <<<"$artifact"
    download_and_verify "$filename" "$sha256" "$size_bytes"
done
