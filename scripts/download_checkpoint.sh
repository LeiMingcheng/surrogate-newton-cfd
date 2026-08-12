#!/usr/bin/env bash
set -euo pipefail

artifact_dir=${1:-artifacts}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
manifest=${SURROGATE_NEWTON_MODEL_MANIFEST:-"$repo_root/model-manifest.json"}
python_bin=${SURROGATE_NEWTON_PYTHON:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}
python_bin=${python_bin:-python3}

mkdir -p "$artifact_dir"

file_size() {
    if stat --format='%s' "$1" >/dev/null 2>&1; then
        stat --format='%s' "$1"
    else
        stat -f '%z' "$1"
    fi
}

file_sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        printf 'ERROR: sha256sum or shasum is required\n' >&2
        return 1
    fi
}

download_and_verify() {
    filename=$1
    expected_sha256=$2
    expected_size=$3
    target="$artifact_dir/$filename"
    partial="$target.part"

    if [[ -f "$target" ]]; then
        actual_size=$(file_size "$target")
        if [[ "$actual_size" == "$expected_size" ]] \
            && [[ "$(file_sha256 "$target")" == "$expected_sha256" ]]; then
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

    actual_size=$(file_size "$partial")
    if [[ "$actual_size" != "$expected_size" ]] \
        || [[ "$(file_sha256 "$partial")" != "$expected_sha256" ]]; then
        rm -f "$partial"
        printf 'ERROR: downloaded artifact failed size or SHA-256 verification: %s\n' \
            "$filename" >&2
        exit 1
    fi

    mv "$partial" "$target"
    printf 'Verified: %s\n' "$target"
}

manifest_output=$(
    "$python_bin" - "$manifest" <<'PY'
from pathlib import Path
import json
import sys
from urllib.parse import urlparse

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
base_url = manifest["hosting"]["base_url"]
parsed = urlparse(base_url)
if parsed.scheme != "https" or not parsed.netloc:
    raise SystemExit(f"manifest hosting.base_url must be an HTTPS URL: {base_url}")
print(base_url)
for key in ("model", "normalization_statistics"):
    item = manifest[key]
    filename = item["filename"]
    if Path(filename).name != filename:
        raise SystemExit(f"unsafe artifact filename: {filename}")
    print(f"{filename}\t{item['sha256']}\t{item['size_bytes']}")
PY
)
manifest_base_url=${manifest_output%%$'\n'*}
base_url=${SURROGATE_NEWTON_ASSET_BASE_URL:-$manifest_base_url}
artifact_output=${manifest_output#*$'\n'}

while IFS=$'\t' read -r filename sha256 size_bytes; do
    [[ -n "$filename" ]] || continue
    download_and_verify "$filename" "$sha256" "$size_bytes"
done <<<"$artifact_output"
