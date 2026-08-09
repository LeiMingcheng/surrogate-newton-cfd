#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
lock_file=${SURROGATE_NEWTON_SOLVER_LOCK:-"$repo_root/solver-stack.lock.yaml"}
python_bin=${SURROGATE_NEWTON_PYTHON:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}
python_bin=${python_bin:-python3}
pyhyp_dir=
adflow_dir=
cgns_dir=
cgnsutilities_dir=
output_dir=

while (($#)); do
    case "$1" in
        --pyhyp)
            pyhyp_dir=${2:?--pyhyp requires a path}
            shift 2
            ;;
        --adflow)
            adflow_dir=${2:?--adflow requires a path}
            shift 2
            ;;
        --cgns)
            cgns_dir=${2:?--cgns requires a path}
            shift 2
            ;;
        --cgnsutilities)
            cgnsutilities_dir=${2:?--cgnsutilities requires a path}
            shift 2
            ;;
        --output)
            output_dir=${2:?--output requires a path}
            shift 2
            ;;
        *)
            printf 'ERROR: unknown argument: %s\n' "$1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$pyhyp_dir" || -z "$adflow_dir" || -z "$cgns_dir" \
    || -z "$cgnsutilities_dir" || -z "$output_dir" ]]; then
    printf '%s\n' \
        'Usage: scripts/package_solver_bundles.sh --pyhyp PATH --adflow PATH' \
        '       --cgns PATH --cgnsutilities PATH --output PATH' >&2
    exit 1
fi

repo_root=$(realpath -m "$repo_root")
output_dir=$(realpath -m "$output_dir")
if [[ "$output_dir/" == "$repo_root/"* ]]; then
    printf 'ERROR: bundle output must be outside the Git repository.\n' >&2
    exit 1
fi
if [[ -e "$output_dir" && ! -d "$output_dir" ]]; then
    printf 'ERROR: output exists and is not a directory: %s\n' "$output_dir" >&2
    exit 1
fi
if [[ -d "$output_dir" && -n $(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
    printf 'ERROR: refusing to overwrite non-empty release directory: %s\n' \
        "$output_dir" >&2
    exit 1
fi

locked_source_output=$(
    "$python_bin" - \
        "$lock_file" "$pyhyp_dir" "$adflow_dir" "$cgns_dir" "$cgnsutilities_dir" <<'PY'
from pathlib import Path
import re
import sys

import yaml

lock = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
paths = {
    "pyhyp": sys.argv[2],
    "adflow": sys.argv[3],
    "cgns": sys.argv[4],
    "cgnsutilities": sys.argv[5],
}
entries = (
    ("pyhyp", lock["pyhyp"], "fork_commit"),
    ("adflow", lock["adflow"], "fork_commit"),
    ("cgns", lock["cgns"], "commit"),
    ("cgnsutilities", lock["cgnsutilities"], "commit"),
)
for name, item, commit_key in entries:
    commit = item[commit_key]
    bundle_ref = item["bundle_ref"]
    bundle_file = item["bundle_file"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise SystemExit(f"{name}: expected a full 40-character commit SHA")
    if Path(bundle_file).name != bundle_file or not bundle_file.endswith(".bundle"):
        raise SystemExit(f"{name}: unsafe bundle filename")
    print(f"{name}\t{Path(paths[name]).expanduser().resolve()}\t{commit}\t{bundle_ref}\t{bundle_file}")
PY
)
mapfile -t locked_sources <<<"$locked_source_output"

for source_record in "${locked_sources[@]}"; do
    IFS=$'\t' read -r name source_dir expected_commit bundle_ref bundle_file \
        <<<"$source_record"
    if [[ $(git -C "$source_dir" rev-parse --is-inside-work-tree 2>/dev/null) != true ]]; then
        printf 'ERROR: %s is not a Git worktree: %s\n' "$name" "$source_dir" >&2
        exit 1
    fi
    actual_ref_commit=$(git -C "$source_dir" rev-parse "$bundle_ref^{commit}")
    if [[ "$actual_ref_commit" != "$expected_commit" ]]; then
        printf 'ERROR: %s ref %s is at %s, expected %s.\n' \
            "$name" "$bundle_ref" "$actual_ref_commit" "$expected_commit" >&2
        exit 1
    fi
done

mkdir -p "$output_dir"
manifest_args=()
bundle_files=()
for source_record in "${locked_sources[@]}"; do
    IFS=$'\t' read -r name source_dir expected_commit bundle_ref bundle_file \
        <<<"$source_record"
    bundle_path="$output_dir/$bundle_file"
    printf 'Packaging %s at %s...\n' "$name" "$expected_commit"
    git -C "$source_dir" bundle create "$bundle_path" "$bundle_ref"
    git -C "$source_dir" bundle verify "$bundle_path"
    size_bytes=$(stat --format='%s' "$bundle_path")
    sha256=$(sha256sum "$bundle_path" | cut -d' ' -f1)
    manifest_args+=("$name" "$expected_commit" "$bundle_ref" "$bundle_file" "$size_bytes" "$sha256")
    bundle_files+=("$bundle_file")
done

"$python_bin" - "$lock_file" "$output_dir/manifest.json" "${manifest_args[@]}" <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys

lock_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
values = sys.argv[3:]
if len(values) % 6:
    raise SystemExit("invalid bundle manifest arguments")
bundles = {}
for index in range(0, len(values), 6):
    name, commit, bundle_ref, filename, size_bytes, sha256 = values[index : index + 6]
    bundles[name] = {
        "commit": commit,
        "bundle_ref": bundle_ref,
        "filename": filename,
        "size_bytes": int(size_bytes),
        "sha256": sha256,
    }
payload = {
    "schema_version": 1,
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "solver_stack_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
    "bundles": bundles,
}
manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

(
    cd "$output_dir"
    sha256sum "${bundle_files[@]}" manifest.json > SHA256SUMS
    sha256sum --check SHA256SUMS
)

printf 'Solver bundle release ready: %s\n' "$output_dir"
