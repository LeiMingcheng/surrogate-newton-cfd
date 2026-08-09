#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
stack_dir=${1:?Usage: scripts/fetch_solver_stack.sh /path/to/solver-stack [--source-mode git|bundle] [--bundle-dir /path]}
shift
lock_file=${SURROGATE_NEWTON_SOLVER_LOCK:-"$repo_root/solver-stack.lock.yaml"}
python_bin=${SURROGATE_NEWTON_PYTHON:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}
python_bin=${python_bin:-python3}
source_mode=${SURROGATE_NEWTON_SOLVER_SOURCE_MODE:-git}
bundle_dir=${SURROGATE_NEWTON_SOLVER_BUNDLE_DIR:-}

while (($#)); do
    case "$1" in
        --source-mode)
            source_mode=${2:?--source-mode requires git or bundle}
            shift 2
            ;;
        --bundle-dir)
            bundle_dir=${2:?--bundle-dir requires a path}
            shift 2
            ;;
        *)
            printf 'ERROR: unknown argument: %s\n' "$1" >&2
            exit 1
            ;;
    esac
done

if [[ "$source_mode" != git && "$source_mode" != bundle ]]; then
    printf 'ERROR: source mode must be git or bundle, got %s.\n' "$source_mode" >&2
    exit 1
fi
if [[ "$source_mode" == bundle && ! -d "$bundle_dir" ]]; then
    printf 'ERROR: bundle mode requires an existing --bundle-dir.\n' >&2
    exit 1
fi
if [[ "$source_mode" == bundle ]]; then
    bundle_dir=$(realpath "$bundle_dir")
fi

mkdir -p "$stack_dir"

locked_repository_output=$(
    "$python_bin" - "$lock_file" "$source_mode" "$bundle_dir" <<'PY'
from pathlib import Path
import hashlib
import json
import re
import sys
from urllib.parse import urlsplit

import yaml

lock_path = Path(sys.argv[1])
source_mode = sys.argv[2]
bundle_dir = Path(sys.argv[3]) if source_mode == "bundle" else None
lock_bytes = lock_path.read_bytes()
lock = yaml.safe_load(lock_bytes)
entries = (
    ("pyhyp", lock["pyhyp"], "fork_commit"),
    ("adflow", lock["adflow"], "fork_commit"),
    ("cgns", lock["cgns"], "commit"),
    ("cgnsutilities", lock["cgnsutilities"], "commit"),
)
manifest = None
if source_mode == "bundle":
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1:
        raise SystemExit("bundle manifest schema_version must be 1")
    lock_sha256 = hashlib.sha256(lock_bytes).hexdigest()
    if manifest["solver_stack_lock_sha256"] != lock_sha256:
        raise SystemExit("bundle manifest does not match solver-stack.lock.yaml")
    if set(manifest["bundles"]) != {name for name, _, _ in entries}:
        raise SystemExit(
            "bundle manifest must contain exactly pyhyp, adflow, cgns, and cgnsutilities"
        )

for name, item, commit_key in entries:
    url = item["repository"]
    commit = item[commit_key]
    bundle_ref = item["bundle_ref"]
    bundle_file = item["bundle_file"]
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.username or parsed.password:
        raise SystemExit(f"{name}: repository URL must be credential-free HTTPS")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise SystemExit(f"{name}: expected a full 40-character commit SHA")
    if Path(bundle_file).name != bundle_file or not bundle_file.endswith(".bundle"):
        raise SystemExit(f"{name}: unsafe bundle filename")

    expected_sha256 = "-"
    expected_size = "-"
    if manifest is not None:
        bundle = manifest["bundles"][name]
        if bundle["commit"] != commit:
            raise SystemExit(f"{name}: bundle manifest commit does not match lock")
        if bundle["bundle_ref"] != bundle_ref:
            raise SystemExit(f"{name}: bundle manifest ref does not match lock")
        if bundle["filename"] != bundle_file:
            raise SystemExit(f"{name}: bundle manifest filename does not match lock")
        if not re.fullmatch(r"[0-9a-f]{64}", bundle["sha256"]):
            raise SystemExit(f"{name}: invalid bundle SHA-256")
        expected_sha256 = bundle["sha256"]
        expected_size = str(bundle["size_bytes"])
    print(
        f"{name}\t{url}\t{commit}\t{bundle_ref}\t{bundle_file}\t"
        f"{expected_sha256}\t{expected_size}"
    )
PY
)
mapfile -t locked_repositories <<<"$locked_repository_output"

verification_repo=
if [[ "$source_mode" == bundle ]]; then
    verification_repo="$stack_dir/.bundle-verification.git"
    if [[ ! -d "$verification_repo" ]]; then
        git init --bare --quiet "$verification_repo"
    fi
    for repository in "${locked_repositories[@]}"; do
        IFS=$'\t' read -r \
            name _ expected_commit bundle_ref bundle_file expected_sha256 expected_size \
            <<<"$repository"
        bundle_path="$bundle_dir/$bundle_file"
        if [[ ! -f "$bundle_path" ]]; then
            printf 'ERROR: missing bundle for %s: %s\n' "$name" "$bundle_file" >&2
            exit 1
        fi
        actual_size=$(stat --format='%s' "$bundle_path")
        if [[ "$actual_size" != "$expected_size" ]]; then
            printf 'ERROR: bundle size mismatch for %s.\n' "$name" >&2
            exit 1
        fi
        if ! printf '%s  %s\n' "$expected_sha256" "$bundle_path" \
            | sha256sum --check --status -; then
            printf 'ERROR: bundle SHA-256 mismatch for %s.\n' "$name" >&2
            exit 1
        fi
        git -C "$verification_repo" bundle verify "$bundle_path"
        git -C "$verification_repo" fetch --quiet --no-tags "$bundle_path" "$bundle_ref"
        if ! git -C "$verification_repo" cat-file -e "$expected_commit^{commit}"; then
            printf 'ERROR: %s bundle does not contain commit %s.\n' \
                "$name" "$expected_commit" >&2
            exit 1
        fi
    done
fi

for repository in "${locked_repositories[@]}"; do
    IFS=$'\t' read -r \
        name url expected_commit bundle_ref bundle_file expected_sha256 expected_size \
        <<<"$repository"
    target="$stack_dir/$name"

    if [[ "$source_mode" == git ]]; then
        if [[ ! -e "$target" ]]; then
            printf 'Fetching %s source from HTTPS Git...\n' "$name"
            git clone --no-checkout "$url" "$target"
        elif [[ ! -d "$target/.git" ]]; then
            printf 'ERROR: %s exists but is not a Git checkout.\n' "$target" >&2
            exit 1
        fi
        git -C "$target" fetch --no-tags origin "$expected_commit"
    else
        bundle_path="$bundle_dir/$bundle_file"
        if [[ ! -e "$target" ]]; then
            printf 'Restoring %s source from verified bundle...\n' "$name"
            git clone --no-checkout "$bundle_path" "$target"
        elif [[ ! -d "$target/.git" ]]; then
            printf 'ERROR: %s exists but is not a Git checkout.\n' "$target" >&2
            exit 1
        fi
    fi

    git -C "$target" checkout --detach "$expected_commit"
    actual_commit=$(git -C "$target" rev-parse HEAD)
    if [[ "$actual_commit" != "$expected_commit" ]]; then
        printf 'ERROR: %s resolved to %s, expected %s.\n' \
            "$name" "$actual_commit" "$expected_commit" >&2
        exit 1
    fi
    printf '%s %s\n' "$name" "$actual_commit"
done
