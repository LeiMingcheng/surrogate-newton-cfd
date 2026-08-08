#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
stack_dir=${1:?Usage: scripts/fetch_solver_stack.sh /path/to/solver-stack}
lock_file=${SURROGATE_NEWTON_SOLVER_LOCK:-"$repo_root/solver-stack.lock.yaml"}
python_bin=${SURROGATE_NEWTON_PYTHON:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}
python_bin=${python_bin:-python3}

mkdir -p "$stack_dir"

locked_repository_output=$(
    "$python_bin" - "$lock_file" <<'PY'
from pathlib import Path
import sys
from urllib.parse import urlsplit

import yaml

lock = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
entries = (
    ("pyhyp", lock["pyhyp"]["repository"], lock["pyhyp"]["fork_commit"]),
    ("adflow", lock["adflow"]["repository"], lock["adflow"]["fork_commit"]),
    ("cgns", lock["cgns"]["repository"], lock["cgns"]["commit"]),
)
for name, url, commit in entries:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.username or parsed.password:
        raise SystemExit(f"{name}: repository URL must be credential-free HTTPS")
    if len(commit) != 40:
        raise SystemExit(f"{name}: expected a full 40-character commit SHA")
    print(f"{name}\t{url}\t{commit}")
PY
)
mapfile -t locked_repositories <<<"$locked_repository_output"

for repository in "${locked_repositories[@]}"; do
    IFS=$'\t' read -r name url expected_commit <<<"$repository"
    target="$stack_dir/$name"
    if [[ ! -e "$target" ]]; then
        printf 'Fetching %s source...\n' "$name"
        git clone --no-checkout "$url" "$target"
    elif [[ ! -d "$target/.git" ]]; then
        printf 'ERROR: %s exists but is not a Git checkout.\n' "$target" >&2
        exit 1
    fi

    git -C "$target" fetch --no-tags origin "$expected_commit"
    git -C "$target" checkout --detach "$expected_commit"
    actual_commit=$(git -C "$target" rev-parse HEAD)
    if [[ "$actual_commit" != "$expected_commit" ]]; then
        printf 'ERROR: %s resolved to %s, expected %s.\n' \
            "$name" "$actual_commit" "$expected_commit" >&2
        exit 1
    fi
    printf '%s %s\n' "$name" "$actual_commit"
done
