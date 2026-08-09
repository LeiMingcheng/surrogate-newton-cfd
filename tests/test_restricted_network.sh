#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${SURROGATE_NEWTON_PYTHON:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}
python_bin=${python_bin:-python3}
test_root=$(mktemp -d "$repo_root/../.surrogate-newton-test.XXXXXX")
source_root="$test_root/sources"
lock_file="$test_root/solver-stack.lock.yaml"
release_dir="$test_root/solver-bundles-test"
mkdir -p "$source_root"

expect_failure() {
    local label=$1
    shift
    if "$@" >"$test_root/$label.log" 2>&1; then
        printf 'ERROR: %s unexpectedly succeeded.\n' "$label" >&2
        exit 1
    fi
}

declare -A expected_commits=()
for name in pyhyp adflow cgns; do
    source_dir="$source_root/$name"
    git init --quiet --initial-branch=main "$source_dir"
    git -C "$source_dir" config user.name "Surrogate Newton test"
    git -C "$source_dir" config user.email "test@example.invalid"
    printf '%s fixture\n' "$name" > "$source_dir/source.txt"
    git -C "$source_dir" add source.txt
    git -C "$source_dir" commit --quiet -m "test: add $name fixture"
    if [[ "$name" == cgns ]]; then
        git -C "$source_dir" branch -M develop
    else
        git -C "$source_dir" tag surrogate-newton-2608.04400
    fi
    expected_commits["$name"]=$(git -C "$source_dir" rev-parse HEAD)
done

"$python_bin" - "$lock_file" \
    "${expected_commits[pyhyp]}" \
    "${expected_commits[adflow]}" \
    "${expected_commits[cgns]}" <<'PY'
from pathlib import Path
import sys

import yaml

path = Path(sys.argv[1])
pyhyp_commit, adflow_commit, cgns_commit = sys.argv[2:]
lock = {
    "pyhyp": {
        "repository": "https://example.invalid/pyhyp.git",
        "fork_commit": pyhyp_commit,
        "bundle_ref": "refs/tags/surrogate-newton-2608.04400",
        "bundle_file": "pyhyp-test.bundle",
    },
    "adflow": {
        "repository": "https://example.invalid/adflow.git",
        "fork_commit": adflow_commit,
        "bundle_ref": "refs/tags/surrogate-newton-2608.04400",
        "bundle_file": "adflow-test.bundle",
    },
    "cgns": {
        "repository": "https://example.invalid/cgns.git",
        "commit": cgns_commit,
        "bundle_ref": "refs/heads/develop",
        "bundle_file": "cgns-test.bundle",
    },
}
path.write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")
PY

env \
    SURROGATE_NEWTON_PYTHON="$python_bin" \
    SURROGATE_NEWTON_SOLVER_LOCK="$lock_file" \
    "$repo_root/scripts/package_solver_bundles.sh" \
    --pyhyp "$source_root/pyhyp" \
    --adflow "$source_root/adflow" \
    --cgns "$source_root/cgns" \
    --output "$release_dir"
(
    cd "$release_dir"
    sha256sum --check --strict SHA256SUMS
)

expect_failure package-nonempty env \
    SURROGATE_NEWTON_PYTHON="$python_bin" \
    SURROGATE_NEWTON_SOLVER_LOCK="$lock_file" \
    "$repo_root/scripts/package_solver_bundles.sh" \
    --pyhyp "$source_root/pyhyp" \
    --adflow "$source_root/adflow" \
    --cgns "$source_root/cgns" \
    --output "$release_dir"
grep -q 'refusing to overwrite non-empty release directory' \
    "$test_root/package-nonempty.log"

mismatched_lock="$test_root/mismatched-solver-stack.lock.yaml"
cp "$lock_file" "$mismatched_lock"
printf '\n# changed after bundle packaging\n' >> "$mismatched_lock"
expect_failure manifest-lock-mismatch env \
    SURROGATE_NEWTON_PYTHON="$python_bin" \
    SURROGATE_NEWTON_SOLVER_LOCK="$mismatched_lock" \
    "$repo_root/scripts/fetch_solver_stack.sh" \
    "$test_root/mismatch-fetch" \
    --source-mode bundle \
    --bundle-dir "$release_dir"
grep -q 'bundle manifest does not match solver-stack.lock.yaml' \
    "$test_root/manifest-lock-mismatch.log"
test ! -e "$test_root/mismatch-fetch/pyhyp"

corrupt_release="$test_root/corrupt-release"
cp -a "$release_dir" "$corrupt_release"
printf X | dd \
    of="$corrupt_release/adflow-test.bundle" \
    bs=1 seek=128 conv=notrunc status=none
expect_failure corrupt-bundle env \
    SURROGATE_NEWTON_PYTHON="$python_bin" \
    SURROGATE_NEWTON_SOLVER_LOCK="$lock_file" \
    "$repo_root/scripts/fetch_solver_stack.sh" \
    "$test_root/corrupt-fetch" \
    --source-mode bundle \
    --bundle-dir "$corrupt_release"
grep -q 'bundle SHA-256 mismatch for adflow' "$test_root/corrupt-bundle.log"
test ! -e "$test_root/corrupt-fetch/pyhyp"

env \
    SURROGATE_NEWTON_PYTHON="$python_bin" \
    SURROGATE_NEWTON_SOLVER_LOCK="$lock_file" \
    "$repo_root/scripts/fetch_solver_stack.sh" \
    "$test_root/restored" \
    --source-mode bundle \
    --bundle-dir "$release_dir"
for name in pyhyp adflow cgns; do
    actual_commit=$(git -C "$test_root/restored/$name" rev-parse HEAD)
    if [[ "$actual_commit" != "${expected_commits[$name]}" ]]; then
        printf 'ERROR: %s restored at %s, expected %s.\n' \
            "$name" "$actual_commit" "${expected_commits[$name]}" >&2
        exit 1
    fi
done

wrapper_repo="$test_root/wrapper-repo"
mkdir -p "$wrapper_repo/scripts"
cp "$repo_root/scripts/build_restricted_server.sh" "$wrapper_repo/scripts/"
cp "$lock_file" "$wrapper_repo/solver-stack.lock.yaml"
git init --quiet --initial-branch=main "$wrapper_repo"
git -C "$wrapper_repo" config user.name "Surrogate Newton test"
git -C "$wrapper_repo" config user.email "test@example.invalid"
git -C "$wrapper_repo" add scripts/build_restricted_server.sh solver-stack.lock.yaml
git -C "$wrapper_repo" commit --quiet -m "test: add wrapper fixture"
wrapper_commit=$(git -C "$wrapper_repo" rev-parse HEAD)
wrapper="$wrapper_repo/scripts/build_restricted_server.sh"

expect_failure wrapper-floating-base \
    "$wrapper" \
    --bundle-dir "$release_dir" \
    --vcs-ref "$wrapper_commit" \
    --image-version test \
    --ubuntu-image ubuntu:24.04
grep -q 'must be pinned by a sha256 digest' "$test_root/wrapper-floating-base.log"

expect_failure wrapper-short-vcs \
    "$wrapper" \
    --bundle-dir "$release_dir" \
    --vcs-ref deadbeef \
    --image-version test
grep -q 'full 40-character lowercase commit SHA' "$test_root/wrapper-short-vcs.log"

mkdir -p "$test_root/incomplete-release"
expect_failure wrapper-incomplete-release \
    "$wrapper" \
    --bundle-dir "$test_root/incomplete-release" \
    --vcs-ref "$wrapper_commit" \
    --image-version test
grep -q 'must contain manifest.json and SHA256SUMS' \
    "$test_root/wrapper-incomplete-release.log"

printf 'dirty\n' > "$wrapper_repo/dirty-marker"
expect_failure wrapper-dirty-worktree \
    "$wrapper" \
    --bundle-dir "$release_dir" \
    --vcs-ref "$wrapper_commit" \
    --image-version test
grep -q 'require a clean Git worktree' "$test_root/wrapper-dirty-worktree.log"

printf 'restricted-network script tests: PASS\n'
printf 'test artifacts: %s\n' "$test_root"
