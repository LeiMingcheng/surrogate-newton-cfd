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
for name in pyhyp adflow cgns cgnsutilities; do
    source_dir="$source_root/$name"
    git init --quiet --initial-branch=main "$source_dir"
    git -C "$source_dir" config user.name "Surrogate Newton test"
    git -C "$source_dir" config user.email "test@example.invalid"
    printf '%s fixture\n' "$name" > "$source_dir/source.txt"
    git -C "$source_dir" add source.txt
    git -C "$source_dir" commit --quiet -m "test: add $name fixture"
    if [[ "$name" == cgns ]]; then
        git -C "$source_dir" branch -M develop
    elif [[ "$name" == cgnsutilities ]]; then
        git -C "$source_dir" branch -M main
    else
        git -C "$source_dir" tag surrogate-newton-2608.04400
    fi
    expected_commits["$name"]=$(git -C "$source_dir" rev-parse HEAD)
done

no_git_root="$test_root/no-git-solvers"
mkdir -p \
    "$no_git_root/pyhyp/pyhyp" \
    "$no_git_root/adflow/src/f2py" \
    "$no_git_root/cgnsutilities/cgnsutilities" \
    "$test_root/no-git-path"
printf '%s\n' '04af13e4e59d4a113d7def96b6b5b2dbf6fa9ed9' \
    > "$no_git_root/pyhyp/.surrogate-newton-revision"
printf '%s\n' '4ad0091910d1d885f1f5ddcf41b96b5950fa16c9' \
    > "$no_git_root/adflow/.surrogate-newton-revision"
printf '%s\n' 'c321af3951432193fa9ead289dd6f88ee20c44e9' \
    > "$no_git_root/cgnsutilities/.surrogate-newton-revision"
printf '%s\n' 'def resetForNewSurface(): pass' \
    > "$no_git_root/pyhyp/pyhyp/pyHyp.py"
printf '%s\n' 'reinitafterinjection rebuildrestartderivedstateaftersetinfo' \
    > "$no_git_root/adflow/src/f2py/adflow.pyf"
printf '%s\n' 'class Grid: pass' \
    > "$no_git_root/cgnsutilities/cgnsutilities/cgnsutilities.py"
env PATH="$test_root/no-git-path" "$python_bin" \
    "$repo_root/deployment/smoke_check.py" \
    --level source \
    --pyhyp-root "$no_git_root/pyhyp" \
    --adflow-root "$no_git_root/adflow" \
    --cgnsutilities-root "$no_git_root/cgnsutilities" \
    > "$test_root/no-git-smoke.json"
grep -q '"cgnsutilities_exact_commit": true' "$test_root/no-git-smoke.json"
grep -q '"status": "ok"' "$test_root/no-git-smoke.json"

grep -q 'CMAKE_C_COMPILER="\$env_prefix/bin/x86_64-conda-linux-gnu-cc"' \
    "$repo_root/scripts/build_solver_stack.sh"
grep -q 'CMAKE_Fortran_COMPILER="\$env_prefix/bin/x86_64-conda-linux-gnu-gfortran"' \
    "$repo_root/scripts/build_solver_stack.sh"
grep -q 'FF90 = @ENV_PREFIX@/bin/mpifort' \
    "$repo_root/deployment/container/config/adflow.config.mk.in"
grep -q 'FF90 = @ENV_PREFIX@/bin/mpifort' \
    "$repo_root/deployment/container/config/pyhyp.config.mk.in"
grep -q 'SURROGATE_NEWTON_RUNTIME_ROOT=/runtime/tmp' \
    "$repo_root/deployment/container/Dockerfile"
grep -q 'CFD_RUNTIME_TMPDIR=/runtime/tmp/cfd' \
    "$repo_root/deployment/container/Dockerfile"
grep -q 'USER surrogate' "$repo_root/deployment/container/Dockerfile"
grep -q -- '-m pip check' "$repo_root/scripts/verify_solver_stack.sh"

"$python_bin" - "$lock_file" \
    "${expected_commits[pyhyp]}" \
    "${expected_commits[adflow]}" \
    "${expected_commits[cgns]}" \
    "${expected_commits[cgnsutilities]}" <<'PY'
from pathlib import Path
import sys

import yaml

path = Path(sys.argv[1])
pyhyp_commit, adflow_commit, cgns_commit, cgnsutilities_commit = sys.argv[2:]
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
    "cgnsutilities": {
        "repository": "https://example.invalid/cgnsutilities.git",
        "commit": cgnsutilities_commit,
        "bundle_ref": "refs/heads/main",
        "bundle_file": "cgnsutilities-test.bundle",
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
    --cgnsutilities "$source_root/cgnsutilities" \
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
    --cgnsutilities "$source_root/cgnsutilities" \
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

missing_release="$test_root/missing-release"
cp -a "$release_dir" "$missing_release"
rm "$missing_release/cgnsutilities-test.bundle"
expect_failure missing-bundle env \
    SURROGATE_NEWTON_PYTHON="$python_bin" \
    SURROGATE_NEWTON_SOLVER_LOCK="$lock_file" \
    "$repo_root/scripts/fetch_solver_stack.sh" \
    "$test_root/missing-fetch" \
    --source-mode bundle \
    --bundle-dir "$missing_release"
grep -q 'missing bundle for cgnsutilities' "$test_root/missing-bundle.log"
test ! -e "$test_root/missing-fetch/pyhyp"

wrong_size_release="$test_root/wrong-size-release"
cp -a "$release_dir" "$wrong_size_release"
printf X >> "$wrong_size_release/pyhyp-test.bundle"
expect_failure wrong-size-bundle env \
    SURROGATE_NEWTON_PYTHON="$python_bin" \
    SURROGATE_NEWTON_SOLVER_LOCK="$lock_file" \
    "$repo_root/scripts/fetch_solver_stack.sh" \
    "$test_root/wrong-size-fetch" \
    --source-mode bundle \
    --bundle-dir "$wrong_size_release"
grep -q 'bundle size mismatch for pyhyp' "$test_root/wrong-size-bundle.log"
test ! -e "$test_root/wrong-size-fetch/pyhyp"

env \
    SURROGATE_NEWTON_PYTHON="$python_bin" \
    SURROGATE_NEWTON_SOLVER_LOCK="$lock_file" \
    "$repo_root/scripts/fetch_solver_stack.sh" \
    "$test_root/restored" \
    --source-mode bundle \
    --bundle-dir "$release_dir"
for name in pyhyp adflow cgns cgnsutilities; do
    actual_commit=$(git -C "$test_root/restored/$name" rev-parse HEAD)
    if [[ "$actual_commit" != "${expected_commits[$name]}" ]]; then
        printf 'ERROR: %s restored at %s, expected %s.\n' \
            "$name" "$actual_commit" "${expected_commits[$name]}" >&2
        exit 1
    fi
done

wrapper_repo="$test_root/wrapper-repo"
mkdir -p "$wrapper_repo/scripts" "$wrapper_repo/deployment/container"
cp "$repo_root/scripts/build_restricted_server.sh" "$wrapper_repo/scripts/"
cp "$lock_file" "$wrapper_repo/solver-stack.lock.yaml"

miniforge_dir="$test_root/miniforge-input"
python_wheel_dir="$test_root/python-wheel-input"
mkdir -p "$miniforge_dir" "$python_wheel_dir"
printf 'miniforge fixture\n' > "$miniforge_dir/miniforge-test.sh"
printf 'torch fixture\n' > "$python_wheel_dir/torch-test.whl"
miniforge_size=$(stat --format='%s' "$miniforge_dir/miniforge-test.sh")
miniforge_sha256=$(sha256sum "$miniforge_dir/miniforge-test.sh" | cut -d' ' -f1)
torch_size=$(stat --format='%s' "$python_wheel_dir/torch-test.whl")
torch_sha256=$(sha256sum "$python_wheel_dir/torch-test.whl" | cut -d' ' -f1)
printf 'miniforge\tminiforge-test.sh\t%s\t%s\n' \
    "$miniforge_size" "$miniforge_sha256" \
    > "$wrapper_repo/deployment/container/offline-inputs.lock"
printf 'torch\ttorch-test.whl\t%s\t%s\n' "$torch_size" "$torch_sha256" \
    >> "$wrapper_repo/deployment/container/offline-inputs.lock"
git init --quiet --initial-branch=main "$wrapper_repo"
git -C "$wrapper_repo" config user.name "Surrogate Newton test"
git -C "$wrapper_repo" config user.email "test@example.invalid"
git -C "$wrapper_repo" add \
    scripts/build_restricted_server.sh \
    solver-stack.lock.yaml \
    deployment/container/offline-inputs.lock
git -C "$wrapper_repo" commit --quiet -m "test: add wrapper fixture"
wrapper_commit=$(git -C "$wrapper_repo" rev-parse HEAD)
wrapper="$wrapper_repo/scripts/build_restricted_server.sh"

expect_failure wrapper-floating-base \
    "$wrapper" \
    --bundle-dir "$release_dir" \
    --miniforge-dir "$miniforge_dir" \
    --python-wheel-dir "$python_wheel_dir" \
    --vcs-ref "$wrapper_commit" \
    --image-version test \
    --ubuntu-image ubuntu:24.04
grep -q 'must be pinned by a sha256 digest' "$test_root/wrapper-floating-base.log"

expect_failure wrapper-short-vcs \
    "$wrapper" \
    --bundle-dir "$release_dir" \
    --miniforge-dir "$miniforge_dir" \
    --python-wheel-dir "$python_wheel_dir" \
    --vcs-ref deadbeef \
    --image-version test
grep -q 'full 40-character lowercase commit SHA' "$test_root/wrapper-short-vcs.log"

mkdir -p "$test_root/incomplete-release"
expect_failure wrapper-incomplete-release \
    "$wrapper" \
    --bundle-dir "$test_root/incomplete-release" \
    --miniforge-dir "$miniforge_dir" \
    --python-wheel-dir "$python_wheel_dir" \
    --vcs-ref "$wrapper_commit" \
    --image-version test
grep -q 'must contain manifest.json and SHA256SUMS' \
    "$test_root/wrapper-incomplete-release.log"

mkdir -p "$test_root/missing-miniforge"
expect_failure wrapper-missing-miniforge \
    "$wrapper" \
    --bundle-dir "$release_dir" \
    --miniforge-dir "$test_root/missing-miniforge" \
    --python-wheel-dir "$python_wheel_dir" \
    --vcs-ref "$wrapper_commit" \
    --image-version test
grep -q 'missing miniforge input file' "$test_root/wrapper-missing-miniforge.log"

bad_miniforge_dir="$test_root/bad-miniforge"
cp -a "$miniforge_dir" "$bad_miniforge_dir"
printf X | dd \
    of="$bad_miniforge_dir/miniforge-test.sh" bs=1 seek=0 conv=notrunc status=none
expect_failure wrapper-bad-miniforge-sha \
    "$wrapper" \
    --bundle-dir "$release_dir" \
    --miniforge-dir "$bad_miniforge_dir" \
    --python-wheel-dir "$python_wheel_dir" \
    --vcs-ref "$wrapper_commit" \
    --image-version test
grep -q 'miniforge input SHA-256 mismatch' \
    "$test_root/wrapper-bad-miniforge-sha.log"

mkdir -p "$test_root/missing-torch"
expect_failure wrapper-missing-torch \
    "$wrapper" \
    --bundle-dir "$release_dir" \
    --miniforge-dir "$miniforge_dir" \
    --python-wheel-dir "$test_root/missing-torch" \
    --vcs-ref "$wrapper_commit" \
    --image-version test
grep -q 'missing torch input file' "$test_root/wrapper-missing-torch.log"

wrong_torch_name_dir="$test_root/wrong-torch-name"
mkdir -p "$wrong_torch_name_dir"
cp "$python_wheel_dir/torch-test.whl" "$wrong_torch_name_dir/not-torch.whl"
expect_failure wrapper-wrong-torch-name \
    "$wrapper" \
    --bundle-dir "$release_dir" \
    --miniforge-dir "$miniforge_dir" \
    --python-wheel-dir "$wrong_torch_name_dir" \
    --vcs-ref "$wrapper_commit" \
    --image-version test
grep -q 'missing torch input file: torch-test.whl' \
    "$test_root/wrapper-wrong-torch-name.log"

bad_torch_dir="$test_root/bad-torch"
cp -a "$python_wheel_dir" "$bad_torch_dir"
printf X | dd of="$bad_torch_dir/torch-test.whl" bs=1 seek=0 conv=notrunc status=none
expect_failure wrapper-bad-torch-sha \
    "$wrapper" \
    --bundle-dir "$release_dir" \
    --miniforge-dir "$miniforge_dir" \
    --python-wheel-dir "$bad_torch_dir" \
    --vcs-ref "$wrapper_commit" \
    --image-version test
grep -q 'torch input SHA-256 mismatch' "$test_root/wrapper-bad-torch-sha.log"

printf 'dirty\n' > "$wrapper_repo/dirty-marker"
expect_failure wrapper-dirty-worktree \
    "$wrapper" \
    --bundle-dir "$release_dir" \
    --miniforge-dir "$miniforge_dir" \
    --python-wheel-dir "$python_wheel_dir" \
    --vcs-ref "$wrapper_commit" \
    --image-version test
grep -q 'require a clean Git worktree' "$test_root/wrapper-dirty-worktree.log"

printf 'restricted-network script tests: PASS\n'
printf 'test artifacts: %s\n' "$test_root"
