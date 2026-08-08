#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
stack_dir=${1:?Usage: scripts/build_solver_stack.sh /path/to/solver-stack}
env_prefix=${SURROGATE_NEWTON_ENV_PREFIX:-${CONDA_PREFIX:-}}
cgns_prefix=${SURROGATE_NEWTON_CGNS_PREFIX:-"$stack_dir/install/cgns"}
jobs=${SURROGATE_NEWTON_BUILD_JOBS:-4}
lock_file=${SURROGATE_NEWTON_SOLVER_LOCK:-"$repo_root/solver-stack.lock.yaml"}

if [[ -z "$env_prefix" ]]; then
    printf 'ERROR: set SURROGATE_NEWTON_ENV_PREFIX or activate the target Conda environment.\n' >&2
    exit 1
fi
if [[ ! -x "$env_prefix/bin/python" || ! -x "$env_prefix/bin/mpifort" ]]; then
    printf 'ERROR: %s does not contain the required Python and MPICH compiler wrappers.\n' \
        "$env_prefix" >&2
    exit 1
fi

locked_commit_output=$(
    "$env_prefix/bin/python" - "$lock_file" <<'PY'
from pathlib import Path
import sys

import yaml

lock = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"pyhyp\t{lock['pyhyp']['fork_commit']}")
print(f"adflow\t{lock['adflow']['fork_commit']}")
print(f"cgns\t{lock['cgns']['commit']}")
PY
)
mapfile -t locked_commits <<<"$locked_commit_output"
for record in "${locked_commits[@]}"; do
    IFS=$'\t' read -r name expected_commit <<<"$record"
    actual_commit=$(git -C "$stack_dir/$name" rev-parse HEAD)
    if [[ "$actual_commit" != "$expected_commit" ]]; then
        printf 'ERROR: %s is at %s, expected %s.\n' \
            "$name" "$actual_commit" "$expected_commit" >&2
        exit 1
    fi
done

render_config() {
    input=$1
    output=$2
    "$env_prefix/bin/python" - "$input" "$output" "$env_prefix" "$cgns_prefix" "$jobs" <<'PY'
from pathlib import Path
import sys

source, destination, env_prefix, cgns_prefix, jobs = sys.argv[1:]
text = Path(source).read_text(encoding="utf-8")
text = text.replace("@ENV_PREFIX@", env_prefix)
text = text.replace("@CGNS_PREFIX@", cgns_prefix)
text = text.replace("@JOBS@", jobs)
Path(destination).write_text(text, encoding="utf-8")
PY
}

mkdir -p "$stack_dir/logs" "$stack_dir/build/cgns" "$cgns_prefix"

printf 'Building CGNS at %s...\n' "$(git -C "$stack_dir/cgns" rev-parse HEAD)"
cmake \
    -S "$stack_dir/cgns" \
    -B "$stack_dir/build/cgns" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$cgns_prefix" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_PREFIX_PATH="$env_prefix" \
    -DCMAKE_C_COMPILER="$env_prefix/bin/mpicc" \
    -DCMAKE_Fortran_COMPILER="$env_prefix/bin/mpifort" \
    -DBUILD_SHARED_LIBS=ON \
    -DCGNS_ENABLE_64BIT=ON \
    -DCGNS_ENABLE_FORTRAN=ON \
    -DCGNS_ENABLE_HDF5=ON \
    -DHDF5_PREFER_PARALLEL=ON \
    2>&1 | tee "$stack_dir/logs/cgns-configure.log"
cmake --build "$stack_dir/build/cgns" --parallel "$jobs" \
    2>&1 | tee "$stack_dir/logs/cgns-build.log"
cmake --install "$stack_dir/build/cgns" \
    2>&1 | tee "$stack_dir/logs/cgns-install.log"

render_config \
    "$repo_root/deployment/container/config/pyhyp.config.mk.in" \
    "$stack_dir/pyhyp/config/config.mk"
render_config \
    "$repo_root/deployment/container/config/adflow.config.mk.in" \
    "$stack_dir/adflow/config/config.mk"

export PETSC_DIR="$env_prefix"
export PETSC_ARCH=
export CGNS_HOME="$cgns_prefix"
export PATH="$env_prefix/bin:$PATH"
export LD_LIBRARY_PATH="$cgns_prefix/lib:$env_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

printf 'Building pyHyp at %s...\n' "$(git -C "$stack_dir/pyhyp" rev-parse HEAD)"
make -C "$stack_dir/pyhyp" 2>&1 | tee "$stack_dir/logs/pyhyp-build.log"
test -f "$stack_dir/pyhyp/pyhyp/hyp.so"

printf 'Building ADflow at %s...\n' "$(git -C "$stack_dir/adflow" rev-parse HEAD)"
make -C "$stack_dir/adflow" 2>&1 | tee "$stack_dir/logs/adflow-build.log"
test -f "$stack_dir/adflow/adflow/libadflow.so"

printf 'Solver stack build complete. Logs: %s\n' "$stack_dir/logs"
