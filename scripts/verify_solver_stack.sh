#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
stack_dir=${1:?Usage: scripts/verify_solver_stack.sh /path/to/solver-stack}
env_prefix=${SURROGATE_NEWTON_ENV_PREFIX:-${CONDA_PREFIX:-}}
cgns_prefix=${SURROGATE_NEWTON_CGNS_PREFIX:-"$stack_dir/install/cgns"}
lock_file=${SURROGATE_NEWTON_SOLVER_LOCK:-"$repo_root/solver-stack.lock.yaml"}

if [[ -z "$env_prefix" ]]; then
    printf 'ERROR: set SURROGATE_NEWTON_ENV_PREFIX or activate the target Conda environment.\n' >&2
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

python_bin="$env_prefix/bin/python"
mpiexec_bin="$env_prefix/bin/mpiexec"
pyhyp_library="$stack_dir/pyhyp/pyhyp/hyp.so"
adflow_library="$stack_dir/adflow/adflow/libadflow.so"
export PATH="$env_prefix/bin:$PATH"
export LD_LIBRARY_PATH="$cgns_prefix/lib:$env_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$stack_dir/pyhyp:$stack_dir/adflow:$repo_root${PYTHONPATH:+:$PYTHONPATH}"

for library in "$pyhyp_library" "$adflow_library"; do
    test -f "$library"
    dependencies=$(ldd "$library")
    if grep -q 'not found' <<<"$dependencies"; then
        printf 'ERROR: unresolved dynamic dependency in %s:\n%s\n' "$library" "$dependencies" >&2
        exit 1
    fi
    mpi_line=$(grep -m1 'libmpi\.so' <<<"$dependencies" || true)
    if [[ "$mpi_line" != *"$env_prefix/"* ]]; then
        printf 'ERROR: %s does not resolve MPI from %s.\n' "$library" "$env_prefix" >&2
        exit 1
    fi
    cgns_line=$(grep -m1 'libcgns\.so' <<<"$dependencies" || true)
    if [[ "$cgns_line" != *"$cgns_prefix/"* ]]; then
        printf 'ERROR: %s does not resolve CGNS from %s.\n' "$library" "$cgns_prefix" >&2
        exit 1
    fi
done

petsc_line=$(ldd "$adflow_library" | grep -m1 'libpetsc\.so' || true)
if [[ "$petsc_line" != *"$env_prefix/"* ]]; then
    printf 'ERROR: ADflow does not resolve PETSc from %s.\n' "$env_prefix" >&2
    exit 1
fi

if [[ $(readlink -f "$mpiexec_bin") != "$env_prefix/"* ]]; then
    printf 'ERROR: MPI launcher is not owned by the target environment.\n' >&2
    exit 1
fi

"$python_bin" - <<'PY'
from adflow import libadflow
from pyhyp import pyHyp

assert hasattr(pyHyp, "resetForNewSurface")
assert hasattr(libadflow.initializeflow, "reinitafterinjection")
assert hasattr(libadflow.initializeflow, "rebuildrestartderivedstateaftersetinfo")
print("compiled solver hooks: ok")
PY

"$mpiexec_bin" -n 2 "$python_bin" -c \
    'from mpi4py import MPI; assert MPI.COMM_WORLD.Get_size() == 2'

"$python_bin" - <<'PY'
import json
import sys

from mpi4py import MPI

print(json.dumps({
    "python": sys.version.split()[0],
    "mpi_vendor": MPI.get_vendor(),
}, sort_keys=True))
PY
"$mpiexec_bin" --version
printf 'Solver stack verification complete.\n'
