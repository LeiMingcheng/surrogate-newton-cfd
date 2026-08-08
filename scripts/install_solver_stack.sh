#!/usr/bin/env bash
set -euo pipefail

stack_dir=${1:?Usage: scripts/install_solver_stack.sh /path/to/workspace}

mkdir -p "$stack_dir"

git clone https://github.com/LeiMingcheng/pyhyp.git "$stack_dir/pyhyp"
git -C "$stack_dir/pyhyp" checkout --detach 04af13e4e59d4a113d7def96b6b5b2dbf6fa9ed9

git clone https://github.com/LeiMingcheng/adflow.git "$stack_dir/adflow"
git -C "$stack_dir/adflow" checkout --detach 4ad0091910d1d885f1f5ddcf41b96b5950fa16c9

git -C "$stack_dir/pyhyp" rev-parse HEAD
git -C "$stack_dir/adflow" rev-parse HEAD

printf '%s\n' \
    'Exact solver sources are ready.' \
    'Configure CGNS, PETSc, MPI, and compiler paths before building.' \
    'Then follow docs/solver_stack.md from the Surrogate-Newton CFD repository.'
