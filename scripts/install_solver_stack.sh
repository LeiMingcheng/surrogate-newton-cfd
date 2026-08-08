#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
stack_dir=${1:?Usage: scripts/install_solver_stack.sh /path/to/solver-stack}

printf '%s\n' \
    'install_solver_stack.sh is retained as a fetch-only compatibility alias.' \
    'Use fetch_solver_stack.sh, build_solver_stack.sh, and verify_solver_stack.sh explicitly.'
exec "$repo_root/scripts/fetch_solver_stack.sh" "$stack_dir"
