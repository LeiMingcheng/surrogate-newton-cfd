#!/usr/bin/env bash
set -euo pipefail

repo_root=/opt/surrogate-newton/src
pyhyp_root=/opt/solver-stack/pyhyp
adflow_root=/opt/solver-stack/adflow
cgnsutilities_root=/opt/solver-stack/cgnsutilities
model_dir=${SURROGATE_NEWTON_MODEL_DIR:-/models}
runtime_dir=${SURROGATE_NEWTON_RUNTIME_DIR:-/runtime}
checkpoint="$model_dir/fsb-dit-airfoil-2608.04400-inference.pt"
stats="$model_dir/turbulent-scale-stats.json"
result_dir="$runtime_dir/rae2822"

case "${1:-}" in
    source-smoke)
        exec python "$repo_root/deployment/smoke_check.py" \
            --level source \
            --pyhyp-root "$pyhyp_root" \
            --adflow-root "$adflow_root" \
            --cgnsutilities-root "$cgnsutilities_root"
        ;;
    runtime-smoke)
        exec python "$repo_root/deployment/smoke_check.py" \
            --level runtime \
            --pyhyp-root "$pyhyp_root" \
            --adflow-root "$adflow_root" \
            --cgnsutilities-root "$cgnsutilities_root" \
            --checkpoint "$checkpoint" \
            --stats "$stats"
        ;;
    rae2822-acceptance)
        python "$repo_root/deployment/run.py" \
            --checkpoint "$checkpoint" \
            --stats "$stats" \
            --output-dir "$result_dir"
        python "$repo_root/deployment/smoke_check.py" \
            --level result \
            --pyhyp-root "$pyhyp_root" \
            --adflow-root "$adflow_root" \
            --cgnsutilities-root "$cgnsutilities_root" \
            --checkpoint "$checkpoint" \
            --stats "$stats" \
            --result-dir "$result_dir"
        exec python "$repo_root/deployment/compare_acceptance.py" \
            --result-dir "$result_dir" \
            --baseline "$repo_root/deployment/acceptance/rae2822-baseline.json"
        ;;
    "")
        exec python "$repo_root/deployment/smoke_check.py" \
            --level source \
            --pyhyp-root "$pyhyp_root" \
            --adflow-root "$adflow_root" \
            --cgnsutilities-root "$cgnsutilities_root"
        ;;
    *)
        exec "$@"
        ;;
esac
