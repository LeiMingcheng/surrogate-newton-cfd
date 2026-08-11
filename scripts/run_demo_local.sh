#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)

python_bin=${SURROGATE_NEWTON_PYTHON:-${DEMO_PYTHON:-}}
if [[ -z "$python_bin" ]]; then
    python_bin=$(command -v python)
fi
if [[ ! -x "$python_bin" ]]; then
    echo "The active Python interpreter is not executable: $python_bin" >&2
    exit 1
fi

web_host=${DEMO_WEB_HOST:-127.0.0.1}
web_port=${DEMO_WEB_PORT:-8080}
surrogate_host=${DEMO_SURROGATE_HOST:-127.0.0.1}
surrogate_port=${DEMO_SURROGATE_PORT:-65432}
gpu_id=${DEMO_GPU_ID:-0}
mpi_ranks=${DEMO_MPI_RANKS:-8}
mpi_launcher=${DEMO_MPI_LAUNCHER:-auto}
prewarm=${DEMO_PREWARM:-1}
start_surrogate=${DEMO_START_SURROGATE:-1}
compute_residuals=${DEMO_COMPUTE_RESIDUALS:-1}

if [[ "$web_host" != "127.0.0.1" && "$web_host" != "localhost" ]]; then
    echo "DEMO_WEB_HOST must remain on the loopback interface." >&2
    exit 1
fi
if [[ "$surrogate_host" != "127.0.0.1" && "$surrogate_host" != "localhost" ]]; then
    echo "DEMO_SURROGATE_HOST must remain on the loopback interface." >&2
    exit 1
fi

runtime_base=${SURROGATE_NEWTON_RUNTIME_DIR:-${SURROGATE_NEWTON_RUNTIME_ROOT:-}}
if [[ -z "$runtime_base" ]]; then
    runtime_base=${TMPDIR:-/tmp}/surrogate-newton-cfd-runtime
fi
runtime_root=${DEMO_RUNTIME_ROOT:-$runtime_base/demo}
model_dir=${SURROGATE_NEWTON_MODEL_DIR:-$repo_root/artifacts}
airfoil_root=${DEMO_AIRFOIL_LIBRARY_ROOT:-$repo_root/demo/airfoils/uiuc}
model_config_source=$repo_root/surrogate/configs/training/fsb_dit.yaml

mapfile -t model_names < <(
    "$python_bin" -c \
        'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); print(p["model"]["filename"]); print(p["normalization_statistics"]["filename"])' \
        "$repo_root/model-manifest.json"
)
checkpoint=${DEMO_CHECKPOINT:-$model_dir/${model_names[0]}}
statistics=${DEMO_STATISTICS:-$model_dir/${model_names[1]}}

mkdir -p "$runtime_root/logs" "$runtime_root/meshes" "$runtime_root/cases"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export SURROGATE_NEWTON_RUNTIME_ROOT="$runtime_root"
export DEMO_RUNTIME_ROOT="$runtime_root"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

surrogate_pid=""
web_pid=""

cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [[ -n "$web_pid" ]] && kill -0 "$web_pid" 2>/dev/null; then
        kill -TERM "$web_pid"
        wait "$web_pid" 2>/dev/null || true
    fi
    if [[ -n "$surrogate_pid" ]] && kill -0 "$surrogate_pid" 2>/dev/null; then
        kill -TERM "$surrogate_pid"
        wait "$surrogate_pid" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

runtime_model_config=$model_config_source
if [[ "$start_surrogate" == "1" ]]; then
    if [[ ! -f "$checkpoint" || ! -f "$statistics" ]]; then
        echo "The released checkpoint and statistics must exist under SURROGATE_NEWTON_MODEL_DIR." >&2
        exit 1
    fi
    runtime_model_config=$(
        "$python_bin" -c \
            'from pathlib import Path; import sys; from deployment.run import _runtime_model_config; print(_runtime_model_config(Path(sys.argv[1]), checkpoint=Path(sys.argv[2]), stats=Path(sys.argv[3]), output_dir=Path(sys.argv[4])))' \
            "$model_config_source" "$checkpoint" "$statistics" "$runtime_root"
    )
    residual_args=()
    if [[ "$compute_residuals" == "1" ]]; then
        residual_args=(--compute-residuals --residual-only-momentum)
    fi
    CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" -m surrogate.serving.cli \
        --config "$runtime_model_config" \
        --checkpoint "$checkpoint" \
        --host "$surrogate_host" \
        --port "$surrogate_port" \
        --device cuda:0 \
        --authority-cgns-dir "$runtime_root/meshes" \
        --max-batch-size 4 \
        --request-timeout-s 240 \
        "${residual_args[@]}" \
        >"$runtime_root/logs/surrogate.log" 2>&1 &
    surrogate_pid=$!

    ready=0
    for _ in $(seq 1 180); do
        if "$python_bin" -c \
            'from surrogate.serving.client import SurrogateClient,SurrogateClientConfig; import sys; SurrogateClient(SurrogateClientConfig(host=sys.argv[1],port=int(sys.argv[2]),timeout_s=2)).ping()' \
            "$surrogate_host" "$surrogate_port" >/dev/null 2>&1; then
            ready=1
            break
        fi
        if ! kill -0 "$surrogate_pid" 2>/dev/null; then
            echo "The Surrogate service exited during startup; inspect the runtime log." >&2
            exit 1
        fi
        sleep 1
    done
    if [[ "$ready" != "1" ]]; then
        echo "The Surrogate service did not become ready within 180 seconds." >&2
        exit 1
    fi
fi

prewarm_args=()
if [[ "$prewarm" != "1" ]]; then
    prewarm_args=(--skip-prewarm)
fi

echo "Starting loopback demo at http://$web_host:$web_port/"
echo "MPI ranks per solver case: $mpi_ranks"
"$python_bin" -m demo.server \
    --host "$web_host" \
    --port "$web_port" \
    --surrogate-host "$surrogate_host" \
    --surrogate-port "$surrogate_port" \
    --runtime-root "$runtime_root" \
    --airfoil-library-root "$airfoil_root" \
    --mpi-launcher "$mpi_launcher" \
    --mpi-ranks "$mpi_ranks" \
    --model-dir "$model_dir" \
    --model-config "$runtime_model_config" \
    --checkpoint "$checkpoint" \
    --statistics "$statistics" \
    --device cuda:0 \
    "${prewarm_args[@]}" &
web_pid=$!
wait "$web_pid"
