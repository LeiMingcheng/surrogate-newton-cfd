#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bundle_dir=
miniforge_dir=
python_wheel_dir=
vcs_ref=
image_version=
image_ref=
ubuntu_image="public.ecr.aws/docker/library/ubuntu:24.04@sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea"
apt_mirror="http://mirrors.tuna.tsinghua.edu.cn/ubuntu"
pip_index_url="https://pypi.tuna.tsinghua.edu.cn/simple"
pip_timeout=120
pip_retries=10
build_log=
offline_input_lock="$repo_root/deployment/container/offline-inputs.lock"

usage() {
    cat >&2 <<'EOF'
Usage: scripts/build_restricted_server.sh \
  --bundle-dir PATH \
  --miniforge-dir PATH \
  --python-wheel-dir PATH \
  --vcs-ref FULL_COMMIT \
  --image-version VERSION \
  [--image IMAGE:TAG] \
  [--ubuntu-image NAME@sha256:DIGEST] \
  [--apt-mirror URL] \
  [--pip-index-url URL] \
  [--pip-timeout SECONDS] \
  [--pip-retries COUNT] \
  [--build-log PATH]
EOF
}

while (($#)); do
    case "$1" in
        --bundle-dir)
            bundle_dir=${2:?--bundle-dir requires a path}
            shift 2
            ;;
        --miniforge-dir)
            miniforge_dir=${2:?--miniforge-dir requires a path}
            shift 2
            ;;
        --python-wheel-dir)
            python_wheel_dir=${2:?--python-wheel-dir requires a path}
            shift 2
            ;;
        --vcs-ref)
            vcs_ref=${2:?--vcs-ref requires a full commit}
            shift 2
            ;;
        --image-version)
            image_version=${2:?--image-version requires a value}
            shift 2
            ;;
        --image)
            image_ref=${2:?--image requires a tagged image reference}
            shift 2
            ;;
        --ubuntu-image)
            ubuntu_image=${2:?--ubuntu-image requires a digest-pinned reference}
            shift 2
            ;;
        --apt-mirror)
            apt_mirror=${2:?--apt-mirror requires a URL}
            shift 2
            ;;
        --pip-index-url)
            pip_index_url=${2:?--pip-index-url requires a public URL}
            shift 2
            ;;
        --pip-timeout)
            pip_timeout=${2:?--pip-timeout requires seconds}
            shift 2
            ;;
        --pip-retries)
            pip_retries=${2:?--pip-retries requires a count}
            shift 2
            ;;
        --build-log)
            build_log=${2:?--build-log requires a path}
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'ERROR: unknown argument: %s\n' "$1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$bundle_dir" || -z "$miniforge_dir" || -z "$python_wheel_dir" \
    || -z "$vcs_ref" || -z "$image_version" ]]; then
    usage
    exit 1
fi
if [[ ! "$vcs_ref" =~ ^[0-9a-f]{40}$ ]]; then
    printf 'ERROR: --vcs-ref must be a full 40-character lowercase commit SHA.\n' >&2
    exit 1
fi
if [[ ! "$ubuntu_image" =~ ^[^[:space:]@]+@sha256:[0-9a-f]{64}$ ]]; then
    printf 'ERROR: --ubuntu-image must be pinned by a sha256 digest.\n' >&2
    exit 1
fi
if [[ "$image_version" =~ [[:space:]] ]]; then
    printf 'ERROR: --image-version cannot contain whitespace.\n' >&2
    exit 1
fi
if [[ ! "$apt_mirror" =~ ^https?://[A-Za-z0-9._:-]+(/[A-Za-z0-9._~/-]*)?$ ]]; then
    printf 'ERROR: --apt-mirror must be a credential-free public HTTP(S) URL.\n' >&2
    exit 1
fi
if [[ ! "$pip_index_url" =~ ^https?://[A-Za-z0-9._:-]+(/[A-Za-z0-9._~/-]*)?$ ]]; then
    printf 'ERROR: --pip-index-url must be a credential-free public HTTP(S) URL.\n' >&2
    exit 1
fi
if [[ ! "$pip_timeout" =~ ^[1-9][0-9]*$ || ! "$pip_retries" =~ ^[0-9]+$ ]]; then
    printf 'ERROR: pip timeout must be positive and retries must be non-negative integers.\n' >&2
    exit 1
fi

actual_vcs_ref=$(git -C "$repo_root" rev-parse HEAD)
if [[ "$actual_vcs_ref" != "$vcs_ref" ]]; then
    printf 'ERROR: current checkout is %s, but --vcs-ref is %s.\n' \
        "$actual_vcs_ref" "$vcs_ref" >&2
    exit 1
fi
if [[ -n $(git -C "$repo_root" status --porcelain --untracked-files=normal) ]]; then
    printf 'ERROR: formal server builds require a clean Git worktree.\n' >&2
    exit 1
fi

bundle_dir=$(realpath "$bundle_dir")
if [[ ! -f "$bundle_dir/manifest.json" || ! -f "$bundle_dir/SHA256SUMS" ]]; then
    printf 'ERROR: bundle directory must contain manifest.json and SHA256SUMS.\n' >&2
    exit 1
fi

mapfile -t bundle_files < <(awk '$1 == "bundle_file:" {print $2}' "$repo_root/solver-stack.lock.yaml")
if ((${#bundle_files[@]} != 4)); then
    printf 'ERROR: solver-stack.lock.yaml must name exactly four bundle files.\n' >&2
    exit 1
fi

declare -A expected_files=( [manifest.json]=1 )
for bundle_file in "${bundle_files[@]}"; do
    if [[ "$bundle_file" == */* || "$bundle_file" != *.bundle ]]; then
        printf 'ERROR: unsafe bundle filename in solver lock: %s\n' "$bundle_file" >&2
        exit 1
    fi
    expected_files["$bundle_file"]=1
done

validate_offline_input() {
    local input_name=$1
    local input_dir=$2
    local record filename expected_size expected_sha256 extra actual_size
    if [[ ! -f "$offline_input_lock" ]]; then
        printf 'ERROR: missing offline input lock: %s\n' "$offline_input_lock" >&2
        exit 1
    fi
    record=$(awk -F '\t' -v name="$input_name" '$1 == name {print}' "$offline_input_lock")
    IFS=$'\t' read -r _ filename expected_size expected_sha256 extra <<<"$record"
    if [[ -z "$filename" || "$filename" == */* || -n "${extra:-}" \
        || ! "$expected_size" =~ ^[1-9][0-9]*$ \
        || ! "$expected_sha256" =~ ^[0-9a-f]{64}$ ]]; then
        printf 'ERROR: invalid %s record in offline-inputs.lock.\n' "$input_name" >&2
        exit 1
    fi
    if [[ ! -d "$input_dir" ]]; then
        printf 'ERROR: missing %s input directory: %s\n' "$input_name" "$input_dir" >&2
        exit 1
    fi
    input_dir=$(realpath "$input_dir")
    if [[ ! -f "$input_dir/$filename" ]]; then
        printf 'ERROR: missing %s input file: %s\n' "$input_name" "$filename" >&2
        exit 1
    fi
    actual_size=$(stat --format='%s' "$input_dir/$filename")
    if [[ "$actual_size" != "$expected_size" ]]; then
        printf 'ERROR: %s input size mismatch.\n' "$input_name" >&2
        exit 1
    fi
    if ! printf '%s  %s\n' "$expected_sha256" "$input_dir/$filename" \
        | sha256sum --check --status -; then
        printf 'ERROR: %s input SHA-256 mismatch.\n' "$input_name" >&2
        exit 1
    fi
    printf '%s\t%s\t%s\n' "$input_dir" "$filename" "$expected_sha256"
}

miniforge_record=$(validate_offline_input miniforge "$miniforge_dir")
IFS=$'\t' read -r miniforge_dir miniforge_filename miniforge_sha256 \
    <<<"$miniforge_record"
torch_record=$(validate_offline_input torch "$python_wheel_dir")
IFS=$'\t' read -r python_wheel_dir torch_filename torch_sha256 <<<"$torch_record"

declare -A recorded_files=()
while read -r sha256 filename extra; do
    if [[ -n "${extra:-}" || ! "$sha256" =~ ^[0-9a-f]{64}$ \
        || -z "${expected_files[${filename:-}]:-}" \
        || -n "${recorded_files[${filename:-}]:-}" ]]; then
        printf 'ERROR: invalid or unexpected SHA256SUMS entry.\n' >&2
        exit 1
    fi
    recorded_files["$filename"]=1
done < "$bundle_dir/SHA256SUMS"
if ((${#recorded_files[@]} != ${#expected_files[@]})); then
    printf 'ERROR: SHA256SUMS must cover exactly manifest.json and all locked bundles.\n' >&2
    exit 1
fi

(
    cd "$bundle_dir"
    sha256sum --check --strict SHA256SUMS
)
for bundle_file in "${bundle_files[@]}"; do
    git -C "$repo_root" bundle verify "$bundle_dir/$bundle_file"
done

image_ref=${image_ref:-"surrogate-newton-cfd-runtime:$image_version"}
build_log=${build_log:-"$repo_root/build-logs/restricted-$image_version.build.log"}
mkdir -p "$(dirname "$build_log")"

docker buildx version
printf 'Building %s from %s...\n' "$image_ref" "$vcs_ref"
docker buildx build \
    --load \
    --network=host \
    --progress=plain \
    --build-context "solver-bundles=$bundle_dir" \
    --build-context "miniforge-installer=$miniforge_dir" \
    --build-context "python-wheels-cu128=$python_wheel_dir" \
    --build-arg "UBUNTU_IMAGE=$ubuntu_image" \
    --build-arg "UBUNTU_APT_MIRROR=$apt_mirror" \
    --build-arg SOLVER_SOURCE_MODE=bundle \
    --build-arg MINIFORGE_SOURCE_MODE=bundle \
    --build-arg PYTHON_WHEEL_SOURCE_MODE=bundle \
    --build-arg "MINIFORGE_FILENAME=$miniforge_filename" \
    --build-arg "MINIFORGE_SHA256=$miniforge_sha256" \
    --build-arg "TORCH_WHEEL_FILENAME=$torch_filename" \
    --build-arg "TORCH_WHEEL_SHA256=$torch_sha256" \
    --build-arg "PIP_INDEX_URL=$pip_index_url" \
    --build-arg "PIP_DEFAULT_TIMEOUT=$pip_timeout" \
    --build-arg "PIP_RETRIES=$pip_retries" \
    --build-arg "VCS_REF=$vcs_ref" \
    --build-arg "IMAGE_VERSION=$image_version" \
    --tag "$image_ref" \
    --file "$repo_root/deployment/container/Dockerfile" \
    "$repo_root" 2>&1 | tee "$build_log"

image_id=$(docker image inspect --format '{{.Id}}' "$image_ref")
repo_digests=$(docker image inspect --format '{{json .RepoDigests}}' "$image_ref")
printf '%s\n' \
    "Image: $image_ref" \
    "Image ID/local digest: $image_id" \
    "RepoDigests: $repo_digests" \
    "Source commit: $vcs_ref" \
    "Ubuntu base: $ubuntu_image" \
    "Build log: $build_log"
