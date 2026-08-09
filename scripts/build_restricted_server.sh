#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bundle_dir=
vcs_ref=
image_version=
image_ref=
ubuntu_image="public.ecr.aws/docker/library/ubuntu:24.04@sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea"
apt_mirror="http://mirrors.tuna.tsinghua.edu.cn/ubuntu"
build_log=

usage() {
    cat >&2 <<'EOF'
Usage: scripts/build_restricted_server.sh \
  --bundle-dir PATH \
  --vcs-ref FULL_COMMIT \
  --image-version VERSION \
  [--image IMAGE:TAG] \
  [--ubuntu-image NAME@sha256:DIGEST] \
  [--apt-mirror URL] \
  [--build-log PATH]
EOF
}

while (($#)); do
    case "$1" in
        --bundle-dir)
            bundle_dir=${2:?--bundle-dir requires a path}
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

if [[ -z "$bundle_dir" || -z "$vcs_ref" || -z "$image_version" ]]; then
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
if ((${#bundle_files[@]} != 3)); then
    printf 'ERROR: solver-stack.lock.yaml must name exactly three bundle files.\n' >&2
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
    --build-arg "UBUNTU_IMAGE=$ubuntu_image" \
    --build-arg "UBUNTU_APT_MIRROR=$apt_mirror" \
    --build-arg SOLVER_SOURCE_MODE=bundle \
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
