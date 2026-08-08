#!/usr/bin/env bash
set -euo pipefail

artifact_dir=${1:-artifacts}
base_url=${SURROGATE_NEWTON_ASSET_BASE_URL:-https://github.com/LeiMingcheng/surrogate-newton-cfd/releases/download/model-2608.04400}

mkdir -p "$artifact_dir"

download_and_verify() {
    filename=$1
    expected_sha256=$2
    target="$artifact_dir/$filename"
    curl --fail --location --output "$target" "$base_url/$filename"
    printf '%s  %s\n' "$expected_sha256" "$target" | sha256sum --check -
}

download_and_verify \
    fsb-dit-airfoil-2608.04400.pt \
    0b8be8a31cc972fb817f46369c1d39efd39b703f307ba28eb85be388c7b2d942
download_and_verify \
    turbulent-scale-stats.json \
    2c5589d891aac5d7b0ce621b4c4b044da81e630bae720515f0831555d440ef97
