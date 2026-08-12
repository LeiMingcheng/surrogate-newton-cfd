# Container build and remote acceptance

This directory defines the first-class, single-node NVIDIA container route.
It deliberately does not replace MDO Lab's native/HPC installation guidance,
and it does not claim multi-node MPI support.

## Locked baseline

- Ubuntu 24.04 build/runtime base (pass a digest through `UBUNTU_IMAGE` for a
  formal release build);
- Python 3.10.18;
- PyTorch 2.8.0+cu128 from the official CUDA 12.8 wheel, including its wheel
  SHA-256 in the environment file;
- MPICH 4.3.1, mpi4py 4.1.0, PETSc 3.19.6, and HDF5 1.14.3 from conda-forge;
- CGNS and cgnsutilities from their exact upstream commits; and
- the exact pyHyp and ADflow fork commits in the same lock file.

The build uses one environment prefix for MPICH, mpi4py, PETSc, the compiler
wrappers, and both Fortran extensions. The ADflow config intentionally omits
`-march=native`. The runtime stage contains neither compilers, tests, build
caches, nor Git metadata. It retains the license-bearing Python runtime trees,
the ADflow f2py interface required for runtime selection, and exact revision
files under `/opt/solver-stack`; complete corresponding sources remain at the
locked public repository URLs.

PETSc 3.19.6's archived conda-forge dependency set requires zlib 1.2.13, while
the validated Python 3.10.18 build comes from the `defaults` channel. Both are
selected by exact package/build constraints; Conda channel priority is flexible
only to permit this documented hybrid, not to choose the MPI or PETSc variant.

## Before the first remote build

The public pyHyp and ADflow forks and the upstream CGNS/cgnsutilities
repositories expose the commits recorded in `solver-stack.lock.yaml`. The
model release is independent: download the paired inference checkpoint and
statistics into one host directory, with the filenames and digests from
`model-manifest.json`:

```bash
scripts/download_checkpoint.sh /absolute/path/to/model-release
```

The remote host needs a working NVIDIA driver, Docker Engine, Compose, and the
NVIDIA Container Toolkit. Keep at least 100 GB free for build layers, Conda and
PyTorch packages, the 362 MB inference checkpoint, runtime data, and one
rollback image.

Keep the model release outside the runtime image and mount it read-only at
`/models`. This avoids duplicating weights in image layers, allows code and
model rollbacks to remain independent, and supports digest verification before
container startup. For a genuinely disconnected one-file demonstration, build
a separately tagged derived image that copies the already verified artifact
pair; do not replace the normal external-mount release with that image.

## Build

Run from the repository root on the remote server:

```bash
export VCS_REF=$(git rev-parse HEAD)
docker build \
  --build-arg VCS_REF="$VCS_REF" \
  --build-arg IMAGE_VERSION=0.1.0-rc1 \
  --tag surrogate-newton-cfd-runtime:0.1.0-rc1 \
  --file deployment/container/Dockerfile \
  .
```

For a release build, resolve `ubuntu:24.04` to an approved immutable digest and
pass the full reference with `--build-arg UBUNTU_IMAGE=ubuntu@sha256:...`.
Record that reference, the final image digest, and `VCS_REF` in the acceptance
report. Do not publish `latest` as the production reference.

## Restricted-network server build

When the server can reach package mirrors but GitHub source and large-file
transfer is slow or unavailable, package the four locked solver repositories
on aerolab3:

```bash
scripts/package_solver_bundles.sh \
  --pyhyp /root/shared-nvme/public-repos/pyhyp \
  --adflow /root/shared-nvme/public-repos/adflow \
  --cgns /root/shared-nvme/build_libraries/CGNS \
  --cgnsutilities /root/shared-nvme/build_libraries/cgnsutilities \
  --output /root/shared-nvme/release-inputs/solver-bundles-2608.04400-v2

cd /root/shared-nvme/release-inputs/solver-bundles-2608.04400-v2
sha256sum --check SHA256SUMS
```

The formal large inputs are kept outside Git:

```text
/root/shared-nvme/release-inputs/miniforge-26.3.2-2/
  Miniforge3-26.3.2-2-Linux-x86_64.sh
/root/shared-nvme/release-inputs/python-wheels-cu128-2.8.0/
  torch-2.8.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl
```

Their filenames, byte sizes, and SHA-256 values are pinned in
`offline-inputs.lock`. Do not copy either file into the repository.

Transfer the clean `surrogate-newton-cfd` checkout and the complete bundle
directory through the local machine. For example, run equivalent `rsync -a`
commands from the relay machine to place them under the server's data disk;
keep the bundle directory outside the Git checkout.

On the server, build from the exact checkout. The wrapper validates the Git
revision, release checksums and bundles, selects the Public ECR Ubuntu 24.04
digest below, uses the Tsinghua Ubuntu HTTP mirror for the first APT bootstrap,
and invokes Buildx with host networking and plain logs:

```bash
cd /data/build/surrogate-newton-cfd
VCS_REF=$(git rev-parse HEAD)
scripts/build_restricted_server.sh \
  --bundle-dir /data/build/solver-bundles-2608.04400-v2 \
  --miniforge-dir /data/build/miniforge-26.3.2-2 \
  --python-wheel-dir /data/build/python-wheels-cu128-2.8.0 \
  --vcs-ref "$VCS_REF" \
  --image-version 0.1.0-rc1 \
  --pip-index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

The formal base is
`public.ecr.aws/docker/library/ubuntu:24.04@sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea`.
The wrapper defaults to `http://mirrors.tuna.tsinghua.edu.cn/ubuntu` because the
fixed base does not yet have system CA certificates when its first
`apt-get update` runs. Ubuntu archive signatures remain enabled; do not add
`--allow-unauthenticated` or disable signature verification. Override the
mirror with `--apt-mirror` when another bootstrap endpoint is required.
The four Git bundles, Miniforge installer, and Torch wheel are exposed as
separate read-only BuildKit named contexts. Solver restoration runs without a
network. The installer and wheel are verified before use and none of these
release inputs is copied into the runtime image. Remaining Conda and pip
packages still come from configured public mirrors.

After the build, run the compiled and model-pair smoke check:

```bash
docker run --rm --gpus all \
  --volume /absolute/path/to/model-release:/models:ro \
  surrogate-newton-cfd-runtime:0.1.0-rc1 \
  runtime-smoke
```

Then run Compose against that already-built image. `--no-build` is required so
Compose cannot fall back to the default online Git source stage:

```bash
cd deployment/container
MODEL_DIR=/absolute/path/to/model-release \
VCS_REF=$(git -C ../.. rev-parse HEAD) \
RUNTIME_IMAGE=surrogate-newton-cfd-runtime:0.1.0-rc1 \
docker compose -f compose.smoke.yaml up --no-build --abort-on-container-exit
```

The pip mirror must be a credential-free public HTTP(S) URL; timeout and retry
counts can be changed with `--pip-timeout` and `--pip-retries`. An HTTP(S)
proxy may be configured outside the repository as an optional
transfer accelerator. It is not a build dependency; never place proxy URLs
containing credentials in this repository, Dockerfile, build arguments, or
image layers.

The default container command runs the source-level smoke check. The following
also checks the compiled hooks and model pair:

```bash
docker run --rm --gpus all \
  --volume /absolute/path/to/model-release:/models:ro \
  surrogate-newton-cfd-runtime:0.1.0-rc1 \
  runtime-smoke
```

The container runs as UID 10001. Its writable roots are `/runtime`,
`/runtime/tmp`, and `/runtime/tmp/cfd`; the source tree remains read-only to the
runtime user. When `/runtime` is a bind mount rather than the supplied named
volume, give UID 10001 write permission on the host directory before launch.

## Internal interactive demo

A runtime image rebuilt from the demo integration commit contains the
`surrogate-newton-demo` entry point and `scripts/run_demo_local.sh`. Keep both
HTTP services on loopback by using Linux host networking rather than publishing
a container port:

```bash
docker run --rm --gpus all --network host \
  --name surrogate-newton-demo \
  --volume /absolute/path/to/model-release:/models:ro \
  --volume /absolute/path/to/writable-runtime:/runtime \
  --volume /absolute/path/to/demo-assets-uiuc:/demo-assets:ro \
  --env DEMO_AIRFOIL_LIBRARY_ROOT=/demo-assets/uiuc \
  --env DEMO_WEB_HOST=127.0.0.1 \
  --env DEMO_WEB_PORT=8080 \
  --env DEMO_SURROGATE_HOST=127.0.0.1 \
  --env DEMO_SURROGATE_PORT=65432 \
  <new-demo-image> \
  /opt/surrogate-newton/src/scripts/run_demo_local.sh
```

The full UIUC bundle remains outside the image and is mounted read-only. The
launcher writes meshes, cases and logs only below `/runtime/demo`. From a local
workstation, reach the host-loopback service with:

```bash
ssh -N -L 8080:127.0.0.1:8080 your-server-alias
```

Do not replace host networking with a public bind address. Public deployment
requires the separate authentication, proxy, queue, rate-limit and TLS stage.

## Full RAE2822 acceptance

For the normal-network route, set a model directory and the exact source
revision, then let Compose build and run the complete chain:

```bash
export MODEL_DIR=/absolute/path/to/model-release
export VCS_REF=$(git -C ../.. rev-parse HEAD)
docker compose -f compose.smoke.yaml up --build --abort-on-container-exit
```

The command runs the RAE2822 mesh, surrogate, and ADflow correction workflow,
then applies both the structural smoke check and the tolerances in
`deployment/acceptance/rae2822-baseline.json`. That acceptance entrypoint
selects `repeated_nk` because the recorded numerical baseline uses that mode;
ordinary deployment configuration defaults to `ank_nk`. Runtime output is
stored in the named `rae2822-runtime` volume. The model directory remains
read-only and is never copied into the image.

## Native/HPC boundary

Native and cluster installations should follow the upstream MDO Lab guides,
then apply only this project's exact commits, config deltas, and smoke checks:

- ADflow installation: <https://mdolab-adflow.readthedocs-hosted.com/en/latest/install.html>
- pyHyp installation: <https://mdolab-pyhyp.readthedocs-hosted.com/en/latest/install.html>
- MACH-Aero Docker guidance: <https://mdolab-mach-aero.readthedocs-hosted.com/en/latest/installInstructions/dockerInstructions.html>

Do not mix a system OpenMPI launcher with this runtime's MPICH libraries. An
Apptainer conversion for a particular cluster is a downstream deployment task,
not a second solver build recipe maintained here.
