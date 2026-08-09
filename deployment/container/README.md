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
- CGNS from the exact commit in `solver-stack.lock.yaml`; and
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

The public pyHyp and ADflow forks expose the commits and release tags recorded
in `solver-stack.lock.yaml`. The model release is independent: place the paired
checkpoint and statistics in one host directory, with the filenames and
digests from `model-manifest.json`.

The remote host needs a working NVIDIA driver, Docker Engine, Compose, and the
NVIDIA Container Toolkit. Keep at least 100 GB free for build layers, Conda and
PyTorch packages, the 1.4 GB checkpoint, runtime data, and one rollback image.

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

When the server can reach package mirrors but GitHub source transfer is slow or
unavailable, package the three locked solver repositories on aerolab3:

```bash
scripts/package_solver_bundles.sh \
  --pyhyp /root/shared-nvme/public-repos/pyhyp \
  --adflow /root/shared-nvme/public-repos/adflow \
  --cgns /root/shared-nvme/build_libraries/CGNS \
  --output /root/shared-nvme/release-inputs/solver-bundles-2608.04400

cd /root/shared-nvme/release-inputs/solver-bundles-2608.04400
sha256sum --check SHA256SUMS
```

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
  --bundle-dir /data/build/solver-bundles-2608.04400 \
  --vcs-ref "$VCS_REF" \
  --image-version 0.1.0-rc1
```

The formal base is
`public.ecr.aws/docker/library/ubuntu:24.04@sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea`.
The wrapper defaults to `http://mirrors.tuna.tsinghua.edu.cn/ubuntu` because the
fixed base does not yet have system CA certificates when its first
`apt-get update` runs. Ubuntu archive signatures remain enabled; do not add
`--allow-unauthenticated` or disable signature verification. Override the
mirror with `--apt-mirror` when another bootstrap endpoint is required.
The bundle is exposed as a read-only BuildKit named context, checked while the
source stage has no network, and never copied into the runtime image.

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

An HTTP(S) proxy may be configured outside the repository as an optional
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
`deployment/acceptance/rae2822-baseline.json`. Runtime output is stored in the
named `rae2822-runtime` volume. The model directory remains read-only and is
never copied into the image.

## Native/HPC boundary

Native and cluster installations should follow the upstream MDO Lab guides,
then apply only this project's exact commits, config deltas, and smoke checks:

- ADflow installation: <https://mdolab-adflow.readthedocs-hosted.com/en/latest/install.html>
- pyHyp installation: <https://mdolab-pyhyp.readthedocs-hosted.com/en/latest/install.html>
- MACH-Aero Docker guidance: <https://mdolab-mach-aero.readthedocs-hosted.com/en/latest/installInstructions/dockerInstructions.html>

Do not mix a system OpenMPI launcher with this runtime's MPICH libraries. An
Apptainer conversion for a particular cluster is a downstream deployment task,
not a second solver build recipe maintained here.
