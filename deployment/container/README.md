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

The `repository` URLs in `solver-stack.lock.yaml` must exist and expose the
locked commits. At present that requires creating and pushing the two solver
fork repositories. The model release URL is independent: place the paired
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

The default container command runs the source-level smoke check. The following
also checks the compiled hooks and model pair:

```bash
docker run --rm --gpus all \
  --volume /absolute/path/to/model-release:/models:ro \
  surrogate-newton-cfd-runtime:0.1.0-rc1 \
  runtime-smoke
```

## Full RAE2822 acceptance

From `deployment/container`, set a model directory and the exact source
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
