# Surrogate–Newton CFD

Surrogate–Newton CFD is the reference implementation accompanying
[*Reliable and efficient steady CFD from surrogate predictions through
Newton–Krylov correction*](https://arxiv.org/abs/2608.04400).

The surrogate supplies a global estimate of a steady turbulent flow field;
ADflow then evaluates and corrects that state with its native Newton–Krylov
solver. The first public release is intentionally limited to the complete
two-dimensional airfoil workflow.

## Scope

Included:

- Direct and flow-field Schrödinger bridge surrogate models;
- two-dimensional CST geometry and persistent pyHyp O-grid generation;
- model serving for fixed-angle and target-lift requests;
- solver-consistent field injection and terminal or staged Newton correction;
- the unified airfoil optimization runtime; and
- one RAE2822 deployment workflow under `deployment/`.

Not included in this release:

- three-dimensional surrogate or wing support;
- paper figures, source data, training datasets, or experiment campaign
  scripts (the inference checkpoint is a separate release asset); and
- the browser demo or project-page implementation.

## Repository layout

```text
surrogate/      model, training, inference, mesh, physics, and serving code
NK_resume/      canonical ADflow correction contracts and MPI execution
optimization/   shared two-dimensional optimization driver and evaluators
data/common/    CFD field, mesh, and reference-condition utilities
deployment/     one end-to-end RAE2822 deployment and its smoke check
docs/           maintained architecture and runtime documentation
```

The modified solver stack is released as two sibling repositories, not copied
into this repository:

```text
workspace/
├── surrogate-newton-cfd/
├── pyhyp/                 # 04af13e, tagged surrogate-newton-2608.04400
└── adflow/                # 4ad00919, tagged surrogate-newton-2608.04400
```

The same lock file also pins upstream CGNS and cgnsutilities; the fetch script
places all four sources in its requested solver-stack directory.

See `solver-stack.lock.yaml` and [solver stack installation](docs/solver_stack.md)
for the exact source revisions and the distinction between numerical-runtime
changes and environment-dependent build fixes.

## Quick start

The maintained reproducibility route is the single-node NVIDIA runtime image.
It rebuilds CGNS, pyHyp, and ADflow from the commits in
`solver-stack.lock.yaml`, keeps the model outside the image, and runs as a
non-root user. Remote build and full RAE2822 acceptance instructions are in
[the container guide](deployment/container/README.md).

For a native development environment, create the locked Python/MPI/PETSc
environment and fetch the exact solver sources:

```bash
conda env create -f environment.yml
conda activate surrogate-newton-cfd
scripts/fetch_solver_stack.sh ../solver-stack
scripts/build_solver_stack.sh ../solver-stack
scripts/verify_solver_stack.sh ../solver-stack
```

The root environment and container both select Python 3.10.18, PyTorch
2.8.0+cu128, and MPICH 4.3.1. The PyTorch wheel is referenced by its official
CUDA 12.8 URL and SHA-256 so installation cannot silently select a CPU build.
For an existing HPC toolchain, follow the upstream MDO Lab installation guides
and apply only this project's locked revisions and verification checks; see
[the solver-stack guide](docs/solver_stack.md).

After the model release is published, download and verify the paired model
artifacts:

```bash
scripts/download_checkpoint.sh artifacts
```

Until the model release URL and license are finalized, set
`SURROGATE_NEWTON_ASSET_BASE_URL` to the approved release-pair location. The
downloader resumes a partial transfer, verifies size and SHA-256, and only then
renames it to the final filename.

Run the lightweight structural check:

```bash
python deployment/smoke_check.py --level runtime \
  --checkpoint artifacts/fsb-dit-airfoil-2608.04400.pt \
  --stats artifacts/turbulent-scale-stats.json
```

The full deployment additionally needs a released FSB-DiT checkpoint and its
matching normalization-statistics JSON:

```bash
python deployment/run.py \
  --checkpoint artifacts/fsb-dit-airfoil-2608.04400.pt \
  --stats artifacts/turbulent-scale-stats.json \
  --output-dir outputs/rae2822
```

This launches the local surrogate service, builds the RAE2822 authority mesh,
predicts a five-channel physical flow field, and runs terminal Newton
correction through ADflow. Detailed prerequisites and outputs are documented in
[deployment/README.md](deployment/README.md).

Compare a completed run with the sanitized aerolab3 baseline using:

```bash
python deployment/compare_acceptance.py \
  --result-dir outputs/rae2822 \
  --baseline deployment/acceptance/rae2822-baseline.json
```

## Model artifacts and data

Model weights, normalization statistics, and training data are deliberately
kept outside the Git repository. The released checkpoint and statistics are
recorded in `model-manifest.json` and verified by SHA-256 during download. A
checkpoint must always be used with the configuration and statistics from the
same model release.

## Citation

If this code is useful in your research, cite the accompanying paper. A
machine-readable entry is provided in `CITATION.cff`.

## License

Original code in this repository is released under BSD-3-Clause. The modified
pyHyp and ADflow repositories retain their upstream Apache-2.0 and LGPL-2.1
licenses, and cgnsutilities retains Apache-2.0. See `THIRD_PARTY_NOTICES.md` for dependency boundaries and the
unresolved license metadata of the optional legacy AeroOpt integration.
