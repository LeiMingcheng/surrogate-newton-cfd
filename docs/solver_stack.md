# Solver stack

The reference runtime uses separate modified forks of pyHyp and ADflow. Keep
them as sibling repositories so their upstream histories and licenses remain
clear.

## pyHyp

The release state is exactly commit
`04af13e4e59d4a113d7def96b6b5b2dbf6fa9ed9`, tagged
`surrogate-newton-2608.04400` and based on pyHyp 2.6.2. Its functional commit
adds
`resetForNewSurface`, resets the existing volume coordinates, and allows a
fixed-topology pyHyp instance to march a new airfoil surface.

No later local three-dimensional marching-direction work belongs to this
release.

## ADflow

The ADflow release state is exactly commit
`4ad0091910d1d885f1f5ddcf41b96b5950fa16c9`, tagged
`surrogate-newton-2608.04400` and based on upstream commit `03da3b97`. Its
changes fall into two groups.

Numerical-runtime changes:

- expose reinitialization hooks through f2py;
- rebuild dependent variables and halos after external state injection; and
- rebuild restart-derived state after `_setInfo()` restoration.

Build-compatibility changes:

- import PETSc and MPI symbols locally in stricter Fortran compilation units;
- remove broad PETSc symbol exposure from the constants module; and
- make the bundled METIS header paths explicit.

The second group is known to depend on local compiler, MPI, PETSc, and METIS
versions. Treat it as a documented reference patch, not a promise that every
toolchain needs the identical edits. The debug ADflow tree is excluded.

## Runtime discovery

Install both forks into the active Python environment. For source-runtime
discovery, either place `adflow/` beside this repository or set:

```bash
export SURROGATE_NEWTON_ADFLOW_ROOT=/path/to/adflow
export SURROGATE_NEWTON_PYTHON=/path/to/environment/bin/python
```

The MPI launcher defaults to `auto` and is resolved from the active Python or
conda environment. The Python environment, `mpi4py`, PETSc, and launcher must
share a compatible MPI implementation.

Exact revisions are recorded in `solver-stack.lock.yaml`.

## Fetch and build

The helper script checks out the immutable revisions into a chosen workspace:

```bash
scripts/install_solver_stack.sh /path/to/workspace
```

Both projects contain compiled Fortran extensions. Configure each checkout's
`config/config.mk` for the same compiler, MPI, CGNS, PETSc, and Python
environment, then build and install it according to its upstream installation
guide. A source checkout alone is not a valid runtime installation. Confirm the
compiled imports and exact commits with:

```bash
python deployment/smoke_check.py --level runtime \
  --pyhyp-root /path/to/workspace/pyhyp \
  --adflow-root /path/to/workspace/adflow \
  --checkpoint artifacts/fsb-dit-airfoil-2608.04400.pt \
  --stats artifacts/turbulent-scale-stats.json
```
