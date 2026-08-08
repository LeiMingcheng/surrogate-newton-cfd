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

## Maintained container build

The release container is the maintained reproducibility path. Its scripts
separate source retrieval, compilation, and binary verification:

```bash
scripts/fetch_solver_stack.sh /path/to/solver-stack
scripts/build_solver_stack.sh /path/to/solver-stack
scripts/verify_solver_stack.sh /path/to/solver-stack
```

`fetch_solver_stack.sh` reads full commits from `solver-stack.lock.yaml` and
checks them after checkout. `build_solver_stack.sh` renders the versioned GNU
Fortran/MPICH config templates, builds the locked CGNS source, then compiles
both extensions against one environment prefix. `verify_solver_stack.sh`
checks commits, dynamic library resolution, the MPICH launcher, a two-rank
mpi4py process, and the three modified solver hooks. Build logs remain beside
the chosen solver-stack workspace.

The compatibility script `install_solver_stack.sh` is fetch-only. New
automation should call the three explicit phases above.

## Native and HPC installations

We do not maintain a second general-purpose HPC installation manual. Follow
the upstream [ADflow installation guide](https://mdolab-adflow.readthedocs-hosted.com/en/latest/install.html)
and [pyHyp installation guide](https://mdolab-pyhyp.readthedocs-hosted.com/en/latest/install.html),
then use the exact source revisions, build-template deltas, and verification
commands in this repository. In particular, the Python environment, mpi4py,
PETSc, compiler wrappers, compiled extensions, and launcher must use the same
MPI implementation.

A source checkout alone is not a valid runtime installation. After a native
build, also confirm the public runtime and model pair with:

```bash
python deployment/smoke_check.py --level runtime \
  --pyhyp-root /path/to/solver-stack/pyhyp \
  --adflow-root /path/to/solver-stack/adflow \
  --checkpoint artifacts/fsb-dit-airfoil-2608.04400.pt \
  --stats artifacts/turbulent-scale-stats.json
```
