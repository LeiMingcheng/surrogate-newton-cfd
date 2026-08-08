# RAE2822 deployment

This directory is the single maintained example and lightweight test surface
for the first release. `run.py` executes the real two-dimensional workflow:

```text
RAE2822 CST -> pyHyp CGNS -> FSB-DiT service -> physical field -> ADflow NK
```

## Prerequisites

- a CUDA-capable PyTorch environment for the released checkpoint;
- the modified pyHyp checkout at commit `04af13e` installed in that environment;
- the modified ADflow checkout built against a compatible MPI/PETSc stack;
- the matching FSB-DiT checkpoint and normalization-statistics JSON; and
- enough CPU ranks for the configured ADflow pool.

Run the repository and exact solver-source check first:

```bash
python deployment/smoke_check.py \
  --level source \
  --pyhyp-root ../pyhyp \
  --adflow-root ../adflow
```

For the release acceptance check, use `--level runtime` and provide the two
artifact paths. This additionally verifies their SHA-256 digests and imports
the compiled solver hooks from the requested checkouts.

Then execute the complete workflow:

```bash
export SURROGATE_NEWTON_ADFLOW_ROOT="$PWD/../adflow"
python deployment/run.py \
  --checkpoint /path/to/final_model.pt \
  --stats /path/to/turbulent_scale_stats.json \
  --output-dir /path/to/rae2822_run
```

The output directory contains the derived model configuration, authority CGNS
mesh, surrogate state, correction manifest and result, service log, and a
compact `summary.json`. Use `--surrogate-only` to verify model and mesh serving
without launching ADflow; this is a diagnostic subset, not the complete
deployment result.

Validate the completed output schema, mesh and field shapes, finite forces,
and strict residual decrease with:

```bash
python deployment/smoke_check.py \
  --level result \
  --pyhyp-root ../pyhyp \
  --adflow-root ../adflow \
  --checkpoint artifacts/fsb-dit-airfoil-2608.04400.pt \
  --stats artifacts/turbulent-scale-stats.json \
  --result-dir outputs/rae2822
```

For release and container acceptance, follow the structural check with the
sanitized aerolab3 numerical baseline:

```bash
python deployment/compare_acceptance.py \
  --result-dir outputs/rae2822 \
  --baseline deployment/acceptance/rae2822-baseline.json
```

The comparison fixes shapes and the physical case exactly, applies explicit
absolute/relative tolerances to surrogate and corrected force coefficients,
allows a bounded cross-machine change in the pre-NK residual, and still
requires the configured final residual threshold. Recorded timing is
informational because it is hardware-dependent.

`config.yaml` controls the flow condition, inference schedule, and solver
budget. Paths to released model artifacts remain explicit command-line inputs
so the example never silently selects an unrelated checkpoint or statistics
file.
