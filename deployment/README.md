# RAE2822 deployment

This directory is the single maintained example and lightweight test surface
for the first release. `run.py` executes the real two-dimensional workflow:

```text
RAE2822 CST -> pyHyp CGNS -> FSB-DiT service -> physical field -> ADflow ANK-to-NK
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

Terminal correction supports two modes. `ank_nk` is selected by default and
runs one uninterrupted production ADflow solve with the work ceiling in
`newton.max_work`, wall-time ceiling in `newton.time_limit_s`, and ANK-to-NK
transition threshold in `newton.nk_switch_tolerance`. The supplied defaults are
2000 work units, 10 s, and `1e-4`, respectively; the residual target remains
`1e-8`. `repeated_nk` runs the cumulative Direct-NK schedule in
`newton.repeated_nk_cycles`. Select that mode explicitly with
`--resume-mode repeated_nk`; residual and work metadata identify the selected
mode in every result.

Validate the completed output schema, mesh and field shapes, finite forces,
and strict residual decrease with:

```bash
python deployment/smoke_check.py \
  --level result \
  --pyhyp-root ../pyhyp \
  --adflow-root ../adflow \
  --checkpoint artifacts/fsb-dit-airfoil-2608.04400-inference.pt \
  --stats artifacts/turbulent-scale-stats.json \
  --result-dir outputs/rae2822
```

The sanitized aerolab3 numerical baseline records `repeated_nk`. Generate the
matching result explicitly before the comparison:

```bash
python deployment/run.py \
  --checkpoint artifacts/fsb-dit-airfoil-2608.04400-inference.pt \
  --stats artifacts/turbulent-scale-stats.json \
  --resume-mode repeated_nk \
  --output-dir outputs/rae2822-repeated
python deployment/compare_acceptance.py \
  --result-dir outputs/rae2822-repeated \
  --baseline deployment/acceptance/rae2822-baseline.json
```

The comparison fixes shapes and the physical case exactly, applies explicit
absolute/relative tolerances to surrogate and corrected force coefficients,
allows a bounded cross-machine change in the pre-NK residual, and still
requires the configured final residual threshold. Recorded timing is
informational because it is hardware-dependent.

`config.yaml` controls the flow condition, inference schedule, mode-specific
solver work, and residual threshold. Paths to released model artifacts remain
explicit command-line inputs so the example never silently selects an
unrelated checkpoint or statistics file.
