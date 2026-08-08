# Architecture

Surrogate–Newton CFD treats a neural prediction as an initial state for the
target discrete CFD solver, not as a solver-accepted solution by itself.

```text
airfoil geometry + Mach/AoA/Re
  -> pyHyp authority O-grid
  -> Direct or FSB surrogate
  -> five-channel physical field
  -> ADflow state injection
  -> residual evaluation and Newton-Krylov correction
  -> corrected field, forces, residual trajectory, and provenance
```

## Stable module boundaries

- `surrogate.inference` loads Direct and FSB checkpoints and returns physical
  fields through a shared backend contract.
- `surrogate.serving` owns geometry preparation, batching, fixed-AoA and
  target-lift requests, and the local socket protocol.
- `surrogate.utils.mesh_generation` owns the two-dimensional pyHyp O-grid
  contract and persistent fixed-topology mesh generation.
- `surrogate.physics` owns force and PDE-residual calculations shared by
  training, evaluation, and serving.
- `NK_resume` owns the solver-side case schema, payload export, MPI execution,
  state injection, correction plans, and result aggregation.
- `surrogate.nk_resume` connects model scheduler states to the solver-side
  `NK_resume` contract for terminal and staged coupling.
- `optimization` reuses these public boundaries; it does not implement another
  mesh, surrogate, or Newton runtime.

## Two-dimensional field contract

The current model grid has 84 radial cells and 304 circumferential cells:

```text
geometry:         (B, 27)
coords:           (B, 4, 84, 304)
coords_vertex:    (B, 2, 85, 305)
flow_conditions:  (B, 3) = [Mach, AoA_degrees, Reynolds]
fields:           (B, 5, 84, 304)
```

The field order is density, x velocity, y velocity, pressure, and the
Spalart–Allmaras working variable. Production model configurations use wall
coordinates for geometry conditioning. CST coefficients remain useful for
geometry generation and optimization, but they are not an additional model
condition in those configurations.

## Release boundary

This repository contains the reusable two-dimensional implementation. It does
not contain campaign launchers, paper-specific analysis, archived comparisons,
three-dimensional code, datasets, or result artifacts.
