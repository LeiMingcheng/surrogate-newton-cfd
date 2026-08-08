# Newton correction

`NK_resume` is the solver-side boundary between a physical surrogate field and
ADflow. It records the mesh, flow conditions, model provenance, MPI layout,
solver plan, injected field, and result paths in a canonical `ResumeCase`.

## Terminal correction

For a final FSB field:

```python
from NK_resume import NKWorkPlan, SolverPreset, finalonly_plan

plan = finalonly_plan(
    "fsb",
    work=NKWorkPlan.fixed(6, solver_preset=SolverPreset.NK),
)
```

The fixed budget represents six consecutive one-cycle ADflow calls. It is not
equivalent to one call with `nCycles=6`, and it should not be described as six
linear Krylov iterations.

An adaptive plan records cumulative budgets and a residual-ratio threshold:

```python
plan = finalonly_plan(
    "fsb",
    work=NKWorkPlan.adaptive(
        range(1, 11),
        threshold=1.0e-8,
        solver_preset=SolverPreset.NK,
    ),
)
```

## Staged correction

FSB intermediate-state correction must use `surrogate.nk_resume.alternating`.
The corrected physical field is written back into the scheduler state before
the next bridge transition. Exporting unrelated solver manifests for each
intermediate field does not reproduce this coupling.

## Modified ADflow interface

The public ADflow fork adds two small solver-state hooks:

- `reinitAfterInjection` rebuilds dependent variables and halo values after an
  externally supplied five-channel field;
- `rebuildRestartDerivedStateAfterSetInfo` rebuilds the restart-derived state
  after restoring ADflow's serialized information.

These numerical-runtime changes are separate from the compiler, PETSc, MPI,
and METIS compatibility edits in the same fork. See
[solver_stack.md](solver_stack.md) before building on a different system.

## Execution

`create_pipeline().export_cases(...)` creates immutable payload and manifest
artifacts. `run_manifest(...)` can execute them sequentially, in static MPI
pools, or in resident warm pools. Deployment uses one resident pool so solver
construction can be reused across requests with compatible topology.
