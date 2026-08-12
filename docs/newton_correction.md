# Newton correction

`NK_resume` is the solver-side boundary between a physical surrogate field and
ADflow. It records the mesh, flow conditions, model provenance, MPI layout,
solver plan, injected field, and result paths in a canonical `ResumeCase`.

## Terminal correction modes

Terminal correction exposes two modes. `ank_nk` is the default: it makes one
uninterrupted call with ADflow's production ANK-to-NK options and treats the
configured work and wall time as independent ceilings. The standard fixed-AoA
defaults are a residual target of `1e-8`, `NKSwitchTol=1e-4`,
`max_work=2000`, and `time_limit_s=10`.

```python
from NK_resume import NKWorkPlan, finalonly_plan

plan = finalonly_plan("fsb")
# Equivalent explicit form:
plan = finalonly_plan(
    "fsb",
    work=NKWorkPlan.ank_nk(
        max_work=2000,
        time_limit_s=10.0,
        nk_switch_tolerance=1.0e-4,
    ),
)
```

`repeated_nk` is the explicit cumulative Direct-NK controller. Each schedule
entry is a separate solver call budget; the controller can stop after a call
when the residual-ratio threshold is met.

```python
plan = finalonly_plan(
    "fsb",
    work=NKWorkPlan.repeated_nk(
        (6, 8, 10),
        threshold=1.0e-8,
    ),
)
```

A cumulative budget is a solver-call contract, not a count of linear Krylov
iterations. Results identify the selected `resume_mode` and report call count,
requested work, approximate total nonlinear work, verified residuals, and the
termination reason in `metrics.solver_work`. The repeated mode additionally
records its per-call residual trajectory in `metrics.nk_residual_contract`.

## Fixed-lift correction

An `ank_nk` case with `FixedLiftContext` delegates angle-of-attack convergence
to native `ADFLOW.solveCL`. Each flow solve defaults to a work ceiling of 1000,
the driver may make at most 5 flow solves, and the complete fixed-lift case has
a 30 s wall-time limit. `NKSwitchTol=1e-4` and `cl_tolerance=0.01` apply
throughout. These limits are serialized with the case. The `repeated_nk`
optimization path retains its explicit external AoA controller.

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
