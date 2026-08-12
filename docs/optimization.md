# Airfoil optimization

The optional `optimization` package provides one driver and three evaluators:

```text
AeroOpt differential evolution
  -> shared geometry, constraints, objective, and artifacts
     -> surrogate
     -> CFD
     -> surrogate plus terminal Newton correction
```

All modes enter through:

```bash
surrogate-newton-opt run --config optimization/configs/surrogate.json
```

The evaluator changes, but geometry parameterization, objective calculation,
result contracts, and output structure remain shared. The `surrogate` mode
calls the native model service; `cfd` uses ADflow target-lift solves; and
`surrogate_nk` uses the model service followed by `NK_resume` warm pools.

`surrogate_nk` supports both terminal resume modes. Its default `ank_nk` path
uses native `ADFLOW.solveCL`: each flow solve has the
`max_work_per_flow_solve` ceiling (default 1000), the target-lift driver may
make at most `max_aoa_solves` flow solves (default 5), and
`total_time_limit_s` limits the complete solveCL case (default 30 s, not 30 s
per angle of attack). `cl_tolerance=0.01` is the single lift acceptance
tolerance and `nk_switch_tolerance=1e-4` controls the ANK-to-NK transition.
Selecting `repeated_nk` instead uses `repeated_nk_cycles` and the explicit
external AoA correction controller retained for that mode. Result provenance
records the mode, solver termination, work and time ceilings, final angle of
attack, and target-lift error.

RAE2822 and OAT15A baseline CST assets are under `optimization/baselines/`.
AeroOpt and `cst_modeling` are external dependencies and must be installed in
the active environment. Optimization campaign histories and paper result
directories are not part of this repository.
