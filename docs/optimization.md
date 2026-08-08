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

RAE2822 and OAT15A baseline CST assets are under `optimization/baselines/`.
AeroOpt and `cst_modeling` are external dependencies and must be installed in
the active environment. Optimization campaign histories and paper result
directories are not part of this repository.
