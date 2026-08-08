# Source provenance

The first public release is a scoped export of the two-dimensional workflow
developed in the private PIR-DM research repository.

## Frozen source

- Functional source freeze:
  `6b8af5c55ceab8ea0b4c4727615f2cc574992a96`
- Public allowlist freeze:
  `c80076d3a18f497584e5a8d810cc4b926afd1e1f`
- Included source roots: `surrogate/`, `NK_resume/`, `optimization/`, and
  `data/common/`
- Excluded material: datasets, checkpoints, paper sources and figures,
  experiment campaigns, archived configurations, three-dimensional work,
  browser demo code, logs, caches, and generated outputs

## Public-release adaptations

The exported code was changed only where public packaging required it:

- private filesystem defaults were replaced by active-environment or
  repository-relative paths;
- internal experiment names and artifact paths were replaced by stable public
  names;
- the standalone ADflow case runner was moved from an internal data-generation
  namespace to `optimization/adflow_case.py`;
- user-facing PIR-DM labels were changed to Surrogate–Newton CFD; and
- release metadata, installation scripts, checksums, licensing, and deployment
  checks were added.

The four post-export runtime changes present at the source freeze (zero-copy
EMA swapping, DDP gradient bucket views, removal of a duplicate terminal PDE
calculation, and configurable optimizer tensor-list execution) are included.

## Solver forks

- pyHyp release revision:
  `04af13e4e59d4a113d7def96b6b5b2dbf6fa9ed9`
- ADflow release revision:
  `4ad0091910d1d885f1f5ddcf41b96b5950fa16c9`

Both solver repositories carry their own upstream history, change notes, and
licenses. They are linked by `solver-stack.lock.yaml`, not copied here.
