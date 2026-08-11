# Interactive local demo

The local website exposes two loopback-only routes:

- `/` is the paper project page;
- `/demo` is the interactive two-dimensional airfoil application.

The application preserves the complete research workflow: choose or upload an
airfoil, project edits to the canonical 27-parameter CST representation, build a
pyHyp O-grid, run the released Surrogate model, optionally apply Newton–Krylov
correction, and optionally compare with an ADflow cold start. Mach and angle of
attack are user inputs; Reynolds number follows the shared fixed reference state
in `data/common/flow_conditions.py`.

## Assets

The two project presets and the small upload example are included in the
package. The complete UIUC coordinate library is not redistributed in public
Git while its redistribution terms are being confirmed. Build it from the
official source or mount the frozen external asset directory:

```text
<release-inputs>/demo-assets-uiuc-v1/uiuc
```

Select it with:

```bash
export DEMO_AIRFOIL_LIBRARY_ROOT=/path/to/demo-assets-uiuc-v1/uiuc
```

`catalog.json` contains 1,511 accepted airfoils. The external bundle includes a
machine-readable `manifest.json` and `SHA256SUMS`. The application continues to
offer presets, uploads, geometry projection and status reporting when this
optional library is absent.

The three project-page illustrations under `static/assets/` are original,
non-quantitative placeholders. They intentionally do not copy manuscript
figures whose public-website redistribution has not yet been confirmed.

The optional geometry-distance badge needs the two offline analysis assets
`cst26_coefficients.npz` and `ood_scores.csv`. Point `DEMO_OOD_ASSET_ROOT` at a
directory containing those files. Without them the badge is explicitly marked
unavailable; the compute workflow is unchanged.

## Start

Activate an environment containing the project and solver stack, place the
released model pair under `SURROGATE_NEWTON_MODEL_DIR`, and run:

```bash
export SURROGATE_NEWTON_MODEL_DIR=/path/to/model-2608.04400
export SURROGATE_NEWTON_RUNTIME_DIR=/path/to/writable-runtime
export DEMO_AIRFOIL_LIBRARY_ROOT=/path/to/demo-assets-uiuc-v1/uiuc
./scripts/run_demo_local.sh
```

Open <http://127.0.0.1:8080/>. Stop the launcher with `Ctrl-C`; it terminates
the Web process, native Surrogate service, resident MPI pool and solver workers.

For a geometry-only page check that does not load the model or start MPI:

```bash
DEMO_START_SURROGATE=0 DEMO_PREWARM=0 ./scripts/run_demo_local.sh
```

To reuse an already-running compatible native Surrogate service:

```bash
DEMO_START_SURROGATE=0 DEMO_SURROGATE_PORT=65432 \
  ./scripts/run_demo_local.sh
```

The Web and native Surrogate endpoints remain restricted to `127.0.0.1` or
`localhost`. Access a remote internal deployment through an SSH tunnel:

```bash
ssh -N -L 8080:127.0.0.1:8080 your-server-alias
```

## Configuration

The launcher uses the active Python interpreter by default and accepts these
environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `SURROGATE_NEWTON_MODEL_DIR` | `artifacts/` | Released checkpoint/statistics pair |
| `SURROGATE_NEWTON_RUNTIME_DIR` | external temporary root | Writable runtime parent |
| `SURROGATE_NEWTON_RUNTIME_ROOT` | fallback runtime parent | Shared runtime protocol |
| `DEMO_RUNTIME_ROOT` | `<runtime parent>/demo` | Demo cases, meshes and logs |
| `DEMO_WEB_HOST` | `127.0.0.1` | Loopback Web address |
| `DEMO_WEB_PORT` | `8080` | Web port |
| `DEMO_SURROGATE_HOST` | `127.0.0.1` | Loopback native service address |
| `DEMO_SURROGATE_PORT` | `65432` | Native service port |
| `DEMO_GPU_ID` | `0` | GPU made visible to the model service |
| `DEMO_MPI_RANKS` | `8` | ADflow ranks per case |
| `DEMO_MPI_LAUNCHER` | `auto` | Active-environment MPI launcher |
| `DEMO_PREWARM` | `1` | Prewarm model, pyHyp and resident pool |
| `DEMO_COMPUTE_RESIDUALS` | `1` | Enable model residual diagnostics |
| `DEMO_AIRFOIL_LIBRARY_ROOT` | package asset path | Optional UIUC library mount |

`SURROGATE_NEWTON_PYTHON` can explicitly select a Python executable. Solver
modules and `mpiexec`/`mpirun` are resolved from the installed solver stack and
active environment; no development checkout is assumed.

## Coordinate import

Uploads are limited to 2 MB and extensions `.dat`, `.txt` or `.csv`. Accepted
formats contain finite `x y` pairs in either a closed contour or two `zone`
sections. Commas and whitespace are accepted. Both surfaces must follow a
monotonic edge-to-edge sequence and pass the normalized chord, intersection,
thickness and camber limits. The projection always returns exactly 27 CST
parameters, including the fixed `0.002c` trailing-edge thickness.

## Local API

- `GET /api/status`
- `GET /api/presets`
- `GET /api/uiuc/catalog`
- `GET /api/uiuc/airfoil/{filename}`
- `POST /api/geometry/import`
- `POST /api/geometry/project`
- `POST /api/mesh`
- `POST /api/predict`
- `GET /api/cases/{case_id}`
- `POST /api/cases/{case_id}/recover`
- `POST /api/cases/{case_id}/reference`

The browser talks only to the same-origin JSON API. It never connects directly
to the native pickle transport.
