# UIUC airfoil assets

The complete UIUC Airfoil Data Site coordinate library is an external runtime
asset until its public redistribution conditions are confirmed. It is not
stored in this public repository or built into the runtime image.

Source pages:

- <https://m-selig.ae.illinois.edu/ads.html>
- <https://m-selig.ae.illinois.edu/ads/coord_database.html>
- <https://m-selig.ae.illinois.edu/ads/archives/coord_seligFmt.zip>

Build a new external library with:

```bash
python -m demo.build_uiuc_library --output-root /absolute/path/to/demo-assets-uiuc
```

The builder downloads the official Selig-format archive, parses single-body
coordinates, chord-normalizes them, projects them to the same 27-parameter CST
representation used by the model, and retains only geometries with maximum
thickness no greater than `0.35c` and CST reconstruction MSE no greater than
`1e-5`. The current frozen external asset contains 1,511 accepted files and 154
rejection records from 1,665 source coordinates.

The bundle root contains `manifest.json`, `SHA256SUMS`, and `uiuc/`. At runtime
set `DEMO_AIRFOIL_LIBRARY_ROOT` to its `uiuc/` directory, which contains
`catalog.json`, `rejected.json`, and `coordinates/`. Mount it read-only in a
container. Tests use a minimal synthetic fixture rather than the full library.
