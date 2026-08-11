# Third-party notices

This repository contains original Surrogate–Newton CFD code under the
BSD-3-Clause license. Its dependencies are not relicensed by that license.

## Modified sibling solver repositories

- **pyHyp** is maintained in a separate sibling repository and remains under
  the Apache License 2.0. Copyright and license text are preserved in that
  repository.
- **ADflow** is maintained in a separate sibling repository and remains under
  the GNU Lesser General Public License 2.1. Copyright and license text are
  preserved in that repository.
- **cgnsutilities** is built from its separate upstream repository and remains
  under the Apache License 2.0. Copyright and license text are preserved in
  that repository.

No solver source is vendored into this repository. Exact revisions are recorded
in `solver-stack.lock.yaml`.

## Python dependencies

Runtime packages including PyTorch, NumPy, SciPy, mpi4py, h5py, pandas,
PyYAML, einops, tqdm, cst-modeling3d, TensorBoard, PyVista, and VTK retain their
own upstream licenses.
Installing this project does not change those terms.

## Optional UIUC Airfoil Data Site assets

`demo/build_uiuc_library.py` can download and screen coordinate files from the
UIUC Airfoil Data Site maintained by Michael Selig at
<https://m-selig.ae.illinois.edu/ads.html>. The complete coordinate library is
not redistributed in this public Git repository or runtime image while its
redistribution conditions are being confirmed. It remains an external,
read-only runtime asset with its own source metadata and checksums.

## Optional legacy AeroOpt integration

`optimization/aeroopt.py` is an adapter to the legacy `AeroOpt` 0.1.1 API; no
AeroOpt source is included here. The authors' legacy local distribution has
conflicting license declarations (`LICENSE` says LGPL-3.0 while package
metadata says MIT). It must not be redistributed with this repository until
its upstream license is clarified. The newer `aeroopt` package published on
PyPI exposes a different API and is not a drop-in replacement for this
adapter.
