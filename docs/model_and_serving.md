# Model inference and serving

## Configuration contract

Training definitions are under `surrogate/configs/training/`. A checkpoint,
configuration, and normalization-statistics file form one inseparable model
release. Mixing these artifacts can yield numerically plausible but invalid
physical fields.

The standard physical input and output are:

```text
coords            [x_center, y_center, i_normalized, j_normalized]
flow_conditions   [Mach, AoA_degrees, Reynolds]
fields            [density, velocity_x, velocity_y, pressure, sa_nu_tilde]
```

The DiT configuration may declare an internal padded height of 88. External
arrays remain `(84, 304)`; padding and cropping are model internals.

## Starting the service

```bash
surrogate-serving \
  --config surrogate/configs/training/fsb_dit.yaml \
  --checkpoint /path/to/final_model.pt \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port 65432 \
  --authority-cgns-dir /path/to/meshes
```

The service accepts exactly one mapping per socket connection. A request may
provide already prepared arrays, a CST27 geometry, or upper/lower CST10
vectors. Its prediction modes are raw flow-condition prediction, fixed AoA,
and target lift. The stable client is:

```python
from surrogate.serving.client import SurrogateClient, SurrogateClientConfig

client = SurrogateClient(SurrogateClientConfig(port=65432))
metadata = client.ping()
result = client.request(payload)
```

## Geometry and mesh reuse

`GeometryPreparer` generates one authority CGNS mesh and retains the associated
cell-centre and vertex arrays. Different flow conditions on the same geometry
reuse that mesh. With the modified pyHyp checkout, a process also reuses its
pyHyp object while the surface topology and all marching options remain fixed;
every geometry still performs a complete hyperbolic march.

The authority CGNS file and surrogate arrays must come from the same mesh
generation call before a field is passed to ADflow.
