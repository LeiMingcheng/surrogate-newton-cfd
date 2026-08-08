# Airfoil baselines

Each baseline contains the upper and lower ten-coefficient CST vectors and a
target maximum thickness:

```text
cst_u0.txt
cst_l0.txt
t0.txt
```

The optimization driver resolves these assets through `optimization.config`.
They are geometry inputs, not paper experiment outputs.
