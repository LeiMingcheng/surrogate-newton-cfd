# Demo refinement requirements and decision record

Date: 2026-08-14  
Status: decisions confirmed; implementation and release validation completed  
Scope: interactive Demo page, public result contract, and force/post-processing conventions

This document records the requested refinements, the agreed technical
interpretation, the implemented contracts, and the release validation. It is
the decision record for this Demo revision.

## 中文结论摘要

本轮已按统一意见完成 Demo、后端和服务器候选镜像修改，口径如下：

1. 几何 OOD 使用训练集 `d5` 分布的 P99 作为操作阈值，界面直接显示
   `ID/OOD + 精确分位数 + d5`；同时明确它只是几何邻域预警，不是模型失效概率。
2. 顶部增加共享服务器硬件、排队及算例依赖说明；把排队等待时间和实际求解时间
   区分开，避免用户误解论文中的正式计时口径。
3. Cm 全链路统一为四分之一弦长参考点 `(0.25, 0)`、抬头为正；旧口径结果应失效
   或归档，避免同一表格混用两种定义。
4. 云图改为曲线网格上的连续插值着色，不用加强模糊来掩盖单元格；激波仍需保留
   为较尖锐的物理梯度。
5. Cp 和 Cf 都保留原始壁面离散值，不做数值平滑，以保留激波间断；Cp 保留
   `x/c=1` 的上下表面尾缘平均闭合点，并明确该点是闭合构造点而非求解器壁面采样。
6. Cp 纵轴保持传统倒置方向，下端默认固定 `Cp=+1.0`（数据明显超出时才放宽），
   必含 `Cp=0.0` 刻度，上端自动调整，纵轴标签保留一位小数。
7. Existing/UIUC 处说明：UIUC 是供浏览选择的外部几何目录，并非以一个具名数据集
   被专门列入训练集；这不等同于其中每个几何都必然是 OOD。
8. 网格标注为 `84 × 304 cells (radial × circumferential)`，合计 25,536 个单元；
   坐标顶点数组对应 `85 × 305`。
9. 结果表把合并 Force 误差拆成 `ΔCL`、`ΔCD` 两行，显示“有符号差值（绝对相对
   误差百分比）”；参考值接近零时不显示易误导的百分比。CD 可展开显示 CDp 和
   黏性阻力分量。
10. Cp/Cf 左右并列；Cf 复用与 ADflow 对齐的壁面黏性牵引算子，按朝向尾缘的
    表面切向投影并用自由来流动压无量纲化，保留原始壁面离散值。
11. 用户原始清单中的第 11 项为空，本文保留占位，待补充。

已确认的主要选择是：P99 二分类、黏性阻力显示为 `CDν`，以及 Cp 标尺固定包含
`Cp=+1.0` 和 `Cp=0.0`，同时在 `+1.0` 下方保留少量视觉空间。

## 1. Decision summary

| Item | Previous behavior | Implemented target | Decision |
| --- | --- | --- | --- |
| Geometry OOD | Shows an integer percentile; P90–P99 is `distribution edge`, P99+ is `out-of-distribution` | Show a binary `ID` / `OOD` badge together with a more precise percentile; use training P99 as the operational threshold | Confirm P99 and whether the intermediate edge state should disappear |
| Timing notice | Only says that a shared server may queue and run slowly | Add the requested hardware and case-dependence timing caveat at the top | Confirm final wording |
| Pitching moment | Field integration and ADflow cases default to leading-edge reference, and public ADflow `cmz` is not sign-converted | Use quarter-chord `(x/c, y/c)=(0.25, 0)`, with nose-up positive, everywhere | Recommended as a required scientific contract migration |
| Field rendering | Cellwise flat-filled quadrilaterals followed by a 0.7 px blur | Continuous vertex-interpolated rendering; preserve shocks without visible cell edges | Confirm WebGL/native-browser implementation |
| Cp smoothing | No numerical smoothing; an averaged trailing-edge endpoint is appended | Keep raw wall-cell values and retain the averaged trailing-edge closure point, identified in the UI | Confirmed |
| Cp axes | P1–P99 auto range plus margin; tick labels use two significant digits | Conventional inverted Cp axis, bottom fixed at `Cp=1.0` unless data exceed it, include `Cp=0.0`, auto top, one-decimal y ticks | Confirm “下端的 1” means `Cp=+1.0` |
| Existing/UIUC label | Existing contains project presets and the UIUC catalog, without a training-set disclaimer | State that UIUC is an exploration catalog and was not explicitly included as a named training dataset | Recommended |
| Mesh description | Grid dimensions are not shown prominently | Show `84 × 304 cells (radial × circumferential), 25,536 cells`; vertices are `85 × 305` | Recommended |
| Force errors | One combined `10|ΔCD| + |ΔCL|` row | Separate signed `ΔCL` and `ΔCD` rows, each shown as `value (absolute relative error)` | Confirm signed-value/absolute-percent convention |
| Drag components | Only total CD is public | Expand total CD to show pressure `CDp` and viscous `CDν` (API field remains `cdv`) | Confirmed |
| Cf plot | No Cf distribution is returned or plotted | Raw ADflow-aligned downstream-tangent skin-friction distribution beside Cp | Confirmed |
| Item 11 | No requirement text was supplied | Reserved | User to complete |

## 2. Geometry OOD: current definition and interpretation

### 2.1 Current calculation

The production asset contains 31,999 geometries, of which 28,544 are marked
as training geometries. The runtime uses only the first 26 enhanced-CST
coefficients; the 27th trailing-edge thickness parameter is currently omitted.

For each geometry, the 26 coefficients are converted to an embedding whose
Euclidean distance is the combined physical surface RMS distance. The upper
and lower surfaces are evaluated at 201 cosine-spaced chordwise positions. For
two geometries `a` and `b`, the intended distance is equivalent to

```text
d(a,b) = sqrt(mean([Δyu(x)^2, Δyl(x)^2]))
```

where coordinates are chord-normalized. Cosine spacing gives increased
sampling density near the leading and trailing edges.

For a query geometry `q`, the runtime calculates its distance to every
training geometry, selects the five nearest distances, and reports

```text
d5(q) = mean(five smallest d(q, train_i)).
```

Using five neighbours makes the score less sensitive than a one-neighbour
distance to one duplicate or isolated training sample, while retaining a
local-density interpretation.

The displayed percentile is the empirical CDF rank of the query score in the
offline training `d5` distribution:

```text
percentile(q) = count(training_d5 <= d5(q)) / 28,544.
```

Therefore `P99` means that the query is farther from the training geometry
neighbourhood than approximately 99% of the training reference scores. It is
not a 99% probability that the geometry is OOD, not model confidence, and not
a percentage prediction error.

The frozen production distribution currently has approximately:

| Quantile | d5 | Chord-relative scale |
| --- | ---: | ---: |
| P90 | 0.000472 | 0.047% c |
| P95 | 0.001216 | 0.122% c |
| P99 | 0.002955 | 0.296% c |

The percentage-of-chord column is only a scale interpretation of the RMS
distance. It is not the empirical percentile.

The offline training scores appear to be leave-one-out scores: training rows
have nonzero nearest-neighbour distances. Runtime queries do not know whether
they are training samples, so an exact training duplicate can include a zero
nearest distance. This does not materially affect ordinary uploaded/custom
queries but should be documented and covered by an asset-generation contract.

### 2.2 Is this a reasonable geometry OOD test?

It is reasonable as a lightweight **geometry-neighbourhood warning** because:

- the metric is expressed in physical surface displacement rather than raw,
  unequally scaled CST coefficient distance;
- it uses the same chord normalization and enhanced-CST representation as the
  Demo geometry pipeline;
- five-neighbour averaging provides a local training-density measure; and
- a training empirical percentile is understandable and reproducible.

It must not be presented as a complete model OOD decision because:

- the 27th trailing-edge thickness parameter is not included;
- Mach, angle of attack, Reynolds number, and their joint support are absent;
- mesh validity and mesh-quality degradation are absent;
- surface RMS can underweight small but aerodynamically important local
  curvature or leading-edge changes; and
- P99 is an operational threshold: by construction, about 1% of the training
  reference scores also lie at or beyond it.

### 2.3 Recommended UI contract

Use a binary operational label while preserving the continuous evidence:

```text
ID  · P96.2     d5 = 0.00154 c
OOD · P99.4     d5 = 0.00321 c
```

Recommended rule:

```text
ID  : percentile < P99
OOD : percentile >= P99
```

The tooltip/help text should say that this is a geometry-only P99 warning, not
a guarantee of prediction accuracy. Percentiles should retain one decimal near
the upper tail rather than rounding P99.5 to the misleading “100th percentile”.

Future scientific hardening, separate from the first UI change:

1. include trailing-edge thickness in the physical geometry distance;
2. freeze and test leave-one-out training score generation in this repository;
3. add a separate flow-condition support indicator; and
4. validate percentile bins against held-out error and convergence statistics.

## 3. Top timing and hardware notice

Recommended English copy:

> **Timing on this shared server.** Limited public-server hardware and queueing
> can make this Demo slower than the reported benchmark setting. Because CFD
> convergence is case-dependent, cold-start ADflow wall time may vary widely,
> typically from 10–100 s; Surrogate + NK typically takes 3–10 s. Optimization
> speedup is also strongly dependent on the case and optimization trajectory.

This should appear directly below the current shared-server notice. Queue wait
and solver wall time should remain separately identifiable in job state and the
result table, so the notice does not conflate scheduling delay with numerical
solver time.

## 4. Unified pitching-moment convention

### 4.1 Required public definition

Adopt the PIR-DM convention as the only public convention:

```text
reference point: (x/c, y/c) = (0.25, 0.0)
sign:             nose-up positive, nose-down negative
public symbol:    Cm
```

PIR-DM defines:

```python
STANDARD_MOMENT_REFERENCE = (0.25, 0.0)
STANDARD_MOMENT_SIGN_CONVENTION = "nose_up_positive"
public_Cm = -native_right_hand_positive_z_Cmz
```

### 4.2 Current mismatch

The current public repository still defaults field integration and serving to
`moment_center=(0.0, 0.0)`. Demo-generated ADflow cases do not supply `x_ref`,
so the solver also receives `xRef=0.0`. Native ADflow `cmz` is currently exposed
as `cm` without the PIR-DM sign conversion.

This is a scientific contract migration, not a label-only UI edit.

### 4.3 Required migration scope

- Add a shared convention module equivalent to PIR-DM.
- Change NumPy and Torch field-force defaults to quarter chord.
- Change serving/AoA force configuration to quarter chord.
- Put `x_ref=0.25`, `y_ref=0.0` explicitly into every Demo/NK case.
- Convert native ADflow `cmz` to nose-up-positive exactly once at the adapter
  boundary; do not negate downstream a second time.
- Add convention metadata to every result and release record.
- Add reference-shift and ADflow sign-conversion tests copied conceptually from
  PIR-DM.
- Archive or invalidate persistent cases produced under the old convention;
  they must not be compared in one table with new results.

The neural model predicts fields rather than a direct Cm output, so this is a
post-processing/solver-contract migration and should not require retraining the
field model. It does require revalidation of public forces.

## 5. Continuous field rendering

### 5.1 Current behavior

The browser currently paints every `(84, 304)` cell as a flat-colour physical
quadrilateral, strokes its edge with the same colour, and applies a 0.7 px
canvas blur. This can hide small raster gaps but does not create a truly
continuous scalar field. Strong shocks make cell-to-cell colour changes and
mesh traces especially visible.

### 5.2 Recommended renderer

Use self-contained browser WebGL rendering:

1. convert cell-centred values to periodic grid-vertex values using a documented
   adjacent-cell interpolation;
2. triangulate each curvilinear quadrilateral consistently;
3. pass physical vertex coordinates and scalar values to WebGL;
4. let the fragment shader interpolate scalar values continuously inside each
   triangle and apply the existing colour map; and
5. draw the white airfoil mask and outline above the rendered field.

The grid is only 25,536 cells (about 51,000 triangles), which is modest for a
browser GPU. A Canvas2D flat-cell fallback should remain available.

This is preferable to increasing Gaussian blur: interpolation removes visible
cell boundaries, whereas strong blur can move or weaken the apparent shock.
The same renderer should be used for physical fields and absolute-error maps.

Acceptance checks:

- no visible white seams or cell-edge lattice at normal and Retina scaling;
- shock position remains aligned with the raw solution;
- common colour limits remain identical across methods;
- no data smoothing is applied to the Cp or Cf line plots; and
- fallback rendering remains functional without WebGL.

## 6. Cp plot contract

### 6.1 Smoothing status

The current Cp arrays come directly from wall-cell pressure and are not passed
through a moving average, spline, or other numerical smoother. Rounded SVG
line joins and sparse markers change appearance only; they do not alter data.

One intentional exception exists: values with `x/c >= 0.9995` are trimmed and
an `x/c=1` closure point is appended using the mean of the last upper/lower Cp
values. This trailing-edge average is retained by decision and is identified
as a constructed closure point; all preceding Cp values are raw solver-wall
samples and are not numerically smoothed.

### 6.2 Recommended axes

Keep the conventional inverted Cp direction: suction/negative Cp at the top,
positive Cp at the bottom.

- Always include and label `Cp = +1.0`, with a small visual margin below it.
- If observed data clearly exceed +1.0, expand the bottom to a rounded “nice”
  limit with a small margin.
- Top bound: automatic from robust minimum data plus margin.
- Always include a labelled `Cp = 0.0` tick/gridline.
- Always label the bottom `Cp = 1.0` when it is the active bound.
- Format y-axis tick labels with one decimal place.
- Keep all raw wall points and do not smooth the shock discontinuity.

The exact robust top rule should avoid clipping a real suction peak while not
allowing one invalid outlier to destroy the scale. A P1-based proposal is
acceptable only if out-of-range points are visibly indicated; otherwise use
the actual finite minimum with a bounded margin.

## 7. Geometry source and mesh description

Recommended Existing-mode note:

> Project presets and the UIUC coordinate catalog are provided for exploration.
> The UIUC catalog was not explicitly included as a named training dataset;
> use the geometry-distance indicator to assess proximity to training shapes.

This wording avoids the incorrect inference that every UIUC airfoil is either
known training data or automatically OOD.

Recommended mesh status text after preparation:

```text
Structured O-grid ready · 84 × 304 cells
(radial × circumferential; 25,536 cells, 85 × 305 vertices)
```

The main UI can show only `84 × 304 cells`; the expanded explanation can live
in the status/help text.

## 8. Result table and drag decomposition

### 8.1 Replace combined force MAE

Remove:

```text
10|ΔCD| + |ΔCL|
```

Add separate rows after the ADflow reference is available:

```text
ΔCL = CL_method - CL_ADflow
ΔCD = CD_method - CD_ADflow
```

Recommended display:

```text
−0.00321 (0.40%)
+0.00018 (1.52%)
```

The signed numerical delta shows direction. The percentage in parentheses is
recommended to be the absolute relative error:

```text
100 * |method - reference| / |reference|.
```

If the reference magnitude is effectively zero, show `—` for the percentage
rather than an unstable large value. The ADflow reference column should show
`0 (0.0%)`.

### 8.2 Expand total drag

Make the total-CD row expandable to two subordinate rows:

- pressure drag coefficient `CDp`;
- viscous drag coefficient, internally/native-ADflow `CDv`.

The public mathematical notation is `CDν`, with the native/internal API field
remaining `cdv`. This avoids changing the ADflow contract while giving the
page the requested notation.

The solver adapter already requests native `cdp` and `cdv`, but the Demo drops
them from its public payload. The field-force calculator also computes CDp and
CDv, while the serving contract currently returns only total CL/CD/Cm. The
backend contract must therefore be expanded for all three columns before the
UI disclosure is added.

## 9. Add a Cf plot beside Cp

This is feasible, and a two-column Cp/Cf row would use the current space well.
Recommended layout on desktop:

```text
+-------------------------+-------------------------+
| Cp comparison           | Cf comparison           |
+-------------------------+-------------------------+
```

On narrow screens, stack the plots vertically.

Recommended scientific definition:

```text
Cf = tau_t / (0.5 * rho_inf * U_inf^2)
```

where `tau_t` is signed wall shear along a clearly defined downstream surface
tangent. The plot should use raw wall-face values without numerical smoothing.
Upper and lower branches should use the same style convention as Cp.

The implementation derives Cf from the existing ADflow-aligned wall viscous
traction path. The face-integrated Cartesian viscous force is divided by face
length and free-stream dynamic pressure, then projected onto a surface tangent
oriented from the leading edge toward the trailing edge. Positive Cf therefore
denotes downstream wall shear on both branches. The same post-processing
operator is applied to Surrogate, Surrogate + NK, and ADflow fields, so the
comparison is operator-consistent. No smoothing or synthetic Cf endpoint is
applied.

`foundational_modules.md` records the ADflow-aligned wall-traction and viscous
force decomposition used here, although it does not expose a separate local-Cf
API. The Demo therefore reuses that aligned operator rather than inventing an
independent shear model.

## 10. Implemented and validated release scope

### Phase A — scientific contracts (complete)

1. Port the PIR-DM Cm convention and tests.
2. Add explicit moment metadata and invalidate/archive old-convention cases.
3. Expose CL/CD/Cm/CDp/CDv consistently for Surrogate, NK, and ADflow.
4. Derive local Cf through the ADflow-aligned wall-traction path and validate
   finite, consistent distributions for all three result stages.

### Phase B — data/API contracts (complete)

5. Return binary OOD label, exact percentile, threshold, d5 units, and method
   limitations in the geometry response.
6. Return mesh dimensions and force-component metadata.
7. Return raw Cp and validated Cf surface branches without smoothing.

### Phase C — visual changes (complete)

8. Add timing, UIUC, OOD, and mesh copy.
9. Implement Cp axes and separate ΔCL/ΔCD rows.
10. Add expandable CD components and the two-column Cp/Cf layout.
11. Replace flat-cell/blur rendering with continuous WebGL interpolation.

### Phase D — release validation (complete)

12. Unit-test quarter-chord and ADflow sign conversions.
13. Run the real Surrogate → Surrogate + NK → ADflow chain on the isolated
    server candidate, checking CL/CD/Cm/CDp/CDν and Cp/Cf payloads.
14. Visually verify raw Cp shock discontinuity, retained trailing-edge closure,
    raw Cf, continuous WebGL field interpolation, and common colour scales.
15. Run the API and 12-client concurrency suites before publishing the new
    immutable image.

## 11. Final decisions

1. **OOD:** binary `ID/OOD` at P99, with percentile and d5 retained; remove the
   public intermediate `distribution edge` category.
2. **OOD wording:** explicitly call it a geometry-only neighbourhood warning.
3. **Cm:** quarter-chord, nose-up positive, with old cases invalidated.
4. **Cp bottom:** interpret the requested fixed lower plot bound as `Cp=+1.0`.
5. **Cp data:** retain the averaged trailing-edge closure point; do not smooth
   the raw curve or shock discontinuity.
6. **Force error:** signed Δ value plus absolute relative percentage.
7. **Viscous drag notation:** display `CDν`; native/API field remains `CDv`/`cdv`.
8. **Cf:** signed downstream-tangent Cf from the ADflow-aligned wall-traction
   operator, raw and unsmoothed.
9. **Field rendering:** use WebGL vertex interpolation with Canvas fallback,
   not stronger blur.
10. **Item 11:** provide the missing requirement text.
