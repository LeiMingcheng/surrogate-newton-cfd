"use strict";

const COLORS = { surrogate: "#c66a2b", recovery: "#326a9a", reference: "#253543" };
const STAGE_LABELS = { surrogate: "Surrogate", recovery: "Surrogate + NK", reference: "ADflow" };
const NS = "http://www.w3.org/2000/svg";
const DEFAULT_REYNOLDS_PER_MACH = 22132436.863567192;
const FIELD_SMOOTHING_PX = 0.7;

const state = {
  presets: {}, geometry: null, editorGeometry: null,
  uploadedGeometry: null, customGeometry: null, geometryMode: "existing",
  existingGeometry: null, selectedAirfoil: "local:rae2822", uiucCatalog: [], uiucSource: null,
  mesh: null, case: null, stages: {}, surrogateOnline: false, solverReady: false,
  busy: false, drag: null, handleCount: 6,
  reynoldsPerMach: DEFAULT_REYNOLDS_PER_MACH,
  mpiRanks: 8,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function svgElement(name, attrs = {}) {
  const element = document.createElementNS(NS, name);
  Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
  return element;
}

async function api(path, options = {}) {
  const response = await fetch(path, { headers: options.body ? { "Content-Type": "application/json" } : {}, ...options });
  const payload = await response.json();
  if (!response.ok || payload.ok === false) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

const clone = (value) => JSON.parse(JSON.stringify(value));

function formatNumber(value, digits = 4) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
  const number = Number(value);
  if (number !== 0 && (Math.abs(number) < 1e-3 || Math.abs(number) >= 1e4)) return number.toExponential(2);
  return number.toFixed(digits);
}

function syncReferenceState() {
  const reynolds = state.reynoldsPerMach * Number($("#mach").value);
  const exponent = Math.floor(Math.log10(reynolds));
  $("#reynolds-value").innerHTML = `${(reynolds / (10 ** exponent)).toFixed(3)} × 10<sup>${exponent}</sup>`;
}

function setMessage(message, error = false) {
  const element = $("#case-message");
  element.textContent = message;
  element.classList.toggle("error", error);
}

function setStageCard(stage, message, className) {
  const card = $(`#stage-${stage}`);
  card.className = `stage-card ${className}`.trim();
  card.querySelector(".stage-state").textContent = message;
}

function clearCase() {
  state.case = null;
  state.stages = {};
  setStageCard("surrogate", "Ready", "active");
  setStageCard("recovery", "Awaiting Surrogate", "");
  setStageCard("reference", "Optional", "");
  $("#recover-button").disabled = true;
  $("#reference-button").disabled = true;
  renderResults();
}

function invalidateMesh() {
  state.mesh = null;
  $("#mesh-status").textContent = "Mesh not generated.";
  $("#mesh-status").classList.remove("ready");
  updateEnabledState();
}

function activateMode(mode, { loadGeometry = true } = {}) {
  state.geometryMode = mode;
  $$(".mode-tab").forEach((button) => {
    const active = button.dataset.mode === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  $$(".mode-panel").forEach((panel) => { panel.hidden = panel.dataset.panel !== mode; });
  if (loadGeometry) {
    if (mode === "existing") {
      if (state.existingGeometry) setGeometry(state.existingGeometry);
    } else if (mode === "upload" && state.uploadedGeometry) {
      setGeometry(state.uploadedGeometry);
    } else if (mode === "custom" && state.customGeometry) {
      setGeometry(state.customGeometry);
    } else {
      drawGeometry();
    }
  } else {
    drawGeometry();
  }
}

function setGeometry(geometry) {
  state.geometry = clone(geometry);
  state.editorGeometry = clone(geometry);
  const tail = Number(state.editorGeometry.geometry27[26]);
  state.editorGeometry.upper = state.editorGeometry.upper.map((value, index) => Number(value) - 0.5 * tail * Number(state.editorGeometry.x[index]));
  state.editorGeometry.lower = state.editorGeometry.lower.map((value, index) => Number(value) + 0.5 * tail * Number(state.editorGeometry.x[index]));
  state.editorGeometry.upper[0] = 0; state.editorGeometry.lower[0] = 0;
  state.editorGeometry.upper[state.editorGeometry.upper.length - 1] = 0;
  state.editorGeometry.lower[state.editorGeometry.lower.length - 1] = 0;
  $("#geometry-name").textContent = geometry.name;
  $("#editor-title").textContent = `${geometry.name} geometry`;
  $("#thickness-value").textContent = `t/c ${(100 * geometry.metrics.max_thickness).toFixed(2)}%`;
  const badge = $("#ood-badge");
  const ood = geometry.ood;
  badge.textContent = ood ? `${Math.round(100 * ood.percentile)}th percentile` : "OOD —";
  badge.className = "";
  if (ood?.label === "distribution edge") badge.classList.add("edge");
  if (ood?.label === "out-of-distribution") badge.classList.add("out");
  badge.title = ood ? `${ood.label}: ${ood.definition}` : "";
  $("#d5-value").innerHTML = ood ? `d<sub>5</sub> ${formatNumber(ood.distance_k5, 6)}` : "d<sub>5</sub> —";
  $("#fit-note").textContent = ood
    ? "d₅ is the mean surface-RMS distance to the five nearest training geometries; the percentile is its rank in the training d₅ distribution."
    : "Optional offline geometry-distance assets are not mounted; geometry editing and computation remain available.";
  drawGeometry();
  clearCase();
  invalidateMesh();
}

const mapEditorX = (x) => 70 + 860 * x;
const mapEditorY = (y) => 180 - 900 * y;

function pathFromSurface(x, y) {
  return x.map((value, index) => `${index === 0 ? "M" : "L"}${mapEditorX(value).toFixed(2)},${mapEditorY(y[index]).toFixed(2)}`).join(" ");
}

function drawEditorGrid() {
  const group = $(".editor-grid");
  group.replaceChildren();
  for (let index = 0; index <= 10; index += 1) {
    group.appendChild(svgElement("line", { x1: mapEditorX(index / 10), x2: mapEditorX(index / 10), y1: 45, y2: 315, class: index % 5 === 0 ? "major" : "" }));
  }
  [-0.1, -0.05, 0, 0.05, 0.1].forEach((y) => {
    group.appendChild(svgElement("line", { x1: 70, x2: 930, y1: mapEditorY(y), y2: mapEditorY(y), class: y === 0 ? "major" : "" }));
    const label = svgElement("text", { x: 59, y: mapEditorY(y) + 4, "text-anchor": "end" });
    label.textContent = y === 0 ? "0" : y.toFixed(2); group.appendChild(label);
  });
}

function interpolate(x, y, target) {
  let index = 1;
  while (index < x.length && x[index] < target) index += 1;
  if (index >= x.length) return y[y.length - 1];
  const ratio = (target - x[index - 1]) / (x[index] - x[index - 1]);
  return y[index - 1] + ratio * (y[index] - y[index - 1]);
}

function handleLocations(count) {
  return Array.from({ length: count }, (_, index) => 0.5 * (1 - Math.cos(Math.PI * (index + 1) / (count + 1))));
}

function smoothstep(value) {
  const clamped = Math.max(0, Math.min(1, value));
  return clamped * clamped * (3 - 2 * clamped);
}

function handleInfluence(x, handleIndex, count) {
  const locations = handleLocations(count);
  const center = locations[handleIndex];
  const leftSpacing = center - (handleIndex === 0 ? 0 : locations[handleIndex - 1]);
  const rightSpacing = (handleIndex === count - 1 ? 1 : locations[handleIndex + 1]) - center;
  const sigma = 0.275 * (leftSpacing + rightSpacing);
  const leftWindow = x < center ? smoothstep(x / center) : 1;
  const rightWindow = x > center ? smoothstep((1 - x) / (1 - center)) : 1;
  return Math.exp(-0.5 * ((x - center) / sigma) ** 2) * leftWindow * rightWindow;
}

function drawGeometry() {
  if (!state.editorGeometry) return;
  const { x, upper, lower } = state.editorGeometry;
  $("#upper-path").setAttribute("d", pathFromSurface(x, upper));
  $("#lower-path").setAttribute("d", pathFromSurface(x, lower));
  const fill = [pathFromSurface(x, upper), ...lower.slice().reverse().map((value, reverseIndex) => {
    const index = lower.length - 1 - reverseIndex;
    return `L${mapEditorX(x[index]).toFixed(2)},${mapEditorY(value).toFixed(2)}`;
  }), "Z"].join(" ");
  $("#airfoil-fill").setAttribute("d", fill);

  const lines = $("#handle-lines");
  const upperGroup = $("#upper-handles");
  const lowerGroup = $("#lower-handles");
  lines.replaceChildren(); upperGroup.replaceChildren(); lowerGroup.replaceChildren();
  if (state.geometryMode !== "custom") return;
  handleLocations(state.handleCount).forEach((handleX, index) => {
    const upperY = interpolate(x, upper, handleX);
    const lowerY = interpolate(x, lower, handleX);
    lines.appendChild(svgElement("line", { x1: mapEditorX(handleX), x2: mapEditorX(handleX), y1: mapEditorY(upperY), y2: mapEditorY(lowerY) }));
    [["upper", upperY, upperGroup], ["lower", lowerY, lowerGroup]].forEach(([side, value, group]) => {
      const handle = svgElement("g");
      const hit = svgElement("circle", { cx: mapEditorX(handleX), cy: mapEditorY(value), r: 18, class: "handle-hit", tabindex: 0, "data-side": side, "data-index": index, "aria-label": `${side} deformation handle ${index + 1}` });
      hit.addEventListener("pointerdown", beginDrag);
      handle.append(hit, svgElement("circle", { cx: mapEditorX(handleX), cy: mapEditorY(value), r: 7, class: "handle-dot" }));
      group.appendChild(handle);
    });
  });
}

function beginDrag(event) {
  if (state.busy || !state.editorGeometry) return;
  event.preventDefault();
  state.drag = { pointerId: event.pointerId, side: event.currentTarget.dataset.side, index: Number(event.currentTarget.dataset.index), startClientY: event.clientY, originalUpper: state.editorGeometry.upper.slice(), originalLower: state.editorGeometry.lower.slice(), moved: false };
  clearCase(); invalidateMesh();
  window.addEventListener("pointermove", continueDrag);
  window.addEventListener("pointerup", finishDrag);
  window.addEventListener("pointercancel", cancelDrag);
}

function continueDrag(event) {
  if (!state.drag || event.pointerId !== state.drag.pointerId) return;
  event.preventDefault();
  const rect = $("#airfoil-svg").getBoundingClientRect();
  const clientDelta = event.clientY - state.drag.startClientY;
  if (Math.abs(clientDelta) > 1) state.drag.moved = true;
  const delta = -(clientDelta / rect.height) * 0.4;
  const original = state.drag.side === "upper" ? state.drag.originalUpper : state.drag.originalLower;
  const deformed = original.map((value, index) => value + delta * handleInfluence(state.editorGeometry.x[index], state.drag.index, state.handleCount));
  state.editorGeometry.upper = state.drag.side === "upper" ? deformed : state.drag.originalUpper.slice();
  state.editorGeometry.lower = state.drag.side === "lower" ? deformed : state.drag.originalLower.slice();
  drawGeometry();
}

function removeDragListeners() {
  window.removeEventListener("pointermove", continueDrag);
  window.removeEventListener("pointerup", finishDrag);
  window.removeEventListener("pointercancel", cancelDrag);
}

async function finishDrag(event) {
  if (!state.drag || event.pointerId !== state.drag.pointerId) return;
  removeDragListeners();
  const drag = state.drag; state.drag = null;
  if (!drag.moved) return;
  setBusy(true, "Projecting the edited surface to CST…");
  try {
    const geometry = await api("/api/geometry/project", { method: "POST", body: JSON.stringify({ x: state.editorGeometry.x, upper: state.editorGeometry.upper, lower: state.editorGeometry.lower, name: "Custom airfoil" }) });
    state.customGeometry = clone(geometry);
    activateMode("custom", { loadGeometry: false });
    setGeometry(geometry);
    setMessage("Custom geometry accepted. Generate its mesh before prediction.");
  } catch (error) {
    state.editorGeometry.upper = drag.originalUpper; state.editorGeometry.lower = drag.originalLower;
    drawGeometry(); setMessage(error.message, true);
  } finally { setBusy(false); }
}

function cancelDrag() {
  if (!state.drag) return;
  state.editorGeometry.upper = state.drag.originalUpper; state.editorGeometry.lower = state.drag.originalLower;
  state.drag = null; removeDragListeners(); drawGeometry();
}

function updateEnabledState() {
  $$(".mode-tab").forEach((button) => { button.disabled = state.busy; });
  $("#airfoil-preset").disabled = state.busy || Object.keys(state.presets).length === 0;
  $("#airfoil-search").disabled = state.busy || Object.keys(state.presets).length === 0;
  $("#coordinate-file").disabled = state.busy;
  $("#mesh-button").disabled = state.busy || !state.geometry || !state.solverReady;
  $("#predict-button").disabled = state.busy || !state.geometry || !state.mesh || !state.surrogateOnline;
  $("#recover-button").disabled = state.busy || !state.case?.stage;
  $("#reference-button").disabled = state.busy || !state.case?.stage;
}

function setBusy(busy, message = "") {
  state.busy = busy; updateEnabledState();
  if (message) setMessage(message);
}

function residualThresholdLabel(exponent) {
  const superscript = { "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
  return `10⁻${String(exponent).split("").map((digit) => superscript[digit]).join("")}`;
}

function syncMethodLabels() {
  const surrogateSteps = Number($("#surrogate-steps").value);
  const nkCycles = Number($("#nk-cycles").value);
  const stopExponent = Number($("#nk-stop-exponent").value);
  $("#surrogate-stage-detail").textContent = `${surrogateSteps} prediction steps`;
  $("#nk-stage-detail").textContent = `Up to ${nkCycles} cycles · stop at ${residualThresholdLabel(stopExponent)}`;
  $("#recover-button").textContent = `Run Surrogate + NK (max ${nkCycles})`;
}

function syncCase(payload) {
  state.case = payload; state.stages = {};
  if (payload.stage) state.stages.surrogate = payload.stage;
  if (payload.recovery) state.stages.recovery = payload.recovery;
  if (payload.reference) state.stages.reference = payload.reference;
  setStageCard("surrogate", payload.stage ? "Complete" : "Ready", payload.stage ? "complete" : "active");
  const recoveryState = payload.recovery
    ? (payload.recovery.converged
      ? `Converged in ${payload.recovery.executed_cycles} cycles`
      : `Reached ${payload.recovery.cycle_limit}-cycle limit`)
    : "Ready to run";
  setStageCard("recovery", recoveryState, payload.recovery ? "complete" : "active");
  setStageCard("reference", payload.reference ? (payload.reference.converged ? "Converged" : "Solve complete") : "Optional", payload.reference ? "complete" : "");
  updateEnabledState(); renderResults();
}

function plotPath(xValues, yValues, mapX, mapY) {
  return xValues.map((value, index) => `${index === 0 ? "M" : "L"}${mapX(Number(value)).toFixed(2)},${mapY(Number(yValues[index])).toFixed(2)}`).join(" ");
}

function drawAxes(svg, { xTicks, yTicks, mapX, mapY, xLabel, yLabel }) {
  const grid = svg.querySelector(".chart-grid"); const labels = svg.querySelector(".chart-labels");
  grid.replaceChildren(); labels.replaceChildren();
  xTicks.forEach((tick) => {
    grid.appendChild(svgElement("line", { x1: mapX(tick), x2: mapX(tick), y1: 25, y2: svg.viewBox.baseVal.height - 45, class: tick === xTicks[0] ? "axis" : "" }));
    const label = svgElement("text", { x: mapX(tick), y: svg.viewBox.baseVal.height - 23, "text-anchor": "middle" }); label.textContent = formatNumber(tick, 2); labels.appendChild(label);
  });
  yTicks.forEach((tick) => {
    grid.appendChild(svgElement("line", { x1: 58, x2: svg.viewBox.baseVal.width - 22, y1: mapY(tick), y2: mapY(tick), class: tick === 0 ? "axis" : "" }));
    const label = svgElement("text", { x: 50, y: mapY(tick) + 4, "text-anchor": "end" }); label.textContent = Number(tick).toPrecision(2); labels.appendChild(label);
  });
  const xTitle = svgElement("text", { x: svg.viewBox.baseVal.width / 2, y: svg.viewBox.baseVal.height - 3, "text-anchor": "middle" }); xTitle.textContent = xLabel; labels.appendChild(xTitle);
  const yTitle = svgElement("text", { x: 14, y: svg.viewBox.baseVal.height / 2, transform: `rotate(-90 14 ${svg.viewBox.baseVal.height / 2})`, "text-anchor": "middle" }); yTitle.textContent = yLabel; labels.appendChild(yTitle);
}

function legendItem(label, color, { dashed = false, marker = "", wide = false } = {}) {
  const item = document.createElement("span"); item.className = "legend-item";
  const swatch = document.createElement("i"); swatch.style.borderColor = color; swatch.classList.toggle("dashed", dashed); swatch.classList.toggle("marked-circle", marker === "circle"); swatch.classList.toggle("marked-square", marker === "square"); swatch.classList.toggle("wide", wide);
  item.append(swatch, document.createTextNode(label)); return item;
}

function cpMarkerIndices(length, key) {
  const closureIndex = Math.max(0, length - 1);
  const stride = Math.max(2, Math.floor(closureIndex / 12));
  const phase = key === "surrogate" ? Math.floor(stride / 2) : 0;
  const indices = [];
  for (let index = phase; index < closureIndex; index += stride) indices.push(index);
  return indices;
}

function renderCp() {
  const svg = $("#cp-chart"); const series = svg.querySelector(".chart-series"); const legend = $("#cp-legend");
  series.replaceChildren(); legend.replaceChildren();
  const available = Object.entries(state.stages).filter(([, stage]) => stage.cp);
  $("#cp-empty").hidden = available.length > 0; svg.hidden = available.length === 0;
  if (!available.length) return;
  const allCp = available.flatMap(([, stage]) => [...stage.cp.upper.cp, ...stage.cp.lower.cp]).map(Number).filter(Number.isFinite).sort((a, b) => a - b);
  const low = allCp[Math.floor(0.01 * (allCp.length - 1))]; const high = allCp[Math.ceil(0.99 * (allCp.length - 1))];
  const margin = Math.max(0.1, 0.08 * (high - low)); const yMin = low - margin; const yMax = high + margin;
  const mapX = (value) => 58 + value * 820; const mapY = (value) => 25 + ((value - yMin) / (yMax - yMin || 1)) * 317;
  drawAxes(svg, { xTicks: [0, 0.25, 0.5, 0.75, 1], yTicks: Array.from({ length: 5 }, (_, index) => yMin + index * (yMax - yMin) / 4), mapX, mapY, xLabel: "x / c", yLabel: "Cp" });
  const byKey = Object.fromEntries(available);
  const drawOrder = ["reference", "recovery", "surrogate"].filter((key) => byKey[key]);
  drawOrder.forEach((key) => {
    const stage = byKey[key];
    ["upper", "lower"].forEach((side) => {
      const values = stage.cp[side];
      series.appendChild(svgElement("path", {
        d: plotPath(values.x, values.cp, mapX, mapY), fill: "none", stroke: COLORS[key],
        "stroke-width": key === "reference" ? 5.5 : 2.4,
        "stroke-opacity": key === "reference" ? 0.72 : 0.96,
        "stroke-dasharray": side === "lower" ? "7 5" : "",
        "stroke-linecap": "round", "stroke-linejoin": "round",
      }));
    });
  });
  ["recovery", "surrogate"].filter((key) => byKey[key]).forEach((key) => {
    ["upper", "lower"].forEach((side) => {
      const values = byKey[key].cp[side];
      cpMarkerIndices(values.x.length, key).forEach((index) => {
        const x = values.x[index];
        const cx = mapX(Number(x)); const cy = mapY(Number(values.cp[index]));
        if (key === "recovery") {
          series.appendChild(svgElement("circle", { cx, cy, r: 4, fill: "#fff", stroke: COLORS[key], "stroke-width": 2 }));
        } else {
          series.appendChild(svgElement("rect", { x: cx - 2.6, y: cy - 2.6, width: 5.2, height: 5.2, fill: "#fff", stroke: COLORS[key], "stroke-width": 1.8 }));
        }
      });
    });
  });
  drawOrder.forEach((key) => {
    legend.appendChild(legendItem(STAGE_LABELS[key], COLORS[key], {
      marker: key === "recovery" ? "circle" : key === "surrogate" ? "square" : "",
      wide: key === "reference",
    }));
  });
  const separator = document.createElement("span"); separator.className = "legend-separator";
  legend.append(separator, legendItem("Upper surface", "#65717c"), legendItem("Lower surface", "#65717c", { dashed: true }));
}

function interpolateColor(value, stops) {
  const clamped = Math.max(0, Math.min(0.999, value)); const scaled = clamped * (stops.length - 1); const index = Math.floor(scaled); const ratio = scaled - index;
  return `rgb(${stops[index].map((channel, channelIndex) => Math.round(channel + ratio * (stops[Math.min(index + 1, stops.length - 1)][channelIndex] - channel))).join(",")})`;
}
const colorMap = (value) => interpolateColor(value, [[24, 62, 115], [65, 142, 183], [107, 184, 213], [238, 228, 185], [232, 119, 60], [158, 38, 53]]);
const errorColorMap = (value) => interpolateColor(value, [[255, 252, 245], [254, 224, 168], [245, 150, 99], [206, 73, 112], [111, 31, 123]]);

function drawAirfoil(context, mapX, mapY) {
  if (!state.geometry) return;
  context.beginPath(); state.geometry.x.forEach((x, index) => index === 0 ? context.moveTo(mapX(x), mapY(state.geometry.upper[index])) : context.lineTo(mapX(x), mapY(state.geometry.upper[index])));
  for (let index = state.geometry.x.length - 1; index >= 0; index -= 1) context.lineTo(mapX(state.geometry.x[index]), mapY(state.geometry.lower[index]));
  context.closePath(); context.fillStyle = "#ffffff"; context.strokeStyle = "#1d2c38"; context.lineWidth = 1.1; context.fill(); context.stroke();
}

function drawPressureField(canvas, field, range, values, mapColor = colorMap) {
  const ratio = window.devicePixelRatio || 1; const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width * ratio)); canvas.height = Math.max(1, Math.round(rect.height * ratio));
  const context = canvas.getContext("2d"); context.setTransform(ratio, 0, 0, ratio, 0, 0); context.fillStyle = "#f2f4f6"; context.fillRect(0, 0, rect.width, rect.height);
  const fieldLayer = document.createElement("canvas"); fieldLayer.width = canvas.width; fieldLayer.height = canvas.height;
  const fieldContext = fieldLayer.getContext("2d"); fieldContext.setTransform(ratio, 0, 0, ratio, 0, 0);
  const [pMin, pMax] = range; const xMin = -0.2; const xMax = 1.2; const yMin = -0.35; const yMax = 0.35;
  const mapX = (value) => ((value - xMin) / (xMax - xMin)) * rect.width; const mapY = (value) => rect.height - ((value - yMin) / (yMax - yMin)) * rect.height;
  const nodeAt = (i, j) => i * field.node_width + j; const cellAt = (i, j) => i * field.width + j;
  for (let i = field.height - 1; i >= 0; i -= 1) {
    for (let j = 0; j < field.width; j += 1) {
      const indices = [nodeAt(i, j), nodeAt(i, j + 1), nodeAt(i + 1, j + 1), nodeAt(i + 1, j)];
      const xs = indices.map((index) => Number(field.x[index])); const ys = indices.map((index) => Number(field.y[index]));
      if (![...xs, ...ys].every(Number.isFinite)) continue;
      if (Math.max(...xs) < xMin || Math.min(...xs) > xMax || Math.max(...ys) < yMin || Math.min(...ys) > yMax) continue;
      const fill = mapColor((Number(values[cellAt(i, j)]) - pMin) / (pMax - pMin || 1));
      fieldContext.beginPath(); fieldContext.moveTo(mapX(xs[0]), mapY(ys[0])); for (let point = 1; point < 4; point += 1) fieldContext.lineTo(mapX(xs[point]), mapY(ys[point])); fieldContext.closePath();
      fieldContext.fillStyle = fill; fieldContext.strokeStyle = fill; fieldContext.lineWidth = 0.7; fieldContext.fill(); fieldContext.stroke();
    }
  }
  context.save(); context.setTransform(1, 0, 0, 1, 0, 0); context.imageSmoothingEnabled = true; context.filter = `blur(${FIELD_SMOOTHING_PX * ratio}px)`; context.drawImage(fieldLayer, 0, 0); context.restore();
  drawAirfoil(context, mapX, mapY);
}

function percentile(values, fraction) {
  const sorted = values.slice().sort((a, b) => a - b); return sorted[Math.min(sorted.length - 1, Math.floor(fraction * (sorted.length - 1)))];
}

function channelData(field, key) { return field.channels[key]; }

function configuredRange(autoRange, minimumSelector, maximumSelector) {
  const minimumInput = $(minimumSelector); const maximumInput = $(maximumSelector);
  const minimum = minimumInput.value === "" ? autoRange[0] : Number(minimumInput.value);
  const maximum = maximumInput.value === "" ? autoRange[1] : Number(maximumInput.value);
  const valid = Number.isFinite(minimum) && Number.isFinite(maximum) && minimum < maximum;
  minimumInput.setCustomValidity(valid ? "" : "Minimum must be smaller than maximum.");
  maximumInput.setCustomValidity(valid ? "" : "Maximum must be larger than minimum.");
  return valid ? [minimum, maximum] : autoRange;
}

function meanSquaredError(values) {
  return values.reduce((sum, value) => sum + Number(value) ** 2, 0) / values.length;
}

function renderFields() {
  const available = Object.entries(state.stages).filter(([, stage]) => stage.field); const colorbar = $("#shared-colorbar"); colorbar.hidden = available.length === 0;
  const channelKey = $("#field-channel").value;
  const channelLabel = $("#field-channel").selectedOptions[0].textContent;
  $("#field-title").textContent = `${channelLabel} fields`;
  let sharedRange = [0, 1];
  if (available.length) {
    const autoRange = [Math.min(...available.map(([, stage]) => Number(channelData(stage.field, channelKey).range[0]))), Math.max(...available.map(([, stage]) => Number(channelData(stage.field, channelKey).range[1])))];
    sharedRange = configuredRange(autoRange, "#field-scale-min", "#field-scale-max");
    $("#color-min").textContent = formatNumber(sharedRange[0], 3); $("#color-max").textContent = formatNumber(sharedRange[1], 3);
  }
  ["surrogate", "recovery", "reference"].forEach((key) => {
    const canvas = $(`#field-${key}`); const empty = $(`#field-${key}-empty`); const field = state.stages[key]?.field;
    canvas.hidden = !field; empty.hidden = Boolean(field); if (field) drawPressureField(canvas, field, sharedRange, channelData(field, channelKey).values);
  });
  const reference = state.stages.reference?.field; const surrogate = state.stages.surrogate?.field; const recovery = state.stages.recovery?.field;
  const showErrors = Boolean(reference && surrogate); $("#error-section").hidden = !showErrors;
  if (!showErrors) return;
  $("#error-title").textContent = `Absolute ${channelLabel.toLowerCase()} error against ADflow`;
  const referenceValues = channelData(reference, channelKey).values;
  const surrogateError = channelData(surrogate, channelKey).values.map((value, index) => Math.abs(Number(value) - Number(referenceValues[index])));
  const recoveryError = recovery ? channelData(recovery, channelKey).values.map((value, index) => Math.abs(Number(value) - Number(referenceValues[index]))) : null;
  const combined = recoveryError ? surrogateError.concat(recoveryError) : surrogateError;
  const autoErrorRange = [0, Math.max(1e-12, percentile(combined, 0.99))];
  const errorRange = configuredRange(autoErrorRange, "#error-scale-min", "#error-scale-max");
  $("#error-min").textContent = formatNumber(errorRange[0], 3); $("#error-max").textContent = formatNumber(errorRange[1], 3);
  drawPressureField($("#error-surrogate"), surrogate, errorRange, surrogateError, errorColorMap);
  $("#surrogate-error-mse").textContent = `(MSE ${formatNumber(meanSquaredError(surrogateError), 6)})`;
  const recoveryCanvas = $("#error-recovery"); recoveryCanvas.parentElement.parentElement.hidden = !recovery;
  if (recovery) {
    drawPressureField(recoveryCanvas, recovery, errorRange, recoveryError, errorColorMap);
    $("#recovery-error-mse").textContent = `(MSE ${formatNumber(meanSquaredError(recoveryError), 6)})`;
  }
}

function stageWallTime(key, stage) {
  if (!stage) return null;
  return key === "surrogate" ? stage.timing?.inference_wall_sec : stage.timing?.solver_wall_sec;
}

function renderMetrics() {
  const rows = [
    ["Lift coefficient, CL", (key, stage) => formatNumber(stage?.forces?.cl, 5)],
    ["Drag coefficient, CD", (key, stage) => formatNumber(stage?.forces?.cd, 6)],
    ["Moment coefficient, CM", (key, stage) => formatNumber(stage?.forces?.cm, 5)],
    ["Residual L2 ratio", (key, stage) => formatNumber(stage?.residual?.final, 4)],
    ["NK cycles executed / limit", (key, stage) => key === "recovery" && stage ? `${stage.executed_cycles} / ${stage.cycle_limit}` : "—"],
    ["Full-field MSE vs ADflow", (key, stage) => formatNumber(stage?.reference_mse, 6)],
    ["Force MAE vs ADflow (10|ΔCD| + |ΔCL|)", (key, stage) => formatNumber(stage?.force_mae_reference, 6)],
    ["Model / solver wall time", (key, stage) => { const seconds = stageWallTime(key, stage); return seconds === null || seconds === undefined ? "—" : `${formatNumber(seconds, 3)} s`; }],
    ["End-to-end request time", (key, stage) => { const seconds = key === "surrogate" ? stage?.timing?.inference_wall_sec : stage?.timing?.request_wall_sec; return seconds === null || seconds === undefined ? "—" : `${formatNumber(seconds, 3)} s`; }],
  ];
  const body = $("#comparison-body"); body.replaceChildren();
  rows.forEach(([label, formatter]) => {
    const row = document.createElement("tr"); const heading = document.createElement("th"); heading.scope = "row"; heading.textContent = label; row.appendChild(heading);
    ["surrogate", "recovery", "reference"].forEach((key) => { const cell = document.createElement("td"); cell.textContent = formatter(key, state.stages[key]); row.appendChild(cell); }); body.appendChild(row);
  });
}

function renderResults() { renderCp(); renderMetrics(); renderFields(); }

async function loadRuntime() {
  try {
    const status = await api("/api/status"); state.surrogateOnline = Boolean(status.surrogate_online); state.solverReady = Boolean(status.solver_ready); state.mpiRanks = Number(status.resources?.cpu_ranks_per_case || 8);
    state.reynoldsPerMach = Number(status.resources?.reference_state?.reynolds) || DEFAULT_REYNOLDS_PER_MACH;
    syncReferenceState();
    const element = $("#system-status"); element.dataset.state = status.surrogate_online && status.solver_ready ? "online" : "offline";
    element.querySelector("span:last-child").textContent = status.surrogate_online ? `Surrogate ready${status.prewarm ? " · runtime prewarmed" : ""}` : "Surrogate service offline";
    if (!status.surrogate_online) setMessage("Geometry editing is available, but the local Surrogate service is offline.", true);
    updateEnabledState();
  } catch (error) {
    $("#system-status").dataset.state = "offline"; $("#system-status span:last-child").textContent = "Runtime unavailable"; setMessage(error.message, true);
  }
}

async function loadPresets() {
  const payload = await api("/api/presets"); payload.presets.forEach((preset) => { state.presets[preset.name.toLowerCase()] = preset; });
  state.existingGeometry = clone(state.presets.rae2822);
  renderAirfoilOptions();
  showAirfoilSource(null);
  setGeometry(state.existingGeometry); activateMode("existing", { loadGeometry: false }); updateEnabledState();
}

function appendAirfoilGroup(select, label, entries) {
  if (!entries.length) return;
  const group = document.createElement("optgroup"); group.label = label;
  entries.forEach((entry) => {
    const option = document.createElement("option");
    option.value = entry.key;
    option.textContent = entry.label;
    group.appendChild(option);
  });
  select.appendChild(group);
}

function renderAirfoilOptions() {
  const select = $("#airfoil-preset");
  const query = $("#airfoil-search").value.trim().toLowerCase();
  const matches = (entry) => !query || `${entry.name} ${entry.filename || ""} ${entry.description || ""}`.toLowerCase().includes(query);
  const localEntries = [
    { key: "local:rae2822", name: "RAE2822", label: "RAE2822", description: "Project preset" },
    { key: "local:oat15a", name: "OAT15A", label: "OAT15A", description: "Project preset" },
  ].filter(matches);
  const uiucEntries = state.uiucCatalog.filter(matches).map((entry) => ({
    ...entry,
    label: `${entry.name} — ${entry.description}`,
  }));
  select.replaceChildren();
  appendAirfoilGroup(select, "Project presets", localEntries);
  appendAirfoilGroup(select, "UIUC Airfoil Data Site", uiucEntries);
  if (Array.from(select.options).some((option) => option.value === state.selectedAirfoil)) {
    select.value = state.selectedAirfoil;
  } else {
    const placeholder = document.createElement("option");
    placeholder.value = ""; placeholder.textContent = "Choose a matching airfoil…";
    placeholder.selected = true; placeholder.disabled = true;
    select.prepend(placeholder);
  }
  if (state.uiucCatalog.length) {
    $("#airfoil-catalog-status").textContent = query
      ? `${uiucEntries.length} of ${state.uiucCatalog.length} UIUC airfoils match this search.`
      : `${state.uiucCatalog.length} locally stored UIUC airfoils passed the geometry and CST reconstruction checks.`;
  }
}

function showAirfoilSource(entry) {
  const description = $("#airfoil-description");
  const source = $("#airfoil-source");
  if (!entry) {
    description.textContent = "Project preset.";
    source.hidden = true;
    return;
  }
  description.textContent = entry.description;
  source.href = entry.coordinate_url;
  source.textContent = `UIUC source · ${entry.filename} ↗`;
  source.hidden = false;
}

async function selectExistingAirfoil(key) {
  if (!key) return;
  state.selectedAirfoil = key;
  if (key.startsWith("local:")) {
    const presetKey = key.slice("local:".length);
    state.existingGeometry = clone(state.presets[presetKey]);
    showAirfoilSource(null);
    setGeometry(state.existingGeometry);
    setMessage(`${state.existingGeometry.name} loaded.`);
    return;
  }
  const filename = key.slice("uiuc:".length);
  const entry = state.uiucCatalog.find((candidate) => candidate.filename.toLowerCase() === filename);
  setBusy(true, `Loading ${entry.name} from the local UIUC airfoil library…`);
  try {
    const payload = await api(`/api/uiuc/airfoil/${encodeURIComponent(entry.filename)}`);
    state.existingGeometry = clone(payload.geometry);
    showAirfoilSource(entry);
    setGeometry(state.existingGeometry);
    setMessage(`${entry.name} loaded from the local UIUC library.`);
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    setBusy(false);
  }
}

async function loadUiucCatalog() {
  try {
    const payload = await api("/api/uiuc/catalog");
    state.uiucCatalog = payload.airfoils;
    state.uiucSource = payload.source;
    renderAirfoilOptions();
  } catch (error) {
    $("#airfoil-catalog-status").textContent = `Local UIUC catalog unavailable: ${error.message}`;
    renderAirfoilOptions();
  }
}

async function generateMesh() {
  setBusy(true, `Generating the pyHyp O-grid and preparing the resident ${state.mpiRanks}-rank ADflow solver…`);
  try {
    const payload = await api("/api/mesh", { method: "POST", body: JSON.stringify({ geometry27: state.geometry.geometry27, name: state.geometry.name }) });
    state.mesh = payload; $("#mesh-status").classList.add("ready");
    $("#mesh-status").textContent = `Mesh ${payload.mesh_wall_sec.toFixed(2)} s · ADflow preparation ${payload.adflow_prepare_wall_sec.toFixed(2)} s · MPI ${payload.mpi_ranks}`;
    setMessage("Mesh generated and the geometry-matched ADflow solver is resident. Surrogate prediction is ready.");
  } catch (error) { setMessage(error.message, true); } finally { setBusy(false); }
}

async function runPrediction() {
  setBusy(true, "Evaluating the Surrogate and formal force/residual metrics…"); setStageCard("surrogate", "Running", "running");
  try {
    const steps = Number($("#surrogate-steps").value);
    const payload = await api("/api/predict", { method: "POST", body: JSON.stringify({ geometry27: state.geometry.geometry27, name: state.geometry.name, mach: Number($("#mach").value), aoa: Number($("#aoa").value), n_inference_steps: steps }) });
    syncCase(payload); setMessage(`Surrogate complete in ${payload.stage.timing.inference_wall_sec.toFixed(2)} s using ${steps} prediction steps.`);
  } catch (error) { setStageCard("surrogate", "Failed", "active"); setMessage(error.message, true); } finally { setBusy(false); }
}

async function runRecovery() {
  if (!state.case) return; const cycles = Number($("#nk-cycles").value); const residualExponent = Number($("#nk-stop-exponent").value); const thresholdLabel = residualThresholdLabel(residualExponent);
  setBusy(true, `Running up to ${cycles} Newton–Krylov correction cycles; stopping automatically at residual L2 ratio ${thresholdLabel}…`); setStageCard("recovery", "Running", "running");
  try {
    const payload = await api(`/api/cases/${state.case.case_id}/recover`, { method: "POST", body: JSON.stringify({ cycles, residual_exponent: residualExponent }) });
    syncCase(payload);
    const recovery = payload.recovery;
    setMessage(recovery.converged
      ? `Surrogate + NK converged in ${recovery.executed_cycles} of at most ${recovery.cycle_limit} cycles (MPI ${state.mpiRanks}).`
      : `Surrogate + NK reached the ${recovery.cycle_limit}-cycle limit without meeting residual L2 ratio ${thresholdLabel} (MPI ${state.mpiRanks}).`);
  } catch (error) { setStageCard("recovery", "Failed", "active"); setMessage(error.message, true); } finally { setBusy(false); }
}

async function runReference() {
  if (!state.case) return;
  if (!window.confirm(`Cold-start ADflow uses ${state.mpiRanks} MPI ranks and may take several minutes. Run the reference solve now?`)) return;
  setBusy(true, `Running cold-start ADflow from uniform flow on ${state.mpiRanks} MPI ranks…`); setStageCard("reference", "Running", "running");
  try {
    const payload = await api(`/api/cases/${state.case.case_id}/reference`, { method: "POST", body: JSON.stringify({ max_cycles: 3000 }) });
    syncCase(payload); setMessage("ADflow reference complete. Field-error maps and all three comparisons are now available.");
  } catch (error) { setStageCard("reference", "Failed", ""); setMessage(error.message, true); } finally { setBusy(false); }
}

function bindRange(rangeSelector, numberSelector, callback = null) {
  const range = $(rangeSelector); const number = $(numberSelector);
  range.addEventListener("input", () => { number.value = range.value; if (callback) callback(); clearCase(); });
  number.addEventListener("input", () => { range.value = number.value; if (callback) callback(); clearCase(); });
}

function wireEvents() {
  $("#mesh-button").addEventListener("click", generateMesh); $("#predict-button").addEventListener("click", runPrediction); $("#recover-button").addEventListener("click", runRecovery); $("#reference-button").addEventListener("click", runReference);
  $$(".mode-tab").forEach((button) => button.addEventListener("click", () => activateMode(button.dataset.mode)));
  $("#airfoil-search").addEventListener("input", renderAirfoilOptions);
  $("#airfoil-preset").addEventListener("change", (event) => selectExistingAirfoil(event.target.value));
  $("#handle-count").addEventListener("input", (event) => { state.handleCount = Number(event.target.value); drawGeometry(); });
  $("#reset-geometry").addEventListener("click", () => {
    const baseGeometry = state.uploadedGeometry || state.existingGeometry;
    if (!baseGeometry) return;
    state.customGeometry = clone(baseGeometry);
    activateMode("custom", { loadGeometry: false });
    setGeometry(baseGeometry);
    setMessage(state.uploadedGeometry
      ? "Custom geometry reset to the latest uploaded airfoil."
      : `Custom geometry reset to the current Existing airfoil (${state.existingGeometry.name}).`);
  });
  $("#coordinate-file").addEventListener("change", async (event) => {
    const [file] = event.target.files; if (!file) return; setBusy(true, `Importing ${file.name}…`);
    try {
      const geometry = await api("/api/geometry/import", { method: "POST", body: JSON.stringify({ filename: file.name, content: await file.text() }) });
      state.uploadedGeometry = clone(geometry); activateMode("upload", { loadGeometry: false }); setGeometry(geometry); setMessage(`${file.name} imported. Generate its mesh before prediction.`);
    } catch (error) { setMessage(error.message, true); } finally { event.target.value = ""; setBusy(false); }
  });
  bindRange("#mach-range", "#mach", syncReferenceState); bindRange("#aoa-range", "#aoa");
  $("#surrogate-steps").addEventListener("input", () => { syncMethodLabels(); clearCase(); });
  $("#nk-cycles").addEventListener("input", syncMethodLabels);
  $("#nk-stop-exponent").addEventListener("input", syncMethodLabels);
  $("#field-channel").addEventListener("change", renderFields);
  ["#field-scale-min", "#field-scale-max", "#error-scale-min", "#error-scale-max"].forEach((selector) => $(selector).addEventListener("input", renderFields));
  $("#field-scale-auto").addEventListener("click", () => { $("#field-scale-min").value = ""; $("#field-scale-max").value = ""; renderFields(); });
  $("#error-scale-auto").addEventListener("click", () => { $("#error-scale-min").value = ""; $("#error-scale-max").value = ""; renderFields(); });
  window.addEventListener("resize", () => { if (Object.values(state.stages).some((stage) => stage.field)) renderFields(); });
}

async function init() {
  drawEditorGrid(); wireEvents(); syncMethodLabels(); syncReferenceState(); await Promise.all([loadRuntime(), loadPresets(), loadUiucCatalog()]); updateEnabledState();
  if (state.surrogateOnline && state.solverReady) setMessage("Runtime ready. Choose a geometry, then generate its mesh.");
}

init().catch((error) => setMessage(error.message, true));
