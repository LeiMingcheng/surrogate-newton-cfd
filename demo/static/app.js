"use strict";

const COLORS = { surrogate: "#c66a2b", recovery: "#326a9a", reference: "#253543" };
const STAGE_LABELS = { surrogate: "Surrogate", recovery: "Surrogate + NK", reference: "ADflow" };
const NS = "http://www.w3.org/2000/svg";
const DEFAULT_REYNOLDS_PER_MACH = 22132436.863567192;
const JOB_POLL_INTERVAL_MS = 750;
const DEFAULT_HANDLE_COUNT = 6;
const MIN_HANDLE_COUNT = 3;
const MAX_HANDLE_COUNT = 16;
const ACTIVE_JOB_STORAGE_KEY = "surrogate-newton-active-job";
const JOB_LABELS = {
  mesh: "Mesh generation and solver preparation",
  predict: "Surrogate prediction",
  recover: "Newton–Krylov correction",
  reference: "Cold-start ADflow reference",
};

const state = {
  presets: {}, geometry: null, editorGeometry: null,
  uploadedGeometry: null, customGeometry: null, geometryMode: "existing",
  existingGeometry: null, selectedAirfoil: "local:rae2822", uiucCatalog: [], uiucSource: null,
  mesh: null, case: null, stages: {}, surrogateOnline: false, solverReady: false,
  busy: false, drag: null, handleCount: DEFAULT_HANDLE_COUNT,
  activeJobId: null, activeJobAction: null,
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

const delay = (milliseconds) => new Promise((resolve) => window.setTimeout(resolve, milliseconds));

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

function rememberActiveJob(job) {
  state.activeJobId = job.job_id;
  state.activeJobAction = job.action;
  window.sessionStorage.setItem(ACTIVE_JOB_STORAGE_KEY, JSON.stringify({ jobId: job.job_id, action: job.action }));
}

function forgetActiveJob() {
  state.activeJobId = null;
  state.activeJobAction = null;
  window.sessionStorage.removeItem(ACTIVE_JOB_STORAGE_KEY);
}

function updateJobPanel(job) {
  const panel = $("#active-job");
  panel.hidden = false;
  panel.dataset.state = job.state;
  $("#active-job-title").textContent = JOB_LABELS[job.action] || "Compute job";
  const cancelButton = $("#cancel-job");
  cancelButton.disabled = !["queued", "running"].includes(job.state) || job.cancel_requested;
  cancelButton.textContent = job.cancel_requested ? "Cancellation requested" : "Cancel job";
  if (job.state === "queued") {
    $("#active-job-detail").textContent = "Waiting for the single compute worker.";
    $("#active-job-position").textContent = job.queue_position
      ? `Queue position ${job.queue_position}`
      : "Queued";
  } else if (job.state === "running") {
    $("#active-job-detail").textContent = job.cancel_requested
      ? "Finishing the current solver call before cancellation."
      : "The server is executing this job now.";
    $("#active-job-position").textContent = "Running · concurrency 1";
  } else {
    $("#active-job-detail").textContent = job.error || `Job ${job.state}.`;
    $("#active-job-position").textContent = job.state;
  }
}

async function waitForJob(jobId) {
  let connectionFailures = 0;
  while (true) {
    let job;
    try {
      const payload = await api(`/api/jobs/${jobId}`);
      job = payload.job;
      connectionFailures = 0;
    } catch (error) {
      connectionFailures += 1;
      $("#active-job-detail").textContent = "Temporarily unable to reach the scheduler; retrying…";
      if (connectionFailures >= 8) throw error;
      await delay(JOB_POLL_INTERVAL_MS);
      continue;
    }
    updateJobPanel(job);
    if (job.state === "succeeded") {
      const payload = await api(job.result_url || `/api/jobs/${jobId}/result`);
      forgetActiveJob();
      $("#active-job").hidden = true;
      return payload.result;
    }
    if (["failed", "cancelled", "expired"].includes(job.state)) {
      forgetActiveJob();
      const error = new Error(job.error || `Compute job ${job.state}.`);
      error.jobState = job.state;
      throw error;
    }
    await delay(JOB_POLL_INTERVAL_MS);
  }
}

async function runQueuedJob(action, payload) {
  const submitted = await api("/api/jobs", {
    method: "POST",
    body: JSON.stringify({ action, payload }),
  });
  rememberActiveJob(submitted.job);
  updateJobPanel(submitted.job);
  return waitForJob(submitted.job.job_id);
}

async function cancelActiveJob() {
  if (!state.activeJobId) return;
  const button = $("#cancel-job");
  button.disabled = true;
  try {
    const payload = await api(`/api/jobs/${state.activeJobId}`, { method: "DELETE" });
    updateJobPanel(payload.job);
  } catch (error) {
    button.disabled = false;
    setMessage(error.message, true);
  }
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
  badge.textContent = ood ? `${ood.is_ood ? "OOD" : "ID"} · P${(100 * ood.percentile).toFixed(1)}` : "OOD —";
  badge.className = "";
  if (ood) badge.classList.add(ood.is_ood ? "out" : "id");
  badge.title = ood ? `${ood.scope || "Geometry-only warning"}: ${ood.definition}` : "";
  $("#d5-value").innerHTML = ood ? `d<sub>5</sub> ${formatNumber(ood.distance_k5, 6)} c` : "d<sub>5</sub> —";
  $("#fit-note").textContent = ood
    ? "Geometry-only warning: d₅ is the mean surface-RMS distance to the five nearest training geometries. P99 is the ID/OOD threshold; the percentile is a rank, not a failure probability."
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
  const source = event.currentTarget;
  const captureTarget = $("#airfoil-svg");
  captureTarget.setPointerCapture?.(event.pointerId);
  state.drag = { pointerId: event.pointerId, target: captureTarget, side: source.dataset.side, index: Number(source.dataset.index), startClientY: event.clientY, originalUpper: state.editorGeometry.upper.slice(), originalLower: state.editorGeometry.lower.slice(), moved: false };
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
  const requestedDelta = -(clientDelta / rect.height) * 0.4;
  const original = state.drag.side === "upper" ? state.drag.originalUpper : state.drag.originalLower;
  const other = state.drag.side === "upper" ? state.drag.originalLower : state.drag.originalUpper;
  const isUpper = state.drag.side === "upper";
  const deltaLimits = state.editorGeometry.x.map((xValue, index) => {
    const influence = handleInfluence(xValue, state.drag.index, state.handleCount);
    if (influence < 1.0e-6 || xValue <= 1.0e-3 || xValue >= 0.995) return null;
    const lowerBound = isUpper ? other[index] + 3.0e-4 : -0.25;
    const upperBound = isUpper ? 0.25 : other[index] - 3.0e-4;
    return [(lowerBound - original[index]) / influence, (upperBound - original[index]) / influence];
  }).filter(Boolean);
  const minDelta = Math.max(-0.18, ...deltaLimits.map(([minimum]) => minimum));
  const maxDelta = Math.min(0.18, ...deltaLimits.map(([, maximum]) => maximum));
  const delta = Math.max(minDelta, Math.min(maxDelta, requestedDelta));
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

function releaseDragPointer(drag) {
  if (drag?.target?.hasPointerCapture?.(drag.pointerId)) drag.target.releasePointerCapture(drag.pointerId);
}

async function finishDrag(event) {
  if (!state.drag || event.pointerId !== state.drag.pointerId) return;
  removeDragListeners();
  const drag = state.drag; state.drag = null;
  releaseDragPointer(drag);
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
  const drag = state.drag;
  state.editorGeometry.upper = drag.originalUpper; state.editorGeometry.lower = drag.originalLower;
  state.drag = null; removeDragListeners(); releaseDragPointer(drag); drawGeometry();
}

function updateEnabledState() {
  $$(".mode-tab").forEach((button) => { button.disabled = state.busy; });
  $("#airfoil-preset").disabled = state.busy || Object.keys(state.presets).length === 0;
  $("#airfoil-search").disabled = state.busy || Object.keys(state.presets).length === 0;
  $("#coordinate-file").disabled = state.busy;
  $("#handle-count").disabled = state.busy;
  $("#reset-geometry").disabled = state.busy;
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
  const stopExponent = Number($("#nk-stop-exponent").value);
  $("#surrogate-stage-detail").textContent = `${surrogateSteps} prediction steps`;
  $("#nk-stage-detail").textContent = `Target ${residualThresholdLabel(stopExponent)} · fixed compute budget`;
  $("#recover-button").textContent = "Run Surrogate + NK";
}

function syncCase(payload) {
  state.case = payload; state.stages = {};
  if (payload.stage) state.stages.surrogate = payload.stage;
  if (payload.recovery) state.stages.recovery = payload.recovery;
  if (payload.reference) state.stages.reference = payload.reference;
  setStageCard("surrogate", payload.stage ? "Complete" : "Ready", payload.stage ? "complete" : "active");
  const recoveryState = payload.recovery
    ? (payload.recovery.converged
      ? "Converged"
      : `Stopped · ${payload.recovery.termination}`)
    : "Ready to run";
  setStageCard("recovery", recoveryState, payload.recovery ? "complete" : "active");
  setStageCard("reference", payload.reference ? (payload.reference.converged ? "Converged" : "Solve complete") : "Optional", payload.reference ? "complete" : "");
  updateEnabledState(); renderResults();
}

function plotPath(xValues, yValues, mapX, mapY) {
  return xValues.map((value, index) => `${index === 0 ? "M" : "L"}${mapX(Number(value)).toFixed(2)},${mapY(Number(yValues[index])).toFixed(2)}`).join(" ");
}

function drawAxes(svg, { xTicks, yTicks, mapX, mapY, xLabel, yLabel, yFormatter = (tick) => Number(tick).toPrecision(2) }) {
  const grid = svg.querySelector(".chart-grid"); const labels = svg.querySelector(".chart-labels");
  grid.replaceChildren(); labels.replaceChildren();
  xTicks.forEach((tick) => {
    grid.appendChild(svgElement("line", { x1: mapX(tick), x2: mapX(tick), y1: 25, y2: svg.viewBox.baseVal.height - 45, class: tick === xTicks[0] ? "axis" : "" }));
    const label = svgElement("text", { x: mapX(tick), y: svg.viewBox.baseVal.height - 23, "text-anchor": "middle" }); label.textContent = formatNumber(tick, 2); labels.appendChild(label);
  });
  yTicks.forEach((tick) => {
    grid.appendChild(svgElement("line", { x1: 58, x2: svg.viewBox.baseVal.width - 22, y1: mapY(tick), y2: mapY(tick), class: tick === 0 ? "axis" : "" }));
    const label = svgElement("text", { x: 50, y: mapY(tick) + 4, "text-anchor": "end" }); label.textContent = yFormatter(tick); labels.appendChild(label);
  });
  const xTitle = svgElement("text", { x: svg.viewBox.baseVal.width / 2, y: svg.viewBox.baseVal.height - 3, "text-anchor": "middle" }); xTitle.textContent = xLabel; labels.appendChild(xTitle);
  const yTitle = svgElement("text", { x: 14, y: svg.viewBox.baseVal.height / 2, transform: `rotate(-90 14 ${svg.viewBox.baseVal.height / 2})`, "text-anchor": "middle" }); yTitle.textContent = yLabel; labels.appendChild(yTitle);
}

function niceStep(span, targetCount = 6) {
  const raw = Math.max(Math.abs(span), 1e-12) / targetCount;
  const power = 10 ** Math.floor(Math.log10(raw));
  const fraction = raw / power;
  const niceFraction = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10;
  return niceFraction * power;
}

function niceTicks(minimum, maximum, { include = [], targetCount = 6 } = {}) {
  const step = niceStep(maximum - minimum, targetCount);
  const start = Math.ceil((minimum - 1e-12) / step) * step;
  const ticks = [];
  for (let value = start; value <= maximum + 1e-12; value += step) ticks.push(Number(value.toPrecision(12)));
  include.forEach((value) => { if (value >= minimum && value <= maximum && !ticks.some((tick) => Math.abs(tick - value) < 1e-10)) ticks.push(value); });
  return ticks.sort((a, b) => a - b);
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
  const allCp = available.flatMap(([, stage]) => [...stage.cp.upper.cp, ...stage.cp.lower.cp]).map(Number).filter(Number.isFinite);
  const actualMin = Math.min(...allCp); const actualMax = Math.max(...allCp);
  const margin = Math.max(0.08, 0.05 * (actualMax - actualMin));
  const yMin = actualMin - margin;
  const yMax = actualMax > 1.05 ? actualMax + margin : 1.12;
  const mapX = (value) => 58 + value * 820; const mapY = (value) => 25 + ((value - yMin) / (yMax - yMin || 1)) * 317;
  drawAxes(svg, { xTicks: [0, 0.25, 0.5, 0.75, 1], yTicks: niceTicks(yMin, yMax, { include: [0, 1] }), mapX, mapY, xLabel: "x / c", yLabel: "Cp", yFormatter: (tick) => Number(tick).toFixed(1) });
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

function renderCf() {
  const svg = $("#cf-chart"); const series = svg.querySelector(".chart-series"); const legend = $("#cf-legend");
  series.replaceChildren(); legend.replaceChildren();
  const available = Object.entries(state.stages).filter(([, stage]) => stage.cf);
  $("#cf-empty").hidden = available.length > 0; svg.hidden = available.length === 0;
  if (!available.length) return;
  const allCf = available.flatMap(([, stage]) => [...stage.cf.upper.cf, ...stage.cf.lower.cf]).map(Number).filter(Number.isFinite);
  const actualMin = Math.min(...allCf, 0); const actualMax = Math.max(...allCf, 0);
  const margin = Math.max(5e-5, 0.08 * (actualMax - actualMin || 1e-3));
  const yMin = actualMax > 0 && actualMin >= 0 ? 0 : actualMin - margin;
  const yMax = actualMax + margin;
  const mapX = (value) => 58 + value * 820; const mapY = (value) => 25 + ((yMax - value) / (yMax - yMin || 1)) * 317;
  drawAxes(svg, { xTicks: [0, 0.25, 0.5, 0.75, 1], yTicks: niceTicks(yMin, yMax, { include: [0] }), mapX, mapY, xLabel: "x / c", yLabel: "Cf", yFormatter: (tick) => formatNumber(tick, 4) });
  const byKey = Object.fromEntries(available);
  const drawOrder = ["reference", "recovery", "surrogate"].filter((key) => byKey[key]);
  drawOrder.forEach((key) => {
    ["upper", "lower"].forEach((side) => {
      const values = byKey[key].cf[side];
      series.appendChild(svgElement("path", {
        d: plotPath(values.x, values.cf, mapX, mapY), fill: "none", stroke: COLORS[key],
        "stroke-width": key === "reference" ? 5.5 : 2.4,
        "stroke-opacity": key === "reference" ? 0.72 : 0.96,
        "stroke-dasharray": side === "lower" ? "7 5" : "",
        "stroke-linecap": "round", "stroke-linejoin": "round",
      }));
    });
  });
  ["recovery", "surrogate"].filter((key) => byKey[key]).forEach((key) => {
    ["upper", "lower"].forEach((side) => {
      const values = byKey[key].cf[side];
      cpMarkerIndices(values.x.length, key).forEach((index) => {
        const cx = mapX(Number(values.x[index])); const cy = mapY(Number(values.cf[index]));
        if (key === "recovery") series.appendChild(svgElement("circle", { cx, cy, r: 4, fill: "#fff", stroke: COLORS[key], "stroke-width": 2 }));
        else series.appendChild(svgElement("rect", { x: cx - 2.6, y: cy - 2.6, width: 5.2, height: 5.2, fill: "#fff", stroke: COLORS[key], "stroke-width": 1.8 }));
      });
    });
  });
  drawOrder.forEach((key) => legend.appendChild(legendItem(STAGE_LABELS[key], COLORS[key], { marker: key === "recovery" ? "circle" : key === "surrogate" ? "square" : "", wide: key === "reference" })));
  const separator = document.createElement("span"); separator.className = "legend-separator";
  legend.append(separator, legendItem("Upper surface", "#65717c"), legendItem("Lower surface", "#65717c", { dashed: true }));
}

function interpolateColorChannels(value, stops) {
  const clamped = Math.max(0, Math.min(0.999, value)); const scaled = clamped * (stops.length - 1); const index = Math.floor(scaled); const ratio = scaled - index;
  return stops[index].map((channel, channelIndex) => channel + ratio * (stops[Math.min(index + 1, stops.length - 1)][channelIndex] - channel));
}
function interpolateColor(value, stops) { return `rgb(${interpolateColorChannels(value, stops).map(Math.round).join(",")})`; }
const FIELD_COLOR_STOPS = [[24, 62, 115], [65, 142, 183], [107, 184, 213], [238, 228, 185], [232, 119, 60], [158, 38, 53]];
const ERROR_COLOR_STOPS = [[255, 252, 245], [254, 224, 168], [245, 150, 99], [206, 73, 112], [111, 31, 123]];
const colorMap = (value) => interpolateColor(value, FIELD_COLOR_STOPS); colorMap.stops = FIELD_COLOR_STOPS;
const errorColorMap = (value) => interpolateColor(value, ERROR_COLOR_STOPS); errorColorMap.stops = ERROR_COLOR_STOPS;

function drawAirfoil(context, mapX, mapY) {
  if (!state.geometry) return;
  context.beginPath(); state.geometry.x.forEach((x, index) => index === 0 ? context.moveTo(mapX(x), mapY(state.geometry.upper[index])) : context.lineTo(mapX(x), mapY(state.geometry.upper[index])));
  for (let index = state.geometry.x.length - 1; index >= 0; index -= 1) context.lineTo(mapX(state.geometry.x[index]), mapY(state.geometry.lower[index]));
  context.closePath(); context.fillStyle = "#ffffff"; context.strokeStyle = "#1d2c38"; context.lineWidth = 1.1; context.fill(); context.stroke();
}

function compileFieldShader(gl, type, source) {
  const shader = gl.createShader(type); gl.shaderSource(shader, source); gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader) || "Field shader compilation failed");
  return shader;
}

function drawContinuousField(fieldLayer, field, range, values, colorStops) {
  const gl = fieldLayer.getContext("webgl", { antialias: true, alpha: false, preserveDrawingBuffer: true });
  if (!gl) return false;
  const vertexShader = compileFieldShader(gl, gl.VERTEX_SHADER, "attribute vec2 a_position; attribute vec3 a_color; varying vec3 v_color; void main(){ gl_Position=vec4(a_position,0.0,1.0); v_color=a_color; }");
  const fragmentShader = compileFieldShader(gl, gl.FRAGMENT_SHADER, "precision mediump float; varying vec3 v_color; void main(){ gl_FragColor=vec4(v_color,1.0); }");
  const program = gl.createProgram(); gl.attachShader(program, vertexShader); gl.attachShader(program, fragmentShader); gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) || "Field shader link failed");

  const nodeCount = field.node_height * field.node_width;
  const nodeSums = new Float64Array(nodeCount); const nodeWeights = new Uint8Array(nodeCount);
  const nodeAt = (i, j) => i * field.node_width + j; const cellAt = (i, j) => i * field.width + j;
  for (let i = 0; i < field.height; i += 1) {
    for (let j = 0; j < field.width; j += 1) {
      const value = Number(values[cellAt(i, j)]);
      [nodeAt(i, j), nodeAt(i, j + 1), nodeAt(i + 1, j + 1), nodeAt(i + 1, j)].forEach((index) => { nodeSums[index] += value; nodeWeights[index] += 1; });
    }
  }
  const nodeValues = new Float64Array(nodeCount);
  for (let index = 0; index < nodeCount; index += 1) nodeValues[index] = nodeSums[index] / Math.max(1, nodeWeights[index]);

  const [pMin, pMax] = range; const xMin = -0.2; const xMax = 1.2; const yMin = -0.35; const yMax = 0.35;
  const vertices = new Float32Array(field.height * field.width * 6 * 5); let offset = 0;
  const appendNode = (index) => {
    const x = Number(field.x[index]); const y = Number(field.y[index]);
    const color = interpolateColorChannels((nodeValues[index] - pMin) / (pMax - pMin || 1), colorStops);
    vertices[offset++] = 2 * (x - xMin) / (xMax - xMin) - 1;
    vertices[offset++] = 2 * (y - yMin) / (yMax - yMin) - 1;
    vertices[offset++] = color[0] / 255; vertices[offset++] = color[1] / 255; vertices[offset++] = color[2] / 255;
  };
  for (let i = 0; i < field.height; i += 1) {
    for (let j = 0; j < field.width; j += 1) {
      const n0 = nodeAt(i, j); const n1 = nodeAt(i, j + 1); const n2 = nodeAt(i + 1, j + 1); const n3 = nodeAt(i + 1, j);
      [n0, n1, n2, n0, n2, n3].forEach(appendNode);
    }
  }
  gl.viewport(0, 0, fieldLayer.width, fieldLayer.height); gl.clearColor(242 / 255, 244 / 255, 246 / 255, 1); gl.clear(gl.COLOR_BUFFER_BIT);
  gl.useProgram(program); const buffer = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
  const position = gl.getAttribLocation(program, "a_position"); const color = gl.getAttribLocation(program, "a_color");
  gl.enableVertexAttribArray(position); gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 20, 0);
  gl.enableVertexAttribArray(color); gl.vertexAttribPointer(color, 3, gl.FLOAT, false, 20, 8);
  gl.drawArrays(gl.TRIANGLES, 0, vertices.length / 5);
  return true;
}

function drawFlatFieldFallback(fieldContext, field, range, values, mapColor, mapX, mapY) {
  const [pMin, pMax] = range; const nodeAt = (i, j) => i * field.node_width + j; const cellAt = (i, j) => i * field.width + j;
  for (let i = field.height - 1; i >= 0; i -= 1) for (let j = 0; j < field.width; j += 1) {
    const indices = [nodeAt(i, j), nodeAt(i, j + 1), nodeAt(i + 1, j + 1), nodeAt(i + 1, j)];
    fieldContext.beginPath(); fieldContext.moveTo(mapX(Number(field.x[indices[0]])), mapY(Number(field.y[indices[0]])));
    for (let point = 1; point < 4; point += 1) fieldContext.lineTo(mapX(Number(field.x[indices[point]])), mapY(Number(field.y[indices[point]])));
    fieldContext.closePath(); fieldContext.fillStyle = mapColor((Number(values[cellAt(i, j)]) - pMin) / (pMax - pMin || 1)); fieldContext.fill();
  }
}

function drawPressureField(canvas, field, range, values, mapColor = colorMap) {
  const ratio = window.devicePixelRatio || 1; const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width * ratio)); canvas.height = Math.max(1, Math.round(rect.height * ratio));
  const context = canvas.getContext("2d"); context.setTransform(ratio, 0, 0, ratio, 0, 0); context.fillStyle = "#f2f4f6"; context.fillRect(0, 0, rect.width, rect.height);
  const fieldLayer = document.createElement("canvas"); fieldLayer.width = canvas.width; fieldLayer.height = canvas.height;
  const xMin = -0.2; const xMax = 1.2; const yMin = -0.35; const yMax = 0.35;
  const mapX = (value) => ((value - xMin) / (xMax - xMin)) * rect.width; const mapY = (value) => rect.height - ((value - yMin) / (yMax - yMin)) * rect.height;
  let continuous = false;
  try { continuous = drawContinuousField(fieldLayer, field, range, values, mapColor.stops || FIELD_COLOR_STOPS); } catch (_error) { continuous = false; }
  if (!continuous) {
    const fallback = fieldLayer.getContext("2d"); fallback.setTransform(ratio, 0, 0, ratio, 0, 0);
    drawFlatFieldFallback(fallback, field, range, values, mapColor, mapX, mapY);
  }
  context.drawImage(fieldLayer, 0, 0, rect.width, rect.height);
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

function dragBreakdown(stage) {
  if (!stage?.forces || !Number.isFinite(Number(stage.forces.cd))) return "—";
  const details = document.createElement("details"); details.className = "drag-breakdown";
  const summary = document.createElement("summary"); summary.textContent = formatNumber(stage.forces.cd, 6);
  const values = document.createElement("div");
  [["CDp", stage.forces.cdp], ["CDν", stage.forces.cdv]].forEach(([label, value]) => {
    const name = document.createElement("span"); name.textContent = label;
    const number = document.createElement("span"); number.textContent = formatNumber(value, 6);
    values.append(name, number);
  });
  details.append(summary, values);
  return details;
}

function forceDelta(stage, reference, key, digits) {
  const value = Number(stage?.forces?.[key]); const referenceValue = Number(reference?.forces?.[key]);
  if (!Number.isFinite(value) || !Number.isFinite(referenceValue)) return "—";
  const delta = value - referenceValue;
  const percent = Math.abs(referenceValue) < 1e-10 ? "—" : `${(100 * Math.abs(delta) / Math.abs(referenceValue)).toFixed(2)}%`;
  return `${formatNumber(delta, digits)} (${percent})`;
}

function renderMetrics() {
  const reference = state.stages.reference;
  const rows = [
    ["Lift coefficient, CL", (key, stage) => formatNumber(stage?.forces?.cl, 5)],
    ["Drag coefficient, CD", (_key, stage) => dragBreakdown(stage)],
    ["Moment coefficient, Cm", (key, stage) => formatNumber(stage?.forces?.cm, 5)],
    ["Residual L2 ratio", (key, stage) => formatNumber(stage?.residual?.final, 4)],
    ["Full-field MSE vs ADflow", (key, stage) => formatNumber(stage?.reference_mse, 6)],
    ["ΔCL vs ADflow", (_key, stage) => forceDelta(stage, reference, "cl", 5)],
    ["ΔCD vs ADflow", (_key, stage) => forceDelta(stage, reference, "cd", 6)],
    ["Model / solver wall time", (key, stage) => { const seconds = stageWallTime(key, stage); return seconds === null || seconds === undefined ? "—" : `${formatNumber(seconds, 3)} s`; }],
    ["End-to-end request time", (key, stage) => { const seconds = key === "surrogate" ? stage?.timing?.inference_wall_sec : stage?.timing?.request_wall_sec; return seconds === null || seconds === undefined ? "—" : `${formatNumber(seconds, 3)} s`; }],
  ];
  const body = $("#comparison-body"); body.replaceChildren();
  rows.forEach(([label, formatter]) => {
    const row = document.createElement("tr"); const heading = document.createElement("th"); heading.scope = "row"; heading.textContent = label; row.appendChild(heading);
    ["surrogate", "recovery", "reference"].forEach((key) => {
      const cell = document.createElement("td"); const content = formatter(key, state.stages[key]);
      if (content instanceof Node) cell.appendChild(content); else cell.textContent = content;
      row.appendChild(cell);
    }); body.appendChild(row);
  });
}

function renderResults() { renderCp(); renderCf(); renderMetrics(); renderFields(); }

async function loadRuntime() {
  try {
    const status = await api("/api/status"); state.surrogateOnline = Boolean(status.surrogate_online); state.solverReady = Boolean(status.solver_ready); state.mpiRanks = Number(status.resources?.cpu_ranks_per_case || 8);
    state.reynoldsPerMach = Number(status.resources?.reference_state?.reynolds) || DEFAULT_REYNOLDS_PER_MACH;
    syncReferenceState();
    syncMethodLabels();
    const element = $("#system-status"); element.dataset.state = status.surrogate_online && status.solver_ready ? "online" : "offline";
    const queued = Number(status.scheduler?.queue_depth || 0);
    element.querySelector("span:last-child").textContent = status.surrogate_online
      ? `Surrogate ready${status.prewarm ? " · runtime prewarmed" : ""}${queued ? ` · ${queued} queued` : ""}`
      : "Surrogate service offline";
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

function applyMeshResult(payload) {
  state.mesh = payload;
  $("#mesh-status").classList.add("ready");
  const shape = Array.isArray(payload.mesh_shape) ? `${payload.mesh_shape[0]} × ${payload.mesh_shape[1]} cells` : "84 × 304 cells";
  $("#mesh-status").textContent = `${shape} · ${Number(payload.cell_count || 25536).toLocaleString()} cells · mesh ${payload.mesh_wall_sec.toFixed(2)} s · ADflow preparation ${payload.adflow_prepare_wall_sec.toFixed(2)} s · MPI ${payload.mpi_ranks}`;
}

async function resumeActiveJob() {
  let saved;
  try {
    saved = JSON.parse(window.sessionStorage.getItem(ACTIVE_JOB_STORAGE_KEY) || "null");
  } catch (_error) {
    window.sessionStorage.removeItem(ACTIVE_JOB_STORAGE_KEY);
    return;
  }
  if (!saved?.jobId || !saved?.action) return;
  state.activeJobId = saved.jobId;
  state.activeJobAction = saved.action;
  setBusy(true, `Restoring ${JOB_LABELS[saved.action] || "compute job"} status…`);
  try {
    const result = await waitForJob(saved.jobId);
    if (saved.action === "mesh") {
      applyMeshResult(result);
      setMessage("The queued mesh job completed while this page was open.");
    } else {
      syncCase(result);
      setMessage(`${JOB_LABELS[saved.action] || "Compute job"} complete.`);
    }
  } catch (error) {
    setMessage(error.message, error.jobState !== "cancelled");
  } finally {
    setBusy(false);
  }
}

async function generateMesh() {
  setBusy(true, `Generating the pyHyp O-grid and preparing the resident ${state.mpiRanks}-rank ADflow solver…`);
  try {
    const payload = await runQueuedJob("mesh", { geometry27: state.geometry.geometry27, name: state.geometry.name });
    applyMeshResult(payload);
    setMessage("Mesh generated and the geometry-matched ADflow solver is resident. Surrogate prediction is ready.");
  } catch (error) { setMessage(error.message, error.jobState !== "cancelled"); } finally { setBusy(false); }
}

async function runPrediction() {
  setBusy(true, "Evaluating the Surrogate and formal force/residual metrics…"); setStageCard("surrogate", "Running", "running");
  try {
    const steps = Number($("#surrogate-steps").value);
    const payload = await runQueuedJob("predict", { geometry27: state.geometry.geometry27, name: state.geometry.name, mach: Number($("#mach").value), aoa: Number($("#aoa").value), n_inference_steps: steps });
    syncCase(payload); setMessage(`Surrogate complete in ${payload.stage.timing.inference_wall_sec.toFixed(2)} s using ${steps} prediction steps.`);
  } catch (error) {
    setStageCard("surrogate", error.jobState === "cancelled" ? "Ready" : "Failed", "active");
    setMessage(error.message, error.jobState !== "cancelled");
  } finally { setBusy(false); }
}

async function runRecovery() {
  if (!state.case) return; const residualExponent = Number($("#nk-stop-exponent").value); const thresholdLabel = residualThresholdLabel(residualExponent);
  setBusy(true, `Running NK correction toward residual L2 ratio ${thresholdLabel} with a fixed compute budget…`); setStageCard("recovery", "Running", "running");
  try {
    const payload = await runQueuedJob("recover", { case_id: state.case.case_id, residual_exponent: residualExponent });
    syncCase(payload);
    const recovery = payload.recovery;
    setMessage(recovery.converged
      ? `Surrogate + NK converged to the requested target (MPI ${state.mpiRanks}).`
      : `Surrogate + NK used its fixed compute budget and stopped with ${recovery.termination}; target ${thresholdLabel} (MPI ${state.mpiRanks}).`);
  } catch (error) {
    setStageCard("recovery", error.jobState === "cancelled" ? "Awaiting request" : "Failed", "active");
    setMessage(error.message, error.jobState !== "cancelled");
  } finally { setBusy(false); }
}

async function runReference() {
  if (!state.case) return;
  if (!window.confirm(`Cold-start ADflow uses ${state.mpiRanks} MPI ranks and may take several minutes. Run the reference solve now?`)) return;
  setBusy(true, `Running cold-start ADflow from uniform flow on ${state.mpiRanks} MPI ranks…`); setStageCard("reference", "Running", "running");
  try {
    const payload = await runQueuedJob("reference", { case_id: state.case.case_id, max_cycles: 3000 });
    syncCase(payload); setMessage("ADflow reference complete. Field-error maps and all three comparisons are now available.");
  } catch (error) {
    setStageCard("reference", error.jobState === "cancelled" ? "Optional" : "Failed", "");
    setMessage(error.message, error.jobState !== "cancelled");
  } finally { setBusy(false); }
}

function bindRange(rangeSelector, numberSelector, callback = null) {
  const range = $(rangeSelector); const number = $(numberSelector);
  range.addEventListener("input", () => { number.value = range.value; if (callback) callback(); clearCase(); });
  number.addEventListener("input", () => { range.value = number.value; if (callback) callback(); clearCase(); });
}

function updateHandleCount(rawValue, { commit = false } = {}) {
  const parsed = Number(rawValue);
  if (!Number.isFinite(parsed)) return;
  if (!commit && (parsed < MIN_HANDLE_COUNT || parsed > MAX_HANDLE_COUNT)) return;
  const count = Math.max(MIN_HANDLE_COUNT, Math.min(MAX_HANDLE_COUNT, Math.round(parsed)));
  state.handleCount = count;
  if (commit) $("#handle-count").value = String(count);
  drawGeometry();
}

function wireEvents() {
  $("#mesh-button").addEventListener("click", generateMesh); $("#predict-button").addEventListener("click", runPrediction); $("#recover-button").addEventListener("click", runRecovery); $("#reference-button").addEventListener("click", runReference);
  $("#cancel-job").addEventListener("click", cancelActiveJob);
  $$(".mode-tab").forEach((button) => button.addEventListener("click", () => activateMode(button.dataset.mode)));
  $("#airfoil-search").addEventListener("input", renderAirfoilOptions);
  $("#airfoil-preset").addEventListener("change", (event) => selectExistingAirfoil(event.target.value));
  $("#handle-count").addEventListener("input", (event) => updateHandleCount(event.target.value));
  $("#handle-count").addEventListener("change", (event) => updateHandleCount(event.target.value, { commit: true }));
  $("#reset-geometry").addEventListener("click", () => {
    const baseGeometry = state.uploadedGeometry || state.existingGeometry;
    if (!baseGeometry) return;
    state.handleCount = DEFAULT_HANDLE_COUNT;
    $("#handle-count").value = String(DEFAULT_HANDLE_COUNT);
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
  $("#nk-stop-exponent").addEventListener("input", syncMethodLabels);
  $("#field-channel").addEventListener("change", renderFields);
  ["#field-scale-min", "#field-scale-max", "#error-scale-min", "#error-scale-max"].forEach((selector) => $(selector).addEventListener("input", renderFields));
  $("#field-scale-auto").addEventListener("click", () => { $("#field-scale-min").value = ""; $("#field-scale-max").value = ""; renderFields(); });
  $("#error-scale-auto").addEventListener("click", () => { $("#error-scale-min").value = ""; $("#error-scale-max").value = ""; renderFields(); });
  window.addEventListener("resize", () => { if (Object.values(state.stages).some((stage) => stage.field)) renderFields(); });
}

async function init() {
  drawEditorGrid(); wireEvents(); syncMethodLabels(); syncReferenceState(); await Promise.all([loadRuntime(), loadPresets(), loadUiucCatalog()]); updateEnabledState();
  if (window.sessionStorage.getItem(ACTIVE_JOB_STORAGE_KEY)) {
    await resumeActiveJob();
  } else if (state.surrogateOnline && state.solverReady) {
    setMessage("Runtime ready. Choose a geometry, then generate its mesh.");
  }
}

init().catch((error) => setMessage(error.message, true));
