import { getDashboardElements } from "./app_shell/dom.js";
import { createDashboardState } from "./app_shell/state.js";
import { loadDashboardBundle } from "./data/load_bundle.js";
import {
  CENTRAL_FILTER_AXES,
  applyCentralFilters,
  centralEvalSets,
  centralMetricRows,
  pruneCentralFilters,
} from "./features/central_ssl/index.js";
import { applyFlFilters, flFilterAxes, flSslRows, pruneFlFilters, sortedFlRows } from "./features/fl_ssl/index.js";
import { renderFilterPanel } from "./ui/controls/filter_panel.js";
import { fillSelect, checkedValues } from "./ui/controls/form_controls.js";
import { storeRunAliases, storeSeriesColors } from "./state/preferences.js";
import { centralEvalSetLabel } from "./features/central_ssl/logic/labels.js";
import { normalizeOverviewSelection, renderOverviewPage } from "./features/central_ssl/ui/overview_page.js";
import { normalizeCompareSelection, renderComparePage } from "./features/central_ssl/ui/compare_page.js";
import { normalizeDetailSelection, renderDetailPage } from "./features/central_ssl/ui/detail_page.js";
import { normalizeProjectionSelection, renderProjectionPage } from "./features/central_ssl/ui/projection_page.js";
import { normalizeFlRunSelection, renderFlRunsPage } from "./features/fl_ssl/ui/runs_page.js";
import { normalizeRoundSelection, renderRoundsPage } from "./features/fl_ssl/ui/rounds_page.js";
import { normalizeClientSelections, renderClientsPage } from "./features/fl_ssl/ui/clients_page.js";
import { normalizeSplitSelection, renderSplitsPage } from "./features/fl_ssl/ui/splits_page.js";
import { normalizeFlProjectionSelection, renderFlProjectionPage } from "./features/fl_ssl/ui/projection_page.js";

const DATA_URL = "./data/experiment_dashboard.json";
const elements = getDashboardElements();
const state = createDashboardState();

init();

async function init() {
  bindEvents();
  try {
    state.bundle = await loadDashboardBundle(DATA_URL);
    elements.loadState.hidden = true;
    hydrateEvalFilters();
    render();
  } catch (error) {
    elements.loadState.hidden = false;
    elements.loadState.className = "notice warning";
    elements.loadState.innerHTML = `
      <strong>dashboard data를 찾지 못했습니다.</strong>
      <span>먼저 result index export를 실행하세요: <code>uv run python -m scripts.workflows.result_index.ingest --dashboard-json apps/experiment_dashboard/data/experiment_dashboard.json</code></span>
    `;
  }
}

function hydrateEvalFilters() {
  const centralEvalValues = centralEvalSets(state.bundle);
  const defaultCentralEval = centralEvalValues.includes("validation")
    ? "validation"
    : centralEvalValues[0] ?? "validation";
  for (const key of [
    "overviewEvalSet",
    "compareEvalSet",
    "classEvalSet",
    "projectionEvalSet",
  ]) {
    if (!centralEvalValues.includes(state.central[key])) {
      state.central[key] = defaultCentralEval;
    }
  }
  for (const select of [
    elements.overviewEvalFilter,
    elements.comparisonEvalFilter,
    elements.classEvalFilter,
    elements.projectionEvalFilter,
  ]) {
    fillSelect(
      select,
      centralEvalValues,
      selectDefault(select.id),
      "eval 없음",
      centralEvalSetLabel,
    );
  }
}

function selectDefault(id) {
  if (id === "overview-eval-filter") return state.central.overviewEvalSet;
  if (id === "comparison-eval-filter") return state.central.compareEvalSet;
  if (id === "class-eval-filter") return state.central.classEvalSet;
  if (id === "projection-eval-filter") return state.central.projectionEvalSet;
  return "validation";
}

function bindEvents() {
  elements.trackButtons.forEach((button) => {
    button.addEventListener("click", () => {
      state.activeTrack = button.dataset.track;
      renderShell();
    });
  });
  elements.tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      state.activeCentralTab = button.dataset.tab;
      renderShell();
    });
  });
  elements.flTabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      state.activeFlTab = button.dataset.flTab;
      renderShell();
    });
  });
  bindCentralEvents();
  bindFlEvents();
  document.addEventListener("input", handleLiveInput);
  document.addEventListener("change", handleLiveInput);
}

function bindCentralEvents() {
  elements.centralFilterToggle.addEventListener("click", () => {
    state.central.filterPanelOpen = !state.central.filterPanelOpen;
    renderFilterVisibility();
  });
  elements.centralFilterAxisPicker.addEventListener("change", () => {
    state.central.filterAxisIds = checkedValues(
      elements.centralFilterAxisPicker,
      "centralFilterAxis",
    );
    for (const axisId of Object.keys(state.central.filterValues)) {
      if (!state.central.filterAxisIds.includes(axisId)) {
        delete state.central.filterValues[axisId];
      }
    }
    resetCentralSelections();
    render();
  });
  elements.centralActiveFilters.addEventListener("change", (event) => {
    const axisId = event.target.dataset.centralFilterValueAxis;
    if (!axisId) return;
    state.central.filterValues[axisId] = checkedValuesForAxis(
      elements.centralActiveFilters,
      "centralFilterValueAxis",
      axisId,
      "centralFilterValue",
    );
    resetCentralSelections();
    render();
  });
  elements.centralFilterReset.addEventListener("click", () => {
    state.central.filterAxisIds = [];
    state.central.filterValues = {};
    resetCentralSelections();
    render();
  });
  elements.overviewEvalFilter.addEventListener("change", (event) => {
    state.central.overviewEvalSet = event.target.value;
    state.central.overviewRunIds = [];
    render();
  });
  elements.overviewColumnTabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const columnTab = button.dataset.overviewColumnTab;
      if (columnTab !== "metric" && columnTab !== "axis") return;
      state.central.overviewColumnTab = columnTab;
      syncOverviewColumnTabUI();
      render();
    });
  });
  elements.overviewMetricPicker.addEventListener("change", () => {
    applyCentralOverviewColumnVisibility();
    render();
  });
  elements.overviewAxisPicker.addEventListener("change", () => {
    applyCentralOverviewColumnVisibility();
    render();
  });
  elements.overviewRunCheckboxes.addEventListener("change", () => {
    state.central.overviewRunIds = checkedValues(
      elements.overviewRunCheckboxes,
      "overviewRunId",
    );
    render();
  });
  elements.overviewSelectedRunCards.addEventListener("click", (event) => {
    const runId = event.target.dataset.removeOverviewRunId;
    if (!runId) return;
    state.central.overviewRunIds = state.central.overviewRunIds.filter((id) => id !== runId);
    render();
  });
  elements.comparisonEvalFilter.addEventListener("change", (event) => {
    state.central.compareEvalSet = event.target.value;
    state.central.compareRunIds = [];
    render();
  });
  elements.comparisonChartType.addEventListener("change", (event) => {
    state.central.compareChartType = event.target.value;
    render();
  });
  elements.comparisonIncludeInitial.addEventListener("change", (event) => {
    state.central.compareIncludeInitial = event.target.checked;
    render();
  });
  elements.metricPicker.addEventListener("change", (event) => {
    const metric = event.target.dataset.comparisonStepMetric;
    if (!metric) return;
    state.central.compareMetric = metric;
    render();
  });
  elements.comparisonRunCheckboxes.addEventListener("change", () => {
    state.central.compareRunIds = checkedValues(
      elements.comparisonRunCheckboxes,
      "runId",
    );
    render();
  });
  elements.selectedRunCards.addEventListener("click", (event) => {
    const runId = event.target.dataset.removeRunId;
    if (!runId) return;
    state.central.compareRunIds = state.central.compareRunIds.filter((id) => id !== runId);
    render();
  });
  elements.classEvalFilter.addEventListener("change", (event) => {
    state.central.classEvalSet = event.target.value;
    state.central.detailRunId = null;
    render();
  });
  elements.detailMethodFilter.addEventListener("change", (event) => {
    state.central.detailAlgorithm = event.target.value || null;
    state.central.detailRunId = null;
    render();
  });
  elements.detailRunFilter.addEventListener("change", (event) => {
    state.central.detailRunId = event.target.value || null;
    render();
  });
  elements.classMetricFilter.addEventListener("change", (event) => {
    state.central.classMetric = event.target.value;
    render();
  });
  elements.projectionEvalFilter.addEventListener("change", (event) => {
    state.central.projectionEvalSet = event.target.value;
    state.central.projectionRunIds = [];
    render();
  });
  elements.projectionMethodFilter.addEventListener("change", (event) => {
    state.central.projectionAlgorithm = event.target.value || null;
    state.central.projectionRunIds = [];
    render();
  });
  elements.projectionRunCheckboxes.addEventListener("change", () => {
    state.central.projectionRunIds = checkedValues(
      elements.projectionRunCheckboxes,
      "projectionRunId",
    );
    render();
  });
  elements.projectionGallery.addEventListener("click", (event) => {
    const runId = event.target.dataset.removeProjectionRunId;
    if (!runId) return;
    state.central.projectionRunIds = state.central.projectionRunIds.filter((id) => id !== runId);
    render();
  });
}

function bindFlEvents() {
  elements.flFilterToggle.addEventListener("click", () => {
    state.fl.filterPanelOpen = !state.fl.filterPanelOpen;
    renderFilterVisibility();
  });
  elements.flFilterAxisPicker.addEventListener("change", () => {
    state.fl.filterAxisIds = checkedValues(elements.flFilterAxisPicker, "flFilterAxis");
    for (const axisId of Object.keys(state.fl.filterValues)) {
      if (!state.fl.filterAxisIds.includes(axisId)) delete state.fl.filterValues[axisId];
    }
    resetFlSelectionsAfterFilterChange();
    render();
  });
  elements.flActiveFilters.addEventListener("change", (event) => {
    const axisId = event.target.dataset.flFilterValueAxis;
    if (!axisId) return;
    state.fl.filterValues[axisId] = checkedValuesForAxis(
      elements.flActiveFilters,
      "flFilterValueAxis",
      axisId,
      "flFilterValue",
    );
    resetFlSelectionsAfterFilterChange();
    render();
  });
  elements.flFilterReset.addEventListener("click", () => {
    state.fl.filterAxisIds = [];
    state.fl.filterValues = {};
    resetFlSelectionsAfterFilterChange();
    render();
  });
  elements.flRunColumnTabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const columnTab = button.dataset.flRunColumnTab;
      if (columnTab !== "metric" && columnTab !== "axis") return;
      state.fl.runColumnTab = columnTab;
      syncFlRunColumnTabUI();
      render();
    });
  });
  elements.flRunMetricPicker.addEventListener("change", () => {
    applyFlRunColumnVisibility();
    render();
  });
  elements.flRunAxisPicker.addEventListener("change", () => {
    applyFlRunColumnVisibility();
    render();
  });
  elements.flRunCheckboxes.addEventListener("change", () => {
    state.fl.runIds = checkedValues(elements.flRunCheckboxes, "flRunId");
    render();
  });
  elements.flRunSelectedRunCards.addEventListener("click", (event) => {
    const runId = event.target.dataset.removeFlRunId;
    if (!runId) return;
    state.fl.runIds = state.fl.runIds.filter((id) => id !== runId);
    render();
  });
  elements.flRoundRunCheckboxes.addEventListener("change", () => {
    state.fl.roundRunIds = checkedValues(elements.flRoundRunCheckboxes, "flRoundRunId");
    render();
  });
  elements.flRoundSelectedRunCards.addEventListener("click", (event) => {
    const runId = event.target.dataset.removeFlRoundRunId;
    if (!runId) return;
    state.fl.roundRunIds = state.fl.roundRunIds.filter((id) => id !== runId);
    render();
  });
  elements.flRoundIncludeInitial.addEventListener("change", (event) => {
    state.fl.roundIncludeInitial = event.target.checked;
    render();
  });
  elements.flRoundTableMetricPicker.addEventListener("change", () => {
    applyFlRoundTableColumnVisibility();
    render();
  });
  elements.flClientValidationRunFilter.addEventListener("change", (event) => {
    state.fl.clientValidationRunId = event.target.value || null;
    render();
  });
  elements.flClientRoundRunFilter.addEventListener("change", (event) => {
    state.fl.clientRoundRunId = event.target.value || null;
    state.fl.clientRoundIndex = "__latest__";
    render();
  });
  elements.flClientRoundFilter.addEventListener("change", (event) => {
    state.fl.clientRoundIndex = event.target.value || "__latest__";
    render();
  });
  elements.flSplitRunFilter.addEventListener("change", (event) => {
    state.fl.splitRunId = event.target.value || null;
    render();
  });
  elements.flProjectionEvalFilter.addEventListener("change", (event) => {
    state.fl.projectionEvalSet = event.target.value;
    state.fl.projectionRunIds = [];
    render();
  });
  elements.flProjectionRunCheckboxes.addEventListener("change", () => {
    state.fl.projectionRunIds = checkedValues(
      elements.flProjectionRunCheckboxes,
      "flProjectionRunId",
    );
    render();
  });
  elements.flProjectionGallery.addEventListener("click", (event) => {
    const runId = event.target.dataset.removeFlProjectionRunId;
    if (!runId) return;
    state.fl.projectionRunIds = state.fl.projectionRunIds.filter((id) => id !== runId);
    render();
  });
}

function handleLiveInput(event) {
  if (!(event.target instanceof HTMLInputElement)) return;
  const input = event.target;
  if (input.dataset.overviewAliasRunId) {
    updateAlias(state.central.overviewRunAliases, "central_overview", input.dataset.overviewAliasRunId, input.value);
    return;
  }
  if (input.dataset.comparisonAliasRunId) {
    updateAlias(state.central.compareRunAliases, "central_compare", input.dataset.comparisonAliasRunId, input.value);
    return;
  }
  if (input.dataset.flRunAliasRunId) {
    updateAlias(state.fl.runAliases, "fl_runs", input.dataset.flRunAliasRunId, input.value);
    return;
  }
  if (input.dataset.flRoundAliasRunId) {
    updateAlias(state.fl.roundRunAliases, "fl_round", input.dataset.flRoundAliasRunId, input.value);
    return;
  }
  if (input.dataset.chartAxisLabelScope) {
    updateAxisLabel(input.dataset.chartAxisLabelScope, input.value);
    return;
  }
  if (input.dataset.seriesColorScope && input.dataset.seriesColorKey) {
    updateSeriesColor(input.dataset.seriesColorScope, input.dataset.seriesColorKey, input.value);
  }
}

function render() {
  if (!state.bundle) return;
  renderShell();
  renderCentral();
  renderFl();
}

function renderShell() {
  elements.trackButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.track === state.activeTrack);
  });
  elements.trackPanels.forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.trackPanel === state.activeTrack);
  });
  elements.tabButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === state.activeCentralTab);
  });
  elements.tabPanels.forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.panel === state.activeCentralTab);
  });
  elements.flTabButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.flTab === state.activeFlTab);
  });
  elements.flTabPanels.forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.flPanel === state.activeFlTab);
  });
  syncOverviewColumnTabUI();
  syncFlRunColumnTabUI();
  renderFilterVisibility();
}

function renderFilterVisibility() {
  elements.centralFilterCard.classList.toggle("open", state.central.filterPanelOpen);
  elements.centralFilterToggle.setAttribute("aria-expanded", String(state.central.filterPanelOpen));
  elements.flFilterCard.classList.toggle("open", state.fl.filterPanelOpen);
  elements.flFilterToggle.setAttribute("aria-expanded", String(state.fl.filterPanelOpen));
}

function renderCentral() {
  const overviewRowsAll = centralMetricRows(
    state.bundle,
    state.central.overviewEvalSet,
    state.central.overviewMetricIds[0] ?? "macro_f1",
  );
  pruneCentralFilters(overviewRowsAll, state.central);
  const overviewRows = applyCentralFilters(overviewRowsAll, state.central);
  const compareRows = applyCentralFilters(
    centralMetricRows(state.bundle, state.central.compareEvalSet),
    state.central,
  );
  const classRows = applyCentralFilters(
    centralMetricRows(state.bundle, state.central.classEvalSet),
    state.central,
  );
  const projectionRows = applyCentralFilters(
    centralMetricRows(state.bundle, state.central.projectionEvalSet),
    state.central,
  );
  renderFilterPanel({
    axisPicker: elements.centralFilterAxisPicker,
    activeFilters: elements.centralActiveFilters,
    summary: elements.centralFilterSummary,
    rows: overviewRowsAll,
    filteredRows: overviewRows,
    axes: CENTRAL_FILTER_AXES,
    selectedAxisIds: state.central.filterAxisIds,
    selectedValues: state.central.filterValues,
    dataPrefix: "central",
  });
  normalizeOverviewSelection(overviewRows, state.central);
  normalizeCompareSelection(compareRows, state.central, state.bundle);
  normalizeDetailSelection(classRows, state.central);
  normalizeProjectionSelection(state.bundle, projectionRows, state.central);
  renderOverviewPage(elements, overviewRows, state.central, state.bundle, render);
  renderComparePage(elements, compareRows, state.central, state.bundle);
  renderDetailPage(elements, classRows, state.central, state.bundle, render);
  renderProjectionPage(elements, projectionRows, state.central, state.bundle);
}

function renderFl() {
  const allRows = sortedFlRows(flSslRows(state.bundle));
  pruneFlFilters(state.bundle, allRows, state.fl);
  const rows = applyFlFilters(state.bundle, allRows, state.fl);
  renderFilterPanel({
    axisPicker: elements.flFilterAxisPicker,
    activeFilters: elements.flActiveFilters,
    summary: elements.flFilterSummary,
    rows: allRows,
    filteredRows: rows,
    axes: flFilterAxes(state.bundle),
    selectedAxisIds: state.fl.filterAxisIds,
    selectedValues: state.fl.filterValues,
    dataPrefix: "fl",
  });
  normalizeFlRunSelection(rows, state.fl);
  normalizeRoundSelection(rows, state.fl);
  normalizeClientSelections(rows, state.fl, state.bundle);
  normalizeSplitSelection(rows, state.fl, state.bundle);
  normalizeFlProjectionSelection(state.bundle, rows, state.fl);
  elements.flRoundIncludeInitial.checked = state.fl.roundIncludeInitial;
  renderFlRunsPage(elements, rows, state.fl, state.bundle, render);
  renderRoundsPage(elements, rows, state.fl, state.bundle, render);
  renderClientsPage(elements, rows, state.fl, state.bundle, render);
  renderSplitsPage(elements, rows, state.fl, state.bundle, render);
  renderFlProjectionPage(elements, rows, state.fl, state.bundle);
}

function checkedValuesForAxis(container, axisDatasetKey, axisId, valueDatasetKey) {
  return Array.from(container.querySelectorAll("input[type='checkbox']:checked"))
    .filter((input) => input.dataset[axisDatasetKey] === axisId)
    .map((input) => input.dataset[valueDatasetKey])
    .filter(Boolean);
}

function syncTableVisibility(columnState, visibleColumnIds) {
  if (!columnState) return;
  const requested = Array.from(new Set((Array.isArray(visibleColumnIds) ? visibleColumnIds : []).filter(Boolean)));
  const requestedSet = new Set(requested);
  columnState.visible = requested;
  const currentOrder = Array.isArray(columnState.order) ? columnState.order.slice() : [];
  const normalizedOrder = currentOrder.filter((id) => requestedSet.has(id));
  for (const id of requested) {
    if (!normalizedOrder.includes(id)) normalizedOrder.push(id);
  }
  columnState.order = normalizedOrder;
}

function applyCentralOverviewColumnVisibility() {
  const selectedColumns = [
    ...checkedValues(elements.overviewMetricPicker, "overviewTableColumn"),
    ...checkedValues(elements.overviewAxisPicker, "overviewTableColumn"),
  ];
  syncTableVisibility(state.central.overviewTableColumns, selectedColumns);
  state.central.overviewMetricIds = checkedValues(elements.overviewMetricPicker, "overviewTableColumn").map((columnId) =>
    columnId.replace(/^metric:/, ""),
  );
}

function applyFlRunColumnVisibility() {
  const selectedColumns = [
    ...checkedValues(elements.flRunMetricPicker, "flRunTableColumn"),
    ...checkedValues(elements.flRunAxisPicker, "flRunTableColumn"),
  ];
  syncTableVisibility(state.fl.runTableColumns, selectedColumns);
  state.fl.runMetricIds = checkedValues(elements.flRunMetricPicker, "flRunTableColumn").map((columnId) =>
    columnId.replace(/^metric:/, ""),
  );
}

function applyFlRoundTableColumnVisibility() {
  const selectedMetricElement = elements.flRoundTableMetricPicker.querySelector("input[name='fl-round-metric']:checked");
  const selectedMetricId = selectedMetricElement?.dataset.flRoundTableMetric ?? "";
  const fallbackMetricId = "metric:macro_f1";
  const metricId = selectedMetricId || fallbackMetricId;
  const selectedMetricColumns = metricId ? [metricId] : [];
  const visibleColumns = ["axis:round", "axis:run", ...selectedMetricColumns];
  syncTableVisibility(state.fl.roundTableColumns, visibleColumns);
  const metricIds = selectedMetricColumns.map((columnId) => columnId.replace(/^metric:/, ""));
  state.fl.roundMetricIds = metricIds;
  if (metricIds.length === 0) {
    state.fl.roundMetric = "macro_f1";
  } else if (!metricIds.includes(state.fl.roundMetric)) {
    state.fl.roundMetric = metricIds[0];
  }
}

function syncOverviewColumnTabUI() {
  elements.overviewMetricPicker.hidden = state.central.overviewColumnTab !== "metric";
  elements.overviewAxisPicker.hidden = state.central.overviewColumnTab !== "axis";
  elements.overviewColumnTabButtons.forEach((button) => {
    const tab = button.dataset.overviewColumnTab;
    button.classList.toggle("active", tab === state.central.overviewColumnTab);
    button.setAttribute("aria-selected", tab === state.central.overviewColumnTab ? "true" : "false");
  });
}

function syncFlRunColumnTabUI() {
  elements.flRunMetricPicker.hidden = state.fl.runColumnTab !== "metric";
  elements.flRunAxisPicker.hidden = state.fl.runColumnTab !== "axis";
  elements.flRunColumnTabButtons.forEach((button) => {
    const tab = button.dataset.flRunColumnTab;
    button.classList.toggle("active", tab === state.fl.runColumnTab);
    button.setAttribute("aria-selected", tab === state.fl.runColumnTab ? "true" : "false");
  });
}

function resetCentralSelections() {
  state.central.overviewRunIds = [];
  state.central.compareRunIds = [];
  state.central.detailAlgorithm = null;
  state.central.detailRunId = null;
  state.central.projectionAlgorithm = null;
  state.central.projectionRunIds = [];
}

function resetFlSelectionsAfterFilterChange() {
  state.fl.runIds = [];
  state.fl.roundRunIds = state.fl.roundRunIds.filter(Boolean);
  state.fl.clientValidationRunId = null;
  state.fl.clientRoundRunId = null;
  state.fl.splitRunId = null;
  state.fl.projectionRunIds = [];
}

function updateAlias(aliasMap, scope, runId, value) {
  const alias = value.trim();
  if (alias) aliasMap[runId] = alias;
  else delete aliasMap[runId];
  storeRunAliases(scope, aliasMap);
}

function updateAxisLabel(scope, value) {
  if (scope === "central_compare") {
    state.central.compareAxisLabel = value.trim();
  } else if (scope === "fl_round") {
    state.fl.roundAxisLabel = value.trim();
  }
  const label = document.querySelector(`[data-chart-axis-label='${scope}']`);
  if (label) {
    label.textContent = value.trim() || label.textContent;
  }
}

function updateSeriesColor(scope, colorKey, value) {
  if (scope === "central_compare") {
    state.central.compareRunColors[colorKey] = value;
    storeSeriesColors(scope, state.central.compareRunColors);
  } else if (scope === "fl_round") {
    state.fl.roundRunColors[colorKey] = value;
    storeSeriesColors(scope, state.fl.roundRunColors);
  }
  document
    .querySelectorAll(`[data-series-color-key='${CSS.escape(colorKey)}']`)
    .forEach((item) => {
      item.style.setProperty("--series-color", value);
    });
}
