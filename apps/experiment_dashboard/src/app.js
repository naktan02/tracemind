const DATA_URL = "./data/experiment_dashboard.json";
const CENTRAL_SSL_TRACK = "central_lora_ssl";
const FL_ROUND_METRICS = [
  "macro_f1",
  "accuracy_top_1",
  "expected_calibration_error",
  "loss",
  "accepted_ratio",
];

const state = {
  bundle: null,
  activeTrack: "central_ssl",
  overviewEvalSet: "validation",
  overviewMetric: "macro_f1",
  comparisonEvalSet: "validation",
  comparisonMetrics: ["macro_f1", "accuracy_top_1"],
  comparisonChartType: "grouped_bar",
  comparisonMethodName: null,
  comparisonSelectionTouched: false,
  showEpochChart: true,
  epochMetric: "selection_macro_f1",
  classEvalSet: "validation",
  classMetric: "f1",
  selectedRunIds: [],
  detailMethodName: null,
  detailRunId: null,
  projectionEvalSet: "validation",
  projectionMethodName: null,
  projectionRunIds: [],
  projectionSelectionTouched: false,
  activeTab: "overview",
  activeFlTab: "runs",
  flRoundCount: "__all__",
  flRoundMethodName: null,
  flRoundRunIds: [],
  flRoundSelectionTouched: false,
  flRoundRunAliases: {},
  flRoundMetric: "macro_f1",
  flClientValidationRunId: null,
  flClientRoundRunId: null,
  flClientRoundIndex: "__latest__",
  flSplitRunId: null,
};

const elements = {
  loadState: document.querySelector("#load-state"),
  trackButtons: document.querySelectorAll("[data-track]"),
  trackPanels: document.querySelectorAll("[data-track-panel]"),
  tabButtons: document.querySelectorAll("[data-tab]"),
  tabPanels: document.querySelectorAll("[data-panel]"),
  flTabButtons: document.querySelectorAll("[data-fl-tab]"),
  flTabPanels: document.querySelectorAll("[data-fl-panel]"),
  overviewEvalFilter: document.querySelector("#overview-eval-filter"),
  overviewMetricFilter: document.querySelector("#overview-metric-filter"),
  comparisonEvalFilter: document.querySelector("#comparison-eval-filter"),
  comparisonChartType: document.querySelector("#comparison-chart-type"),
  metricPicker: document.querySelector("#metric-picker"),
  comparisonMethodFilter: document.querySelector("#comparison-method-filter"),
  comparisonRunCheckboxes: document.querySelector("#comparison-run-checkboxes"),
  selectedRunCards: document.querySelector("#selected-run-cards"),
  epochToggle: document.querySelector("#epoch-toggle"),
  epochPanel: document.querySelector("#epoch-panel"),
  epochMetricFilter: document.querySelector("#epoch-metric-filter"),
  classEvalFilter: document.querySelector("#class-eval-filter"),
  detailMethodFilter: document.querySelector("#detail-method-filter"),
  detailRunFilter: document.querySelector("#detail-run-filter"),
  detailRunSummary: document.querySelector("#detail-run-summary"),
  classMetricFilter: document.querySelector("#class-metric-filter"),
  metricCards: document.querySelector("#metric-cards"),
  comparisonChart: document.querySelector("#comparison-chart"),
  barChart: document.querySelector("#bar-chart"),
  epochChart: document.querySelector("#epoch-chart"),
  classChart: document.querySelector("#class-chart"),
  classTable: document.querySelector("#class-table"),
  confusionMatrix: document.querySelector("#confusion-matrix"),
  projectionEvalFilter: document.querySelector("#projection-eval-filter"),
  projectionMethodFilter: document.querySelector("#projection-method-filter"),
  projectionRunCheckboxes: document.querySelector("#projection-run-checkboxes"),
  projectionGallery: document.querySelector("#projection-gallery"),
  runTable: document.querySelector("#run-table"),
  flMetricCards: document.querySelector("#fl-metric-cards"),
  flRunTable: document.querySelector("#fl-run-table"),
  flRoundCountFilter: document.querySelector("#fl-round-count-filter"),
  flRoundMethodFilter: document.querySelector("#fl-round-method-filter"),
  flRoundRunCheckboxes: document.querySelector("#fl-round-run-checkboxes"),
  flRoundSelectedRunCards: document.querySelector(
    "#fl-round-selected-run-cards",
  ),
  flRoundMetricPicker: document.querySelector("#fl-round-metric-picker"),
  flRoundFlatNote: document.querySelector("#fl-round-flat-note"),
  flRoundChart: document.querySelector("#fl-round-chart"),
  flRoundTable: document.querySelector("#fl-round-table"),
  flClientValidationRunFilter: document.querySelector(
    "#fl-client-validation-run-filter",
  ),
  flClientValidationTable: document.querySelector("#fl-client-validation-table"),
  flClientRoundRunFilter: document.querySelector("#fl-client-round-run-filter"),
  flClientRoundFilter: document.querySelector("#fl-client-round-filter"),
  flClientRoundTable: document.querySelector("#fl-client-round-table"),
  flSplitRunFilter: document.querySelector("#fl-split-run-filter"),
  flSplitTable: document.querySelector("#fl-split-table"),
};

init();

async function init() {
  bindEvents();
  try {
    const response = await fetch(DATA_URL, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    state.bundle = await response.json();
    elements.loadState.className = "notice success";
    elements.loadState.textContent = "dashboard data 로드 완료";
    hydrateFilters();
    render();
  } catch (error) {
    elements.loadState.className = "notice warning";
    elements.loadState.innerHTML = `
      <strong>dashboard data를 찾지 못했습니다.</strong>
      <span>먼저 result index export를 실행하세요: <code>uv run python -m scripts.experiments.result_index.ingest --dashboard-json apps/experiment_dashboard/data/experiment_dashboard.json</code></span>
    `;
  }
}

function bindEvents() {
  elements.trackButtons.forEach((button) => {
    button.addEventListener("click", () => {
      state.activeTrack = button.dataset.track;
      render();
    });
  });
  elements.tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      state.activeTab = button.dataset.tab;
      renderTabs();
    });
  });
  elements.flTabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      state.activeFlTab = button.dataset.flTab;
      renderFlTabs();
    });
  });
  elements.overviewEvalFilter.addEventListener("change", (event) => {
    state.overviewEvalSet = event.target.value;
    render();
  });
  elements.overviewMetricFilter.addEventListener("change", (event) => {
    state.overviewMetric = event.target.value;
    render();
  });
  elements.comparisonEvalFilter.addEventListener("change", (event) => {
    state.comparisonEvalSet = event.target.value;
    resetComparisonSelection();
    render();
  });
  elements.comparisonChartType.addEventListener("change", (event) => {
    state.comparisonChartType = event.target.value;
    render();
  });
  elements.metricPicker.addEventListener("change", (event) => {
    if (!(event.target instanceof HTMLInputElement)) {
      return;
    }
    state.comparisonMetrics = checkedValues(elements.metricPicker, "metric");
    render();
  });
  elements.comparisonMethodFilter.addEventListener("change", (event) => {
    state.comparisonMethodName = event.target.value || null;
    render();
  });
  elements.comparisonRunCheckboxes.addEventListener("change", (event) => {
    if (!(event.target instanceof HTMLInputElement)) {
      return;
    }
    const candidateIds = checkedValues(elements.comparisonRunCheckboxes, "runId");
    const candidateRows = state.comparisonMethodName
      ? rowsForMethods(selectedMetricRows(state.comparisonEvalSet), [
          state.comparisonMethodName,
        ])
      : [];
    const candidateIdSet = new Set(candidateRows.map((row) => row.run_id));
    state.selectedRunIds = uniqueValues([
      ...state.selectedRunIds.filter((runId) => !candidateIdSet.has(runId)),
      ...candidateIds,
    ]);
    state.comparisonSelectionTouched = true;
    render();
  });
  elements.selectedRunCards.addEventListener("click", (event) => {
    if (!(event.target instanceof HTMLButtonElement)) {
      return;
    }
    const runId = event.target.dataset.removeRunId;
    if (!runId) {
      return;
    }
    state.selectedRunIds = state.selectedRunIds.filter(
      (selectedRunId) => selectedRunId !== runId,
    );
    state.comparisonSelectionTouched = true;
    render();
  });
  elements.epochToggle.addEventListener("change", (event) => {
    state.showEpochChart = event.target.checked;
    render();
  });
  elements.epochMetricFilter.addEventListener("change", (event) => {
    state.epochMetric = event.target.value;
    render();
  });
  elements.classEvalFilter.addEventListener("change", (event) => {
    state.classEvalSet = event.target.value;
    state.detailMethodName = null;
    state.detailRunId = null;
    render();
  });
  elements.detailMethodFilter.addEventListener("change", (event) => {
    state.detailMethodName = event.target.value || null;
    state.detailRunId = null;
    render();
  });
  elements.detailRunFilter.addEventListener("change", (event) => {
    state.detailRunId = event.target.value || null;
    render();
  });
  elements.classMetricFilter.addEventListener("change", (event) => {
    state.classMetric = event.target.value;
    render();
  });
  elements.projectionEvalFilter.addEventListener("change", (event) => {
    state.projectionEvalSet = event.target.value;
    state.projectionMethodName = null;
    state.projectionRunIds = [];
    state.projectionSelectionTouched = false;
    render();
  });
  elements.projectionMethodFilter.addEventListener("change", (event) => {
    state.projectionMethodName = event.target.value || null;
    render();
  });
  elements.projectionRunCheckboxes.addEventListener("change", (event) => {
    if (!(event.target instanceof HTMLInputElement)) {
      return;
    }
    const candidateIds = checkedValues(
      elements.projectionRunCheckboxes,
      "runId",
    );
    const candidateRows = state.projectionMethodName
      ? rowsWithProjection(
          rowsForMethods(selectedMetricRows(state.projectionEvalSet), [
            state.projectionMethodName,
          ]),
        )
      : [];
    const candidateIdSet = new Set(candidateRows.map((row) => row.run_id));
    state.projectionRunIds = uniqueValues([
      ...state.projectionRunIds.filter((runId) => !candidateIdSet.has(runId)),
      ...candidateIds,
    ]);
    state.projectionSelectionTouched = true;
    render();
  });
  elements.projectionGallery.addEventListener("click", (event) => {
    if (!(event.target instanceof HTMLButtonElement)) {
      return;
    }
    const runId = event.target.dataset.removeProjectionRunId;
    if (!runId) {
      return;
    }
    state.projectionRunIds = state.projectionRunIds.filter(
      (selectedRunId) => selectedRunId !== runId,
    );
    state.projectionSelectionTouched = true;
    render();
  });
  elements.flRoundCountFilter.addEventListener("change", (event) => {
    state.flRoundCount = event.target.value || "__all__";
    state.flRoundRunIds = [];
    state.flRoundSelectionTouched = false;
    render();
  });
  elements.flRoundMethodFilter.addEventListener("change", (event) => {
    state.flRoundMethodName = event.target.value || null;
    state.flRoundRunIds = [];
    state.flRoundSelectionTouched = false;
    render();
  });
  elements.flRoundRunCheckboxes.addEventListener("change", (event) => {
    if (!(event.target instanceof HTMLInputElement)) {
      return;
    }
    state.flRoundRunIds = checkedValues(
      elements.flRoundRunCheckboxes,
      "flRoundRunId",
    );
    state.flRoundSelectionTouched = true;
    render();
  });
  elements.flRoundSelectedRunCards.addEventListener("click", (event) => {
    if (!(event.target instanceof HTMLButtonElement)) {
      return;
    }
    const runId = event.target.dataset.removeFlRoundRunId;
    if (!runId) {
      return;
    }
    state.flRoundRunIds = state.flRoundRunIds.filter(
      (selectedRunId) => selectedRunId !== runId,
    );
    state.flRoundSelectionTouched = true;
    render();
  });
  elements.flRoundSelectedRunCards.addEventListener("change", (event) => {
    if (!(event.target instanceof HTMLInputElement)) {
      return;
    }
    updateFlRoundRunAlias(event.target);
  });
  elements.flRoundSelectedRunCards.addEventListener("input", (event) => {
    if (!(event.target instanceof HTMLInputElement)) {
      return;
    }
    updateFlRoundRunAlias(event.target);
  });
  elements.flRoundMetricPicker.addEventListener("click", (event) => {
    if (!(event.target instanceof HTMLButtonElement)) {
      return;
    }
    state.flRoundMetric = event.target.dataset.flRoundMetric ?? state.flRoundMetric;
    render();
  });
  elements.flClientValidationRunFilter.addEventListener("change", (event) => {
    state.flClientValidationRunId = event.target.value || null;
    render();
  });
  elements.flClientRoundRunFilter.addEventListener("change", (event) => {
    state.flClientRoundRunId = event.target.value || null;
    state.flClientRoundIndex = "__latest__";
    render();
  });
  elements.flClientRoundFilter.addEventListener("change", (event) => {
    state.flClientRoundIndex = event.target.value;
    render();
  });
  elements.flSplitRunFilter.addEventListener("change", (event) => {
    state.flSplitRunId = event.target.value || null;
    render();
  });
}

function hydrateFilters() {
  const evalSets = centralEvalSets();
  ensureEvalDefaults(evalSets);
  fillSelect(elements.overviewEvalFilter, evalSets, state.overviewEvalSet);
  fillSelect(elements.comparisonEvalFilter, evalSets, state.comparisonEvalSet);
  fillSelect(elements.classEvalFilter, evalSets, state.classEvalSet);
  fillSelect(elements.projectionEvalFilter, evalSets, state.projectionEvalSet);
  renderMetricPicker();
}

function render() {
  if (!state.bundle) {
    return;
  }
  renderTrackPanels();
  renderTabs();
  const overviewRows = selectedMetricRows(
    state.overviewEvalSet,
    state.overviewMetric,
  );
  const comparisonRows = selectedMetricRows(state.comparisonEvalSet);
  const classRows = selectedMetricRows(state.classEvalSet);
  const projectionBaseRows = selectedMetricRows(state.projectionEvalSet);
  normalizeComparisonSelection(comparisonRows);
  normalizeDetailSelection(classRows);
  normalizeProjectionSelection(projectionBaseRows);
  const projectionRows = rowsWithProjection(projectionBaseRows);

  renderEvalFilters();
  renderMetricCards(overviewRows);
  renderMetricPicker();
  renderComparisonRunControls(comparisonRows);
  renderSelectedRunCards(comparisonRows);
  renderDetailRunFilter(classRows);
  renderProjectionRunControls(projectionRows);
  renderComparisonChart(comparisonRows);
  renderBarChart(overviewRows);
  renderEpochChart();
  renderClassChart();
  renderClassTable();
  renderConfusionMatrix();
  renderProjectionGallery();
  renderRunTable(overviewRows);
  renderFlSslPanel();
}

function renderTrackPanels() {
  elements.trackButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.track === state.activeTrack);
  });
  elements.trackPanels.forEach((panel) => {
    panel.classList.toggle(
      "active",
      panel.dataset.trackPanel === state.activeTrack,
    );
  });
}

function renderTabs() {
  elements.tabButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === state.activeTab);
  });
  elements.tabPanels.forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.panel === state.activeTab);
  });
  renderFlTabs();
}

function renderFlTabs() {
  elements.flTabButtons.forEach((button) => {
    button.classList.toggle(
      "active",
      button.dataset.flTab === state.activeFlTab,
    );
  });
  elements.flTabPanels.forEach((panel) => {
    panel.classList.toggle(
      "active",
      panel.dataset.flPanel === state.activeFlTab,
    );
  });
}

function ensureEvalDefaults(evalSets) {
  const fallback = evalSets.includes("validation") ? "validation" : evalSets[0];
  if (!fallback) {
    return;
  }
  for (const key of [
    "overviewEvalSet",
    "comparisonEvalSet",
    "classEvalSet",
    "projectionEvalSet",
  ]) {
    if (!evalSets.includes(state[key])) {
      state[key] = fallback;
    }
  }
}

function renderEvalFilters() {
  const evalSets = centralEvalSets();
  fillSelect(elements.overviewEvalFilter, evalSets, state.overviewEvalSet);
  elements.overviewMetricFilter.value = state.overviewMetric;
  fillSelect(elements.comparisonEvalFilter, evalSets, state.comparisonEvalSet);
  fillSelect(elements.classEvalFilter, evalSets, state.classEvalSet);
  fillSelect(elements.projectionEvalFilter, evalSets, state.projectionEvalSet);
}

function selectedMetricRows(evalSet, sortMetric = "macro_f1") {
  const runsById = new Map(state.bundle.runs.map((run) => [run.run_id, run]));
  return state.bundle.eval_metrics
    .filter((metric) => metric.eval_set === evalSet)
    .map((metric) => ({ ...runsById.get(metric.run_id), ...metric }))
    .filter((row) => row.run_id && row.track === CENTRAL_SSL_TRACK)
    .sort((a, b) => compareMetric(a, b, sortMetric));
}

function centralEvalSets() {
  const runIds = new Set(
    state.bundle.runs
      .filter((run) => run.track === CENTRAL_SSL_TRACK)
      .map((run) => run.run_id),
  );
  return uniqueValues(
    state.bundle.eval_metrics
      .filter((metric) => runIds.has(metric.run_id))
      .map((metric) => metric.eval_set),
  ).sort();
}

function renderFlSslPanel() {
  const rows = sortedFlRows(flSslRows());
  normalizeFlSelections(rows);
  renderFlTabs();

  const methods = new Set(rows.map((row) => flMethodName(row))).size;
  const macroF1Values = rows
    .map((row) => numberOrNull(flMetric(row, "macro_f1")))
    .filter((value) => value !== null);
  const worstClientValues = rows
    .map((row) => numberOrNull(flMetric(row, "worst_client_macro_f1")))
    .filter((value) => value !== null);

  elements.flMetricCards.innerHTML = [
    card("runs", rows.length),
    card("methods", methods),
    card(
      "best macro_f1",
      macroF1Values.length ? formatMetric(Math.max(...macroF1Values)) : "-",
    ),
    card(
      "worst client",
      worstClientValues.length ? formatMetric(Math.min(...worstClientValues)) : "-",
    ),
  ].join("");

  renderFlRunTable(rows);
  renderFlRunSelectors(rows);
  renderFlRoundPanel();
  renderFlClientValidationPanel();
  renderFlClientRoundPanel();
  renderFlSplitPanel();
}

function renderFlRunTable(rows) {
  if (rows.length === 0) {
    elements.flRunTable.innerHTML = emptyTableRow(
      14,
      "아직 dashboard bundle에 FL SSL run이 없습니다.",
    );
    return;
  }

  elements.flRunTable.innerHTML = rows
    .map(
      (row) => `
        <tr>
          <td>${escapeHtml(shortRun(flRunId(row)))}</td>
          <td>${escapeHtml(flMethodName(row))}</td>
          <td>${escapeHtml(shortSplit(row.selection_slug))}</td>
          <td>${formatCount(row.client_count)}</td>
          <td>${formatCount(row.completed_rounds)} / ${formatCount(row.round_budget)}</td>
          <td>${formatCount(row.labeled_row_exposure_count)}</td>
          <td>${formatCount(row.unique_labeled_row_count)}</td>
          <td>${formatMetric(flMetric(row, "macro_f1"))}</td>
          <td>${formatMetric(flMetric(row, "worst_client_macro_f1"))}</td>
          <td>${formatMetric(flMetric(row, "expected_calibration_error"))}</td>
          <td>${formatMetric(row.macro_f1_std)}</td>
          <td>${formatCount(flCostValue(row))}</td>
          <td>${formatBytes(flPosthocCommunicationBytes(row, "c2s_total_bytes"))}</td>
          <td>${formatBytes(flPosthocCommunicationBytes(row, "s2c_total_bytes_estimated"))}</td>
        </tr>
      `,
    )
    .join("");
}

function renderFlRunSelectors(rows) {
  const roundRuns = flRunsWithRows(rows, flRoundRows());
  fillFlRoundCountSelect(elements.flRoundCountFilter, roundRuns);
  const roundCountRuns = flRoundRunsForSelectedRoundCount(roundRuns);
  fillFlMethodSelect(
    elements.flRoundMethodFilter,
    roundCountRuns,
    state.flRoundMethodName,
  );
  renderFlRoundRunControls(roundCountRuns);
  renderFlRoundMetricTabs();
  fillFlRunSelect(
    elements.flClientValidationRunFilter,
    flRunsWithRows(rows, flClientValidationRows()),
    state.flClientValidationRunId,
  );
  fillFlRunSelect(
    elements.flClientRoundRunFilter,
    flRunsWithRows(rows, flClientRoundRows()),
    state.flClientRoundRunId,
  );
  fillFlRunSelect(
    elements.flSplitRunFilter,
    flRunsWithRows(rows, flSplitRows()),
    state.flSplitRunId,
  );
}

function renderFlRoundPanel() {
  const rows = flRoundRowsForSelectedRuns();
  renderFlRoundFlatNote(rows);
  if (rows.length === 0) {
    elements.flRoundChart.innerHTML =
      `<p class="empty">선택한 run들의 round curve가 없습니다.</p>`;
    elements.flRoundTable.innerHTML = emptyTableRow(9, "round row 없음");
    return;
  }
  elements.flRoundChart.innerHTML = drawFlRoundLines(rows, state.flRoundMetric);
  elements.flRoundTable.innerHTML = rows
    .map(
      (row) => `
        <tr>
          <td>${escapeHtml(flRoundTableRoundLabel(row))}</td>
          <td>${formatMetric(row.macro_f1)}</td>
          <td>${formatMetric(row.accuracy_top_1)}</td>
          <td>${formatMetric(row.loss)}</td>
          <td>${formatMetric(row.expected_calibration_error)}</td>
          <td>${formatMetric(row.accepted_ratio)}</td>
          <td>${formatCount(row.update_count)}</td>
          <td>${formatCount(row.total_payload_bytes)}</td>
          <td>${formatSeconds(row.round_time_seconds)}</td>
        </tr>
      `,
    )
    .join("");
}

function renderFlClientValidationPanel() {
  const rows = flClientValidationRows()
    .filter((row) => row.run_id === state.flClientValidationRunId)
    .sort((a, b) =>
      compareNullableNumbers(
        a.client_validation_macro_f1,
        b.client_validation_macro_f1,
      ),
    );
  if (rows.length === 0) {
    elements.flClientValidationTable.innerHTML = emptyTableRow(
      10,
      "선택한 run의 client validation row가 없습니다.",
    );
    return;
  }
  elements.flClientValidationTable.innerHTML = rows
    .map(
      (row) => `
        <tr>
          <td>${escapeHtml(row.client_id ?? "-")}</td>
          <td>${formatCount(row.client_labeled_count)}</td>
          <td>${formatCount(row.client_unlabeled_count)}</td>
          <td>${formatMetric(row.client_validation_macro_f1)}</td>
          <td>${formatMetric(row.client_validation_accuracy_top_1)}</td>
          <td>${formatMetric(row.client_validation_loss)}</td>
          <td>${formatMetric(row.client_validation_ece)}</td>
          <td>${formatMetric(row.client_accepted_ratio)}</td>
          <td>${formatCount(row.update_generated_round_count)}</td>
          <td>${formatMetric(row.client_delta_l2_norm)}</td>
        </tr>
      `,
    )
    .join("");
}

function renderFlClientRoundPanel() {
  const rows = flClientRoundRows()
    .filter((row) => row.run_id === state.flClientRoundRunId)
    .sort(compareClientRoundRows);
  fillRoundSelect(elements.flClientRoundFilter, rows, state.flClientRoundIndex);
  const roundIndex = selectedClientRoundIndex(rows);
  const selectedRows = rows.filter(
    (row) => String(row.round_index) === String(roundIndex),
  );
  if (selectedRows.length === 0) {
    elements.flClientRoundTable.innerHTML = emptyTableRow(
      9,
      "선택한 run/round의 client update row가 없습니다.",
    );
    return;
  }
  elements.flClientRoundTable.innerHTML = selectedRows
    .map(
      (row) => `
        <tr>
          <td>${escapeHtml(row.client_id ?? "-")}</td>
          <td>${formatCount(row.candidate_count)}</td>
          <td>${formatCount(row.accepted_count)}</td>
          <td>${formatMetric(row.accepted_ratio)}</td>
          <td>${boolLabel(row.update_generated)}</td>
          <td>${formatMetric(row.delta_l2_norm)}</td>
          <td>${formatCount(row.client_payload_bytes)}</td>
          <td>${formatSeconds(row.client_train_time_seconds)}</td>
          <td>${formatMetric(row.pseudo_label_accuracy)}</td>
        </tr>
      `,
    )
    .join("");
}

function renderFlSplitPanel() {
  const rows = flSplitRows()
    .filter((row) => row.run_id === state.flSplitRunId)
    .sort((a, b) => String(a.client_id ?? "").localeCompare(String(b.client_id ?? "")));
  if (rows.length === 0) {
    elements.flSplitTable.innerHTML = emptyTableRow(
      6,
      "선택한 run의 client split row가 없습니다.",
    );
    return;
  }
  elements.flSplitTable.innerHTML = rows
    .map(
      (row) => `
        <tr>
          <td>${escapeHtml(row.client_id ?? "-")}</td>
          <td>${formatCount(row.labeled_count)}</td>
          <td>${formatCount(row.unlabeled_count)}</td>
          <td class="distribution-cell">${formatDistribution(row.labeled_label_distribution)}</td>
          <td class="distribution-cell">${formatDistribution(row.unlabeled_label_distribution)}</td>
          <td class="distribution-cell">${formatDistribution(row.label_distribution)}</td>
        </tr>
      `,
    )
    .join("");
}

function renderFlRoundMetricTabs() {
  elements.flRoundMetricPicker.innerHTML = FL_ROUND_METRICS.map(
    (metric) => `
      <button
        type="button"
        data-fl-round-metric="${metric}"
        class="${metric === state.flRoundMetric ? "active" : ""}"
      >${metricLabel(metric)}</button>
    `,
  ).join("");
}

function renderFlRoundRunControls(rows) {
  const candidateRows = flRoundCandidateRuns(rows);
  if (candidateRows.length === 0) {
    elements.flRoundRunCheckboxes.innerHTML =
      `<p class="empty">선택 가능한 run이 없습니다.</p>`;
    renderFlRoundSelectedRunCards(rows);
    return;
  }
  const selectedRunIds = new Set(state.flRoundRunIds);
  elements.flRoundRunCheckboxes.innerHTML = candidateRows
    .map((row) => {
      const runId = flRunId(row);
      return `
        <label class="run-option">
          <input
            type="checkbox"
            data-fl-round-run-id="${runId}"
            ${selectedRunIds.has(runId) ? "checked" : ""}
          />
          <span>
            <strong>${escapeHtml(defaultFlRoundRunLabel(row))}</strong>
            <small>${escapeHtml(flRunDescriptor(row))}</small>
          </span>
        </label>
      `;
    })
    .join("");
  renderFlRoundSelectedRunCards(rows);
}

function renderFlRoundSelectedRunCards(rows) {
  const rowsById = new Map(rows.map((row) => [flRunId(row), row]));
  const selectedRows = state.flRoundRunIds
    .map((runId) => rowsById.get(runId))
    .filter((row) => row);
  if (selectedRows.length === 0) {
    elements.flRoundSelectedRunCards.innerHTML =
      `<p class="empty">선택된 FL round run이 없습니다.</p>`;
    return;
  }
  elements.flRoundSelectedRunCards.innerHTML = selectedRows
    .map((row) => {
      const runId = flRunId(row);
      const detail = [
        defaultFlRoundRunLabel(row),
        flMethodName(row),
        flRunDescriptor(row),
      ].join(" · ");
      return `
        <article class="selected-run-card alias-run-card" title="${escapeHtml(detail)}">
          <strong>${escapeHtml(defaultFlRoundRunLabel(row))}</strong>
          <input
            type="text"
            data-fl-round-alias-run-id="${runId}"
            value="${escapeHtml(state.flRoundRunAliases[runId] ?? "")}"
            placeholder="legend alias"
            aria-label="${escapeHtml(defaultFlRoundRunLabel(row))} 범례 alias"
          />
          <button
            type="button"
            data-remove-fl-round-run-id="${runId}"
            aria-label="${escapeHtml(defaultFlRoundRunLabel(row))} 제거"
          >x</button>
          <span class="selected-run-detail" aria-hidden="true">${escapeHtml(detail)}</span>
        </article>
      `;
    })
    .join("");
}

function updateFlRoundRunAlias(input) {
  const runId = input.dataset.flRoundAliasRunId;
  if (!runId) {
    return;
  }
  const alias = input.value.trim();
  if (alias) {
    state.flRoundRunAliases[runId] = alias;
  } else {
    delete state.flRoundRunAliases[runId];
  }
  renderFlRoundPanel();
}

function renderFlRoundFlatNote(rows) {
  const uniqueValuesForMetric = uniqueValues(
    rows
      .map((row) => numberOrNull(row[state.flRoundMetric]))
      .filter((value) => value !== null),
  );
  if (rows.length === 0 || uniqueValuesForMetric.length !== 1) {
    elements.flRoundFlatNote.hidden = true;
    elements.flRoundFlatNote.textContent = "";
    return;
  }
  elements.flRoundFlatNote.hidden = false;
  elements.flRoundFlatNote.textContent = [
    `선택한 run에서 ${metricLabel(state.flRoundMetric)} 값이 전 라운드 동일합니다.`,
    "현재 기존 FL LoRA-classifier runs는 validation scorer가 prototype_similarity라 shared LoRA/classifier state를 직접 읽지 않아 aggregate 효과가 global validation curve에 반영되지 않습니다.",
  ].join(" ");
}

function normalizeFlSelections(rows) {
  if (!FL_ROUND_METRICS.includes(state.flRoundMetric)) {
    state.flRoundMetric = "macro_f1";
  }
  const roundRuns = flRunsWithRows(rows, flRoundRows());
  const roundCounts = flRoundCountOptions(roundRuns).map((option) => option.count);
  if (
    state.flRoundCount !== "__all__" &&
    !roundCounts.includes(Number(state.flRoundCount))
  ) {
    state.flRoundCount = "__all__";
  }
  const roundCountRuns = flRoundRunsForSelectedRoundCount(roundRuns);
  const methodNames = uniqueValues(roundCountRuns.map((row) => flMethodName(row)));
  if (
    state.flRoundMethodName &&
    !methodNames.includes(state.flRoundMethodName)
  ) {
    state.flRoundMethodName = null;
  }
  if (!state.flRoundMethodName && methodNames.length > 0) {
    state.flRoundMethodName = methodNames[0];
  }
  normalizeFlRoundRunSelection(
    flRoundCandidateRuns(roundCountRuns),
  );
  state.flClientValidationRunId = normalizeFlRunId(
    state.flClientValidationRunId,
    flRunsWithRows(rows, flClientValidationRows()),
  );
  state.flClientRoundRunId = normalizeFlRunId(
    state.flClientRoundRunId,
    flRunsWithRows(rows, flClientRoundRows()),
  );
  state.flSplitRunId = normalizeFlRunId(
    state.flSplitRunId,
    flRunsWithRows(rows, flSplitRows()),
  );

  const clientRoundIndexes = uniqueValues(
    flClientRoundRows()
      .filter((row) => row.run_id === state.flClientRoundRunId)
      .map((row) => String(row.round_index))
      .filter((value) => value !== "null" && value !== "undefined"),
  );
  if (
    state.flClientRoundIndex !== "__latest__" &&
    !clientRoundIndexes.includes(state.flClientRoundIndex)
  ) {
    state.flClientRoundIndex = "__latest__";
  }
}

function normalizeFlRoundRunSelection(candidateRows) {
  const visibleRunIds = new Set(candidateRows.map((row) => flRunId(row)));
  state.flRoundRunIds = state.flRoundRunIds.filter((runId) =>
    visibleRunIds.has(runId),
  );
  for (const runId of Object.keys(state.flRoundRunAliases)) {
    if (!visibleRunIds.has(runId)) {
      delete state.flRoundRunAliases[runId];
    }
  }
  if (!state.flRoundSelectionTouched && state.flRoundRunIds.length === 0) {
    state.flRoundRunIds = defaultFlRoundRunIds(candidateRows);
  }
}

function normalizeFlRunId(selectedRunId, rows) {
  const runIds = new Set(rows.map((row) => flRunId(row)));
  if (selectedRunId && runIds.has(selectedRunId)) {
    return selectedRunId;
  }
  return rows.length > 0 ? flRunId(rows[0]) : null;
}

function flSslRows() {
  if (Array.isArray(state.bundle.fl_ssl_runs)) {
    return state.bundle.fl_ssl_runs;
  }
  return state.bundle.runs.filter((run) => isFlSslTrack(run.track));
}

function flRoundRows() {
  return Array.isArray(state.bundle.fl_ssl_rounds)
    ? state.bundle.fl_ssl_rounds
    : [];
}

function flClientRoundRows() {
  return Array.isArray(state.bundle.fl_ssl_client_rounds)
    ? state.bundle.fl_ssl_client_rounds
    : [];
}

function flClientValidationRows() {
  return Array.isArray(state.bundle.fl_ssl_client_validations)
    ? state.bundle.fl_ssl_client_validations
    : [];
}

function flSplitRows() {
  return Array.isArray(state.bundle.fl_ssl_client_splits)
    ? state.bundle.fl_ssl_client_splits
    : [];
}

function sortedFlRows(rows) {
  return rows.slice().sort((a, b) => compareFlMetric(a, b, "macro_f1"));
}

function flRunsWithRows(runs, dataRows) {
  const runIds = new Set(dataRows.map((row) => row.run_id));
  return runs.filter((row) => runIds.has(flRunId(row)));
}

function flRoundRunsForSelectedRoundCount(rows) {
  if (state.flRoundCount === "__all__") {
    return rows;
  }
  const selectedCount = Number(state.flRoundCount);
  return rows.filter((row) => flRoundCountForRun(row) === selectedCount);
}

function flRoundCountOptions(rows) {
  const countByRoundCount = new Map();
  for (const row of rows) {
    const roundCount = flRoundCountForRun(row);
    if (roundCount === null) {
      continue;
    }
    countByRoundCount.set(
      roundCount,
      (countByRoundCount.get(roundCount) ?? 0) + 1,
    );
  }
  return Array.from(countByRoundCount.entries())
    .map(([count, runCount]) => ({ count, runCount }))
    .sort((a, b) => b.count - a.count);
}

function flRoundCountForRun(row) {
  const completedRounds = numberOrNull(row.completed_rounds);
  if (completedRounds !== null) {
    return completedRounds;
  }
  const roundIndexes = flRoundRowsForRun(flRunId(row))
    .map((roundRow) => numberOrNull(roundRow.round_index))
    .filter((roundIndex) => roundIndex !== null);
  return roundIndexes.length > 0 ? Math.max(...roundIndexes) : null;
}

function flRoundRowsForRun(runId) {
  return flRoundRows()
    .filter((row) => row.run_id === runId)
    .sort(compareRoundRows);
}

function flRoundRowsForSelectedRuns() {
  const selectedRunIds = new Set(state.flRoundRunIds);
  return flRoundRows()
    .filter((row) => selectedRunIds.has(row.run_id))
    .sort(compareFlRoundRows);
}

function flRoundCandidateRuns(rows) {
  if (!state.flRoundMethodName) {
    return rows;
  }
  return rows.filter((row) => flMethodName(row) === state.flRoundMethodName);
}

function isFlSslTrack(track) {
  return String(track ?? "").startsWith("fl_ssl");
}

function flRunId(row) {
  return row.run_id ?? row.id ?? "-";
}

function flMethodName(row) {
  if (
    row.fl_composition_mode === "method_owned" ||
    row.fl_execution_role === "method_owned" ||
    row.method_family === "fedmatch"
  ) {
    return row.fl_descriptor_name ?? row.method_family ?? row.method_name ?? "-";
  }
  return (
    row.method_name ??
    row.ssl_method_name ??
    row.method ??
    row.protocol?.ssl_method?.name ??
    "-"
  );
}

function flMetric(row, metric) {
  if (row[metric] !== undefined) {
    return row[metric];
  }
  if (row.metrics?.primary?.[metric] !== undefined) {
    return row.metrics.primary[metric];
  }
  if (row.metrics?.secondary?.[metric] !== undefined) {
    return row.metrics.secondary[metric];
  }
  if (metric === "expected_calibration_error") {
    return row.metrics?.final_validation?.expected_calibration_error;
  }
  if (metric === "macro_f1") {
    return row.metrics?.final_validation?.macro_f1;
  }
  return null;
}

function flRunDescriptor(row) {
  const protocol = row.protocol ?? {};
  const flMethod = protocol.fl_method ?? {};
  const sslMethod = protocol.ssl_method ?? {};
  const roundRuntime = protocol.round_runtime ?? {};
  const cost = flMetric(row, "communication_cost");
  const costValue = typeof cost === "object" && cost !== null ? cost.value : cost;
  return [
    `mode=${row.fl_composition_mode ?? flMethod.composition_mode ?? "-"}`,
    `descriptor=${row.fl_descriptor_name ?? flMethod.descriptor_name ?? sslMethod.name ?? "-"}`,
    `adapter=${row.adapter_family_name ?? roundRuntime.adapter_family_name ?? "-"}`,
    `agg=${row.aggregation_backend_name ?? roundRuntime.aggregation_backend_name ?? "-"}`,
    `clients=${row.client_count ?? protocol.client_count ?? "-"}`,
    `rounds=${row.completed_rounds ?? protocol.completed_rounds ?? "-"}/${row.round_budget ?? protocol.round_budget ?? "-"}`,
    `updates=${costValue ?? "-"}`,
    `seed=${row.seed ?? protocol.seed ?? "-"}`,
  ].join(" · ");
}

function flCostValue(row) {
  const cost = flMetric(row, "communication_cost");
  return typeof cost === "object" && cost !== null ? cost.value : cost;
}

function flPosthocCommunicationBytes(row, key) {
  const cost = flMetric(row, "communication_cost");
  if (typeof cost !== "object" || cost === null) {
    return null;
  }
  const estimates = cost.posthoc_byte_estimates;
  if (typeof estimates !== "object" || estimates === null) {
    return null;
  }
  return estimates[key] ?? null;
}

function fillFlRoundCountSelect(select, rows) {
  const options = flRoundCountOptions(rows);
  if (options.length === 0) {
    select.innerHTML = `<option value="__all__">round 없음</option>`;
    return;
  }
  const totalRunCount = rows.length;
  select.innerHTML = [
    `<option value="__all__" ${state.flRoundCount === "__all__" ? "selected" : ""}>All rounds (${totalRunCount})</option>`,
    ...options.map(({ count, runCount }) => {
      const selected = String(count) === String(state.flRoundCount) ? "selected" : "";
      return `<option value="${count}" ${selected}>${count} rounds (${runCount})</option>`;
    }),
  ].join("");
}

function fillFlMethodSelect(select, rows, selectedValue) {
  const methodNames = uniqueValues(rows.map((row) => flMethodName(row))).sort();
  if (methodNames.length === 0) {
    select.innerHTML = `<option value="">method 없음</option>`;
    return;
  }
  select.innerHTML = methodNames
    .map((methodName) => {
      const selected = methodName === selectedValue ? "selected" : "";
      const runCount = rows.filter(
        (row) => flMethodName(row) === methodName,
      ).length;
      return `<option value="${methodName}" ${selected}>${methodName} (${runCount})</option>`;
    })
    .join("");
}

function fillFlRunSelect(select, rows, selectedValue) {
  if (rows.length === 0) {
    select.innerHTML = `<option value="">run 없음</option>`;
    return;
  }
  select.innerHTML = rows
    .map((row) => {
      const runId = flRunId(row);
      const selected = runId === selectedValue ? "selected" : "";
      return `<option value="${runId}" ${selected}>${flRunDetailLabel(row)}</option>`;
    })
    .join("");
}

function flRunDetailLabel(row) {
  return [
    `clients=${row.client_count ?? "-"}`,
    `rounds=${row.completed_rounds ?? "-"}/${row.round_budget ?? "-"}`,
    `alpha=${formatMetric(row.shard_alpha)}`,
    `seed=${row.seed ?? "-"}`,
    flRunSuffix(row),
  ].join(" · ");
}

function defaultFlRoundRunIds(rows) {
  return rows.slice(0, 4).map((row) => flRunId(row));
}

function defaultFlRoundRunLabel(row) {
  if (!row) {
    return "run";
  }
  return [
    flMethodName(row),
    flDataSourceLabel(row),
    flLabelBudgetLabel(row),
    `seed${row.seed ?? "?"}`,
    flRunSuffix(row),
  ].join(" · ");
}

function flLabelBudgetLabel(row) {
  const slug = String(row.selection_slug ?? flRunId(row));
  const match = slug.match(/labels_pc(\d+)/);
  if (match) {
    return `pc${match[1]}`;
  }
  return "pc?";
}

function flRoundLegendLabel(runId, row) {
  const metadata = flRunMetadata(runId) ?? row;
  return state.flRoundRunAliases[runId] || defaultFlRoundRunLabel(metadata);
}

function flRunSuffix(row) {
  const runId = flRunId(row);
  const timestampMatch = runId.match(/(\d{8}T\d{6}Z)$/);
  if (timestampMatch) {
    return timestampMatch[1];
  }
  const parts = runId.split("__").filter((part) => part);
  return parts.length > 0 ? parts[parts.length - 1] : shortRun(runId);
}

function flRunMetadata(runId) {
  return flSslRows().find((row) => flRunId(row) === runId) ?? null;
}

function flDataSourceLabel(row) {
  const labeled = row.labeled_dataset_name ?? _extractRunIdPart(row, "labeled");
  const unlabeled = row.unlabeled_dataset_name ?? _extractRunIdPart(row, "unlabeled");
  return `L:${labeled ?? "?"} U:${unlabeled ?? "?"}`;
}

function _extractRunIdPart(row, prefix) {
  const source = String(row.selection_slug ?? flRunId(row));
  const next = prefix === "labeled" ? "unlabeled" : "labels_pc";
  const match = source.match(new RegExp(`${prefix}-(.+?)_${next}`));
  return match ? match[1] : null;
}

function fillRoundSelect(select, rows, selectedValue) {
  const roundIndexes = uniqueValues(
    rows
      .map((row) => String(row.round_index))
      .filter((value) => value !== "null" && value !== "undefined"),
  ).sort((a, b) => Number(a) - Number(b));
  if (roundIndexes.length === 0) {
    select.innerHTML = `<option value="">round 없음</option>`;
    return;
  }
  select.innerHTML = [
    `<option value="__latest__" ${selectedValue === "__latest__" ? "selected" : ""}>Latest</option>`,
    ...roundIndexes.map((roundIndex) => {
      const selected = String(selectedValue) === roundIndex ? "selected" : "";
      return `<option value="${roundIndex}" ${selected}>${roundLabel(roundIndex)}</option>`;
    }),
  ].join("");
}

function selectedClientRoundIndex(rows) {
  if (rows.length === 0) {
    return null;
  }
  if (state.flClientRoundIndex !== "__latest__") {
    return state.flClientRoundIndex;
  }
  const roundIndexes = rows
    .map((row) => numberOrNull(row.round_index))
    .filter((value) => value !== null);
  return roundIndexes.length > 0 ? Math.max(...roundIndexes) : null;
}

function drawFlRoundLines(rows, metric) {
  const runRowsById = new Map();
  for (const row of rows) {
    if (!runRowsById.has(row.run_id)) {
      runRowsById.set(row.run_id, []);
    }
    runRowsById.get(row.run_id).push(row);
  }
  const runMetadataById = new Map(flSslRows().map((row) => [flRunId(row), row]));
  const selectedRunIds = state.flRoundRunIds.filter((runId) =>
    runRowsById.has(runId),
  );
  const series = selectedRunIds
    .flatMap((runId) => {
      const runRows = runRowsById.get(runId).slice().sort(compareRoundRows);
      const runMetadata = runMetadataById.get(runId);
      return [
        {
          runId,
          run: runMetadata,
          metric,
          label: flRoundSeriesLabel(
            runId,
            runMetadata,
            metric,
            selectedRunIds.length,
            1,
          ),
          points: runRows
            .map((row) => ({
              roundIndex: numberOrNull(row.round_index),
              roundId: row.round_id ?? roundLabel(row.round_index),
              value: numberOrNull(row[metric]),
            }))
            .filter((point) => point.roundIndex !== null && point.value !== null),
        },
      ];
    })
    .filter((item) => item.points.length > 0);
  const allPoints = series.flatMap((item) => item.points);
  if (series.length === 0 || allPoints.length === 0) {
    return `<p class="empty">선택한 지표 값이 없습니다.</p>`;
  }
  const roundIndexes = uniqueValues(
    allPoints.map((point) => point.roundIndex),
  ).sort((a, b) => a - b);
  const width = Math.max(760, roundIndexes.length * 82 + 160);
  const height = 320;
  const pad = { top: 28, right: 42, bottom: 56, left: 108 };
  const chartHeight = height - pad.top - pad.bottom;
  const pointInset = 44;
  const chartWidth = width - pad.left - pad.right - pointInset * 2;
  const minRound = Math.min(...allPoints.map((point) => point.roundIndex));
  const maxRound = Math.max(...allPoints.map((point) => point.roundIndex));
  const roundRange = Math.max(maxRound - minRound, 1);
  const minValue = Math.min(...allPoints.map((point) => point.value));
  const maxValue = Math.max(...allPoints.map((point) => point.value));
  const valuePadding = Math.max((maxValue - minValue) * 0.08, 0.02);
  let axisMin = minValue - valuePadding;
  let axisMax = maxValue + valuePadding;
  if (minValue >= 0 && maxValue <= 1) {
    axisMin = Math.max(0, axisMin);
    axisMax = Math.min(1, axisMax);
  }
  const valueRange = Math.max(axisMax - axisMin, 0.000001);
  const colors = seriesColors(series);
  const xForPoint = (point) =>
    pad.left +
    pointInset +
    ((point.roundIndex - minRound) / roundRange) * chartWidth;
  const yForValue = (value) =>
    pad.top + chartHeight - ((value - axisMin) / valueRange) * chartHeight;
  const lines = series
    .map((item) => {
      const color = colors.get(item.label);
      const path = item.points
        .map((point) => `${xForPoint(point)},${yForValue(point.value)}`)
        .join(" ");
      const dots = item.points
        .map(
          (point) => `
            <circle cx="${xForPoint(point)}" cy="${yForValue(point.value)}" r="4" style="--series-color:${color}">
              <title>${escapeHtml(item.label)} · ${escapeHtml(point.roundId)} · ${formatMetric(point.value)}</title>
            </circle>
          `,
        )
        .join("");
      return `<polyline points="${path}" fill="none" style="--series-color:${color}" />${dots}`;
    })
    .join("");
  const labels = roundIndexes
    .filter(
      (_roundIndex, index) =>
        index === 0 ||
        index === roundIndexes.length - 1 ||
        roundIndexes.length <= 12 ||
        index % Math.ceil(roundIndexes.length / 10) === 0,
    )
    .map((roundIndex) => {
      const point = { roundIndex };
      const x = xForPoint(point);
      return `
        <text class="axis-label" x="${x}" y="${height - 24}" text-anchor="middle">
          ${compactRoundLabel(roundIndex)}
        </text>
      `;
    })
    .join("");
  const valueTicks = buildValueTicks(axisMin, axisMax, 5)
    .map((value) => {
      const y = yForValue(value);
      return `
        <line class="grid-line" x1="${pad.left}" y1="${y}" x2="${width - pad.right}" y2="${y}" />
        <text class="axis-label" x="${pad.left - 10}" y="${y + 4}" text-anchor="end">${formatMetric(value)}</text>
      `;
    })
    .join("");
  return `
    ${renderSeriesLegend(series, colors)}
    <div class="chart-scroll line-chart">
      <svg viewBox="0 0 ${width} ${height}" role="img">
        <line class="axis-line" x1="${pad.left}" y1="${pad.top + chartHeight}" x2="${width - pad.right}" y2="${pad.top + chartHeight}" />
        <line class="axis-line" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${pad.top + chartHeight}" />
        ${valueTicks}
        ${lines}
        ${labels}
      </svg>
    </div>
  `;
}

function compareFlMetric(a, b, metric) {
  const left = numberOrNull(flMetric(a, metric));
  const right = numberOrNull(flMetric(b, metric));
  return compareMetricValues(left, right, metric);
}

function compareMetricValues(left, right, metric) {
  if (left === null && right === null) return 0;
  if (left === null) return 1;
  if (right === null) return -1;
  return metric.includes("error") || metric === "loss" ? left - right : right - left;
}

function compareNullableNumbers(a, b) {
  const left = numberOrNull(a);
  const right = numberOrNull(b);
  if (left === null && right === null) return 0;
  if (left === null) return 1;
  if (right === null) return -1;
  return left - right;
}

function compareRoundRows(a, b) {
  return compareNullableNumbers(a.round_index, b.round_index);
}

function compareFlRoundRows(a, b) {
  const runCompare = String(a.run_id ?? "").localeCompare(String(b.run_id ?? ""));
  if (runCompare !== 0) {
    return runCompare;
  }
  return compareRoundRows(a, b);
}

function compareClientRoundRows(a, b) {
  const roundCompare = compareRoundRows(a, b);
  if (roundCompare !== 0) {
    return roundCompare;
  }
  return String(a.client_id ?? "").localeCompare(String(b.client_id ?? ""));
}

function roundLabel(roundIndex) {
  const number = numberOrNull(roundIndex);
  if (number === null) {
    return "round ?";
  }
  return number === 0 ? "initial" : `round ${number}`;
}

function compactRoundLabel(roundIndex) {
  const number = numberOrNull(roundIndex);
  if (number === null) {
    return "?";
  }
  return number === 0 ? "init" : `r${number}`;
}

function flRoundTableRoundLabel(row) {
  const base = row.round_id ?? roundLabel(row.round_index);
  if (state.flRoundRunIds.length <= 1) {
    return base;
  }
  return `${flRoundLegendLabel(row.run_id, row)} · ${base}`;
}

function emptyTableRow(columnCount, message) {
  return `<tr><td class="empty-cell" colspan="${columnCount}">${message}</td></tr>`;
}

function boolLabel(value) {
  if (value === true) return "yes";
  if (value === false) return "no";
  return "-";
}

function resetScopedSelections() {
  state.selectedRunIds = [];
  state.comparisonMethodName = null;
  state.comparisonSelectionTouched = false;
  state.detailMethodName = null;
  state.detailRunId = null;
  state.projectionMethodName = null;
  state.projectionRunIds = [];
  state.projectionSelectionTouched = false;
}

function resetComparisonSelection() {
  state.selectedRunIds = [];
  state.comparisonMethodName = null;
  state.comparisonSelectionTouched = false;
}

function normalizeComparisonSelection(rows) {
  const methodNames = uniqueMethodNames(rows);
  const visibleMethods = new Set(methodNames);
  if (state.comparisonMethodName && !visibleMethods.has(state.comparisonMethodName)) {
    state.comparisonMethodName = null;
  }
  if (!state.comparisonMethodName && methodNames.length > 0) {
    state.comparisonMethodName = methodNames[0];
  }
  const visibleRunIds = new Set(rows.map((row) => row.run_id));
  state.selectedRunIds = state.selectedRunIds.filter((runId) =>
    visibleRunIds.has(runId),
  );
  if (!state.comparisonSelectionTouched && state.selectedRunIds.length === 0) {
    state.selectedRunIds = defaultComparisonRunIds(rows);
  }
}

function normalizeDetailSelection(rows) {
  const visibleMethods = new Set(uniqueMethodNames(rows));
  if (state.detailMethodName && !visibleMethods.has(state.detailMethodName)) {
    state.detailMethodName = null;
    state.detailRunId = null;
  }
  if (!state.detailMethodName) {
    state.detailRunId = null;
    return;
  }
  const detailRunIds = new Set(
    rowsForMethods(rows, [state.detailMethodName]).map((row) => row.run_id),
  );
  if (state.detailRunId && !detailRunIds.has(state.detailRunId)) {
    state.detailRunId = null;
  }
}

function normalizeProjectionSelection(rows) {
  const methodNames = uniqueMethodNames(rowsWithProjection(rows));
  const visibleMethods = new Set(methodNames);
  if (state.projectionMethodName && !visibleMethods.has(state.projectionMethodName)) {
    state.projectionMethodName = null;
  }
  if (!state.projectionMethodName && methodNames.length > 0) {
    state.projectionMethodName = methodNames[0];
  }
  const candidateRows = state.projectionMethodName
    ? rowsWithProjection(rowsForMethods(rows, [state.projectionMethodName]))
    : [];
  const visibleRunIds = new Set(rowsWithProjection(rows).map((row) => row.run_id));
  state.projectionRunIds = state.projectionRunIds.filter((runId) =>
    visibleRunIds.has(runId),
  );
  if (!state.projectionSelectionTouched && state.projectionRunIds.length === 0) {
    state.projectionRunIds = defaultProjectionRunIds(candidateRows);
  }
}

function defaultComparisonRunIds(rows) {
  const runIds = [];
  const seenMethods = new Set();
  for (const row of rows) {
    if (seenMethods.has(row.method_name)) {
      continue;
    }
    runIds.push(row.run_id);
    seenMethods.add(row.method_name);
    if (runIds.length >= 4) {
      break;
    }
  }
  return runIds;
}

function defaultProjectionRunIds(rows) {
  return rows.map((row) => row.run_id);
}

function renderMetricPicker() {
  const metrics = [
    "macro_f1",
    "accuracy_top_1",
    "expected_calibration_error",
    "loss",
    "balanced_accuracy",
    "weighted_f1",
    "worst_category_f1_value",
    "mean_margin_top1_top2",
  ];
  const selectedMetrics = new Set(state.comparisonMetrics);
  elements.metricPicker.innerHTML = metrics
    .map(
      (metric) => `
        <label class="check-row">
          <input
            type="checkbox"
            data-metric="${metric}"
            ${selectedMetrics.has(metric) ? "checked" : ""}
          />
          <span>${metricLabel(metric)}</span>
        </label>
      `,
    )
    .join("");
}

function renderComparisonRunControls(rows) {
  fillMethodSelect(
    elements.comparisonMethodFilter,
    rows,
    state.comparisonMethodName,
  );
  const candidateRows = state.comparisonMethodName
    ? rowsForMethods(rows, [state.comparisonMethodName])
    : [];
  if (candidateRows.length === 0) {
    elements.comparisonRunCheckboxes.innerHTML =
      `<p class="empty">방법론 선택 먼저</p>`;
    return;
  }
  const selectedRunIds = new Set(state.selectedRunIds);
  elements.comparisonRunCheckboxes.innerHTML = candidateRows
    .map(
      (row) => `
        <label class="run-option">
          <input
            type="checkbox"
            data-run-id="${row.run_id}"
            ${selectedRunIds.has(row.run_id) ? "checked" : ""}
          />
          <span>
            <strong>${shortRun(row.run_id)}</strong>
            <small>${runDescriptor(row)}</small>
          </span>
        </label>
      `,
    )
    .join("");
}

function renderSelectedRunCards(rows) {
  const rowsById = new Map(rows.map((row) => [row.run_id, row]));
  const selectedRows = state.selectedRunIds
    .map((runId) => rowsById.get(runId))
    .filter((row) => row);
  if (selectedRows.length === 0) {
    elements.selectedRunCards.innerHTML =
      `<p class="empty">선택된 비교 run이 없습니다.</p>`;
    return;
  }
  elements.selectedRunCards.innerHTML = selectedRows
    .map((row) => {
      const detail = [
        row.method_name,
        shortRun(row.run_id),
        runDescriptor(row),
        `labeled=${row.labeled_dataset_name ?? "-"}`,
        `unlabeled=${row.unlabeled_dataset_name ?? "-"}`,
      ].join(" · ");
      return `
        <article class="selected-run-card" title="${escapeHtml(detail)}">
          <strong>${shortRun(row.run_id)}</strong>
          <button
            type="button"
            data-remove-run-id="${row.run_id}"
            aria-label="${shortRun(row.run_id)} 제거"
          >x</button>
          <span class="selected-run-detail" aria-hidden="true">${escapeHtml(detail)}</span>
        </article>
      `;
    })
    .join("");
}

function renderDetailRunFilter(rows) {
  fillMethodSelect(elements.detailMethodFilter, rows, state.detailMethodName);
  if (!state.detailMethodName) {
    elements.detailRunFilter.innerHTML = `<option value="">방법론 선택 먼저</option>`;
    elements.detailRunSummary.textContent =
      "왼쪽에서 상세 방법론을 선택한 뒤 오른쪽에서 세부 run을 고르세요.";
    return;
  }
  const detailRows = rowsForMethods(rows, [state.detailMethodName]);
  if (detailRows.length === 0) {
    state.detailRunId = null;
    elements.detailRunFilter.innerHTML = `<option value="">run 없음</option>`;
    elements.detailRunSummary.textContent = "현재 필터에 해당하는 상세 run이 없습니다.";
    return;
  }
  fillRunSelect(elements.detailRunFilter, detailRows, state.detailRunId);
  const detailRow = detailRows.find((row) => row.run_id === state.detailRunId);
  elements.detailRunSummary.textContent = detailRow
    ? [
        detailRow.method_name,
        shortRun(detailRow.run_id),
        `labeled=${detailRow.labeled_dataset_name ?? "-"}`,
        `unlabeled=${detailRow.unlabeled_dataset_name ?? "-"}`,
        `eval=${state.classEvalSet}`,
      ].join(" · ")
    : "Per-class와 confusion matrix를 보려면 상세 run을 선택하세요.";
}

function renderProjectionRunControls(rows) {
  fillMethodSelect(
    elements.projectionMethodFilter,
    rows,
    state.projectionMethodName,
  );
  const candidateRows = state.projectionMethodName
    ? rowsForMethods(rows, [state.projectionMethodName])
    : [];
  if (candidateRows.length === 0) {
    elements.projectionRunCheckboxes.innerHTML =
      `<p class="empty">projection 이미지가 있는 method/run이 없습니다.</p>`;
    return;
  }
  const selectedRunIds = new Set(state.projectionRunIds);
  elements.projectionRunCheckboxes.innerHTML = candidateRows
    .map(
      (row) => `
        <label class="run-option">
          <input
            type="checkbox"
            data-run-id="${row.run_id}"
            ${selectedRunIds.has(row.run_id) ? "checked" : ""}
          />
          <span>
            <strong>${shortRun(row.run_id)}</strong>
            <small>${runDescriptor(row)}</small>
          </span>
        </label>
      `,
    )
    .join("");
}

function groupedRowsByMethod(rows) {
  const groups = new Map();
  for (const row of rows) {
    if (!groups.has(row.method_name)) {
      groups.set(row.method_name, []);
    }
    groups.get(row.method_name).push(row);
  }
  return Array.from(groups.entries());
}

function uniqueMethodNames(rows) {
  return Array.from(new Set(rows.map((row) => row.method_name)));
}

function rowsForMethods(rows, methodNames) {
  if (methodNames.length === 0) {
    return [];
  }
  const methodNameSet = new Set(methodNames);
  return rows.filter((row) => methodNameSet.has(row.method_name));
}

function rowsWithProjection(rows) {
  const projectionRunIds = new Set(
    state.bundle.projection_images
      .filter((image) => image.eval_set === state.projectionEvalSet)
      .map((image) => image.run_id),
  );
  return rows.filter((row) => projectionRunIds.has(row.run_id));
}

function runDescriptor(row) {
  return [
    `lr=${formatMetric(row.learning_rate)}`,
    `clf=${formatMetric(row.classifier_learning_rate)}`,
    shortSplit(row.selection_slug),
  ].join(" · ");
}

function compareMetric(a, b, metric) {
  const left = numberOrNull(a[metric]);
  const right = numberOrNull(b[metric]);
  if (left === null && right === null) return 0;
  if (left === null) return 1;
  if (right === null) return -1;
  return metric.includes("error") || metric === "loss" ? left - right : right - left;
}

function renderMetricCards(rows) {
  const best = rows[0];
  const runCount = rows.length;
  const methodCount = new Set(rows.map((row) => row.method_name)).size;
  elements.metricCards.innerHTML = [
    card("runs", runCount),
    card("methods", methodCount),
    card("best metric", best ? formatMetric(best[state.overviewMetric]) : "-"),
    card("selected runs", state.selectedRunIds.length),
  ].join("");
}

function card(label, value) {
  return `<div class="metric-card"><span>${label}</span><strong>${value}</strong></div>`;
}

function renderComparisonChart(rows) {
  const selectedRows = rows.filter((row) => state.selectedRunIds.includes(row.run_id));
  const metrics = state.comparisonMetrics;
  if (metrics.length === 0) {
    elements.comparisonChart.innerHTML = `<p class="empty">비교할 metric을 선택하세요.</p>`;
    return;
  }
  if (selectedRows.length === 0) {
    elements.comparisonChart.innerHTML = `<p class="empty">비교할 run을 선택하세요.</p>`;
    return;
  }
  if (state.comparisonChartType === "vertical_bar") {
    elements.comparisonChart.innerHTML = drawVerticalBarComparison(
      selectedRows,
      metrics,
    );
    return;
  }
  if (state.comparisonChartType === "line") {
    elements.comparisonChart.innerHTML = drawMetricLineComparison(
      selectedRows,
      metrics,
    );
    return;
  }
  elements.comparisonChart.innerHTML = drawGroupedHorizontalComparison(
    selectedRows,
    metrics,
  );
}

function drawGroupedHorizontalComparison(selectedRows, metrics) {
  const series = metrics.map((metric) => {
    const values = selectedRows
      .map((row) => numberOrNull(row[metric]))
      .filter((value) => value !== null);
    return {
      metric,
      max: Math.max(...values, 0.000001),
    };
  });
  return selectedRows
    .map((row) => {
      const metricBars = series
        .map(({ metric, max }) => {
          const value = numberOrNull(row[metric]);
          const width = value === null ? 0 : Math.max(2, (value / max) * 100);
          return `
            <div class="metric-bar">
              <span>${metricLabel(metric)}</span>
              <div class="bar-track"><i style="width:${width}%"></i></div>
              <strong>${formatMetric(value)}</strong>
            </div>
          `;
        })
        .join("");
      return `
        <div class="comparison-run">
          <div>
            <strong>${shortRun(row.run_id)}</strong>
            <span>${row.method_name} · ${formatMetric(row.learning_rate)} / ${formatMetric(row.classifier_learning_rate)}</span>
          </div>
          <div class="comparison-metrics">${metricBars}</div>
        </div>
      `;
    })
    .join("");
}

function drawVerticalBarComparison(selectedRows, metrics) {
  const records = selectedRows.flatMap((row) =>
    metrics.map((metric) => ({
      run: row,
      metric,
      value: numberOrNull(row[metric]),
    })),
  );
  const values = records
    .map((record) => record.value)
    .filter((value) => value !== null);
  if (values.length === 0) {
    return `<p class="empty">선택한 지표 값이 없습니다.</p>`;
  }
  const colors = metricColors(metrics);
  const width = Math.max(760, selectedRows.length * metrics.length * 48 + 120);
  const height = 340;
  const pad = { top: 24, right: 28, bottom: 96, left: 56 };
  const chartHeight = height - pad.top - pad.bottom;
  const chartWidth = width - pad.left - pad.right;
  const maxValue = Math.max(...values, 0.000001);
  const groupWidth = chartWidth / selectedRows.length;
  const barWidth = Math.max(8, Math.min(24, (groupWidth - 16) / metrics.length));
  const bars = records
    .filter((record) => record.value !== null)
    .map((record) => {
      const groupIndex = selectedRows.findIndex(
        (row) => row.run_id === record.run.run_id,
      );
      const metricIndex = metrics.indexOf(record.metric);
      const x = pad.left + groupIndex * groupWidth + 8 + metricIndex * barWidth;
      const barHeight = (record.value / maxValue) * chartHeight;
      const y = pad.top + chartHeight - barHeight;
      return `
        <rect
          x="${x}"
          y="${y}"
          width="${Math.max(6, barWidth - 3)}"
          height="${barHeight}"
          rx="4"
          fill="${colors.get(record.metric)}"
        >
          <title>${shortRun(record.run.run_id)} ${metricLabel(record.metric)} ${formatMetric(record.value)}</title>
        </rect>
      `;
    })
    .join("");
  const labels = selectedRows
    .map((row, index) => {
      const x = pad.left + index * groupWidth + groupWidth / 2;
      return `
        <text class="axis-label" x="${x}" y="${height - 48}" transform="rotate(35 ${x} ${height - 48})">
          ${shortRun(row.run_id)}
        </text>
      `;
    })
    .join("");
  return `
    ${renderMetricLegend(metrics, colors)}
    <div class="chart-scroll">
      <svg viewBox="0 0 ${width} ${height}" role="img">
        <line class="axis-line" x1="${pad.left}" y1="${pad.top + chartHeight}" x2="${width - pad.right}" y2="${pad.top + chartHeight}" />
        <line class="axis-line" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${pad.top + chartHeight}" />
        <text class="axis-label" x="${pad.left}" y="18">${formatMetric(maxValue)}</text>
        ${bars}
        ${labels}
      </svg>
    </div>
  `;
}

function drawMetricLineComparison(selectedRows, metrics) {
  const pointsByMetric = metrics.map((metric) => ({
    metric,
    points: selectedRows
      .map((row, index) => ({
        index,
        runId: row.run_id,
        value: numberOrNull(row[metric]),
      }))
      .filter((point) => point.value !== null),
  }));
  const values = pointsByMetric.flatMap((series) =>
    series.points.map((point) => point.value),
  );
  if (values.length === 0) {
    return `<p class="empty">선택한 지표 값이 없습니다.</p>`;
  }
  const colors = metricColors(metrics);
  const width = Math.max(760, selectedRows.length * 92 + 120);
  const height = 340;
  const pad = { top: 24, right: 28, bottom: 96, left: 56 };
  const chartHeight = height - pad.top - pad.bottom;
  const chartWidth = width - pad.left - pad.right;
  const maxValue = Math.max(...values);
  const minValue = Math.min(...values);
  const valueRange = Math.max(maxValue - minValue, 0.000001);
  const xForIndex = (index) =>
    pad.left +
    (selectedRows.length <= 1
      ? chartWidth / 2
      : (index / (selectedRows.length - 1)) * chartWidth);
  const yForValue = (value) =>
    pad.top + chartHeight - ((value - minValue) / valueRange) * chartHeight;
  const lines = pointsByMetric
    .filter((series) => series.points.length > 0)
    .map((series) => {
      const color = colors.get(series.metric);
      const path = series.points
        .map((point) => `${xForIndex(point.index)},${yForValue(point.value)}`)
        .join(" ");
      const dots = series.points
        .map(
          (point) => `
            <circle cx="${xForIndex(point.index)}" cy="${yForValue(point.value)}" r="4" style="--series-color:${color}">
              <title>${shortRun(point.runId)} ${metricLabel(series.metric)} ${formatMetric(point.value)}</title>
            </circle>
          `,
        )
        .join("");
      return `<polyline points="${path}" fill="none" style="--series-color:${color}" />${dots}`;
    })
    .join("");
  const labels = selectedRows
    .map((row, index) => {
      const x = xForIndex(index);
      return `
        <text class="axis-label" x="${x}" y="${height - 48}" transform="rotate(35 ${x} ${height - 48})">
          ${shortRun(row.run_id)}
        </text>
      `;
    })
    .join("");
  return `
    ${renderMetricLegend(metrics, colors)}
    <div class="chart-scroll line-chart">
      <svg viewBox="0 0 ${width} ${height}" role="img">
        <line class="axis-line" x1="${pad.left}" y1="${pad.top + chartHeight}" x2="${width - pad.right}" y2="${pad.top + chartHeight}" />
        <line class="axis-line" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${pad.top + chartHeight}" />
        <text class="axis-label" x="${pad.left}" y="18">${formatMetric(maxValue)}</text>
        <text class="axis-label" x="${pad.left}" y="${height - 8}">${formatMetric(minValue)}</text>
        ${lines}
        ${labels}
      </svg>
    </div>
  `;
}

function renderBarChart(rows) {
  const values = rows.slice(0, 16).map((row) => ({
    label: `${row.method_name} · ${shortRun(row.run_id)}`,
    value: numberOrNull(row[state.overviewMetric]),
  }));
  const max = Math.max(...values.map((item) => item.value ?? 0), 0.000001);
  elements.barChart.innerHTML = values
    .map((item) => {
      const width = item.value === null ? 0 : Math.max(2, (item.value / max) * 100);
      return `
        <div class="bar-row">
          <span>${item.label}</span>
          <div class="bar-track"><i style="width:${width}%"></i></div>
          <strong>${formatMetric(item.value)}</strong>
        </div>
      `;
    })
    .join("");
}

function renderEpochChart() {
  elements.epochToggle.checked = state.showEpochChart;
  elements.epochPanel.hidden = !state.showEpochChart;
  if (!state.showEpochChart) {
    elements.epochChart.innerHTML = "";
    return;
  }
  const selectedRunIds = state.selectedRunIds.slice(0, 6);
  const series = selectedRunIds
    .map((runId) => {
      const points = state.bundle.epoch_metrics
        .filter((row) => row.run_id === runId)
        .map((row) => ({
          epoch: row.epoch,
          value: numberOrNull(row[state.epochMetric]),
        }))
        .filter((point) => point.value !== null);
      return { runId, points };
    })
    .filter((item) => item.points.length > 0);
  if (series.length === 0) {
    elements.epochChart.innerHTML = `<p class="empty">선택한 run에 ${state.epochMetric} history가 없습니다.</p>`;
    return;
  }
  elements.epochChart.innerHTML = drawMultiLine(series);
}

function renderClassChart() {
  const runId = detailRunId();
  if (!runId) {
    elements.classChart.innerHTML = `<p class="empty">상세 run을 선택하세요.</p>`;
    return;
  }
  const rows = state.bundle.per_class_metrics
    .filter((row) => row.run_id === runId)
    .filter((row) => row.eval_set === state.classEvalSet)
    .sort((a, b) => a.category.localeCompare(b.category));
  const max = Math.max(
    ...rows.map((row) => numberOrNull(row[state.classMetric]) ?? 0),
    0.000001,
  );
  elements.classChart.innerHTML = rows
    .map((row) => {
      const value = numberOrNull(row[state.classMetric]);
      const width = value === null ? 0 : Math.max(2, (value / max) * 100);
      return `
        <div class="bar-row">
          <span>${row.category}</span>
          <div class="bar-track class"><i style="width:${width}%"></i></div>
          <strong>${formatMetric(value)}</strong>
        </div>
      `;
    })
    .join("");
}

function renderClassTable() {
  const runId = detailRunId();
  if (!runId) {
    elements.classTable.innerHTML = "";
    return;
  }
  const rows = state.bundle.per_class_metrics
    .filter((row) => row.run_id === runId)
    .filter((row) => row.eval_set === state.classEvalSet)
    .sort((a, b) => a.category.localeCompare(b.category));
  if (rows.length === 0) {
    elements.classTable.innerHTML = "";
    return;
  }
  elements.classTable.innerHTML = rows
    .map(
      (row) => `
        <tr>
          <td>${row.category}</td>
          <td>${row.support ?? "-"}</td>
          <td>${formatMetric(row.precision)}</td>
          <td>${formatMetric(row.recall)}</td>
          <td>${formatMetric(row.f1)}</td>
          <td>${formatMetric(row.mean_true_label_probability)}</td>
          <td>${formatMetric(row.mean_top_1_probability)}</td>
          <td>${formatMetric(row.mean_margin_top1_top2)}</td>
        </tr>
      `,
    )
    .join("");
}

function renderConfusionMatrix() {
  const runId = detailRunId();
  if (!runId) {
    elements.confusionMatrix.innerHTML = `<p class="empty">상세 run을 선택하세요.</p>`;
    return;
  }
  const cells = state.bundle.confusion_matrix_cells
    .filter((cell) => cell.run_id === runId)
    .filter((cell) => cell.eval_set === state.classEvalSet);
  if (cells.length === 0) {
    elements.confusionMatrix.innerHTML = `<p class="empty">confusion matrix가 없습니다.</p>`;
    return;
  }
  const categories = [...new Set(cells.map((cell) => cell.actual_category))].sort();
  const max = Math.max(...cells.map((cell) => cell.count), 1);
  const countByPair = new Map(
    cells.map((cell) => [
      `${cell.actual_category}::${cell.predicted_category}`,
      cell.count,
    ]),
  );
  const body = categories
    .map((actual) => {
      const cols = categories
        .map((predicted) => {
          const count = countByPair.get(`${actual}::${predicted}`) ?? 0;
          const intensity = count / max;
          return `<td style="--heat:${intensity}">${count}</td>`;
        })
        .join("");
      return `<tr><th>${actual}</th>${cols}</tr>`;
    })
    .join("");
  elements.confusionMatrix.innerHTML = `
    <table class="matrix">
      <thead><tr><th>actual \\ predicted</th>${categories
        .map((category) => `<th>${category}</th>`)
        .join("")}</tr></thead>
      <tbody>${body}</tbody>
    </table>
  `;
}

function renderProjectionGallery() {
  const runIds = state.projectionRunIds;
  const runsById = new Map(state.bundle.runs.map((run) => [run.run_id, run]));
  const images = state.bundle.projection_images
    .filter((image) => runIds.includes(image.run_id))
    .filter((image) => image.eval_set === state.projectionEvalSet);
  if (images.length === 0) {
    elements.projectionGallery.innerHTML = `<p class="empty">선택한 run/eval set의 projection image가 없습니다. dashboard JSON을 다시 export했는지 확인하세요.</p>`;
    return;
  }
  elements.projectionGallery.innerHTML = images
    .map((image) => {
      const run = runsById.get(image.run_id);
      const dataLabel = [
        `labeled=${run?.labeled_dataset_name ?? "-"}`,
        `unlabeled=${run?.unlabeled_dataset_name ?? "-"}`,
        `eval=${image.eval_set}`,
      ].join(" · ");
      return `
        <figure>
          <button
            class="projection-remove"
            type="button"
            data-remove-projection-run-id="${image.run_id}"
            aria-label="${shortRun(image.run_id)} projection 제거"
          >x</button>
          <img src="${image.image_src}" alt="${shortRun(image.run_id)} ${image.eval_set} projection" loading="lazy" />
          <figcaption>
            <strong>${shortRun(image.run_id)}</strong>
            <span>${run?.method_name ?? "-"} · ${dataLabel}</span>
            <span>${image.reducer ?? "projection"}${image.fallback_reason ? ` · ${image.fallback_reason}` : ""}</span>
          </figcaption>
        </figure>
      `;
    })
    .join("");
}

function renderRunTable(rows) {
  elements.runTable.innerHTML = rows
    .map(
      (row) => `
      <tr class="${state.selectedRunIds.includes(row.run_id) ? "selected" : ""}">
        <td><button type="button" data-run-id="${row.run_id}">${shortRun(row.run_id)}</button></td>
        <td>${row.method_name}</td>
        <td>${shortSplit(row.selection_slug)}</td>
        <td>${formatMetric(row.learning_rate)}</td>
        <td>${formatMetric(row.classifier_learning_rate)}</td>
        <td>${formatMetric(row.macro_f1)}</td>
        <td>${formatMetric(row.accuracy_top_1)}</td>
        <td>${formatMetric(row.expected_calibration_error)}</td>
        <td>${row.rows_total ?? "-"}</td>
        <td>${row.worst_category_f1 ?? "-"}</td>
      </tr>
    `,
    )
    .join("");
  elements.runTable.querySelectorAll("button[data-run-id]").forEach((button) => {
    button.addEventListener("click", () => {
      const runId = button.dataset.runId;
      const row = rows.find((candidate) => candidate.run_id === runId);
      state.comparisonSelectionTouched = true;
      state.detailMethodName = row?.method_name ?? null;
      state.detailRunId = runId;
      if (state.selectedRunIds.includes(runId)) {
        state.selectedRunIds =
          state.selectedRunIds.length > 1
            ? state.selectedRunIds.filter((selectedRunId) => selectedRunId !== runId)
            : state.selectedRunIds;
      } else {
        state.selectedRunIds = [...state.selectedRunIds, runId];
      }
      render();
    });
  });
}

function drawMultiLine(series) {
  const width = 680;
  const height = 260;
  const pad = 28;
  const allPoints = series.flatMap((item) => item.points);
  const minValue = Math.min(...allPoints.map((point) => point.value));
  const maxValue = Math.max(...allPoints.map((point) => point.value));
  const valueRange = Math.max(maxValue - minValue, 0.000001);
  const epochMin = Math.min(...allPoints.map((point) => point.epoch));
  const epochMax = Math.max(...allPoints.map((point) => point.epoch));
  const epochRange = Math.max(epochMax - epochMin, 1);
  const colors = ["#23766f", "#d97732", "#527a45", "#8f5b3d", "#6c7a89", "#a94f1f"];
  const lines = series
    .map((item, index) => {
      const color = colors[index % colors.length];
      const path = item.points
        .map((point) => {
          const x = pad + ((point.epoch - epochMin) / epochRange) * (width - pad * 2);
          const y =
            height -
            pad -
            ((point.value - minValue) / valueRange) * (height - pad * 2);
          return `${x},${y}`;
        })
        .join(" ");
      const dots = item.points
        .map((point) => {
          const x = pad + ((point.epoch - epochMin) / epochRange) * (width - pad * 2);
          const y =
            height -
            pad -
            ((point.value - minValue) / valueRange) * (height - pad * 2);
          return `<circle cx="${x}" cy="${y}" r="4" style="--series-color:${color}"><title>${shortRun(item.runId)} epoch ${point.epoch}: ${formatMetric(point.value)}</title></circle>`;
        })
        .join("");
      return `<polyline points="${path}" fill="none" style="--series-color:${color}" />${dots}`;
    })
    .join("");
  const legend = series
    .map((item, index) => {
      const color = colors[index % colors.length];
      return `<span><i style="background:${color}"></i>${shortRun(item.runId)}</span>`;
    })
    .join("");
  return `
    <div class="chart-legend">${legend}</div>
    <svg viewBox="0 0 ${width} ${height}" role="img">
      ${lines}
      <text x="${pad}" y="20">${formatMetric(maxValue)}</text>
      <text x="${pad}" y="${height - 6}">${formatMetric(minValue)}</text>
    </svg>
  `;
}

function metricColors(metrics) {
  const palette = ["#23766f", "#d97732", "#527a45", "#8f5b3d", "#6c7a89", "#a94f1f"];
  return new Map(
    metrics.map((metric, index) => [metric, palette[index % palette.length]]),
  );
}

function seriesColors(series) {
  const palette = [
    "#23766f",
    "#d97732",
    "#527a45",
    "#8f5b3d",
    "#6c7a89",
    "#a94f1f",
    "#5a6fb0",
    "#b55a7a",
  ];
  return new Map(
    series.map((item, index) => [item.label, palette[index % palette.length]]),
  );
}

function renderMetricLegend(metrics, colors) {
  return `
    <div class="chart-legend">
      ${metrics
        .map(
          (metric) =>
            `<span><i style="background:${colors.get(metric)}"></i>${metricLabel(metric)}</span>`,
        )
        .join("")}
    </div>
  `;
}

function renderSeriesLegend(series, colors) {
  return `
    <div class="chart-legend editable-legend">
      ${series
        .map(
          (item) =>
            `<span><i style="background:${colors.get(item.label)}"></i>${escapeHtml(item.label)}</span>`,
        )
        .join("")}
    </div>
  `;
}

function flRoundSeriesLabel(runId, row, metric, runCount, metricCount) {
  const runLabel = flRoundLegendLabel(runId, row);
  const hasAlias = Boolean(state.flRoundRunAliases[runId]);
  if (runCount <= 1 && !hasAlias) {
    return metricLabel(metric);
  }
  if (metricCount <= 1) {
    return runLabel;
  }
  return `${runLabel} · ${metricLabel(metric)}`;
}

function buildValueTicks(minValue, maxValue, tickCount) {
  if (tickCount <= 1) {
    return [minValue];
  }
  const step = (maxValue - minValue) / (tickCount - 1);
  return Array.from({ length: tickCount }, (_item, index) => minValue + step * index);
}

function fillSelect(select, values, selectedValue) {
  const selectedValues = Array.isArray(selectedValue)
    ? new Set(selectedValue)
    : new Set([selectedValue]);
  select.innerHTML = values
    .map((value) => {
      const label = value === "__all__" ? "All" : value;
      const selected = selectedValues.has(value) ? "selected" : "";
      return `<option value="${value}" ${selected}>${label}</option>`;
    })
    .join("");
}

function fillMethodSelect(select, rows, selectedMethodName) {
  const options = uniqueMethodNames(rows)
    .map((methodName) => {
      const selected = methodName === selectedMethodName ? "selected" : "";
      return `<option value="${methodName}" ${selected}>${methodName}</option>`;
    })
    .join("");
  select.innerHTML = `<option value="">방법론 선택</option>${options}`;
}

function fillRunSelect(select, rows, selectedRunId) {
  const options = rows
    .map((row) => {
      const selected = row.run_id === selectedRunId ? "selected" : "";
      return `<option value="${row.run_id}" ${selected}>${shortRun(row.run_id)} · ${runDescriptor(row)}</option>`;
    })
    .join("");
  select.innerHTML = `<option value="">세부 run 선택</option>${options}`;
}

function checkedValues(container, dataKey) {
  return Array.from(container.querySelectorAll("input[type='checkbox']:checked"))
    .map((input) => input.dataset[dataKey])
    .filter((value) => value);
}

function uniqueValues(values) {
  return Array.from(new Set(values));
}

function detailRunId() {
  return state.detailRunId;
}

function numberOrNull(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function formatMetric(value) {
  const number = numberOrNull(value);
  if (number === null) {
    return "-";
  }
  if (Math.abs(number) < 0.001 && number !== 0) {
    return number.toExponential(2);
  }
  return number.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

function formatCount(value) {
  const number = numberOrNull(value);
  if (number === null) {
    return "-";
  }
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 0,
  }).format(number);
}

function formatBytes(value) {
  const number = numberOrNull(value);
  if (number === null) {
    return "-";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  let scaled = Math.abs(number);
  let unitIndex = 0;
  while (scaled >= 1024 && unitIndex < units.length - 1) {
    scaled /= 1024;
    unitIndex += 1;
  }
  const signed = number < 0 ? -scaled : scaled;
  return `${formatMetric(signed)} ${units[unitIndex]}`;
}

function formatSeconds(value) {
  const number = numberOrNull(value);
  if (number === null) {
    return "-";
  }
  return `${formatMetric(number)}s`;
}

function formatDistribution(value) {
  if (!value || typeof value !== "object") {
    return "-";
  }
  const entries = Object.entries(value).sort(([left], [right]) =>
    left.localeCompare(right),
  );
  if (entries.length === 0) {
    return "-";
  }
  return entries
    .map(([label, count]) => `${escapeHtml(label)}: ${formatCount(count)}`)
    .join(" · ");
}

function shortRun(runId) {
  if (!runId) return "-";
  return runId.replace(/^lora_/, "").replace(/_2026_.*/, "");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function metricLabel(metric) {
  if (metric === "expected_calibration_error") {
    return "ece";
  }
  return metric;
}

function shortSplit(selectionSlug) {
  if (!selectionSlug) return "-";
  return selectionSlug
    .replace(/^labeled-/, "L:")
    .replace("_unlabeled-", " U:")
    .replace("_validation-", " V:")
    .replace("_test-", " T:");
}
