import {
  buildValueTicks,
  renderSeriesLegend,
  seriesColors,
} from "./dashboard_charting.js";
import {
  loadStoredRunAliases,
  loadStoredSeriesColors,
  storeRunAliases,
  storeSeriesColors,
} from "./dashboard_preferences.js";

const DATA_URL = "./data/experiment_dashboard.json";
const CENTRAL_SSL_TRACK = "central_peft_ssl";
const CENTRAL_INITIAL_EVAL_TRACK = "central_peft_initial_eval";
const CENTRAL_EPOCH_METRICS = [
  "selection_macro_f1",
  "selection_accuracy_top_1",
  "selection_expected_calibration_error",
  "selection_worst_category_f1_value",
  "selection_worst_category_f1",
  "selection_loss",
  "train_loss",
  "train_sup_loss",
  "train_unsup_loss",
  "train_util_ratio",
];
const CENTRAL_INITIAL_METRIC_MAP = {
  selection_accuracy_top_1: "accuracy_top_1",
  selection_macro_f1: "macro_f1",
  selection_expected_calibration_error: "expected_calibration_error",
  selection_worst_category_f1_value: "worst_category_f1_value",
  selection_loss: "loss",
};
const CENTRAL_OVERVIEW_METRICS = [
  "macro_f1",
  "accuracy_top_1",
  "loss",
  "expected_calibration_error",
  "weighted_f1",
  "balanced_accuracy",
  "worst_category_f1_value",
  "max_calibration_error",
  "rows_total",
];
const DEFAULT_CENTRAL_OVERVIEW_METRICS = [
  "macro_f1",
  "accuracy_top_1",
  "loss",
  "expected_calibration_error",
];
const FL_RUN_METRICS = [
  "macro_f1",
  "final_macro_f1",
  "accuracy_top_1",
  "final_accuracy_top_1",
  "loss",
  "final_loss",
  "expected_calibration_error",
  "final_expected_calibration_error",
  "worst_client_macro_f1",
  "best_client_macro_f1",
  "weighted_f1",
  "balanced_accuracy",
  "worst_category_f1_value",
  "max_calibration_error",
  "macro_f1_std",
  "loss_std",
  "fairness_gap",
  "per_client_macro_f1_variance",
  "communication_cost",
  "c2s_total_bytes",
  "s2c_total_bytes_estimated",
  "labeled_row_exposure_count",
  "unlabeled_row_exposure_count",
  "total_row_exposure_count",
  "unique_labeled_row_count",
  "unique_unlabeled_row_count",
  "unique_total_row_count",
  "unlabeled_row_count",
  "evaluated_client_count",
  "client_count",
  "completed_rounds",
  "round_budget",
  "epochs",
  "max_train_steps",
  "train_batch_size",
  "learning_rate",
  "shard_alpha",
  "initial_macro_f1",
  "initial_loss",
  "initial_expected_calibration_error",
];
const DEFAULT_FL_RUN_METRICS = [
  "macro_f1",
  "worst_client_macro_f1",
  "expected_calibration_error",
  "macro_f1_std",
  "communication_cost",
];
const FL_ROUND_METRICS = [
  "macro_f1",
  "accuracy_top_1",
  "expected_calibration_error",
  "loss",
  "accepted_ratio",
  "update_count",
  "total_payload_bytes",
  "round_time_seconds",
  "gpu_memory_peak_mb",
  "macro_f1_delta_from_initial",
  "macro_f1_delta_from_previous",
  "loss_delta_from_initial",
  "loss_delta_from_previous",
  "ece_delta_from_initial",
  "accepted_ratio_delta_from_initial",
  "round_update_delta_l2_mean",
  "round_update_delta_l2_max",
  "round_update_delta_to_mean_l2_mean",
  "round_update_delta_to_mean_l2_max",
  "round_update_cosine_to_mean_mean",
  "round_update_cosine_to_mean_min",
];
const DEFAULT_FL_FILTER_AXIS_IDS = ["round_count", "method"];
const FL_FILTER_AXIS_IDS = [
  "method",
  "local_regularizer",
  "peft_adapter",
  "peft_adapter_rank",
  "data_pair",
  "label_budget",
  "round_count",
  "local_epochs",
  "client_count",
  "adapter",
  "aggregation",
  "seed",
  "shard_alpha",
];

const state = {
  bundle: null,
  activeTrack: "central_ssl",
  overviewEvalSet: "validation",
  overviewMethodName: "__all__",
  overviewMetricIds: [...DEFAULT_CENTRAL_OVERVIEW_METRICS],
  overviewRunIds: [],
  overviewSelectionTouched: false,
  overviewRunAliases: loadStoredRunAliases("central_overview"),
  comparisonEvalSet: "validation",
  comparisonMetric: "selection_macro_f1",
  comparisonChartType: "grouped_bar",
  comparisonMethodName: null,
  comparisonSelectionTouched: false,
  comparisonRunColors: loadStoredSeriesColors("central_compare"),
  comparisonIncludeInitial: true,
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
  flRunMetricIds: [...DEFAULT_FL_RUN_METRICS],
  flRunIds: [],
  flRunSelectionTouched: false,
  flRunAliases: loadStoredRunAliases("fl_runs"),
  flFilterPanelOpen: false,
  flFilterAxisIds: [...DEFAULT_FL_FILTER_AXIS_IDS],
  flFilterValues: {},
  flRoundCount: "__all__",
  flRoundMethodName: null,
  flRoundRunIds: [],
  flRoundSelectionTouched: false,
  flRoundRunAliases: loadStoredRunAliases("fl_round"),
  flRoundRunColors: loadStoredSeriesColors("fl_round"),
  flRoundIncludeInitial: true,
  flRoundMetric: "macro_f1",
  flClientValidationRunId: null,
  flClientRoundRunId: null,
  flClientRoundIndex: "__latest__",
  flSplitRunId: null,
  flProjectionEvalSet: "validation",
  flProjectionRunIds: [],
  flProjectionSelectionTouched: false,
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
  overviewMethodFilter: document.querySelector("#overview-method-filter"),
  overviewMetricPicker: document.querySelector("#overview-metric-picker"),
  overviewRunCheckboxes: document.querySelector("#overview-run-checkboxes"),
  overviewSelectedRunCards: document.querySelector("#overview-selected-run-cards"),
  overviewSelectionSummary: document.querySelector("#overview-selection-summary"),
  comparisonEvalFilter: document.querySelector("#comparison-eval-filter"),
  comparisonChartType: document.querySelector("#comparison-chart-type"),
  comparisonIncludeInitial: document.querySelector("#comparison-include-initial"),
  metricPicker: document.querySelector("#metric-picker"),
  comparisonMethodFilter: document.querySelector("#comparison-method-filter"),
  comparisonRunCheckboxes: document.querySelector("#comparison-run-checkboxes"),
  selectedRunCards: document.querySelector("#selected-run-cards"),
  classEvalFilter: document.querySelector("#class-eval-filter"),
  detailMethodFilter: document.querySelector("#detail-method-filter"),
  detailRunFilter: document.querySelector("#detail-run-filter"),
  detailRunSummary: document.querySelector("#detail-run-summary"),
  classMetricFilter: document.querySelector("#class-metric-filter"),
  metricCards: document.querySelector("#metric-cards"),
  comparisonChart: document.querySelector("#comparison-chart"),
  classChart: document.querySelector("#class-chart"),
  classTable: document.querySelector("#class-table"),
  confusionMatrix: document.querySelector("#confusion-matrix"),
  projectionEvalFilter: document.querySelector("#projection-eval-filter"),
  projectionMethodFilter: document.querySelector("#projection-method-filter"),
  projectionRunCheckboxes: document.querySelector("#projection-run-checkboxes"),
  projectionGallery: document.querySelector("#projection-gallery"),
  runTableHead: document.querySelector("#run-table-head"),
  runTable: document.querySelector("#run-table"),
  flMetricCards: document.querySelector("#fl-metric-cards"),
  flFilterToggle: document.querySelector("#fl-filter-toggle"),
  flFilterCard: document.querySelector("#fl-filter-card"),
  flFilterAxisPicker: document.querySelector("#fl-filter-axis-picker"),
  flActiveFilters: document.querySelector("#fl-active-filters"),
  flFilterSummary: document.querySelector("#fl-filter-summary"),
  flFilterReset: document.querySelector("#fl-filter-reset"),
  flRunMetricPicker: document.querySelector("#fl-run-metric-picker"),
  flRunCheckboxes: document.querySelector("#fl-run-checkboxes"),
  flRunSelectedRunCards: document.querySelector("#fl-run-selected-run-cards"),
  flRunSelectionSummary: document.querySelector("#fl-run-selection-summary"),
  flRunTableHead: document.querySelector("#fl-run-table-head"),
  flRunTable: document.querySelector("#fl-run-table"),
  flRoundCountFilter: document.querySelector("#fl-round-count-filter"),
  flRoundMethodFilter: document.querySelector("#fl-round-method-filter"),
  flRoundRunCheckboxes: document.querySelector("#fl-round-run-checkboxes"),
  flRoundSelectedRunCards: document.querySelector(
    "#fl-round-selected-run-cards",
  ),
  flRoundIncludeInitial: document.querySelector("#fl-round-include-initial"),
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
  flProjectionEvalFilter: document.querySelector("#fl-projection-eval-filter"),
  flProjectionRunCheckboxes: document.querySelector(
    "#fl-projection-run-checkboxes",
  ),
  flProjectionGallery: document.querySelector("#fl-projection-gallery"),
};

init();

async function init() {
  bindEvents();
  try {
    const response = await fetch(DATA_URL, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    state.bundle = normalizeDashboardBundle(await response.json());
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

function normalizeDashboardBundle(bundle) {
  const runs = (bundle.runs ?? []).map(normalizeRunRecord);
  const flSslRuns = (bundle.fl_ssl_runs ?? []).map(normalizeRunRecord);
  return {
    ...bundle,
    filters: normalizeDashboardFilters(bundle.filters ?? {}),
    runs,
    fl_ssl_runs: flSslRuns,
  };
}

function normalizeRunRecord(row) {
  const legacyTrackNames = {
    central_lora_ssl: CENTRAL_SSL_TRACK,
    central_lora_initial_eval: CENTRAL_INITIAL_EVAL_TRACK,
  };
  const normalized = {
    ...row,
    track: legacyTrackNames[row.track] ?? row.track,
  };
  if (
    normalized.payload_adapter_kind === undefined &&
    normalized.adapter_family_name !== undefined
  ) {
    normalized.payload_adapter_kind = normalized.adapter_family_name;
  }
  const legacyFieldRenames = {
    lora_rank: "peft_adapter_rank",
    lora_alpha: "peft_adapter_alpha",
    lora_dropout: "peft_adapter_dropout",
    lora_bias: "peft_adapter_bias",
    lora_target_modules: "peft_adapter_target_modules",
    lora_use_rslora: "peft_adapter_use_rslora",
    lora_use_dora: "peft_adapter_use_dora",
  };
  for (const [legacyKey, currentKey] of Object.entries(legacyFieldRenames)) {
    if (normalized[currentKey] === undefined && normalized[legacyKey] !== undefined) {
      normalized[currentKey] = normalized[legacyKey];
    }
  }
  return normalized;
}

function normalizeDashboardFilters(filters) {
  return {
    ...filters,
    peft_adapter_ranks: filters.peft_adapter_ranks ?? filters.lora_ranks ?? [],
    peft_adapter_alphas: filters.peft_adapter_alphas ?? filters.lora_alphas ?? [],
    peft_adapter_use_rslora_values:
      filters.peft_adapter_use_rslora_values ??
      filters.lora_use_rslora_values ??
      [],
    peft_adapter_use_dora_values:
      filters.peft_adapter_use_dora_values ?? filters.lora_use_dora_values ?? [],
  };
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
  elements.flFilterToggle.addEventListener("click", () => {
    state.flFilterPanelOpen = !state.flFilterPanelOpen;
    renderFlFilterPanelVisibility();
  });
  elements.flFilterAxisPicker.addEventListener("change", (event) => {
    if (!(event.target instanceof HTMLInputElement)) {
      return;
    }
    state.flFilterAxisIds = checkedValues(
      elements.flFilterAxisPicker,
      "flFilterAxis",
    );
    for (const axisId of Object.keys(state.flFilterValues)) {
      if (!state.flFilterAxisIds.includes(axisId)) {
        delete state.flFilterValues[axisId];
      }
    }
    state.flRunSelectionTouched = false;
    preserveFlRoundSelectionAfterFilterChange();
    render();
  });
  elements.flActiveFilters.addEventListener("change", (event) => {
    if (!(event.target instanceof HTMLInputElement)) {
      return;
    }
    const axisId = event.target.dataset.flFilterValueAxis;
    if (!axisId) {
      return;
    }
    state.flFilterValues[axisId] = checkedFlFilterValues(axisId);
    state.flRunSelectionTouched = false;
    preserveFlRoundSelectionAfterFilterChange();
    render();
  });
  elements.flFilterReset.addEventListener("click", () => {
    state.flFilterAxisIds = [...DEFAULT_FL_FILTER_AXIS_IDS];
    state.flFilterValues = {};
    state.flRunSelectionTouched = false;
    preserveFlRoundSelectionAfterFilterChange();
    render();
  });
  elements.flRunMetricPicker.addEventListener("change", (event) => {
    if (!(event.target instanceof HTMLInputElement)) {
      return;
    }
    state.flRunMetricIds = checkedValues(
      elements.flRunMetricPicker,
      "flRunMetric",
    );
    render();
  });
  elements.flRunCheckboxes.addEventListener("change", (event) => {
    if (!(event.target instanceof HTMLInputElement)) {
      return;
    }
    state.flRunIds = checkedValues(elements.flRunCheckboxes, "flRunId");
    state.flRunSelectionTouched = true;
    render();
  });
  elements.flRunSelectedRunCards.addEventListener("click", (event) => {
    if (!(event.target instanceof HTMLButtonElement)) {
      return;
    }
    const runId = event.target.dataset.removeFlRunId;
    if (!runId) {
      return;
    }
    state.flRunIds = state.flRunIds.filter(
      (selectedRunId) => selectedRunId !== runId,
    );
    state.flRunSelectionTouched = true;
    render();
  });
  elements.flRunSelectedRunCards.addEventListener("change", (event) => {
    if (!(event.target instanceof HTMLInputElement)) {
      return;
    }
    updateFlRunAlias(event.target);
  });
  elements.flRunSelectedRunCards.addEventListener("input", (event) => {
    if (!(event.target instanceof HTMLInputElement)) {
      return;
    }
    updateFlRunAlias(event.target);
  });
  elements.overviewEvalFilter.addEventListener("change", (event) => {
    state.overviewEvalSet = event.target.value;
    state.overviewSelectionTouched = false;
    render();
  });
  elements.overviewMethodFilter.addEventListener("change", (event) => {
    state.overviewMethodName = event.target.value || "__all__";
    state.overviewSelectionTouched = false;
    render();
  });
  elements.overviewMetricPicker.addEventListener("change", (event) => {
    if (!(event.target instanceof HTMLInputElement)) {
      return;
    }
    state.overviewMetricIds = checkedValues(
      elements.overviewMetricPicker,
      "overviewMetric",
    );
    render();
  });
  elements.overviewRunCheckboxes.addEventListener("change", (event) => {
    if (!(event.target instanceof HTMLInputElement)) {
      return;
    }
    state.overviewRunIds = checkedValues(
      elements.overviewRunCheckboxes,
      "overviewRunId",
    );
    state.overviewSelectionTouched = true;
    render();
  });
  elements.overviewSelectedRunCards.addEventListener("click", (event) => {
    if (!(event.target instanceof HTMLButtonElement)) {
      return;
    }
    const runId = event.target.dataset.removeOverviewRunId;
    if (!runId) {
      return;
    }
    state.overviewRunIds = state.overviewRunIds.filter(
      (selectedRunId) => selectedRunId !== runId,
    );
    state.overviewSelectionTouched = true;
    render();
  });
  elements.overviewSelectedRunCards.addEventListener("change", (event) => {
    if (!(event.target instanceof HTMLInputElement)) {
      return;
    }
    updateOverviewRunAlias(event.target);
  });
  elements.overviewSelectedRunCards.addEventListener("input", (event) => {
    if (!(event.target instanceof HTMLInputElement)) {
      return;
    }
    updateOverviewRunAlias(event.target);
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
  elements.comparisonIncludeInitial.addEventListener("change", (event) => {
    state.comparisonIncludeInitial = event.target.checked;
    render();
  });
  elements.metricPicker.addEventListener("click", (event) => {
    if (!(event.target instanceof HTMLButtonElement)) {
      return;
    }
    const metric = event.target.dataset.metric;
    if (!metric) {
      return;
    }
    state.comparisonMetric = metric;
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
  bindSeriesColorEvents(elements.comparisonChart);
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
  if (elements.flRoundCountFilter) {
    elements.flRoundCountFilter.addEventListener("change", (event) => {
      state.flRoundCount = event.target.value || "__all__";
      resetFlRoundSelection();
      render();
    });
  }
  if (elements.flRoundMethodFilter) {
    elements.flRoundMethodFilter.addEventListener("change", (event) => {
      state.flRoundMethodName = event.target.value || null;
      resetFlRoundSelection();
      render();
    });
  }
  elements.flRoundRunCheckboxes.addEventListener("change", (event) => {
    if (!(event.target instanceof HTMLInputElement)) {
      return;
    }
    const visibleRunIds = new Set(
      Array.from(
        elements.flRoundRunCheckboxes.querySelectorAll("input[type='checkbox']"),
      )
        .map((input) => input.dataset.flRoundRunId)
        .filter((runId) => runId),
    );
    const checkedRunIds = checkedValues(
      elements.flRoundRunCheckboxes,
      "flRoundRunId",
    );
    state.flRoundRunIds = uniqueValues([
      ...state.flRoundRunIds.filter((runId) => !visibleRunIds.has(runId)),
      ...checkedRunIds,
    ]);
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
  elements.flRoundIncludeInitial.addEventListener("change", (event) => {
    state.flRoundIncludeInitial = event.target.checked;
    render();
  });
  bindSeriesColorEvents(elements.flRoundChart);
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
  elements.flProjectionEvalFilter.addEventListener("change", (event) => {
    state.flProjectionEvalSet = event.target.value;
    state.flProjectionRunIds = [];
    state.flProjectionSelectionTouched = false;
    render();
  });
  elements.flProjectionRunCheckboxes.addEventListener("change", (event) => {
    if (!(event.target instanceof HTMLInputElement)) {
      return;
    }
    const visibleRunIds = new Set(
      Array.from(
        elements.flProjectionRunCheckboxes.querySelectorAll(
          "input[type='checkbox']",
        ),
      )
        .map((input) => input.dataset.flProjectionRunId)
        .filter((runId) => runId),
    );
    const checkedRunIds = checkedValues(
      elements.flProjectionRunCheckboxes,
      "flProjectionRunId",
    );
    state.flProjectionRunIds = uniqueValues([
      ...state.flProjectionRunIds.filter((runId) => !visibleRunIds.has(runId)),
      ...checkedRunIds,
    ]);
    state.flProjectionSelectionTouched = true;
    render();
  });
  elements.flProjectionGallery.addEventListener("click", (event) => {
    if (!(event.target instanceof HTMLButtonElement)) {
      return;
    }
    const runId = event.target.dataset.removeFlProjectionRunId;
    if (!runId) {
      return;
    }
    state.flProjectionRunIds = state.flProjectionRunIds.filter(
      (selectedRunId) => selectedRunId !== runId,
    );
    state.flProjectionSelectionTouched = true;
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
  const overviewBaseRows = selectedMetricRows(
    state.overviewEvalSet,
    state.overviewMetricIds[0] ?? "macro_f1",
  );
  normalizeOverviewMethodSelection(overviewBaseRows);
  const overviewRows = centralOverviewRowsForMethod(overviewBaseRows);
  const comparisonRows = selectedMetricRows(state.comparisonEvalSet);
  const classRows = selectedMetricRows(state.classEvalSet);
  const projectionBaseRows = selectedMetricRows(state.projectionEvalSet);
  normalizeComparisonSelection(comparisonRows);
  normalizeDetailSelection(classRows);
  normalizeProjectionSelection(projectionBaseRows);
  const projectionRows = rowsWithProjection(projectionBaseRows);
  normalizeOverviewSelection(overviewRows);

  renderEvalFilters();
  renderOverviewMethodFilter(overviewBaseRows);
  renderOverviewControls(overviewRows);
  renderMetricCards(overviewRows);
  renderMetricPicker();
  renderComparisonRunControls(comparisonRows);
  renderSelectedRunCards(comparisonRows);
  renderDetailRunFilter(classRows);
  renderProjectionRunControls(projectionRows);
  renderComparisonChart(comparisonRows);
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
  const allRows = sortedFlRows(flSslRows());
  normalizeFlFilterState(allRows);
  const rows = applyFlFilters(allRows);
  normalizeFlSelections(rows);
  normalizeFlRunSelection(rows);
  renderFlTabs();
  renderFlFilterControls(allRows, rows);
  renderFlFilterPanelVisibility();
  renderFlRunControls(rows);

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
  renderFlProjectionPanel(rows);
}

function renderFlRunTable(rows) {
  const selectedRows = rows.filter((row) => state.flRunIds.includes(flRunId(row)));
  const metricIds = state.flRunMetricIds;
  const columnCount = 7 + metricIds.length;
  elements.flRunTableHead.innerHTML = `
    <tr>
      <th>run</th>
      <th>method</th>
      <th>regularizer</th>
      <th>rank</th>
      <th>split</th>
      <th>clients</th>
      <th>rounds</th>
      ${metricIds.map((metric) => `<th>${escapeHtml(metricLabel(metric))}</th>`).join("")}
    </tr>
  `;
  if (rows.length === 0) {
    elements.flRunTable.innerHTML = emptyTableRow(
      columnCount,
      "아직 dashboard bundle에 FL SSL run이 없습니다.",
    );
    return;
  }
  if (selectedRows.length === 0) {
    elements.flRunTable.innerHTML = emptyTableRow(
      columnCount,
      "선택한 FL SSL run이 없습니다.",
    );
    return;
  }

  elements.flRunTable.innerHTML = selectedRows
    .map(
      (row) => `
        <tr>
          <td title="${escapeHtml(flRunDescriptor(row))}">${escapeHtml(flRunDisplayLabel(row))}</td>
          <td>${escapeHtml(flMethodName(row))}</td>
          <td>${escapeHtml(flLocalRegularizerLabel(row))}</td>
          <td>${formatCount(row.peft_adapter_rank)}</td>
          <td title="${escapeHtml(shortSplit(row.selection_slug))}">${escapeHtml(flDataSourceLabel(row))} · ${escapeHtml(flLabelBudgetLabel(row))}</td>
          <td>${formatCount(row.client_count)}</td>
          <td>${formatCount(row.completed_rounds)} / ${formatCount(row.round_budget)}</td>
          ${metricIds
            .map((metric) => `<td>${formatFlRunMetric(row, metric)}</td>`)
            .join("")}
        </tr>
      `,
    )
    .join("");
}

function normalizeFlRunSelection(rows) {
  const availableMetrics = flRunMetricKeys(rows);
  state.flRunMetricIds = state.flRunMetricIds.filter((metric) =>
    availableMetrics.includes(metric),
  );
  if (state.flRunMetricIds.length === 0) {
    state.flRunMetricIds = DEFAULT_FL_RUN_METRICS.filter((metric) =>
      availableMetrics.includes(metric),
    );
  }
  if (state.flRunMetricIds.length === 0 && availableMetrics.length > 0) {
    state.flRunMetricIds = [availableMetrics[0]];
  }

  const visibleRunIds = new Set(rows.map((row) => flRunId(row)));
  state.flRunIds = state.flRunIds.filter((runId) => visibleRunIds.has(runId));
  if (!state.flRunSelectionTouched && state.flRunIds.length === 0) {
    state.flRunIds = rows.slice(0, 8).map((row) => flRunId(row));
  }
}

function renderFlRunControls(rows) {
  const availableMetrics = flRunMetricKeys(rows);
  renderCheckboxList(
    elements.flRunMetricPicker,
    availableMetrics,
    new Set(state.flRunMetricIds),
    "flRunMetric",
    (metric) => metricLabel(metric),
  );
  const selectedRunIds = new Set(state.flRunIds);
  if (rows.length === 0) {
    elements.flRunCheckboxes.innerHTML =
      `<p class="empty">선택 가능한 FL SSL run이 없습니다.</p>`;
  } else {
    elements.flRunCheckboxes.innerHTML = rows
      .map((row) => {
        const runId = flRunId(row);
        const detail = flRunDescriptor(row);
        return `
          <label class="run-option" title="${escapeHtml(detail)}">
            <input
              type="checkbox"
              data-fl-run-id="${runId}"
              ${selectedRunIds.has(runId) ? "checked" : ""}
            />
            <span>
              <strong>${escapeHtml(flRunDisplayLabel(row))}</strong>
              <small>${escapeHtml(compactFlRunSubLabel(row))}</small>
            </span>
          </label>
        `;
      })
      .join("");
  }
  renderFlRunSelectedRunCards(rows);
  elements.flRunSelectionSummary.textContent = [
    `selected runs=${state.flRunIds.length}/${rows.length}`,
    `metrics=${state.flRunMetricIds.length}`,
    `basis=final_round`,
  ].join(" · ");
}

function renderFlRunSelectedRunCards(rows) {
  const rowsById = new Map(rows.map((row) => [flRunId(row), row]));
  const selectedRows = state.flRunIds.map((runId) => rowsById.get(runId)).filter(Boolean);
  if (selectedRows.length === 0) {
    elements.flRunSelectedRunCards.innerHTML =
      `<p class="empty">선택된 FL run이 없습니다.</p>`;
    return;
  }
  elements.flRunSelectedRunCards.innerHTML = selectedRows
    .map((row) => {
      const runId = flRunId(row);
      const label = flRunDisplayLabel(row);
      return `
        <article class="selected-run-card alias-run-card">
          <strong>${escapeHtml(label)}</strong>
          <input
            type="text"
            data-fl-run-alias-run-id="${runId}"
            value="${escapeHtml(state.flRunAliases[runId] ?? "")}"
            placeholder="run alias"
            aria-label="${escapeHtml(compactFlRunLabel(row))} 표시명 alias"
          />
          <button
            type="button"
            data-remove-fl-run-id="${runId}"
            aria-label="${escapeHtml(label)} 제거"
          >x</button>
          <span class="selected-run-detail" aria-hidden="true">${escapeHtml(flRunDescriptor(row))}</span>
        </article>
      `;
    })
    .join("");
}

function flRunMetricKeys(rows) {
  const discovered = new Set();
  for (const row of rows) {
    for (const metric of FL_RUN_METRICS) {
      if (flRunMetricValue(row, metric) !== null) {
        discovered.add(metric);
      }
    }
  }
  return FL_RUN_METRICS.filter((metric) => discovered.has(metric));
}

function renderFlFilterControls(allRows, filteredRows) {
  const selectedAxes = new Set(state.flFilterAxisIds);
  elements.flFilterAxisPicker.innerHTML = FL_FILTER_AXIS_IDS.map((axisId) => {
    const axis = flFilterAxis(axisId);
    const valueCount = flFilterOptions(allRows, axisId).length;
    return `
      <label class="check-row compact">
        <input
          type="checkbox"
          data-fl-filter-axis="${axisId}"
          ${selectedAxes.has(axisId) ? "checked" : ""}
        />
        <span>${escapeHtml(axis.label)} (${valueCount})</span>
      </label>
    `;
  }).join("");

  if (state.flFilterAxisIds.length === 0) {
    elements.flActiveFilters.innerHTML =
      `<p class="empty">왼쪽에서 사용할 필터 축을 선택하세요.</p>`;
  } else {
    elements.flActiveFilters.innerHTML = state.flFilterAxisIds
      .map((axisId) => renderFlActiveFilterGroup(allRows, axisId))
      .join("");
  }

  elements.flFilterSummary.textContent = [
    `filtered runs=${filteredRows.length}/${allRows.length}`,
    `axes=${state.flFilterAxisIds.length}`,
  ].join(" · ");
}

function renderFlFilterPanelVisibility() {
  elements.flFilterCard.classList.toggle("open", state.flFilterPanelOpen);
  elements.flFilterToggle.classList.toggle("active", state.flFilterPanelOpen);
  elements.flFilterToggle.setAttribute(
    "aria-expanded",
    String(state.flFilterPanelOpen),
  );
}

function renderFlActiveFilterGroup(rows, axisId) {
  const axis = flFilterAxis(axisId);
  const selectedValues = new Set(state.flFilterValues[axisId] ?? []);
  const options = flFilterOptions(rows, axisId);
  if (options.length === 0) {
    return `
      <section class="filter-group">
        <p class="filter-group-title">${escapeHtml(axis.label)}</p>
        <p class="empty">선택 가능한 값 없음</p>
      </section>
    `;
  }
  const valueRows = options
    .map(
      (option) => `
        <label class="check-row compact">
          <input
            type="checkbox"
            data-fl-filter-value-axis="${axisId}"
            data-fl-filter-value="${escapeHtml(option.value)}"
            ${selectedValues.has(option.value) ? "checked" : ""}
          />
          <span>${escapeHtml(option.label)} (${option.count})</span>
        </label>
      `,
    )
    .join("");
  return `
    <section class="filter-group">
      <p class="filter-group-title">${escapeHtml(axis.label)}</p>
      <div class="filter-value-list">${valueRows}</div>
    </section>
  `;
}

function renderFlRunSelectors(rows) {
  const roundRuns = flRunsWithRows(rows, flRoundRows());
  if (elements.flRoundCountFilter) {
    fillFlRoundCountSelect(elements.flRoundCountFilter, roundRuns);
  }
  if (elements.flRoundMethodFilter) {
    fillFlMethodSelect(
      elements.flRoundMethodFilter,
      roundRuns,
      state.flRoundMethodName,
    );
  }
  renderFlRoundRunControls(roundRuns);
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
  elements.flRoundIncludeInitial.checked = state.flRoundIncludeInitial;
  renderFlRoundFlatNote(rows);
  if (rows.length === 0) {
    elements.flRoundChart.innerHTML =
      `<p class="empty">선택한 run들의 round curve가 없습니다.</p>`;
    elements.flRoundTable.innerHTML = emptyTableRow(13, "round row 없음");
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
          <td>${formatMegabytes(row.gpu_memory_peak_mb)}</td>
          <td>${formatMetric(row.round_update_delta_l2_mean)}</td>
          <td>${formatMetric(row.round_update_delta_l2_max)}</td>
          <td>${formatMetric(row.round_update_cosine_to_mean_mean)}</td>
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
      10,
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
          <td>${formatMetric(row.per_client_delta_l2_norm ?? row.delta_l2_norm)}</td>
          <td>${formatMetric(row.per_client_delta_cosine_to_mean)}</td>
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

function renderFlProjectionPanel(rows) {
  const projectionRows = flRowsWithProjection(rows);
  fillSelect(
    elements.flProjectionEvalFilter,
    flProjectionEvalSets(rows),
    state.flProjectionEvalSet,
  );
  renderFlProjectionRunControls(projectionRows);
  renderFlProjectionGallery();
}

function renderFlProjectionRunControls(rows) {
  if (rows.length === 0) {
    elements.flProjectionRunCheckboxes.innerHTML =
      `<p class="empty">projection 이미지가 있는 FL run이 없습니다.</p>`;
    return;
  }
  const selectedRunIds = new Set(state.flProjectionRunIds);
  elements.flProjectionRunCheckboxes.innerHTML = rows
    .map((row) => {
      const runId = flRunId(row);
      return `
        <label class="run-option">
          <input
            type="checkbox"
            data-fl-projection-run-id="${runId}"
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
}

function renderFlProjectionGallery() {
  const selectedRunIds = new Set(state.flProjectionRunIds);
  const runsById = new Map(flSslRows().map((run) => [flRunId(run), run]));
  const images = flProjectionImagesForEval()
    .filter((image) => selectedRunIds.has(image.run_id))
    .filter((image) => runsById.has(image.run_id));
  if (images.length === 0) {
    elements.flProjectionGallery.innerHTML =
      `<p class="empty">선택한 FL run/eval set의 projection image가 없습니다.</p>`;
    return;
  }
  elements.flProjectionGallery.innerHTML = images
    .map((image) => {
      const run = runsById.get(image.run_id);
      const dataLabel = [
        flDataSourceLabel(run),
        flLabelBudgetLabel(run),
        `eval=${image.eval_set}`,
      ].join(" · ");
      return `
        <figure>
          <button
            class="projection-remove"
            type="button"
            data-remove-fl-projection-run-id="${image.run_id}"
            aria-label="${escapeHtml(defaultFlRoundRunLabel(run))} projection 제거"
          >x</button>
          <img src="${image.image_src}" alt="${escapeHtml(defaultFlRoundRunLabel(run))} ${image.eval_set} projection" loading="lazy" />
          <figcaption>
            <strong>${escapeHtml(defaultFlRoundRunLabel(run))}</strong>
            <span>${escapeHtml(flMethodName(run))} · ${escapeHtml(dataLabel)}</span>
            <span>${escapeHtml(image.reducer ?? "projection")}${image.fallback_reason ? ` · ${escapeHtml(image.fallback_reason)}` : ""}</span>
          </figcaption>
        </figure>
      `;
    })
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
    renderFlRoundSelectedRunCards();
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
  renderFlRoundSelectedRunCards();
}

function renderFlRoundSelectedRunCards() {
  const rowsById = new Map(
    flRunsWithRows(sortedFlRows(flSslRows()), flRoundRows()).map((row) => [
      flRunId(row),
      row,
    ]),
  );
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
        <article class="selected-run-card alias-run-card">
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
  storeRunAliases("fl_round", state.flRoundRunAliases);
  renderFlRoundPanel();
}

function updateOverviewRunAlias(input) {
  const runId = input.dataset.overviewAliasRunId;
  if (!runId) {
    return;
  }
  updateRunAlias(state.overviewRunAliases, "central_overview", runId, input.value);
  const rows = centralOverviewRowsForMethod(
    selectedMetricRows(state.overviewEvalSet, state.overviewMetricIds[0] ?? "macro_f1"),
  );
  const row = rows.find((candidate) => candidate.run_id === runId);
  updateSelectedRunCardLabel(input, row ? centralOverviewDisplayLabel(row) : input.value);
  renderRunTable(rows);
}

function updateFlRunAlias(input) {
  const runId = input.dataset.flRunAliasRunId;
  if (!runId) {
    return;
  }
  updateRunAlias(state.flRunAliases, "fl_runs", runId, input.value);
  const rows = applyFlFilters(sortedFlRows(flSslRows()));
  const row = rows.find((candidate) => flRunId(candidate) === runId);
  updateSelectedRunCardLabel(input, row ? flRunDisplayLabel(row) : input.value);
  renderFlRunTable(rows);
}

function updateRunAlias(aliasMap, scope, runId, rawAlias) {
  const alias = rawAlias.trim();
  if (alias) {
    aliasMap[runId] = alias;
  } else {
    delete aliasMap[runId];
  }
  storeRunAliases(scope, aliasMap);
}

function updateSelectedRunCardLabel(input, label) {
  const cardElement = input.closest(".selected-run-card");
  const labelElement = cardElement?.querySelector("strong");
  if (labelElement) {
    labelElement.textContent = label;
  }
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
  normalizeFlRoundRunSelection(roundRuns);
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
  normalizeFlProjectionSelection(rows);

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
  if (!state.flRoundSelectionTouched && state.flRoundRunIds.length === 0) {
    state.flRoundRunIds = defaultFlRoundRunIds(candidateRows);
  }
  for (const runId of Object.keys(state.flRoundRunAliases)) {
    if (!state.flRoundRunIds.includes(runId) && !visibleRunIds.has(runId)) {
      delete state.flRoundRunAliases[runId];
    }
  }
}

function normalizeFlProjectionSelection(rows) {
  const evalSets = flProjectionEvalSets(rows);
  if (!evalSets.includes(state.flProjectionEvalSet)) {
    state.flProjectionEvalSet = evalSets.includes("validation")
      ? "validation"
      : (evalSets[0] ?? "validation");
    state.flProjectionRunIds = [];
    state.flProjectionSelectionTouched = false;
  }

  const projectionRows = flRowsWithProjection(rows);
  const visibleRunIds = new Set(projectionRows.map((row) => flRunId(row)));
  state.flProjectionRunIds = state.flProjectionRunIds.filter((runId) =>
    visibleRunIds.has(runId),
  );
  if (
    !state.flProjectionSelectionTouched &&
    state.flProjectionRunIds.length === 0
  ) {
    state.flProjectionRunIds = projectionRows.map((row) => flRunId(row));
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

function flProjectionImagesForEval() {
  const flRunIds = new Set(flSslRows().map((run) => flRunId(run)));
  return state.bundle.projection_images.filter(
    (image) =>
      flRunIds.has(image.run_id) && image.eval_set === state.flProjectionEvalSet,
  );
}

function flProjectionEvalSets(rows) {
  const runIds = new Set(rows.map((row) => flRunId(row)));
  return uniqueValues(
    state.bundle.projection_images
      .filter((image) => runIds.has(image.run_id))
      .map((image) => image.eval_set),
  ).sort();
}

function flRowsWithProjection(rows) {
  const projectionRunIds = new Set(
    flProjectionImagesForEval().map((image) => image.run_id),
  );
  return rows.filter((row) => projectionRunIds.has(flRunId(row)));
}

function normalizeFlFilterState(rows) {
  const validAxisIds = new Set(FL_FILTER_AXIS_IDS);
  state.flFilterAxisIds = state.flFilterAxisIds.filter((axisId) =>
    validAxisIds.has(axisId),
  );
  for (const axisId of Object.keys(state.flFilterValues)) {
    if (!state.flFilterAxisIds.includes(axisId)) {
      delete state.flFilterValues[axisId];
      continue;
    }
    const optionValues = new Set(
      flFilterOptions(rows, axisId).map((option) => option.value),
    );
    state.flFilterValues[axisId] = (state.flFilterValues[axisId] ?? []).filter(
      (value) => optionValues.has(value),
    );
  }
}

function applyFlFilters(rows) {
  return rows.filter((row) =>
    state.flFilterAxisIds.every((axisId) => {
      const selectedValues = state.flFilterValues[axisId] ?? [];
      if (selectedValues.length === 0) {
        return true;
      }
      return selectedValues.includes(flFilterValue(row, axisId));
    }),
  );
}

function flFilterAxis(axisId) {
  const labels = {
    method: "Method",
    local_regularizer: "Regularizer",
    peft_adapter: "PEFT Adapter",
    peft_adapter_rank: "Adapter Rank",
    data_pair: "Labeled / Unlabeled",
    label_budget: "Label Budget",
    round_count: "Round Count",
    local_epochs: "Local Epochs",
    client_count: "Client Count",
    adapter: "Adapter",
    aggregation: "Aggregation",
    seed: "Seed",
    shard_alpha: "Shard Alpha",
  };
  return { id: axisId, label: labels[axisId] ?? axisId };
}

function flFilterOptions(rows, axisId) {
  const counts = new Map();
  for (const row of rows) {
    const value = flFilterValue(row, axisId);
    const label = flFilterLabel(row, axisId);
    if (!counts.has(value)) {
      counts.set(value, { value, label, count: 0 });
    }
    counts.get(value).count += 1;
  }
  return Array.from(counts.values()).sort((a, b) =>
    a.label.localeCompare(b.label, undefined, { numeric: true }),
  );
}

function flFilterValue(row, axisId) {
  if (axisId === "method") return flMethodName(row);
  if (axisId === "local_regularizer") return flLocalRegularizerLabel(row);
  if (axisId === "peft_adapter") return peftAdapterVariantLabel(row);
  if (axisId === "peft_adapter_rank") return String(row.peft_adapter_rank ?? "-");
  if (axisId === "data_pair") return flDataSourceLabel(row);
  if (axisId === "label_budget") return flLabelBudgetLabel(row);
  if (axisId === "round_count") return String(flRoundCountForRun(row) ?? "-");
  if (axisId === "local_epochs") return String(row.epochs ?? "-");
  if (axisId === "client_count") return String(row.client_count ?? "-");
  if (axisId === "adapter") return String(flPayloadAdapterKind(row) ?? "-");
  if (axisId === "aggregation") return String(row.aggregation_backend_name ?? "-");
  if (axisId === "seed") return String(row.seed ?? "-");
  if (axisId === "shard_alpha") return String(row.shard_alpha ?? "-");
  return "-";
}

function flFilterLabel(row, axisId) {
  if (axisId === "round_count") return `${flRoundCountForRun(row) ?? "-"} rounds`;
  if (axisId === "local_epochs") return `${row.epochs ?? "-"} local epochs`;
  if (axisId === "client_count") return `${row.client_count ?? "-"} clients`;
  if (axisId === "seed") return `seed ${row.seed ?? "-"}`;
  if (axisId === "shard_alpha") return `alpha ${formatMetric(row.shard_alpha)}`;
  if (axisId === "peft_adapter_rank") return `rank ${row.peft_adapter_rank ?? "-"}`;
  return flFilterValue(row, axisId);
}

function checkedFlFilterValues(axisId) {
  return Array.from(
    elements.flActiveFilters.querySelectorAll(
      `input[type='checkbox'][data-fl-filter-value-axis='${axisId}']:checked`,
    ),
  )
    .map((input) => input.dataset.flFilterValue)
    .filter((value) => value);
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
    .filter(
      (row) =>
        state.flRoundIncludeInitial || numberOrNull(row.round_index) !== 0,
    )
    .sort(compareFlRoundRows);
}

function flRoundCandidateRuns(rows) {
  return rows;
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

function flRunMetricValue(row, metric) {
  if (metric === "accuracy_top_1") {
    return row.final_accuracy_top_1 ?? flMetric(row, metric);
  }
  if (metric === "communication_cost") {
    return flCostValue(row);
  }
  if (metric === "c2s_total_bytes") {
    return flCommunicationEstimateBytes(row, "c2s_total_bytes");
  }
  if (metric === "s2c_total_bytes_estimated") {
    return flCommunicationEstimateBytes(row, "s2c_total_bytes_estimated");
  }
  return flMetric(row, metric);
}

function formatFlRunMetric(row, metric) {
  const value = flRunMetricValue(row, metric);
  if (
    metric === "c2s_total_bytes" ||
    metric === "s2c_total_bytes_estimated"
  ) {
    return formatBytes(value);
  }
  if (
    metric === "communication_cost" ||
    metric.endsWith("_count") ||
    metric.endsWith("_bytes") ||
    metric.includes("labeled_row")
  ) {
    return formatCount(value);
  }
  return formatMetric(value);
}

function flRunDescriptor(row) {
  const protocol = row.protocol ?? {};
  const roundRuntime = protocol.round_runtime ?? {};
  const cost = flMetric(row, "communication_cost");
  const costValue = typeof cost === "object" && cost !== null ? cost.value : cost;
  return [
    flDataSourceLabel(row),
    flLabelBudgetLabel(row),
    `adapter=${flPayloadAdapterKind(row, roundRuntime) ?? "-"}`,
    peftAdapterConfigLabel(row),
    `agg=${row.aggregation_backend_name ?? roundRuntime.aggregation_backend_name ?? "-"}`,
    `regularizer=${flLocalRegularizerLabel(row)}`,
    `clients=${row.client_count ?? protocol.client_count ?? "-"}`,
    `rounds=${row.completed_rounds ?? protocol.completed_rounds ?? "-"}/${row.round_budget ?? protocol.round_budget ?? "-"}`,
    `updates=${costValue ?? "-"}`,
    `seed=${row.seed ?? protocol.seed ?? "-"}`,
  ].join(" · ");
}

function flPayloadAdapterKind(row, roundRuntime = null) {
  const runtime = roundRuntime ?? row.protocol?.round_runtime ?? {};
  return (
    row.payload_adapter_kind ??
    runtime.payload_adapter_kind ??
    null
  );
}

function flLocalRegularizerLabel(row) {
  const name = row.local_regularizer_name ?? inferRegularizerFromRunId(flRunId(row));
  if (!name || name === "none") {
    return "none";
  }
  const mu = row.local_regularizer_mu ?? inferFedProxMuFromRunId(flRunId(row));
  return mu === null || mu === undefined ? name : `${name}_mu${mu}`;
}

function inferRegularizerFromRunId(runId) {
  return String(runId ?? "").includes("fedprox") ? "fedprox" : "none";
}

function inferFedProxMuFromRunId(runId) {
  const match = String(runId ?? "").match(/fedprox_mu([0-9.]+)/);
  return match ? match[1].replace(/\.$/, "") : null;
}

function flCostValue(row) {
  const cost = flMetric(row, "communication_cost");
  return typeof cost === "object" && cost !== null ? cost.value : cost;
}

function flCommunicationEstimateBytes(row, key) {
  const cost = flMetric(row, "communication_cost");
  if (typeof cost !== "object" || cost === null) {
    return null;
  }
  const estimates = cost.artifact_byte_estimates ?? cost.posthoc_byte_estimates;
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
    flMethodName(row),
    flLocalRegularizerLabel(row),
    `clients=${row.client_count ?? "-"}`,
    `rounds=${row.completed_rounds ?? "-"}/${row.round_budget ?? "-"}`,
    `alpha=${formatMetric(row.shard_alpha)}`,
    `seed=${row.seed ?? "-"}`,
    flRunSuffix(row),
  ].join(" · ");
}

function compactFlRunLabel(row) {
  return [
    flMethodName(row),
    flLocalRegularizerLabel(row),
    `r${row.completed_rounds ?? "?"}`,
    `seed${row.seed ?? "?"}`,
  ].join(" · ");
}

function flRunDisplayLabel(row) {
  const alias = state.flRunAliases[flRunId(row)];
  return alias || compactFlRunLabel(row);
}

function compactFlRunSubLabel(row) {
  return [
    flLabelBudgetLabel(row),
    `clients=${row.client_count ?? "-"}`,
    `rank=${row.peft_adapter_rank ?? "-"}`,
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
    flLocalRegularizerLabel(row),
    flDataSourceLabel(row),
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
          colorKey: runId,
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
  const width = 1160;
  const height = 520;
  const pad = { top: 22, right: 52, bottom: 72, left: 112 };
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
  const colors = seriesColors(series, seriesColorOverrides("fl_round"));
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
            <circle cx="${xForPoint(point)}" cy="${yForValue(point.value)}" r="4" style="--series-color:${color}" data-series-color-key="${escapeHtml(item.colorKey ?? item.label)}">
              <title>${escapeHtml(item.label)} · ${escapeHtml(point.roundId)} · ${formatMetric(point.value)}</title>
            </circle>
          `,
        )
        .join("");
      return `<polyline points="${path}" fill="none" style="--series-color:${color}" data-series-color-key="${escapeHtml(item.colorKey ?? item.label)}" />${dots}`;
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
    ${renderSeriesLegend(series, colors, "fl_round")}
    <div class="line-chart fl-round-line-chart">
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

function resetFlRoundSelection() {
  state.flRoundRunIds = [];
  state.flRoundSelectionTouched = false;
}

function preserveFlRoundSelectionAfterFilterChange() {
  state.flRoundSelectionTouched = true;
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
  const metrics = centralEpochMetricKeys();
  if (!metrics.includes(state.comparisonMetric)) {
    state.comparisonMetric = metrics[0] ?? "selection_macro_f1";
  }
  elements.metricPicker.innerHTML = metrics
    .map(
      (metric) => `
        <button
          type="button"
          data-metric="${metric}"
          class="${metric === state.comparisonMetric ? "active" : ""}"
        >${metricLabel(metric)}</button>
      `,
    )
    .join("");
}

function centralEpochMetricKeys() {
  const discovered = new Set();
  for (const row of state.bundle?.epoch_metrics ?? []) {
    for (const key of Object.keys(row)) {
      if (key !== "run_id" && key !== "epoch" && numberOrNull(row[key]) !== null) {
        discovered.add(key);
      }
    }
  }
  return uniqueValues([
    ...CENTRAL_EPOCH_METRICS.filter((metric) => discovered.has(metric)),
    ...Array.from(discovered).sort(),
  ]);
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
        <article class="selected-run-card">
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
  return Array.from(
    new Set(rows.map((row) => row.method_name).filter((methodName) => methodName)),
  );
}

function normalizeOverviewMethodSelection(rows) {
  const methodNames = new Set(uniqueMethodNames(rows));
  if (
    state.overviewMethodName !== "__all__" &&
    !methodNames.has(state.overviewMethodName)
  ) {
    state.overviewMethodName = "__all__";
  }
}

function centralOverviewRowsForMethod(rows) {
  if (state.overviewMethodName === "__all__") {
    return rows;
  }
  return rows.filter((row) => row.method_name === state.overviewMethodName);
}

function renderOverviewMethodFilter(rows) {
  const methodNames = uniqueMethodNames(rows).sort();
  const allSelected = state.overviewMethodName === "__all__" ? "selected" : "";
  elements.overviewMethodFilter.innerHTML = [
    `<option value="__all__" ${allSelected}>All methods (${rows.length})</option>`,
    ...methodNames.map((methodName) => {
      const selected = methodName === state.overviewMethodName ? "selected" : "";
      const runCount = rows.filter((row) => row.method_name === methodName).length;
      return `<option value="${escapeHtml(methodName)}" ${selected}>${escapeHtml(methodName)} (${runCount})</option>`;
    }),
  ].join("");
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
    peftAdapterConfigLabel(row),
    `lr=${formatMetric(row.learning_rate)}`,
    `clf=${formatMetric(row.classifier_learning_rate)}`,
    shortSplit(row.selection_slug),
  ].join(" · ");
}

function peftAdapterConfigLabel(row) {
  const rank = row.peft_adapter_rank ?? "-";
  const alpha = row.peft_adapter_alpha ?? "-";
  const dropout = row.peft_adapter_dropout ?? "-";
  return [
    `peft=${peftAdapterVariantLabel(row)}`,
    `r=${rank}`,
    `alpha=${alpha}`,
    `dropout=${dropout}`,
  ].join(" · ");
}

function peftAdapterVariantLabel(row) {
  const adapterName = row.peft_adapter_name ?? "-";
  const modifiers = [];
  if (row.peft_adapter_use_rslora) {
    modifiers.push("rs");
  }
  if (row.peft_adapter_use_dora) {
    modifiers.push("dora");
  }
  return modifiers.length > 0 ? `${adapterName}+${modifiers.join("+")}` : adapterName;
}

function compareMetric(a, b, metric) {
  const left = numberOrNull(a[metric]);
  const right = numberOrNull(b[metric]);
  if (left === null && right === null) return 0;
  if (left === null) return 1;
  if (right === null) return -1;
  return metric.includes("error") || metric === "loss" ? left - right : right - left;
}

function normalizeOverviewSelection(rows) {
  const availableMetrics = centralOverviewMetricKeys(rows);
  state.overviewMetricIds = state.overviewMetricIds.filter((metric) =>
    availableMetrics.includes(metric),
  );
  if (state.overviewMetricIds.length === 0) {
    state.overviewMetricIds = DEFAULT_CENTRAL_OVERVIEW_METRICS.filter((metric) =>
      availableMetrics.includes(metric),
    );
  }
  if (state.overviewMetricIds.length === 0 && availableMetrics.length > 0) {
    state.overviewMetricIds = [availableMetrics[0]];
  }

  const visibleRunIds = new Set(rows.map((row) => row.run_id));
  state.overviewRunIds = state.overviewRunIds.filter((runId) =>
    visibleRunIds.has(runId),
  );
  if (!state.overviewSelectionTouched && state.overviewRunIds.length === 0) {
    state.overviewRunIds = rows.slice(0, 8).map((row) => row.run_id);
  }
}

function centralOverviewMetricKeys(rows) {
  const discovered = new Set();
  for (const row of rows) {
    for (const [key, value] of Object.entries(row)) {
      if (typeof value !== "boolean" && numberOrNull(value) !== null) {
        discovered.add(key);
      }
    }
  }
  return uniqueValues([
    ...CENTRAL_OVERVIEW_METRICS.filter((metric) => discovered.has(metric)),
    ...Array.from(discovered)
      .filter((metric) => isDisplayMetricKey(metric))
      .sort(),
  ]);
}

function isDisplayMetricKey(metric) {
  return ![
    "seed",
    "learning_rate",
    "classifier_learning_rate",
    "epochs",
    "max_train_steps",
    "train_batch_size",
    "eval_batch_size",
    "peft_adapter_rank",
    "peft_adapter_alpha",
    "peft_adapter_dropout",
    "peft_adapter_use_dora",
    "peft_adapter_use_rslora",
    "created_at",
  ].includes(metric);
}

function renderOverviewControls(rows) {
  const availableMetrics = centralOverviewMetricKeys(rows);
  renderCheckboxList(
    elements.overviewMetricPicker,
    availableMetrics,
    new Set(state.overviewMetricIds),
    "overviewMetric",
    (metric) => metricLabel(metric),
  );
  const selectedRunIds = new Set(state.overviewRunIds);
  if (rows.length === 0) {
    elements.overviewRunCheckboxes.innerHTML =
      `<p class="empty">선택 가능한 중앙 SSL run이 없습니다.</p>`;
  } else {
    elements.overviewRunCheckboxes.innerHTML = rows
      .map(
        (row) => {
          const detail = centralRunDetail(row);
          return `
          <label class="run-option" title="${escapeHtml(detail)}">
            <input
              type="checkbox"
              data-overview-run-id="${row.run_id}"
              ${selectedRunIds.has(row.run_id) ? "checked" : ""}
            />
            <span>
              <strong>${escapeHtml(centralOverviewDisplayLabel(row))}</strong>
              <small>${escapeHtml(centralOverviewRunSubLabel(row))}</small>
            </span>
          </label>
        `;
        },
      )
      .join("");
  }
  renderOverviewSelectedRunCards(rows);
  elements.overviewSelectionSummary.textContent = [
    `selected runs=${state.overviewRunIds.length}/${rows.length}`,
    `metrics=${state.overviewMetricIds.length}`,
    `basis=best_validation_accuracy`,
  ].join(" · ");
}

function renderOverviewSelectedRunCards(rows) {
  const rowsById = new Map(rows.map((row) => [row.run_id, row]));
  const selectedRows = state.overviewRunIds
    .map((runId) => rowsById.get(runId))
    .filter(Boolean);
  if (selectedRows.length === 0) {
    elements.overviewSelectedRunCards.innerHTML =
      `<p class="empty">선택된 중앙 SSL run이 없습니다.</p>`;
    return;
  }
  elements.overviewSelectedRunCards.innerHTML = selectedRows
    .map(
      (row) => {
        const label = centralOverviewDisplayLabel(row);
        return `
        <article class="selected-run-card alias-run-card">
          <strong>${escapeHtml(label)}</strong>
          <input
            type="text"
            data-overview-alias-run-id="${row.run_id}"
            value="${escapeHtml(state.overviewRunAliases[row.run_id] ?? "")}"
            placeholder="run alias"
            aria-label="${escapeHtml(centralOverviewRunLabel(row))} 표시명 alias"
          />
          <button
            type="button"
            data-remove-overview-run-id="${row.run_id}"
            aria-label="${escapeHtml(label)} 제거"
          >x</button>
          <span class="selected-run-detail" aria-hidden="true">${escapeHtml(centralRunDetail(row))}</span>
        </article>
      `;
      },
    )
    .join("");
}

function renderMetricCards(rows) {
  const selectedRows = rows.filter((row) =>
    state.overviewRunIds.includes(row.run_id),
  );
  const primaryMetric = state.overviewMetricIds[0] ?? "macro_f1";
  const best = selectedRows.slice().sort((a, b) => compareMetric(a, b, primaryMetric))[0];
  const runCount = rows.length;
  const methodCount = new Set(rows.map((row) => row.method_name)).size;
  elements.metricCards.innerHTML = [
    card("runs", runCount),
    card("methods", methodCount),
    card(`best ${metricLabel(primaryMetric)}`, best ? formatMetric(best[primaryMetric]) : "-"),
    card("selected runs", state.overviewRunIds.length),
  ].join("");
}

function card(label, value) {
  return `<div class="metric-card"><span>${label}</span><strong>${value}</strong></div>`;
}

function renderComparisonChart(rows) {
  const selectedRows = rows.filter((row) => state.selectedRunIds.includes(row.run_id));
  const metric = state.comparisonMetric;
  elements.comparisonIncludeInitial.checked = state.comparisonIncludeInitial;
  if (!metric) {
    elements.comparisonChart.innerHTML = `<p class="empty">비교할 epoch metric을 선택하세요.</p>`;
    return;
  }
  if (selectedRows.length === 0) {
    elements.comparisonChart.innerHTML = `<p class="empty">비교할 run을 선택하세요.</p>`;
    return;
  }
  if (state.comparisonChartType === "vertical_bar") {
    elements.comparisonChart.innerHTML = drawVerticalBarComparison(
      selectedRows,
      metric,
    );
    return;
  }
  if (state.comparisonChartType === "line") {
    elements.comparisonChart.innerHTML = drawMetricLineComparison(
      selectedRows,
      metric,
    );
    return;
  }
  elements.comparisonChart.innerHTML = drawGroupedHorizontalComparison(
    selectedRows,
    metric,
  );
}

function drawGroupedHorizontalComparison(selectedRows, metric) {
  const values = selectedRows
    .map((row) => centralLatestMetricValue(row.run_id, metric))
    .filter((value) => value !== null);
  if (values.length === 0) {
    return `<p class="empty">선택한 run에 ${metricLabel(metric)} epoch history가 없습니다.</p>`;
  }
  const max = Math.max(...values, 0.000001);
  return selectedRows
    .map((row) => {
      const value = centralLatestMetricValue(row.run_id, metric);
      const width = value === null ? 0 : Math.max(2, (value / max) * 100);
      const metricBars = `
            <div class="metric-bar">
              <span>${metricLabel(metric)}</span>
              <div class="bar-track"><i style="width:${width}%"></i></div>
              <strong>${formatMetric(value)}</strong>
            </div>
          `;
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

function drawVerticalBarComparison(selectedRows, metric) {
  const records = selectedRows.map((row) => ({
    run: row,
    value: centralLatestMetricValue(row.run_id, metric),
  }));
  const values = records
    .map((record) => record.value)
    .filter((value) => value !== null);
  if (values.length === 0) {
    return `<p class="empty">선택한 run에 ${metricLabel(metric)} epoch history가 없습니다.</p>`;
  }
  const width = Math.max(760, selectedRows.length * 128 + 140);
  const height = 360;
  const pad = { top: 32, right: 36, bottom: 76, left: 88 };
  const chartHeight = height - pad.top - pad.bottom;
  const chartWidth = width - pad.left - pad.right;
  const maxValue = Math.max(...values, 0.000001);
  const valueTicks = buildValueTicks(0, maxValue, 5)
    .map((value) => {
      const y = pad.top + chartHeight - (value / maxValue) * chartHeight;
      return `
        <line class="grid-line" x1="${pad.left}" y1="${y}" x2="${width - pad.right}" y2="${y}" />
        <text class="axis-label" x="${pad.left - 10}" y="${y + 4}" text-anchor="end">${formatMetric(value)}</text>
      `;
    })
    .join("");
  const groupWidth = chartWidth / selectedRows.length;
  const bars = records
    .filter((record) => record.value !== null)
    .map((record) => {
      const groupIndex = selectedRows.findIndex(
        (row) => row.run_id === record.run.run_id,
      );
      const barWidth = Math.max(18, Math.min(52, groupWidth * 0.48));
      const x = pad.left + groupIndex * groupWidth + (groupWidth - barWidth) / 2;
      const barHeight = (record.value / maxValue) * chartHeight;
      const y = pad.top + chartHeight - barHeight;
      return `
        <rect
          x="${x}"
          y="${y}"
          width="${barWidth}"
          height="${barHeight}"
          rx="4"
          fill="var(--teal)"
        >
          <title>${escapeHtml(centralRunAxisLabel(record.run))} ${metricLabel(metric)} ${formatMetric(record.value)}</title>
        </rect>
      `;
    })
    .join("");
  const labels = selectedRows
    .map((row, index) => {
      const x = pad.left + index * groupWidth + groupWidth / 2;
      const [methodLabel, runLabel] = centralRunAxisLabel(row).split(" · ");
      return `
        <text class="axis-label" x="${x}" y="${height - 46}" text-anchor="middle">
          ${escapeHtml(methodLabel)}
        </text>
        <text class="axis-label" x="${x}" y="${height - 26}" text-anchor="middle">
          ${escapeHtml(runLabel ?? shortRun(row.run_id))}
        </text>
      `;
    })
    .join("");
  return `
    <div class="chart-legend"><span><i style="background:var(--teal)"></i>${metricLabel(metric)}</span></div>
    <div class="chart-scroll">
      <svg viewBox="0 0 ${width} ${height}" role="img">
        <line class="axis-line" x1="${pad.left}" y1="${pad.top + chartHeight}" x2="${width - pad.right}" y2="${pad.top + chartHeight}" />
        <line class="axis-line" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${pad.top + chartHeight}" />
        ${valueTicks}
        ${bars}
        ${labels}
      </svg>
    </div>
  `;
}

function drawMetricLineComparison(selectedRows, metric) {
  const series = selectedRows
    .map((row) => ({
      label: centralRunAxisLabel(row),
      runId: row.run_id,
      colorKey: row.run_id,
      points: centralLinePoints(row, metric),
    }))
    .filter((item) => item.points.length > 0);
  const values = series.flatMap((item) => item.points.map((point) => point.value));
  if (values.length === 0) {
    return `<p class="empty">선택한 run에 ${metricLabel(metric)} epoch history가 없습니다.</p>`;
  }
  return drawCentralEpochLineChart(series, metric);
}

function drawCentralEpochLineChart(series, metric) {
  const allPoints = series.flatMap((item) => item.points);
  const epochIndexes = uniqueValues(
    allPoints.map((point) => point.epoch),
  ).sort((a, b) => a - b);
  const width = 1040;
  const height = 460;
  const pad = { top: 42, right: 48, bottom: 72, left: 104 };
  const chartHeight = height - pad.top - pad.bottom;
  const pointInset = 36;
  const chartWidth = width - pad.left - pad.right - pointInset * 2;
  const minEpoch = Math.min(...allPoints.map((point) => point.epoch));
  const maxEpoch = Math.max(...allPoints.map((point) => point.epoch));
  const epochRange = Math.max(maxEpoch - minEpoch, 1);
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
  const colors = seriesColors(series, seriesColorOverrides("central_compare"));
  const xForPoint = (point) =>
    pad.left + pointInset + ((point.epoch - minEpoch) / epochRange) * chartWidth;
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
            <circle cx="${xForPoint(point)}" cy="${yForValue(point.value)}" r="4" style="--series-color:${color}" data-series-color-key="${escapeHtml(item.colorKey ?? item.label)}">
              <title>${escapeHtml(item.label)} · epoch ${point.epoch} · ${formatMetric(point.value)}</title>
            </circle>
          `,
        )
        .join("");
      return `<polyline points="${path}" fill="none" style="--series-color:${color}" data-series-color-key="${escapeHtml(item.colorKey ?? item.label)}" />${dots}`;
    })
    .join("");
  const labels = epochIndexes
    .filter(
      (_epoch, index) =>
        index === 0 ||
        index === epochIndexes.length - 1 ||
        epochIndexes.length <= 12 ||
        index % Math.ceil(epochIndexes.length / 10) === 0,
    )
    .map((epoch) => {
      const x = xForPoint({ epoch });
      return `
        <text class="axis-label" x="${x}" y="${height - 24}" text-anchor="middle">
          ${epoch}
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
    <p class="chart-subtitle">${metricLabel(metric)} · x-axis: epoch${seriesHasEpochZero(series) ? " · epoch 0=initial eval" : ""}</p>
    ${renderSeriesLegend(series, colors, "central_compare")}
    <div class="chart-scroll line-chart central-epoch-line-chart">
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

function centralEpochPoints(runId, metric) {
  return (state.bundle.epoch_metrics ?? [])
    .filter((row) => row.run_id === runId)
    .map((row) => ({
      epoch: numberOrNull(row.epoch),
      value: numberOrNull(row[metric]),
    }))
    .filter((point) => point.epoch !== null && point.value !== null)
    .sort((left, right) => left.epoch - right.epoch);
}

function centralLinePoints(row, metric) {
  const points = centralEpochPoints(row.run_id, metric);
  if (!state.comparisonIncludeInitial) {
    return points;
  }
  const initialPoint = centralInitialEpochPoint(row, metric);
  if (!initialPoint || points.some((point) => point.epoch === 0)) {
    return points;
  }
  return [initialPoint, ...points].sort((left, right) => left.epoch - right.epoch);
}

function centralInitialEpochPoint(row, metric) {
  const value = centralInitialMetricValue(row, metric, state.comparisonEvalSet);
  if (value === null) {
    return null;
  }
  return { epoch: 0, value };
}

function centralInitialMetricValue(row, metric, evalSet) {
  const evalMetricKey = CENTRAL_INITIAL_METRIC_MAP[metric];
  if (!evalMetricKey) {
    return null;
  }
  const initialRun = centralInitialRunFor(row);
  if (!initialRun) {
    return null;
  }
  const initialMetric = state.bundle.eval_metrics.find(
    (metricRow) =>
      metricRow.run_id === initialRun.run_id && metricRow.eval_set === evalSet,
  );
  return initialMetric ? numberOrNull(initialMetric[evalMetricKey]) : null;
}

function centralInitialRunFor(row) {
  return state.bundle.runs.find(
    (candidate) =>
      candidate.track === CENTRAL_INITIAL_EVAL_TRACK &&
      candidate.seed === row.seed &&
      candidate.validation_dataset_name === row.validation_dataset_name &&
      candidate.test_dataset_name === row.test_dataset_name &&
      samePeftAdapterConfig(candidate, row),
  );
}

function samePeftAdapterConfig(left, right) {
  const keys = [
    "peft_adapter_name",
    "peft_adapter_rank",
    "peft_adapter_alpha",
    "peft_adapter_dropout",
    "peft_adapter_bias",
    "peft_adapter_target_modules",
    "peft_adapter_use_rslora",
    "peft_adapter_use_dora",
  ];
  return keys.every((key) => String(left[key] ?? "") === String(right[key] ?? ""));
}

function seriesHasEpochZero(series) {
  return series.some((item) => item.points.some((point) => point.epoch === 0));
}

function centralLatestMetricValue(runId, metric) {
  const points = centralEpochPoints(runId, metric);
  return points.length > 0 ? points[points.length - 1].value : null;
}

function centralRunAxisLabel(row) {
  return `${row.method_name} · ${shortRun(row.run_id)}`;
}

function centralOverviewRunLabel(row) {
  return [
    row.method_name ?? "-",
    `rank${row.peft_adapter_rank ?? "?"}`,
    centralRunSuffix(row.run_id),
  ].join(" · ");
}

function centralOverviewDisplayLabel(row) {
  const alias = state.overviewRunAliases[row.run_id];
  return alias || centralOverviewRunLabel(row);
}

function centralOverviewRunSubLabel(row) {
  return [
    row.labeled_dataset_name ?? "?",
    "->",
    row.unlabeled_dataset_name ?? "?",
    `seed${row.seed ?? "?"}`,
  ].join(" ");
}

function centralRunDetail(row) {
  return [
    row.method_name,
    shortRun(row.run_id),
    runDescriptor(row),
    `labeled=${row.labeled_dataset_name ?? "-"}`,
    `unlabeled=${row.unlabeled_dataset_name ?? "-"}`,
    `validation=${row.validation_dataset_name ?? "-"}`,
    `test=${row.test_dataset_name ?? "-"}`,
    `run_id=${row.run_id}`,
  ].join(" · ");
}

function centralDataLabel(row) {
  const labeled = row.labeled_dataset_name ?? "?";
  const unlabeled = row.unlabeled_dataset_name ?? "?";
  return `${labeled} -> ${unlabeled}`;
}

function centralRunSuffix(runId) {
  const match = String(runId ?? "").match(/(\d{4}_\d{2}_\d{2}_\d{6})$/);
  if (match) {
    return match[1].replace(/^2026_/, "");
  }
  return shortRun(runId);
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
  const selectedRows = rows.filter((row) =>
    state.overviewRunIds.includes(row.run_id),
  );
  const metricIds = state.overviewMetricIds;
  const columnCount = 6 + metricIds.length;
  elements.runTableHead.innerHTML = `
    <tr>
      <th>eval</th>
      <th>run</th>
      <th>method</th>
      <th>rank</th>
      <th>split</th>
      <th>basis</th>
      ${metricIds.map((metric) => `<th>${escapeHtml(metricLabel(metric))}</th>`).join("")}
    </tr>
  `;
  if (selectedRows.length === 0) {
    elements.runTable.innerHTML = emptyTableRow(
      columnCount,
      "선택한 중앙 SSL run이 없습니다.",
    );
    return;
  }
  elements.runTable.innerHTML = selectedRows
    .map(
      (row) => `
      <tr>
        <td>${escapeHtml(row.eval_set ?? state.overviewEvalSet)}</td>
        <td title="${escapeHtml(centralRunDetail(row))}"><button type="button" data-run-id="${row.run_id}">${escapeHtml(centralOverviewDisplayLabel(row))}</button></td>
        <td>${escapeHtml(row.method_name)}</td>
        <td>${formatCount(row.peft_adapter_rank)}</td>
        <td title="${escapeHtml(shortSplit(row.selection_slug))}">${escapeHtml(centralDataLabel(row))}</td>
        <td>best_acc</td>
        ${metricIds
          .map((metric) => `<td>${formatOverviewMetric(row, metric)}</td>`)
          .join("")}
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
      state.selectedRunIds = uniqueValues([...state.selectedRunIds, runId]);
      state.activeTab = "class";
      render();
    });
  });
}

function formatOverviewMetric(row, metric) {
  if (metric === "rows_total") {
    return formatCount(row[metric]);
  }
  return formatMetric(row[metric]);
}

function bindSeriesColorEvents(container) {
  for (const eventName of ["input", "change"]) {
    container.addEventListener(eventName, (event) => {
      if (!(event.target instanceof HTMLInputElement)) {
        return;
      }
      updateSeriesColor(event.target);
    });
  }
}

function updateSeriesColor(input) {
  if (input.type !== "color") {
    return;
  }
  const scope = input.dataset.seriesColorScope;
  const colorKey = input.dataset.seriesColorKey;
  if (!scope || !colorKey) {
    return;
  }
  const overrides = seriesColorOverrides(scope);
  overrides[colorKey] = input.value;
  storeSeriesColors(scope, overrides);
  applySeriesColorToCurrentChart(scope, colorKey, input.value);
}

function seriesColorOverrides(scope) {
  if (scope === "fl_round") {
    return state.flRoundRunColors;
  }
  if (scope === "central_compare") {
    return state.comparisonRunColors;
  }
  return {};
}

function applySeriesColorToCurrentChart(scope, colorKey, color) {
  const containers =
    scope === "fl_round"
      ? [elements.flRoundChart]
      : scope === "central_compare"
        ? [elements.comparisonChart]
        : [];
  for (const container of containers) {
    container
      .querySelectorAll(`[data-series-color-key="${cssEscape(colorKey)}"]`)
      .forEach((item) => {
        item.style.setProperty("--series-color", color);
      });
    container
      .querySelectorAll(
        `.legend-color-input[data-series-color-key="${cssEscape(colorKey)}"]`,
      )
      .forEach((item) => {
        item.value = color;
      });
  }
}

function cssEscape(value) {
  if (window.CSS?.escape) {
    return window.CSS.escape(value);
  }
  return String(value).replace(/["\\]/g, "\\$&");
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

function renderCheckboxList(container, values, selectedValues, dataKey, labelForValue) {
  if (values.length === 0) {
    container.innerHTML = `<p class="empty">선택 가능한 값이 없습니다.</p>`;
    return;
  }
  const dataAttribute = dataKey.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`);
  container.innerHTML = values
    .map(
      (value) => `
        <label class="check-row compact">
          <input
            type="checkbox"
            data-${dataAttribute}="${escapeHtml(value)}"
            ${selectedValues.has(value) ? "checked" : ""}
          />
          <span>${escapeHtml(labelForValue(value))}</span>
        </label>
      `,
    )
    .join("");
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

function formatMegabytes(value) {
  const number = numberOrNull(value);
  if (number === null) {
    return "-";
  }
  return `${formatMetric(number)} MB`;
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
  const labels = {
    expected_calibration_error: "ece",
    accuracy_top_1: "accuracy",
    macro_f1: "macro-F1",
    weighted_f1: "weighted-F1",
    balanced_accuracy: "balanced acc",
    loss: "loss",
    rows_total: "rows",
    max_calibration_error: "max ece",
    worst_category_f1_value: "worst category F1",
    worst_client_macro_f1: "worst client F1",
    best_client_macro_f1: "best client F1",
    macro_f1_std: "client F1 std",
    loss_std: "client loss std",
    fairness_gap: "client F1 gap",
    per_client_macro_f1_variance: "client F1 variance",
    final_macro_f1: "final macro-F1",
    final_accuracy_top_1: "final accuracy",
    final_loss: "final loss",
    final_expected_calibration_error: "final ece",
    initial_macro_f1: "initial macro-F1",
    initial_loss: "initial loss",
    initial_expected_calibration_error: "initial ece",
    communication_cost: "updates",
    c2s_total_bytes: "C2S",
    s2c_total_bytes_estimated: "S2C est.",
    labeled_row_exposure_count: "labeled exp",
    unique_labeled_row_count: "unique labeled",
    unlabeled_row_exposure_count: "unlabeled exp",
    total_row_exposure_count: "total exp",
    unique_unlabeled_row_count: "unique unlabeled",
    unique_total_row_count: "unique total",
    unlabeled_row_count: "unlabeled rows",
    evaluated_client_count: "eval clients",
    client_count: "clients",
    completed_rounds: "completed rounds",
    round_budget: "round budget",
    epochs: "epochs",
    max_train_steps: "max steps",
    train_batch_size: "batch",
    learning_rate: "lr",
    shard_alpha: "shard alpha",
    selection_macro_f1: "selection macro-F1",
    selection_accuracy_top_1: "selection accuracy",
    selection_expected_calibration_error: "selection ece",
    selection_worst_category_f1_value: "selection worst F1",
    selection_worst_category_f1: "selection worst F1",
    selection_loss: "selection loss",
    train_loss: "train loss",
    train_sup_loss: "supervised loss",
    train_unsup_loss: "unsupervised loss",
    train_util_ratio: "util ratio",
    update_count: "updates",
    total_payload_bytes: "payload bytes",
    round_time_seconds: "round seconds",
    gpu_memory_peak_mb: "gpu peak MB",
    macro_f1_delta_from_initial: "F1 Δ init",
    macro_f1_delta_from_previous: "F1 Δ prev",
    loss_delta_from_initial: "loss Δ init",
    loss_delta_from_previous: "loss Δ prev",
    ece_delta_from_initial: "ece Δ init",
    accepted_ratio_delta_from_initial: "accepted Δ init",
    round_update_delta_l2_mean: "update L2 mean",
    round_update_delta_l2_max: "update L2 max",
    round_update_delta_to_mean_l2_mean: "to-mean L2 mean",
    round_update_delta_to_mean_l2_max: "to-mean L2 max",
    round_update_cosine_to_mean_mean: "cos-to-mean mean",
    round_update_cosine_to_mean_min: "cos-to-mean min",
  };
  return labels[metric] ?? metric;
}

function shortSplit(selectionSlug) {
  if (!selectionSlug) return "-";
  return selectionSlug
    .replace(/^labeled-/, "L:")
    .replace("_unlabeled-", " U:")
    .replace("_validation-", " V:")
    .replace("_test-", " T:");
}
