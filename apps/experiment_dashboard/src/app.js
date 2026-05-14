const DATA_URL = "./data/experiment_dashboard.json";

const state = {
  bundle: null,
  method: "__all__",
  evalSet: "validation",
  split: "__all__",
  metric: "macro_f1",
  epochMetric: "selection_macro_f1",
  selectedRunId: null,
};

const elements = {
  loadState: document.querySelector("#load-state"),
  methodFilter: document.querySelector("#method-filter"),
  evalFilter: document.querySelector("#eval-filter"),
  splitFilter: document.querySelector("#split-filter"),
  metricFilter: document.querySelector("#metric-filter"),
  runFilter: document.querySelector("#run-filter"),
  epochMetricFilter: document.querySelector("#epoch-metric-filter"),
  metricCards: document.querySelector("#metric-cards"),
  barChart: document.querySelector("#bar-chart"),
  epochChart: document.querySelector("#epoch-chart"),
  classChart: document.querySelector("#class-chart"),
  confusionMatrix: document.querySelector("#confusion-matrix"),
  runTable: document.querySelector("#run-table"),
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
  elements.methodFilter.addEventListener("change", (event) => {
    state.method = event.target.value;
    state.selectedRunId = null;
    render();
  });
  elements.evalFilter.addEventListener("change", (event) => {
    state.evalSet = event.target.value;
    render();
  });
  elements.splitFilter.addEventListener("change", (event) => {
    state.split = event.target.value;
    state.selectedRunId = null;
    render();
  });
  elements.metricFilter.addEventListener("change", (event) => {
    state.metric = event.target.value;
    render();
  });
  elements.runFilter.addEventListener("change", (event) => {
    state.selectedRunId = event.target.value;
    render();
  });
  elements.epochMetricFilter.addEventListener("change", (event) => {
    state.epochMetric = event.target.value;
    render();
  });
}

function hydrateFilters() {
  const filters = state.bundle.filters;
  fillSelect(elements.methodFilter, ["__all__", ...filters.methods], "__all__");
  fillSelect(elements.evalFilter, filters.eval_sets, state.evalSet);
  fillSelect(elements.splitFilter, ["__all__", ...filters.selection_slugs], "__all__");
}

function render() {
  if (!state.bundle) {
    return;
  }
  const rows = selectedMetricRows();
  if (!state.selectedRunId && rows.length > 0) {
    state.selectedRunId = rows[0].run_id;
  }
  fillSelect(
    elements.runFilter,
    rows.map((row) => row.run_id),
    state.selectedRunId,
  );

  renderMetricCards(rows);
  renderBarChart(rows);
  renderEpochChart();
  renderClassChart();
  renderConfusionMatrix();
  renderRunTable(rows);
}

function selectedMetricRows() {
  const runsById = new Map(state.bundle.runs.map((run) => [run.run_id, run]));
  return state.bundle.eval_metrics
    .filter((metric) => metric.eval_set === state.evalSet)
    .map((metric) => ({ ...runsById.get(metric.run_id), ...metric }))
    .filter((row) => row.run_id)
    .filter((row) => state.method === "__all__" || row.method_name === state.method)
    .filter((row) => state.split === "__all__" || row.selection_slug === state.split)
    .sort((a, b) => compareMetric(a, b, state.metric));
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
    card("best metric", best ? formatMetric(best[state.metric]) : "-"),
    card("selected eval", state.evalSet),
  ].join("");
}

function card(label, value) {
  return `<div class="metric-card"><span>${label}</span><strong>${value}</strong></div>`;
}

function renderBarChart(rows) {
  const values = rows.slice(0, 16).map((row) => ({
    label: `${row.method_name} · ${shortRun(row.run_id)}`,
    value: numberOrNull(row[state.metric]),
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
  const rows = state.bundle.epoch_metrics.filter(
    (row) => row.run_id === state.selectedRunId,
  );
  const points = rows
    .map((row) => ({
      epoch: row.epoch,
      value: numberOrNull(row[state.epochMetric]),
    }))
    .filter((point) => point.value !== null);
  if (points.length === 0) {
    elements.epochChart.innerHTML = `<p class="empty">선택한 run에 ${state.epochMetric} history가 없습니다.</p>`;
    return;
  }
  elements.epochChart.innerHTML = drawLine(points);
}

function renderClassChart() {
  const rows = state.bundle.per_class_metrics
    .filter((row) => row.run_id === state.selectedRunId)
    .filter((row) => row.eval_set === state.evalSet)
    .sort((a, b) => a.category.localeCompare(b.category));
  const max = Math.max(...rows.map((row) => row.f1 ?? 0), 0.000001);
  elements.classChart.innerHTML = rows
    .map((row) => {
      const width = Math.max(2, ((row.f1 ?? 0) / max) * 100);
      return `
        <div class="bar-row">
          <span>${row.category}</span>
          <div class="bar-track class"><i style="width:${width}%"></i></div>
          <strong>${formatMetric(row.f1)}</strong>
        </div>
      `;
    })
    .join("");
}

function renderConfusionMatrix() {
  const cells = state.bundle.confusion_matrix_cells
    .filter((cell) => cell.run_id === state.selectedRunId)
    .filter((cell) => cell.eval_set === state.evalSet);
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

function renderRunTable(rows) {
  elements.runTable.innerHTML = rows
    .map(
      (row) => `
      <tr class="${row.run_id === state.selectedRunId ? "selected" : ""}">
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
      state.selectedRunId = button.dataset.runId;
      render();
    });
  });
}

function drawLine(points) {
  const width = 680;
  const height = 260;
  const pad = 28;
  const minValue = Math.min(...points.map((point) => point.value));
  const maxValue = Math.max(...points.map((point) => point.value));
  const valueRange = Math.max(maxValue - minValue, 0.000001);
  const epochMin = Math.min(...points.map((point) => point.epoch));
  const epochMax = Math.max(...points.map((point) => point.epoch));
  const epochRange = Math.max(epochMax - epochMin, 1);
  const path = points
    .map((point) => {
      const x = pad + ((point.epoch - epochMin) / epochRange) * (width - pad * 2);
      const y =
        height - pad - ((point.value - minValue) / valueRange) * (height - pad * 2);
      return `${x},${y}`;
    })
    .join(" ");
  const dots = points
    .map((point) => {
      const x = pad + ((point.epoch - epochMin) / epochRange) * (width - pad * 2);
      const y =
        height - pad - ((point.value - minValue) / valueRange) * (height - pad * 2);
      return `<circle cx="${x}" cy="${y}" r="4"><title>epoch ${point.epoch}: ${formatMetric(point.value)}</title></circle>`;
    })
    .join("");
  return `
    <svg viewBox="0 0 ${width} ${height}" role="img">
      <polyline points="${path}" fill="none" />
      ${dots}
      <text x="${pad}" y="20">${formatMetric(maxValue)}</text>
      <text x="${pad}" y="${height - 6}">${formatMetric(minValue)}</text>
    </svg>
  `;
}

function fillSelect(select, values, selectedValue) {
  select.innerHTML = values
    .map((value) => {
      const label = value === "__all__" ? "All" : value;
      const selected = value === selectedValue ? "selected" : "";
      return `<option value="${value}" ${selected}>${label}</option>`;
    })
    .join("");
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

function shortRun(runId) {
  if (!runId) return "-";
  return runId.replace(/^lora_/, "").replace(/_2026_.*/, "");
}

function shortSplit(selectionSlug) {
  if (!selectionSlug) return "-";
  return selectionSlug
    .replace(/^labeled-/, "L:")
    .replace("_unlabeled-", " U:")
    .replace("_validation-", " V:")
    .replace("_test-", " T:");
}
