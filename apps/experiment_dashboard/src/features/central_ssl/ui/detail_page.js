import { escapeHtml } from "../../../shared/formatting/html.js";
import { formatMetric } from "../../../shared/formatting/numbers.js";
import { fillSelect } from "../../../ui/controls/form_controls.js";
import {
  emptyTableRow,
  moveTableColumn,
  renderSortableTableHeader,
  resolveTableColumns,
} from "../../../ui/tables/table.js";
import { centralEvalSetLabel, runDetail } from "../logic/labels.js";

const CLASS_TABLE_COLUMNS = [
  { id: "category", label: "category", group: "axis", render: (row) => escapeHtml(row.category) },
  { id: "support", label: "support", group: "metric", render: (row) => formatMetric(row.support) },
  { id: "precision", label: "precision", group: "metric", render: (row) => formatMetric(row.precision) },
  { id: "recall", label: "recall", group: "metric", render: (row) => formatMetric(row.recall) },
  { id: "f1", label: "f1", group: "metric", render: (row) => formatMetric(row.f1) },
  {
    id: "mean_true_label_probability",
    label: "true prob",
    group: "metric",
    render: (row) => formatMetric(row.mean_true_label_probability),
  },
  {
    id: "mean_top_1_probability",
    label: "top1 prob",
    group: "metric",
    render: (row) => formatMetric(row.mean_top_1_probability),
  },
  {
    id: "mean_margin_top1_top2",
    label: "margin",
    group: "metric",
    render: (row) => formatMetric(row.mean_margin_top1_top2),
  },
];

export function normalizeDetailSelection(rows, state) {
  const runIds = new Set(rows.map((row) => rowToText(row.run_id)));
  if (state.detailRunId && !runIds.has(rowToText(state.detailRunId))) {
    state.detailRunId = null;
  }
}

export function renderDetailPage(elements, rows, state, bundle, rerender = () => {}) {
  const detailRows = rows;
  fillSelect(
    elements.detailRunFilter,
    detailRows.map((row) => rowToText(row.run_id)),
    rowToText(state.detailRunId),
    "run 없음",
  );
  Array.from(elements.detailRunFilter.options).forEach((option) => {
    const row = detailRows.find((candidate) => rowToText(candidate.run_id) === option.value);
    if (row) option.textContent = runDetail(row);
  });
  const row = detailRows.find((candidate) => rowToText(candidate.run_id) === rowToText(state.detailRunId));
  elements.detailRunSummary.textContent = row
    ? [`eval=${centralEvalSetLabel(state.classEvalSet)}`, rowToText(row.run_id)].join(" · ")
    : "Per-class와 confusion matrix를 보려면 상세 run을 선택하세요.";
  renderClassTable(elements, row, state, bundle, rerender);
  renderClassChart(elements, row, state, bundle);
  renderConfusionMatrix(elements, row, state, bundle);
}

function renderClassTable(elements, row, state, bundle, rerender) {
  const { visibleColumns } = resolveTableColumns(
    state.classTableColumns,
    CLASS_TABLE_COLUMNS,
    CLASS_TABLE_COLUMNS.map((column) => column.id),
  );
  renderSortableTableHeader(elements.classTableHead, visibleColumns, (sourceColumnId, targetColumnId) => {
    if (moveTableColumn(state.classTableColumns, sourceColumnId, targetColumnId)) {
      rerender();
    }
  });

  if (!row) {
    elements.classTable.innerHTML = emptyTableRow(
      visibleColumns.length || 1,
      "선택된 run이 없습니다.",
    );
    return;
  }
  const rows = (bundle.per_class_metrics ?? [])
    .filter((item) =>
      rowToText(item.run_id) === rowToText(row.run_id) &&
      item.eval_set === state.classEvalSet
    )
    .sort((left, right) => String(left.category).localeCompare(String(right.category)));
  if (rows.length === 0) {
    elements.classTable.innerHTML = emptyTableRow(
      visibleColumns.length || 1,
      "per-class metric이 없습니다.",
    );
    return;
  }
  const rowById = new Map(CLASS_TABLE_COLUMNS.map((column) => [column.id, column]));
  elements.classTable.innerHTML = rows
    .map(
      (item) => `
        <tr>
          ${visibleColumns
            .map((column) => `<td>${rowById.get(column.id)?.render(item)}</td>`)
            .join("")}
        </tr>
      `,
    )
    .join("");
}

function renderClassChart(elements, row, state, bundle) {
  if (!row) {
    elements.classChart.innerHTML = `<p class="empty">선택된 run이 없습니다.</p>`;
    return;
  }
  const rows = (bundle.per_class_metrics ?? []).filter(
    (item) =>
      rowToText(item.run_id) === rowToText(row.run_id) && item.eval_set === state.classEvalSet,
  );
  const max = Math.max(...rows.map((item) => Number(item[state.classMetric]) || 0), 0.000001);
  elements.classChart.innerHTML = rows
    .map((item) => {
      const value = Number(item[state.classMetric]) || 0;
      return `
        <div class="metric-bar">
          <span>${escapeHtml(item.category)}</span>
          <div class="bar-track"><i style="width:${Math.max(2, (value / max) * 100)}%"></i></div>
          <strong>${formatMetric(value)}</strong>
        </div>
      `;
    })
    .join("");
}

function renderConfusionMatrix(elements, row, state, bundle) {
  if (!row) {
    elements.confusionMatrix.innerHTML = `<p class="empty">선택된 run이 없습니다.</p>`;
    return;
  }
  const cells = (bundle.confusion_matrix_cells ?? []).filter(
    (cell) =>
      rowToText(cell.run_id) === rowToText(row.run_id) &&
      cell.eval_set === state.classEvalSet,
  );
  if (cells.length === 0) {
    elements.confusionMatrix.innerHTML = `<p class="empty">confusion matrix가 없습니다.</p>`;
    return;
  }
  const labels = Array.from(
    new Set(
      cells.flatMap((cell) => [
        cell.actual_category ?? cell.true_label,
        cell.predicted_category ?? cell.predicted_label,
      ]),
    ),
  ).sort();
  const max = Math.max(...cells.map((cell) => Number(cell.count) || 0), 1);
  const byKey = new Map(
    cells.map((cell) => [
      `${cell.actual_category ?? cell.true_label}::${cell.predicted_category ?? cell.predicted_label}`,
      cell,
    ]),
  );
  elements.confusionMatrix.innerHTML = `
    <div class="matrix-grid" style="--matrix-category-count:${labels.length}">
      <span class="matrix-corner">actual \\ predicted</span>
      ${labels.map((label) => `<strong class="matrix-column-label">${escapeHtml(label)}</strong>`).join("")}
      ${labels
        .map(
          (trueLabel) => `
            <strong class="matrix-row-label">${escapeHtml(trueLabel)}</strong>
            ${labels
              .map((predictedLabel) => {
                const cell = byKey.get(`${trueLabel}::${predictedLabel}`);
                const count = Number(cell?.count) || 0;
                const intensity = Math.min(1, Math.max(0, count / max));
                return `
                  <span
                    class="matrix-cell"
                    style="--cell-alpha:${intensity}"
                    title="${escapeHtml(trueLabel)} -> ${escapeHtml(predictedLabel)}: ${formatMetric(count)}"
                  >
                    ${formatMetric(count)}
                  </span>
                `;
              })
              .join("")}
          `,
        )
              .join("")}
    </div>
  `;
}

function rowToText(value) {
  return value === undefined || value === null ? "" : String(value);
}
