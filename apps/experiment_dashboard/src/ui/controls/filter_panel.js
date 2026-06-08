import { escapeHtml } from "../../shared/formatting/html.js";
import {
  optionsForAxis,
  visibleFacetedAxes,
} from "../../shared/filters/faceted_filters.js";

export function renderFilterPanel({
  axisPicker,
  activeFilters,
  summary,
  rows,
  filteredRows,
  axes,
  selectedAxisIds,
  selectedValues,
  dataPrefix,
}) {
  const visibleAxes = visibleFacetedAxes(rows, axes, selectedAxisIds, selectedValues);
  const selectedAxes = new Set(selectedAxisIds);
  axisPicker.innerHTML = visibleAxes
    .map((axis) => {
      const valueCount = optionsForAxis(
        rows,
        axis,
        axes,
        selectedAxisIds,
        selectedValues,
      ).length;
      return `
        <label class="check-row compact">
          <input
            type="checkbox"
            data-${dataPrefix}-filter-axis="${escapeHtml(axis.id)}"
            ${selectedAxes.has(axis.id) ? "checked" : ""}
          />
          <span>${escapeHtml(axis.label)} (${valueCount})</span>
        </label>
      `;
    })
    .join("");

  if (visibleAxes.length === 0) {
    activeFilters.innerHTML = `<p class="empty">현재 조건에서 더 나눌 필터 축이 없습니다.</p>`;
  } else if (selectedAxisIds.length === 0) {
    activeFilters.innerHTML = `<p class="empty">왼쪽에서 사용할 필터 축을 선택하세요.</p>`;
  } else {
    activeFilters.innerHTML = selectedAxisIds
      .map((axisId) => {
        const axis = axes.find((candidate) => candidate.id === axisId);
        if (!axis) return "";
        const selected = new Set(selectedValues[axisId] ?? []);
        const options = optionsForAxis(rows, axis, axes, selectedAxisIds, selectedValues);
        return `
          <section class="filter-group">
            <p class="filter-group-title">${escapeHtml(axis.label)}</p>
            <div class="filter-value-list">
              ${options
                .map(
                  (option) => `
                    <label class="check-row compact">
                      <input
                        type="checkbox"
                        data-${dataPrefix}-filter-value-axis="${escapeHtml(axis.id)}"
                        data-${dataPrefix}-filter-value="${escapeHtml(option.value)}"
                        ${selected.has(option.value) ? "checked" : ""}
                      />
                      <span>${escapeHtml(option.label)} (${option.count})</span>
                    </label>
                  `,
                )
                .join("")}
            </div>
          </section>
        `;
      })
      .join("");
  }

  const selectedValueCount = Object.values(selectedValues).reduce(
    (total, values) => total + values.length,
    0,
  );
  summary.textContent = [
    `filtered runs=${filteredRows.length}/${rows.length}`,
    `axes=${visibleAxes.length}`,
    `selected values=${selectedValueCount}`,
  ].join(" · ");
}
