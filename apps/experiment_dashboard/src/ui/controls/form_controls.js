import { escapeHtml } from "../../shared/formatting/html.js";

export function fillSelect(
  select,
  values,
  selectedValue,
  emptyLabel = "선택",
  labelForValue = null,
) {
  const selectedValues = Array.isArray(selectedValue)
    ? new Set(selectedValue)
    : new Set([selectedValue]);
  const labelResolver = typeof labelForValue === "function" ? labelForValue : null;
  const options = values
    .map((value) => {
      const selected = selectedValues.has(value) ? "selected" : "";
      const label = labelResolver
        ? labelResolver(value)
        : value === "__all__"
          ? "All"
          : value;
      return `<option value="${escapeHtml(value)}" ${selected}>${escapeHtml(label)}</option>`;
    })
    .join("");
  select.innerHTML = values.length > 0 ? options : `<option value="">${emptyLabel}</option>`;
}

export function renderCheckboxList(container, values, selectedValues, dataKey, labelForValue) {
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

export function checkedValues(container, dataKey) {
  return Array.from(container.querySelectorAll("input[type='checkbox']:checked"))
    .map((input) => input.dataset[dataKey])
    .filter((value) => value);
}
