import { escapeHtml } from "../../shared/formatting/html.js";

export function emptyTableRow(columnCount, message) {
  return `<tr><td class="empty-cell" colspan="${columnCount}">${message}</td></tr>`;
}

export function resolveTableColumns(state, columns, defaultVisible = [], requiredVisible = []) {
  const allColumns = Array.isArray(columns) ? columns.slice() : [];
  const columnById = new Map(allColumns.map((column) => [column.id, column]));
  const availableIds = allColumns.map((column) => column.id);
  const availableSet = new Set(availableIds);
  const normalizedDefaultVisible = defaultVisible.filter((id) => availableSet.has(id));
  const normalizedRequiredVisible = requiredVisible.filter((id) => availableSet.has(id));

  const rawOrder = Array.isArray(state?.order) ? state.order : [];
  const rawVisible = Array.isArray(state?.visible) ? state.visible : [];

  const stateObj = state ?? ({});
  const normalizedVisible = rawVisible.filter((id) => availableSet.has(id));

  let nextVisible = normalizedVisible.length > 0 ? normalizedVisible : normalizedDefaultVisible;
  for (const requiredId of normalizedRequiredVisible) {
    if (!nextVisible.includes(requiredId)) {
      nextVisible = [...nextVisible, requiredId];
    }
  }

  const visibleSet = new Set(nextVisible);
  const order = rawOrder.filter((id) => availableSet.has(id) && visibleSet.has(id));
  for (const id of availableIds) {
    if (!order.includes(id) && visibleSet.has(id)) order.push(id);
  }

  stateObj.order = order.length > 0 ? order : nextVisible.slice();
  stateObj.visible = nextVisible.slice();

  return {
    orderedColumns: stateObj.order
      .map((id) => columnById.get(id))
      .filter(Boolean),
    visibleColumns: stateObj.order
      .map((id) => columnById.get(id))
      .filter(Boolean),
    allColumns: allColumns,
    state: stateObj,
  };
}

export function renderTableHeader(headElement, columns) {
  if (!headElement) return;
  headElement.innerHTML = `<tr>${columns
    .map(
      (column) => `
      <th data-table-column-id="${escapeHtml(column.id)}" draggable="true">
        ${escapeHtml(column.label)}
      </th>
    `,
    )
    .join("")}
  </tr>`;
}

export function renderSortableTableHeader(headElement, columns, onMove) {
  if (!headElement) return;
  renderTableHeader(headElement, columns);

  const ths = Array.from(headElement.querySelectorAll("th[data-table-column-id]"));
  let draggingId = null;

  ths.forEach((th) => {
    th.classList.add("draggable-column");
    th.addEventListener("dragstart", (event) => {
      const columnId = th.dataset.tableColumnId;
      if (!columnId || !event.dataTransfer) return;
      draggingId = columnId;
      event.dataTransfer.setData("text/plain", columnId);
      event.dataTransfer.effectAllowed = "move";
      th.classList.add("dragging");
    });

    th.addEventListener("dragend", () => {
      th.classList.remove("dragging");
      draggingId = null;
    });

    th.addEventListener("dragover", (event) => {
      event.preventDefault();
      event.dataTransfer.dropEffect = "move";
    });

    th.addEventListener("drop", (event) => {
      event.preventDefault();
      const sourceId = event.dataTransfer?.getData("text/plain") || draggingId;
      const targetId = th.dataset.tableColumnId;
      if (!sourceId || !targetId || sourceId === targetId || typeof onMove !== "function") return;
      onMove(sourceId, targetId);
    });
  });
}

export function moveTableColumn(state, sourceColumnId, targetColumnId) {
  if (!state || !Array.isArray(state.order)) return false;
  const order = state.order.slice();
  const from = order.indexOf(sourceColumnId);
  const to = order.indexOf(targetColumnId);
  if (from < 0 || to < 0 || from === to) return false;

  const next = order.slice();
  const [moved] = next.splice(from, 1);
  next.splice(to, 0, moved);
  state.order = next;
  return true;
}

export function setTableColumnVisibility(state, columns, visibleIds, defaultVisible = []) {
  if (!state) return;
  const availableSet = new Set((Array.isArray(columns) ? columns : []).map((column) => column.id));
  const nextVisible = (Array.isArray(visibleIds) ? visibleIds : [])
    .filter((id) => availableSet.has(id));
  const requiredVisible = (Array.isArray(defaultVisible) ? defaultVisible : [])
    .filter((id) => availableSet.has(id));

  const forcedVisible = nextVisible.filter((id) => !requiredVisible.includes(id));
  const mergedVisible = [...requiredVisible, ...forcedVisible];
  const fallback = defaultVisible.filter((id) => availableSet.has(id));

  const appliedVisible = mergedVisible.length > 0 ? mergedVisible : fallback;
  const visibleSet = new Set(appliedVisible);
  const order = (Array.isArray(state.order) ? state.order : [])
    .filter((id) => availableSet.has(id) && visibleSet.has(id));

  for (const id of appliedVisible) {
    if (!order.includes(id)) {
      order.push(id);
    }
  }

  state.visible = appliedVisible;
  state.order = order.length > 0 ? order : appliedVisible.slice();
}

export function renderColumnCheckboxes(container, columns, selectedIds, dataKey) {
  if (!container) return;
  const selectedSet = new Set(selectedIds);
  const dataAttribute = dataKey.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`);
  container.innerHTML =
    columns.length === 0
      ? `<p class="empty">선택 가능한 항목이 없습니다.</p>`
      : columns
          .map(
            (column) => `
              <label class="check-row compact">
                <input
                  type="checkbox"
                  data-${dataAttribute}="${escapeHtml(column.id)}"
                  data-table-column-id="${escapeHtml(column.id)}"
                  ${selectedSet.has(column.id) ? "checked" : ""}
                />
                <span>${escapeHtml(column.label)}</span>
              </label>
            `,
          )
          .join("");
}
