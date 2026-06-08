export function applyFacetedFilters(rows, axisDefs, selectedAxisIds, selectedValues) {
  return applyFacetedFiltersExcept(
    rows,
    axisDefs,
    selectedAxisIds,
    selectedValues,
    null,
  );
}

export function visibleFacetedAxes(rows, axisDefs, selectedAxisIds, selectedValues) {
  return axisDefs.filter(
    (axis) =>
      optionsForAxis(rows, axis, axisDefs, selectedAxisIds, selectedValues).length > 1,
  );
}

export function optionsForAxis(rows, axis, axisDefs, selectedAxisIds, selectedValues) {
  const filteredRows = applyFacetedFiltersExcept(
    rows,
    axisDefs,
    selectedAxisIds,
    selectedValues,
    axis.id,
  );
  const counts = new Map();
  for (const row of filteredRows) {
    const value = String(axis.value(row));
    const label = axis.labelForValue ? axis.labelForValue(row) : value;
    if (!counts.has(value)) {
      counts.set(value, { value, label, count: 0 });
    }
    counts.get(value).count += 1;
  }
  return Array.from(counts.values()).sort((a, b) =>
    a.label.localeCompare(b.label, undefined, { numeric: true }),
  );
}

function applyFacetedFiltersExcept(
  rows,
  axisDefs,
  selectedAxisIds,
  selectedValues,
  excludedAxisId,
) {
  const axisById = new Map(axisDefs.map((axis) => [axis.id, axis]));
  return rows.filter((row) =>
    selectedAxisIds.every((axisId) => {
      if (axisId === excludedAxisId) return true;
      const axis = axisById.get(axisId);
      if (!axis) return true;
      const values = selectedValues[axisId] ?? [];
      if (values.length === 0) return true;
      return values.includes(String(axis.value(row)));
    }),
  );
}
