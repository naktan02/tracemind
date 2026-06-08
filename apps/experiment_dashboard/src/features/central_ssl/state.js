import { loadStoredRunAliases, loadStoredSeriesColors } from "../../state/preferences.js";

export function createCentralSslState() {
  return {
    filterPanelOpen: true,
    filterAxisIds: [],
    filterValues: {},
    overviewEvalSet: "validation",
    overviewMetricIds: [],
    overviewRunIds: [],
    overviewRunAliases: loadStoredRunAliases("central_overview"),
    compareEvalSet: "validation",
    compareMetric: "selection_macro_f1",
    compareChartType: "line",
    compareIncludeInitial: true,
    compareRunIds: [],
    compareRunAliases: loadStoredRunAliases("central_compare"),
    compareRunColors: loadStoredSeriesColors("central_compare"),
    compareAxisLabel: "",
    classEvalSet: "validation",
    classMetric: "f1",
    overviewColumnTab: "metric",
    overviewTableColumns: { order: [], visible: [] },
    detailAlgorithm: null,
    detailRunId: null,
    projectionEvalSet: "validation",
    projectionAlgorithm: null,
    projectionRunIds: [],
    classTableColumns: { order: [], visible: [] },
  };
}
