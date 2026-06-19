import { loadStoredRunAliases, loadStoredSeriesColors } from "../../state/preferences.js";

export const DEFAULT_CENTRAL_FILTER_AXIS_IDS = [
  "initial_checkpoint",
  "created_date",
];

export function createCentralSslState() {
  return {
    filterPanelOpen: true,
    filterAxisIds: [...DEFAULT_CENTRAL_FILTER_AXIS_IDS],
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
    detailRunId: null,
    projectionEvalSet: "validation",
    projectionAlgorithm: null,
    projectionRunIds: [],
    classTableColumns: { order: [], visible: [] },
  };
}
