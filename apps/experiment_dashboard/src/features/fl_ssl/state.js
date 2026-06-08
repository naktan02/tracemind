import { loadStoredRunAliases, loadStoredSeriesColors } from "../../state/preferences.js";

export function createFlSslState() {
  return {
    filterPanelOpen: false,
    filterAxisIds: [],
    filterValues: {},
    runColumnTab: "metric",
    runMetricIds: [],
    runIds: [],
    runAliases: loadStoredRunAliases("fl_runs"),
    runTableColumns: { order: [], visible: [] },
    roundRunIds: [],
    roundRunAliases: loadStoredRunAliases("fl_round"),
    roundRunColors: loadStoredSeriesColors("fl_round"),
    roundMetricIds: ["macro_f1"],
    roundAxisLabel: "",
    roundIncludeInitial: true,
    roundMetric: "macro_f1",
    roundTableColumns: { order: [], visible: [] },
    clientValidationRunId: null,
    clientValidationTableColumns: { order: [], visible: [] },
    clientRoundRunId: null,
    clientRoundIndex: "__latest__",
    clientRoundTableColumns: { order: [], visible: [] },
    splitRunId: null,
    splitTableColumns: { order: [], visible: [] },
    projectionEvalSet: "validation",
    projectionRunIds: [],
  };
}
