import { loadStoredRunAliases, loadStoredSeriesColors } from "../../state/preferences.js";

export function createFlSslState() {
  return {
    filterPanelOpen: false,
    filterAxisIds: [],
    filterValues: {},
    runMetricIds: [],
    runIds: [],
    runAliases: loadStoredRunAliases("fl_runs"),
    roundRunIds: [],
    roundRunAliases: loadStoredRunAliases("fl_round"),
    roundRunColors: loadStoredSeriesColors("fl_round"),
    roundAxisLabel: "",
    roundIncludeInitial: true,
    roundMetric: "macro_f1",
    clientValidationRunId: null,
    clientRoundRunId: null,
    clientRoundIndex: "__latest__",
    splitRunId: null,
    projectionEvalSet: "validation",
    projectionRunIds: [],
  };
}
