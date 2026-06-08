import { loadStoredRunAliases, loadStoredSeriesColors } from "../../state/preferences.js";
import { DEFAULT_FL_RUN_METRICS } from "./logic/constants.js";

export function createFlSslState() {
  return {
    filterPanelOpen: false,
    filterAxisIds: [],
    filterValues: {},
    runMetricIds: [...DEFAULT_FL_RUN_METRICS],
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
