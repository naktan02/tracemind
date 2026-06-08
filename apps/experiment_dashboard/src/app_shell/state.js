import { createCentralSslState } from "../features/central_ssl/state.js";
import { createFlSslState } from "../features/fl_ssl/state.js";

export function createDashboardState() {
  return {
    bundle: null,
    activeTrack: "central_ssl",
    activeCentralTab: "overview",
    activeFlTab: "runs",
    central: createCentralSslState(),
    fl: createFlSslState(),
  };
}
