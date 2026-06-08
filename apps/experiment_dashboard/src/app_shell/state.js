import { createCentralSslState } from "../central_ssl/state.js";
import { createFlSslState } from "../fl_ssl/state.js";

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
