import { normalizeDashboardBundle } from "./normalize_bundle.js";

export async function loadDashboardBundle(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return normalizeDashboardBundle(await response.json());
}
