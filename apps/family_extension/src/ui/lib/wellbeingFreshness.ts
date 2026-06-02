const STALE_SIGNAL_THRESHOLD_MINUTES = 360;

export type WellbeingFreshnessState = "fresh" | "stale" | "unknown";

export function getWellbeingFreshnessState(
  computedAt: string,
): WellbeingFreshnessState {
  const date = new Date(computedAt);
  if (Number.isNaN(date.getTime())) {
    return "unknown";
  }

  const ageMinutes = (Date.now() - date.getTime()) / 60_000;
  return ageMinutes >= STALE_SIGNAL_THRESHOLD_MINUTES ? "stale" : "fresh";
}
