import { requestAgentJson } from "./client";

export async function fetchWellbeingSummary(): Promise<unknown> {
  return requestAgentJson("/api/v1/wellbeing/summary");
}

export async function fetchWellbeingTimeseries(
  range: "7d" | "14d" | "30d",
): Promise<unknown> {
  return requestAgentJson(`/api/v1/wellbeing/timeseries?range=${range}`);
}

export async function unlockParentView(pin: string): Promise<unknown> {
  return requestAgentJson("/api/v1/parent/unlock", {
    method: "POST",
    body: JSON.stringify({ pin }),
  });
}

export async function checkLocalProgramHealth(): Promise<boolean> {
  try {
    await requestAgentJson("/api/v1/system/health");
    return true;
  } catch {
    return false;
  }
}
