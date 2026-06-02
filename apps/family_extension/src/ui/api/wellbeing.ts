import type {
  WellbeingSignalRange,
  WellbeingSignalSummaryPayload,
  WellbeingSignalTimeseriesPayload,
} from "../../contracts/generated";
import { requestAgentJson } from "../../common/agentClient";

export async function fetchWellbeingSummary(): Promise<WellbeingSignalSummaryPayload> {
  return requestAgentJson("/api/v1/wellbeing/summary");
}

export async function fetchWellbeingTimeseries(
  range: WellbeingSignalRange,
): Promise<WellbeingSignalTimeseriesPayload> {
  return requestAgentJson(`/api/v1/wellbeing/timeseries?range=${range}`);
}

export async function checkLocalProgramHealth(): Promise<boolean> {
  try {
    await requestAgentJson("/api/v1/system/health");
    return true;
  } catch {
    return false;
  }
}
