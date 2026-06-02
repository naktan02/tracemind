import { useEffect, useState } from "react";

import { fetchWellbeingTimeseries } from "../api/wellbeing";
import type {
  WellbeingSignalRange,
  WellbeingSignalTimeseriesPayload,
} from "../../contracts/generated";

export type WellbeingTimeseriesLoadState =
  | { status: "loading"; timeseries: null; errorMessage: null }
  | {
      status: "loaded";
      timeseries: WellbeingSignalTimeseriesPayload;
      errorMessage: null;
    }
  | { status: "error"; timeseries: null; errorMessage: string };

type UseWellbeingTimeseriesOptions = {
  enabled?: boolean;
  requestedRange: WellbeingSignalRange;
};

export function useWellbeingTimeseries({
  enabled = true,
  requestedRange,
}: UseWellbeingTimeseriesOptions) {
  const [loadState, setLoadState] = useState<WellbeingTimeseriesLoadState>({
    status: "loading",
    timeseries: null,
    errorMessage: null,
  });

  useEffect(() => {
    if (!enabled) {
      setLoadState({
        status: "loading",
        timeseries: null,
        errorMessage: null,
      });
      return;
    }

    let cancelled = false;

    async function loadTimeseries() {
      setLoadState({
        status: "loading",
        timeseries: null,
        errorMessage: null,
      });
      try {
        const timeseries = await fetchWellbeingTimeseries(requestedRange);
        if (!cancelled) {
          setLoadState({
            status: "loaded",
            timeseries,
            errorMessage: null,
          });
        }
      } catch (error) {
        const message =
          error instanceof Error
            ? error.message
            : "최근 추이를 불러오지 못했습니다.";
        if (!cancelled) {
          setLoadState({
            status: "error",
            timeseries: null,
            errorMessage: message,
          });
        }
      }
    }

    void loadTimeseries();
    return () => {
      cancelled = true;
    };
  }, [enabled, requestedRange]);

  return loadState;
}
