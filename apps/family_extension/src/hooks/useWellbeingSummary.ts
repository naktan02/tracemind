import { useEffect, useState } from "react";

import { fetchWellbeingSummary } from "../api/wellbeing";
import type { WellbeingSignalSummaryPayload } from "../contracts/generated";

export type WellbeingSummaryLoadState =
  | { status: "loading"; summary: null; errorMessage: null }
  | { status: "loaded"; summary: WellbeingSignalSummaryPayload; errorMessage: null }
  | { status: "error"; summary: null; errorMessage: string };

export function useWellbeingSummary() {
  const [loadState, setLoadState] = useState<WellbeingSummaryLoadState>({
    status: "loading",
    summary: null,
    errorMessage: null,
  });

  useEffect(() => {
    let cancelled = false;

    async function loadSummary() {
      setLoadState({
        status: "loading",
        summary: null,
        errorMessage: null,
      });
      try {
        const summary = await fetchWellbeingSummary();
        if (!cancelled) {
          setLoadState({
            status: "loaded",
            summary,
            errorMessage: null,
          });
        }
      } catch (error) {
        const message =
          error instanceof Error
            ? error.message
            : "현재 상태를 불러오지 못했습니다.";
        if (!cancelled) {
          setLoadState({
            status: "error",
            summary: null,
            errorMessage: message,
          });
        }
      }
    }

    void loadSummary();
    return () => {
      cancelled = true;
    };
  }, []);

  return loadState;
}
