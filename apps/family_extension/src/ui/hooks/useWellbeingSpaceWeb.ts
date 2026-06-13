import { useEffect, useState } from "react";

import { fetchWellbeingSpaceWeb } from "../api/wellbeing";
import type {
  WellbeingSignalRange,
  WellbeingSpaceWebPayload,
} from "../../contracts/generated";

export type WellbeingSpaceWebLoadState =
  | { status: "loading"; spaceWeb: null; errorMessage: null }
  | {
      status: "loaded";
      spaceWeb: WellbeingSpaceWebPayload;
      errorMessage: null;
    }
  | { status: "error"; spaceWeb: null; errorMessage: string };

type UseWellbeingSpaceWebOptions = {
  enabled?: boolean;
  requestedRange: WellbeingSignalRange;
};

export function useWellbeingSpaceWeb({
  enabled = true,
  requestedRange,
}: UseWellbeingSpaceWebOptions) {
  const [loadState, setLoadState] = useState<WellbeingSpaceWebLoadState>({
    status: "loading",
    spaceWeb: null,
    errorMessage: null,
  });

  useEffect(() => {
    if (!enabled) {
      setLoadState({
        status: "loading",
        spaceWeb: null,
        errorMessage: null,
      });
      return;
    }

    let cancelled = false;

    async function loadSpaceWeb() {
      setLoadState({
        status: "loading",
        spaceWeb: null,
        errorMessage: null,
      });
      try {
        const spaceWeb = await fetchWellbeingSpaceWeb(requestedRange);
        if (!cancelled) {
          setLoadState({
            status: "loaded",
            spaceWeb,
            errorMessage: null,
          });
        }
      } catch (error) {
        const message =
          error instanceof Error
            ? error.message
            : "공간웹 데이터를 불러오지 못했습니다.";
        if (!cancelled) {
          setLoadState({
            status: "error",
            spaceWeb: null,
            errorMessage: message,
          });
        }
      }
    }

    void loadSpaceWeb();
    return () => {
      cancelled = true;
    };
  }, [enabled, requestedRange]);

  return loadState;
}
