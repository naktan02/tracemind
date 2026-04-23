import { useEffect, useMemo, useState } from "react";

import { unlockParentView } from "../api/wellbeing";
import type { ParentUnlockResponsePayload } from "../contracts/generated";

export type ParentUnlockPhase =
  | "idle"
  | "submitting"
  | "granted"
  | "rejected"
  | "locked"
  | "error";

export type ParentUnlockState = {
  phase: ParentUnlockPhase;
  response: ParentUnlockResponsePayload | null;
  errorMessage: string | null;
};

export function useParentUnlock() {
  const [unlockState, setUnlockState] = useState<ParentUnlockState>({
    phase: "idle",
    response: null,
    errorMessage: null,
  });
  const [sessionResponse, setSessionResponse] =
    useState<ParentUnlockResponsePayload | null>(null);
  const [currentTimeMs, setCurrentTimeMs] = useState(() => Date.now());

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      setCurrentTimeMs(Date.now());
    }, 30_000);
    return () => window.clearInterval(intervalId);
  }, []);

  const hasActiveParentSession = useMemo(
    () =>
      sessionResponse?.granted === true &&
      sessionResponse.session_expires_at != null &&
      new Date(sessionResponse.session_expires_at).getTime() > currentTimeMs,
    [currentTimeMs, sessionResponse],
  );

  const activeSessionExpiresAt = useMemo(() => {
    if (!hasActiveParentSession || sessionResponse?.session_expires_at == null) {
      return null;
    }
    return sessionResponse.session_expires_at;
  }, [hasActiveParentSession, sessionResponse]);

  async function submitUnlock(pin: string) {
    setUnlockState({
      phase: "submitting",
      response: null,
      errorMessage: null,
    });

    try {
      const response = await unlockParentView(pin);
      if (response.granted) {
        setSessionResponse(response);
        setUnlockState({
          phase: "granted",
          response,
          errorMessage: null,
        });
        return response;
      }

      setUnlockState({
        phase: response.locked_until == null ? "rejected" : "locked",
        response,
        errorMessage: null,
      });
      return response;
    } catch (error) {
      const errorMessage =
        error instanceof Error
          ? error.message
          : "부모용 PIN 검증 요청에 실패했습니다.";
      setUnlockState({
        phase: "error",
        response: null,
        errorMessage,
      });
      return null;
    }
  }

  return {
    activeSessionExpiresAt,
    hasActiveParentSession,
    submitUnlock,
    unlockState,
  };
}
