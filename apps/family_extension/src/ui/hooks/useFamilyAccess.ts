import { useEffect, useMemo, useState } from "react";

import {
  fetchFamilySetupStatus,
  submitInitialFamilySetup,
  unlockFamilyRole,
} from "../api/familyAccess";
import type {
  FamilyAccessRole,
  FamilySetupResponsePayload,
  FamilySetupStatusPayload,
  FamilyUnlockResponsePayload,
} from "../../contracts/generated";

export type FamilySetupStatusPhase = "loading" | "loaded" | "error";
export type FamilySetupSubmissionPhase =
  | "idle"
  | "submitting"
  | "completed"
  | "error";
export type FamilyUnlockPhase =
  | "idle"
  | "submitting"
  | "granted"
  | "rejected"
  | "locked"
  | "error";

export type FamilySetupStatusState = {
  phase: FamilySetupStatusPhase;
  status: FamilySetupStatusPayload | null;
  errorMessage: string | null;
};

export type FamilySetupSubmissionState = {
  phase: FamilySetupSubmissionPhase;
  response: FamilySetupResponsePayload | null;
  errorMessage: string | null;
};

export type FamilyUnlockState = {
  phase: FamilyUnlockPhase;
  response: FamilyUnlockResponsePayload | null;
  errorMessage: string | null;
};

export type ActiveFamilySession = {
  role: FamilyAccessRole;
  sessionToken: string;
  sessionExpiresAt: string | null;
};

const INITIAL_UNLOCK_STATE: FamilyUnlockState = {
  phase: "idle",
  response: null,
  errorMessage: null,
};

export function useFamilyAccess() {
  const [setupStatusState, setSetupStatusState] = useState<FamilySetupStatusState>({
    phase: "loading",
    status: null,
    errorMessage: null,
  });
  const [setupSubmissionState, setSetupSubmissionState] =
    useState<FamilySetupSubmissionState>({
      phase: "idle",
      response: null,
      errorMessage: null,
    });
  const [unlockStates, setUnlockStates] = useState<
    Record<FamilyAccessRole, FamilyUnlockState>
  >({
    child: INITIAL_UNLOCK_STATE,
    parent: INITIAL_UNLOCK_STATE,
  });
  const [activeSession, setActiveSession] = useState<ActiveFamilySession | null>(null);
  const [currentTimeMs, setCurrentTimeMs] = useState(() => Date.now());

  useEffect(() => {
    void reloadSetupStatus();
  }, []);

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      setCurrentTimeMs(Date.now());
    }, 30_000);
    return () => window.clearInterval(intervalId);
  }, []);

  useEffect(() => {
    if (activeSession?.sessionExpiresAt == null) {
      return;
    }
    if (new Date(activeSession.sessionExpiresAt).getTime() <= currentTimeMs) {
      setActiveSession(null);
    }
  }, [activeSession, currentTimeMs]);

  const activeRole = useMemo<FamilyAccessRole | null>(() => {
    if (activeSession == null) {
      return null;
    }
    return activeSession.role;
  }, [activeSession]);

  async function reloadSetupStatus() {
    setSetupStatusState({
      phase: "loading",
      status: null,
      errorMessage: null,
    });
    try {
      const status = await fetchFamilySetupStatus();
      setSetupStatusState({
        phase: "loaded",
        status,
        errorMessage: null,
      });
      return status;
    } catch (error) {
      const errorMessage =
        error instanceof Error
          ? error.message
          : "초기 설정 상태를 불러오지 못했습니다.";
      setSetupStatusState({
        phase: "error",
        status: null,
        errorMessage,
      });
      return null;
    }
  }

  async function submitSetup(childPin: string, parentPin: string) {
    setSetupSubmissionState({
      phase: "submitting",
      response: null,
      errorMessage: null,
    });
    try {
      const response = await submitInitialFamilySetup({
        child_pin: childPin,
        parent_pin: parentPin,
      });
      setSetupSubmissionState({
        phase: "completed",
        response,
        errorMessage: null,
      });
      setSetupStatusState({
        phase: "loaded",
        status: {
          schema_version: "family_setup_status.v1",
          access_mode: response.access_mode,
          is_setup_complete: response.is_setup_complete,
          configured_roles: response.configured_roles,
        },
        errorMessage: null,
      });
      return response;
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "초기 설정 저장에 실패했습니다.";
      setSetupSubmissionState({
        phase: "error",
        response: null,
        errorMessage,
      });
      return null;
    }
  }

  async function submitRoleUnlock(role: FamilyAccessRole, pin: string) {
    setUnlockStates((prev) => ({
      ...prev,
      [role]: {
        phase: "submitting",
        response: null,
        errorMessage: null,
      },
    }));
    try {
      const response = await unlockFamilyRole(role, pin);
      if (response.granted && response.session_token != null) {
        setActiveSession({
          role: response.role,
          sessionToken: response.session_token,
          sessionExpiresAt: response.session_expires_at,
        });
        setUnlockStates((prev) => ({
          ...prev,
          [role]: {
            phase: "granted",
            response,
            errorMessage: null,
          },
        }));
        return response;
      }
      setUnlockStates((prev) => ({
        ...prev,
        [role]: {
          phase: response.locked_until == null ? "rejected" : "locked",
          response,
          errorMessage: null,
        },
      }));
      return response;
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "PIN 검증 요청에 실패했습니다.";
      setUnlockStates((prev) => ({
        ...prev,
        [role]: {
          phase: "error",
          response: null,
          errorMessage,
        },
      }));
      return null;
    }
  }

  function clearRoleSession() {
    setActiveSession(null);
  }

  function getUnlockState(role: FamilyAccessRole): FamilyUnlockState {
    return unlockStates[role];
  }

  function resetUnlockState(role: FamilyAccessRole) {
    setUnlockStates((prev) => ({
      ...prev,
      [role]: INITIAL_UNLOCK_STATE,
    }));
  }

  return {
    activeRole,
    activeSession,
    clearRoleSession,
    getUnlockState,
    reloadSetupStatus,
    resetUnlockState,
    setupStatusState,
    setupSubmissionState,
    submitRoleUnlock,
    submitSetup,
  };
}
