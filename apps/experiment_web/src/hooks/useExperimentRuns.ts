import { useEffect, useState } from "react";

import { launchExperimentRun, listExperimentRuns } from "../api";
import { asErrorMessage } from "../lib/formatters";
import type {
  ExperimentRunPayload,
  LaunchExperimentRunRequestPayload,
} from "../types";

export interface ExperimentRunsState {
  runs: ExperimentRunPayload[];
  runsError: string | null;
  isRunsLoading: boolean;
  refreshRuns: (options?: { silent?: boolean }) => Promise<void>;
  launchRun: (
    request: LaunchExperimentRunRequestPayload,
  ) => Promise<ExperimentRunPayload>;
}

export function useExperimentRuns(apiBaseUrl: string): ExperimentRunsState {
  const [runs, setRuns] = useState<ExperimentRunPayload[]>([]);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [isRunsLoading, setIsRunsLoading] = useState(false);

  async function refreshRuns(options?: { silent?: boolean }) {
    if (!options?.silent) {
      setIsRunsLoading(true);
    }
    try {
      const payload = await listExperimentRuns(apiBaseUrl);
      setRuns(payload);
      setRunsError(null);
    } catch (error) {
      setRunsError(asErrorMessage(error));
    } finally {
      if (!options?.silent) {
        setIsRunsLoading(false);
      }
    }
  }

  useEffect(() => {
    void refreshRuns();
  }, [apiBaseUrl]);

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      void refreshRuns({ silent: true });
    }, 4000);
    return () => {
      window.clearInterval(intervalId);
    };
  }, [apiBaseUrl]);

  async function launchRun(request: LaunchExperimentRunRequestPayload) {
    return launchExperimentRun(apiBaseUrl, request);
  }

  return {
    runs,
    runsError,
    isRunsLoading,
    refreshRuns,
    launchRun,
  };
}
