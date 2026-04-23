import { useEffect, useState } from "react";

import {
  deleteSavedExperimentWorkspace,
  getSavedExperimentWorkspace,
  listSavedExperimentWorkspaces,
  saveExperimentWorkspace,
} from "../api";
import { asErrorMessage } from "../lib/formatters";
import type {
  SavedWorkspaceDetailPayload,
  SavedWorkspaceSummaryPayload,
  WorkspaceManifestPayload,
} from "../types";

export interface SavedWorkspacesState {
  savedWorkspaces: SavedWorkspaceSummaryPayload[];
  savedWorkspacesError: string | null;
  isSavedWorkspacesLoading: boolean;
  loadingWorkspaceId: string | null;
  deletingWorkspaceId: string | null;
  refreshSavedWorkspaces: () => Promise<void>;
  saveWorkspace: (
    manifest: WorkspaceManifestPayload,
  ) => Promise<SavedWorkspaceDetailPayload>;
  loadWorkspace: (workspaceId: string) => Promise<SavedWorkspaceDetailPayload>;
  deleteWorkspace: (workspaceId: string) => Promise<SavedWorkspaceSummaryPayload>;
}

export function useSavedWorkspaces(
  apiBaseUrl: string,
): SavedWorkspacesState {
  const [savedWorkspaces, setSavedWorkspaces] = useState<
    SavedWorkspaceSummaryPayload[]
  >([]);
  const [savedWorkspacesError, setSavedWorkspacesError] = useState<string | null>(
    null,
  );
  const [isSavedWorkspacesLoading, setIsSavedWorkspacesLoading] = useState(false);
  const [loadingWorkspaceId, setLoadingWorkspaceId] = useState<string | null>(null);
  const [deletingWorkspaceId, setDeletingWorkspaceId] = useState<string | null>(
    null,
  );

  async function refreshSavedWorkspaces() {
    setIsSavedWorkspacesLoading(true);
    try {
      const payload = await listSavedExperimentWorkspaces(apiBaseUrl);
      setSavedWorkspaces(payload);
      setSavedWorkspacesError(null);
    } catch (error) {
      setSavedWorkspacesError(asErrorMessage(error));
    } finally {
      setIsSavedWorkspacesLoading(false);
    }
  }

  useEffect(() => {
    void refreshSavedWorkspaces();
  }, [apiBaseUrl]);

  async function saveWorkspace(manifest: WorkspaceManifestPayload) {
    return saveExperimentWorkspace(apiBaseUrl, manifest);
  }

  async function loadWorkspace(workspaceId: string) {
    setLoadingWorkspaceId(workspaceId);
    try {
      return await getSavedExperimentWorkspace(apiBaseUrl, workspaceId);
    } finally {
      setLoadingWorkspaceId(null);
    }
  }

  async function deleteWorkspace(workspaceId: string) {
    setDeletingWorkspaceId(workspaceId);
    try {
      return await deleteSavedExperimentWorkspace(apiBaseUrl, workspaceId);
    } finally {
      setDeletingWorkspaceId(null);
    }
  }

  return {
    savedWorkspaces,
    savedWorkspacesError,
    isSavedWorkspacesLoading,
    loadingWorkspaceId,
    deletingWorkspaceId,
    refreshSavedWorkspaces,
    saveWorkspace,
    loadWorkspace,
    deleteWorkspace,
  };
}
