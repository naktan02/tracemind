import type {
  ExperimentRunPayload,
  ExperimentCatalogPayload,
  LaunchExperimentRunRequestPayload,
  ResolvedExperimentPlanPayload,
  SavedWorkspaceDetailPayload,
  SavedWorkspaceSummaryPayload,
  WorkspaceManifestPayload,
} from "./types";

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

export function resolveApiBaseUrl(): string {
  const configured = import.meta.env.VITE_API_BASE_URL?.trim();
  if (!configured) {
    return DEFAULT_API_BASE_URL;
  }
  return configured.endsWith("/") ? configured.slice(0, -1) : configured;
}

async function requestJson<T>(
  baseUrl: string,
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${baseUrl}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) {
        detail = payload.detail;
      }
    } catch {
      // Ignore JSON decode failure and use status text fallback.
    }
    throw new Error(detail);
  }

  return (await response.json()) as T;
}

export async function loadExperimentCatalog(
  baseUrl: string,
): Promise<ExperimentCatalogPayload> {
  return requestJson<ExperimentCatalogPayload>(
    baseUrl,
    "/api/v1/experiments/catalog",
    {
      method: "GET",
    },
  );
}

export async function compileExperimentWorkspace(
  baseUrl: string,
  manifest: WorkspaceManifestPayload,
): Promise<ResolvedExperimentPlanPayload> {
  return requestJson<ResolvedExperimentPlanPayload>(
    baseUrl,
    "/api/v1/experiments/compile",
    {
      method: "POST",
      body: JSON.stringify(manifest),
    },
  );
}

export async function listSavedExperimentWorkspaces(
  baseUrl: string,
): Promise<SavedWorkspaceSummaryPayload[]> {
  return requestJson<SavedWorkspaceSummaryPayload[]>(
    baseUrl,
    "/api/v1/experiments/workspaces",
    {
      method: "GET",
    },
  );
}

export async function saveExperimentWorkspace(
  baseUrl: string,
  manifest: WorkspaceManifestPayload,
): Promise<SavedWorkspaceDetailPayload> {
  return requestJson<SavedWorkspaceDetailPayload>(
    baseUrl,
    "/api/v1/experiments/workspaces",
    {
      method: "POST",
      body: JSON.stringify(manifest),
    },
  );
}

export async function getSavedExperimentWorkspace(
  baseUrl: string,
  workspaceId: string,
): Promise<SavedWorkspaceDetailPayload> {
  return requestJson<SavedWorkspaceDetailPayload>(
    baseUrl,
    `/api/v1/experiments/workspaces/${encodeURIComponent(workspaceId)}`,
    {
      method: "GET",
    },
  );
}

export async function deleteSavedExperimentWorkspace(
  baseUrl: string,
  workspaceId: string,
): Promise<SavedWorkspaceSummaryPayload> {
  return requestJson<SavedWorkspaceSummaryPayload>(
    baseUrl,
    `/api/v1/experiments/workspaces/${encodeURIComponent(workspaceId)}`,
    {
      method: "DELETE",
    },
  );
}

export async function listExperimentRuns(
  baseUrl: string,
): Promise<ExperimentRunPayload[]> {
  return requestJson<ExperimentRunPayload[]>(
    baseUrl,
    "/api/v1/experiments/runs",
    {
      method: "GET",
    },
  );
}

export async function launchExperimentRun(
  baseUrl: string,
  request: LaunchExperimentRunRequestPayload,
): Promise<ExperimentRunPayload> {
  return requestJson<ExperimentRunPayload>(
    baseUrl,
    "/api/v1/experiments/runs",
    {
      method: "POST",
      body: JSON.stringify(request),
    },
  );
}

export function buildExperimentRunLogUrl(
  baseUrl: string,
  runId: string,
  streamName: "stdout" | "stderr",
): string {
  return (
    `${baseUrl}/api/v1/experiments/runs/${encodeURIComponent(runId)}/logs/` +
    encodeURIComponent(streamName)
  );
}
