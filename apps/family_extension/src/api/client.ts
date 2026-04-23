const DEFAULT_AGENT_API_BASE_URL = "http://127.0.0.1:8001";

export class AgentApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "AgentApiError";
    this.status = status;
  }
}

export function getAgentApiBaseUrl(): string {
  return import.meta.env.VITE_AGENT_API_BASE_URL ?? DEFAULT_AGENT_API_BASE_URL;
}

export async function requestAgentJson<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${getAgentApiBaseUrl()}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });

  if (!response.ok) {
    throw new AgentApiError(
      `Agent API 요청이 실패했습니다: ${response.status}`,
      response.status,
    );
  }

  return (await response.json()) as T;
}
