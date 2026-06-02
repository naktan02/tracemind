const DEFAULT_AGENT_API_BASE_URL = "http://127.0.0.1:8001";
export type AgentApiErrorKind = "network" | "http";

export class AgentApiError extends Error {
  readonly status: number;
  readonly kind: AgentApiErrorKind;

  constructor(message: string, status: number, kind: AgentApiErrorKind) {
    super(message);
    this.name = "AgentApiError";
    this.status = status;
    this.kind = kind;
  }
}

export function getAgentApiBaseUrl(): string {
  return import.meta.env.VITE_AGENT_API_BASE_URL ?? DEFAULT_AGENT_API_BASE_URL;
}

export async function requestAgentJson<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${getAgentApiBaseUrl()}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers ?? {}),
      },
    });
  } catch {
    throw new AgentApiError(
      "로컬 프로그램에 연결하지 못했습니다. agent API가 실행 중인지, 브라우저 접근이 허용됐는지 확인해 주세요.",
      0,
      "network",
    );
  }

  if (!response.ok) {
    throw new AgentApiError(
      `Agent API 요청이 실패했습니다: ${response.status}`,
      response.status,
      "http",
    );
  }

  return (await response.json()) as T;
}

export function buildAgentApiUrl(path: string): string {
  return `${getAgentApiBaseUrl()}${path}`;
}
