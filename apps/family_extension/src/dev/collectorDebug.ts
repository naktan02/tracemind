import {
  getAgentApiBaseUrl,
  requestAgentJson,
  type AgentApiError,
} from "../common/agentClient";
import type {
  CapturedTextDebugJobConfigRequestPayload,
  CapturedTextDebugJobRunRequestPayload,
  CapturedTextDebugJobRunResultPayload,
  CapturedTextDebugJobStatusPayload,
  RuntimeProfileStatusPayload,
  RuntimeProfileSyncRequest,
  RuntimeProfileSyncResponse,
  TypingSegmentPayload,
} from "../contracts/generated";
import {
  COLLECTOR_DEBUG_ENABLED_STORAGE_KEY,
  COLLECTOR_DEBUG_PIPELINE_ENABLED_STORAGE_KEY,
  COLLECTOR_STATUS_STORAGE_KEY,
  LAST_TYPING_SEGMENT_STORAGE_KEY,
  TYPING_SEGMENT_HISTORY_STORAGE_KEY,
} from "../extension/storageKeys";

type ChromeExtensionApi = {
  storage: {
    local: {
      get: (
        keys: string[],
        callback: (items: Record<string, unknown>) => void,
      ) => void;
      set: (items: Record<string, unknown>, callback?: () => void) => void;
    };
  };
};

const root = document.getElementById("root");
if (!(root instanceof HTMLElement)) {
  throw new Error("collector debug root element를 찾지 못했습니다.");
}
const debugRoot = root;

const extensionApi = getChromeExtensionApi();

if (extensionApi === null) {
  root.innerHTML = `
    <style>
      body {
        margin: 0;
        background: #f6f8fb;
        color: #17202a;
        font-family:
          Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
          "Segoe UI", sans-serif;
      }

      .debug-shell {
        max-width: 720px;
        margin: 0 auto;
        padding: 32px 20px;
      }

      .panel {
        border: 1px solid #d8dee8;
        border-radius: 8px;
        background: #ffffff;
        padding: 18px;
      }

      h1 {
        margin: 0 0 8px;
        font-size: 24px;
      }

      p {
        margin: 0;
        color: #526071;
        line-height: 1.6;
      }
    </style>
    <main class="debug-shell">
      <section class="panel">
        <h1>Collector Debug</h1>
        <p>
          이 페이지는 확장 프로그램 컨텍스트에서만 chrome.storage를 읽을 수 있습니다.
          확장 아이콘 popup의 debug 열기 버튼으로 다시 여세요.
        </p>
      </section>
    </main>
  `;
} else {
  const activeExtensionApi = extensionApi;

  root.innerHTML = `
  <style>
    :root {
      color: #17202a;
      background: #f6f8fb;
      font-family:
        Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
    }

    body {
      margin: 0;
    }

    .debug-shell {
      max-width: 980px;
      margin: 0 auto;
      padding: 28px 20px;
    }

    .debug-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 20px;
    }

    h1 {
      margin: 0 0 8px;
      font-size: 26px;
    }

    p {
      margin: 0;
      color: #526071;
      line-height: 1.6;
    }

    button {
      border: 1px solid #2251a7;
      border-radius: 6px;
      padding: 10px 14px;
      color: #ffffff;
      background: #2251a7;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }

    button.secondary {
      color: #2251a7;
      background: #ffffff;
    }

    .debug-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 16px;
    }

    .panel {
      border: 1px solid #d8dee8;
      border-radius: 8px;
      background: #ffffff;
      padding: 16px;
    }

    .panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }

    .panel h2 {
      margin: 0;
      font-size: 18px;
    }

    .control-row {
      display: flex;
      flex-wrap: wrap;
      align-items: flex-end;
      gap: 10px;
      margin: 12px 0;
    }

    label {
      display: grid;
      gap: 6px;
      color: #526071;
      font-size: 13px;
      font-weight: 700;
    }

    input {
      width: min(420px, 100%);
      box-sizing: border-box;
      border: 1px solid #bac4d2;
      border-radius: 6px;
      padding: 10px 12px;
      color: #17202a;
      background: #ffffff;
      font: inherit;
      font-size: 14px;
    }

    input.numeric {
      width: 120px;
    }

    .copy-button {
      border-color: #bac4d2;
      padding: 6px 10px;
      color: #2251a7;
      background: #ffffff;
      font-size: 13px;
    }

    .state-pill {
      border: 1px solid #bac4d2;
      border-radius: 6px;
      padding: 9px 11px;
      background: #f8fafc;
      color: #17202a;
      font-size: 13px;
      font-weight: 800;
      line-height: 1.2;
    }

    .state-pill.enabled {
      border-color: #0f766e;
      background: #ecfdf5;
      color: #0f5132;
    }

    pre {
      min-height: 180px;
      margin: 0;
      overflow: auto;
      border-radius: 6px;
      background: #111827;
      color: #e5e7eb;
      padding: 14px;
      font-size: 13px;
      line-height: 1.55;
      white-space: pre-wrap;
      word-break: break-word;
      user-select: text;
    }
  </style>
  <div class="debug-shell">
    <div class="debug-header">
      <div>
        <h1>Collector Debug</h1>
        <p>
          debug 저장을 켠 뒤 입력하면 마지막 TypingSegment와 최근 segment
          history가 여기에 표시됩니다.
        </p>
      </div>
      <div>
        <button id="toggle-debug" type="button">debug 켜기</button>
        <button id="refresh-debug" class="secondary" type="button">새로고침</button>
      </div>
    </div>
    <div class="debug-grid">
      <section class="panel">
        <div class="panel-header">
          <h2>Collector Status</h2>
          <button id="copy-status" class="copy-button" type="button">복사</button>
        </div>
        <pre id="collector-status">{}</pre>
      </section>
      <section class="panel">
        <div class="panel-header">
          <h2>Agent Pipeline Job</h2>
          <button id="refresh-job" class="copy-button" type="button">상태 갱신</button>
        </div>
        <p id="agent-api-base"></p>
        <div class="control-row">
          <label>
            server base URL
            <input id="server-url" type="url" value="http://127.0.0.1:8000" />
          </label>
          <button id="sync-runtime-profile" class="secondary" type="button">profile sync</button>
        </div>
        <pre id="runtime-profile-status">{}</pre>
        <div class="control-row">
          <label>
            interval seconds
            <input id="job-interval" class="numeric" type="number" min="5" max="3600" value="30" />
          </label>
          <label>
            batch size
            <input id="job-batch-size" class="numeric" type="number" min="1" max="500" value="100" />
          </label>
          <button id="toggle-job" type="button">job 켜기</button>
          <button id="toggle-debug-pipeline" class="secondary" type="button">입력마다 분석 켜기</button>
          <span id="debug-pipeline-state" class="state-pill">debug_pipeline_enabled=false</span>
          <button id="run-view-generation" class="secondary" type="button">번역/view 생성</button>
          <button id="run-analysis" class="secondary" type="button">분석 실행</button>
        </div>
        <div class="control-row">
          <button id="run-training" class="secondary" type="button">학습 즉시 실행</button>
        </div>
        <pre id="job-status">{}</pre>
      </section>
      <section class="panel">
        <div class="panel-header">
          <h2>Last TypingSegmentPayload</h2>
          <button id="copy-segment" class="copy-button" type="button">복사</button>
        </div>
        <pre id="last-segment">아직 저장된 segment가 없습니다.</pre>
      </section>
      <section class="panel">
        <div class="panel-header">
          <h2>Recent TypingSegment History</h2>
          <button id="copy-history" class="copy-button" type="button">복사</button>
        </div>
        <pre id="segment-history">아직 저장된 history가 없습니다.</pre>
      </section>
    </div>
  </div>
`;

const toggleButton = getElement("toggle-debug", HTMLButtonElement);
const refreshButton = getElement("refresh-debug", HTMLButtonElement);
const refreshJobButton = getElement("refresh-job", HTMLButtonElement);
const toggleJobButton = getElement("toggle-job", HTMLButtonElement);
const toggleDebugPipelineButton = getElement(
  "toggle-debug-pipeline",
  HTMLButtonElement,
);
const debugPipelineStateText = getElement(
  "debug-pipeline-state",
  HTMLSpanElement,
);
const runViewGenerationButton = getElement(
  "run-view-generation",
  HTMLButtonElement,
);
const runAnalysisButton = getElement("run-analysis", HTMLButtonElement);
const syncRuntimeProfileButton = getElement(
  "sync-runtime-profile",
  HTMLButtonElement,
);
const runTrainingButton = getElement("run-training", HTMLButtonElement);
const copyStatusButton = getElement("copy-status", HTMLButtonElement);
const copySegmentButton = getElement("copy-segment", HTMLButtonElement);
const copyHistoryButton = getElement("copy-history", HTMLButtonElement);
const agentApiBaseText = getElement("agent-api-base", HTMLParagraphElement);
const jobIntervalInput = getElement("job-interval", HTMLInputElement);
const jobBatchSizeInput = getElement("job-batch-size", HTMLInputElement);
const serverUrlInput = getElement("server-url", HTMLInputElement);
const statusPre = getElement("collector-status", HTMLPreElement);
const runtimeProfileStatusPre = getElement(
  "runtime-profile-status",
  HTMLPreElement,
);
const jobStatusPre = getElement("job-status", HTMLPreElement);
const segmentPre = getElement("last-segment", HTMLPreElement);
const historyPre = getElement("segment-history", HTMLPreElement);
let runViewGenerationRequestActive = false;
let runAnalysisRequestActive = false;
let syncRuntimeProfileRequestActive = false;

toggleButton.addEventListener("click", () => {
  void toggleDebug();
});
refreshButton.addEventListener("click", () => {
  void refreshDebugView();
});
refreshJobButton.addEventListener("click", () => {
  void refreshJobStatus();
  void refreshRuntimeProfileStatus();
});
toggleJobButton.addEventListener("click", () => {
  void togglePipelineJob();
});
toggleDebugPipelineButton.addEventListener("click", () => {
  void toggleDebugPipeline();
});
runViewGenerationButton.addEventListener("click", () => {
  void runViewGenerationNow();
});
runAnalysisButton.addEventListener("click", () => {
  void runAnalysisNow();
});
syncRuntimeProfileButton.addEventListener("click", () => {
  void syncRuntimeProfileNow();
});
runTrainingButton.addEventListener("click", () => {
  void runTrainingNow();
});
copyStatusButton.addEventListener("click", () => {
  void copyPreText(statusPre, copyStatusButton);
});
copySegmentButton.addEventListener("click", () => {
  void copyPreText(segmentPre, copySegmentButton);
});
copyHistoryButton.addEventListener("click", () => {
  void copyPreText(historyPre, copyHistoryButton);
});

void refreshDebugView();
agentApiBaseText.textContent = `Agent API: ${getAgentApiBaseUrl()}`;
void refreshRuntimeProfileStatus();
void refreshJobStatus();
window.setInterval(() => {
  if (!hasActiveDebugSelection()) {
    void refreshDebugView();
  }
}, 1000);

async function toggleDebug(): Promise<void> {
  const enabled = await loadDebugEnabled();
  await storageSet({ [COLLECTOR_DEBUG_ENABLED_STORAGE_KEY]: !enabled });
  await refreshDebugView();
}

async function refreshDebugView(): Promise<void> {
  const items = await storageGet([
    COLLECTOR_DEBUG_ENABLED_STORAGE_KEY,
    COLLECTOR_DEBUG_PIPELINE_ENABLED_STORAGE_KEY,
    LAST_TYPING_SEGMENT_STORAGE_KEY,
    TYPING_SEGMENT_HISTORY_STORAGE_KEY,
    COLLECTOR_STATUS_STORAGE_KEY,
  ]);
  const enabled = items[COLLECTOR_DEBUG_ENABLED_STORAGE_KEY] === true;
  const pipelineEnabled =
    items[COLLECTOR_DEBUG_PIPELINE_ENABLED_STORAGE_KEY] === true;
  toggleButton.textContent = enabled ? "debug 끄기" : "debug 켜기";
  toggleButton.className = enabled ? "secondary" : "";
  toggleDebugPipelineButton.textContent = pipelineEnabled
    ? "입력마다 분석 끄기 (true)"
    : "입력마다 분석 켜기 (false)";
  toggleDebugPipelineButton.className = pipelineEnabled ? "" : "secondary";
  debugPipelineStateText.textContent = `debug_pipeline_enabled=${String(
    pipelineEnabled,
  )}`;
  debugPipelineStateText.className = pipelineEnabled
    ? "state-pill enabled"
    : "state-pill";

  statusPre.textContent = JSON.stringify(
    {
      debug_enabled: enabled,
      debug_pipeline_enabled: pipelineEnabled,
      ...(isRecord(items[COLLECTOR_STATUS_STORAGE_KEY])
        ? (items[COLLECTOR_STATUS_STORAGE_KEY] as Record<string, unknown>)
        : {}),
    },
    null,
    2,
  );
  const lastSegment = items[LAST_TYPING_SEGMENT_STORAGE_KEY];
  segmentPre.textContent = isTypingSegment(lastSegment)
    ? JSON.stringify(lastSegment, null, 2)
    : "아직 저장된 segment가 없습니다.";
  const history = items[TYPING_SEGMENT_HISTORY_STORAGE_KEY];
  historyPre.textContent = isTypingSegmentArray(history)
    ? JSON.stringify(
        history.map((segment) => ({
          segment_id: segment.segment_id,
          ended_at: segment.ended_at,
          surface_type: segment.surface_type,
          field_hint: segment.field_hint,
          final_text: segment.final_text,
          deleted_text: segment.deleted_text,
        })),
        null,
        2,
      )
    : "아직 저장된 history가 없습니다.";
}

async function toggleDebugPipeline(): Promise<void> {
  const items = await storageGet([COLLECTOR_DEBUG_PIPELINE_ENABLED_STORAGE_KEY]);
  const enabled = items[COLLECTOR_DEBUG_PIPELINE_ENABLED_STORAGE_KEY] === true;
  await storageSet({ [COLLECTOR_DEBUG_PIPELINE_ENABLED_STORAGE_KEY]: !enabled });
  await refreshDebugView();
}

async function refreshJobStatus(): Promise<void> {
  try {
    const status = await requestAgentJson<CapturedTextDebugJobStatusPayload>(
      "/api/v1/captured-text/debug-job/status",
    );
    applyJobStatus(status);
    jobStatusPre.textContent = JSON.stringify(status, null, 2);
  } catch (error) {
    jobStatusPre.textContent = JSON.stringify(formatAgentError(error), null, 2);
  }
}

async function refreshRuntimeProfileStatus(): Promise<void> {
  try {
    const status = await requestAgentJson<RuntimeProfileStatusPayload>(
      "/api/v1/runtime-profile/status",
    );
    runtimeProfileStatusPre.textContent = JSON.stringify(status, null, 2);
  } catch (error) {
    runtimeProfileStatusPre.textContent = JSON.stringify(
      formatAgentError(error),
      null,
      2,
    );
  }
}

async function syncRuntimeProfileNow(): Promise<void> {
  if (syncRuntimeProfileRequestActive || syncRuntimeProfileButton.disabled) {
    return;
  }
  const serverBaseUrl = serverUrlInput.value.trim();
  if (serverBaseUrl.length === 0) {
    runtimeProfileStatusPre.textContent = JSON.stringify(
      { error: "server_base_url을 입력하세요." },
      null,
      2,
    );
    return;
  }
  syncRuntimeProfileRequestActive = true;
  updateSyncRuntimeProfileButton(true);
  const payload: RuntimeProfileSyncRequest = { server_base_url: serverBaseUrl };
  try {
    const result = await requestAgentJson<RuntimeProfileSyncResponse>(
      "/api/v1/runtime-profile/sync",
      {
        method: "POST",
        body: JSON.stringify(payload),
      },
    );
    runtimeProfileStatusPre.textContent = JSON.stringify(result, null, 2);
    await refreshJobStatus();
  } catch (error) {
    runtimeProfileStatusPre.textContent = JSON.stringify(
      formatAgentError(error),
      null,
      2,
    );
  } finally {
    syncRuntimeProfileRequestActive = false;
    updateSyncRuntimeProfileButton(false);
  }
}

async function togglePipelineJob(): Promise<void> {
  const nextEnabled = toggleJobButton.dataset.enabled !== "true";
  const payload: CapturedTextDebugJobConfigRequestPayload = {
    view_generation_enabled: nextEnabled,
    view_generation_interval_seconds: readNumberInput(jobIntervalInput, 30),
    view_generation_batch_size: readNumberInput(jobBatchSizeInput, 100),
  };
  try {
    const status = await requestAgentJson<CapturedTextDebugJobStatusPayload>(
      "/api/v1/captured-text/debug-job/config",
      {
        method: "POST",
        body: JSON.stringify(payload),
      },
    );
    applyJobStatus(status);
    jobStatusPre.textContent = JSON.stringify(status, null, 2);
  } catch (error) {
    jobStatusPre.textContent = JSON.stringify(formatAgentError(error), null, 2);
  }
}

async function runViewGenerationNow(): Promise<void> {
  if (runViewGenerationRequestActive || runViewGenerationButton.disabled) {
    return;
  }
  runViewGenerationRequestActive = true;
  updateRunViewGenerationButton(true);
  const payload: CapturedTextDebugJobRunRequestPayload = {
    limit: readNumberInput(jobBatchSizeInput, 100),
  };
  try {
    const result =
      await requestAgentJson<CapturedTextDebugJobRunResultPayload>(
        "/api/v1/captured-text/debug-job/run-view-generation",
        {
          method: "POST",
          body: JSON.stringify(payload),
        },
      );
    jobStatusPre.textContent = JSON.stringify(result, null, 2);
    await refreshJobStatus();
  } catch (error) {
    jobStatusPre.textContent = JSON.stringify(formatAgentError(error), null, 2);
  } finally {
    runViewGenerationRequestActive = false;
    updateRunViewGenerationButton(false);
  }
}

async function runAnalysisNow(): Promise<void> {
  if (runAnalysisRequestActive || runAnalysisButton.disabled) {
    return;
  }
  runAnalysisRequestActive = true;
  updateRunAnalysisButton(true);
  const payload: CapturedTextDebugJobRunRequestPayload = {
    limit: readNumberInput(jobBatchSizeInput, 100),
  };
  try {
    const result =
      await requestAgentJson<CapturedTextDebugJobRunResultPayload>(
        "/api/v1/captured-text/debug-job/run-analysis",
        {
          method: "POST",
          body: JSON.stringify(payload),
        },
      );
    jobStatusPre.textContent = JSON.stringify(result, null, 2);
    await refreshJobStatus();
  } catch (error) {
    jobStatusPre.textContent = JSON.stringify(formatAgentError(error), null, 2);
  } finally {
    runAnalysisRequestActive = false;
    updateRunAnalysisButton(false);
  }
}

async function runTrainingNow(): Promise<void> {
  const serverBaseUrl = serverUrlInput.value.trim();
  if (serverBaseUrl.length === 0) {
    jobStatusPre.textContent = JSON.stringify(
      { error: "server_base_url을 입력하세요." },
      null,
      2,
    );
    return;
  }
  try {
    const result = await requestAgentJson<Record<string, unknown>>(
      "/api/v1/training/run-current-task",
      {
        method: "POST",
        body: JSON.stringify({
          server_base_url: serverBaseUrl,
          analysis_event_days: 7,
        }),
      },
    );
    jobStatusPre.textContent = JSON.stringify(result, null, 2);
  } catch (error) {
    jobStatusPre.textContent = JSON.stringify(formatAgentError(error), null, 2);
  }
}

function applyJobStatus(status: CapturedTextDebugJobStatusPayload): void {
  toggleJobButton.dataset.enabled = String(status.view_generation_enabled);
  toggleJobButton.textContent = status.view_generation_enabled
    ? "job 끄기"
    : "job 켜기";
  toggleJobButton.className = status.view_generation_enabled ? "secondary" : "";
  jobIntervalInput.value = String(status.view_generation_interval_seconds);
  jobBatchSizeInput.value = String(status.view_generation_batch_size);
  updateRunViewGenerationButton(status.view_generation_running);
  updateRunAnalysisButton(status.view_generation_running);
}

function updateRunViewGenerationButton(isServerJobRunning: boolean): void {
  const isRunning = runViewGenerationRequestActive || isServerJobRunning;
  runViewGenerationButton.disabled = isRunning;
  runViewGenerationButton.textContent = isRunning
    ? "번역/view 생성 중"
    : "번역/view 생성";
}

function updateRunAnalysisButton(isServerJobRunning: boolean): void {
  const isRunning = runAnalysisRequestActive || isServerJobRunning;
  runAnalysisButton.disabled = isRunning;
  runAnalysisButton.textContent = isRunning ? "분석 실행 중" : "분석 실행";
}

function updateSyncRuntimeProfileButton(isRunning: boolean): void {
  syncRuntimeProfileButton.disabled = isRunning;
  syncRuntimeProfileButton.textContent = isRunning
    ? "profile sync 중"
    : "profile sync";
}

function readNumberInput(input: HTMLInputElement, fallback: number): number {
  const parsed = Number.parseInt(input.value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

async function loadDebugEnabled(): Promise<boolean> {
  const items = await storageGet([COLLECTOR_DEBUG_ENABLED_STORAGE_KEY]);
  return items[COLLECTOR_DEBUG_ENABLED_STORAGE_KEY] === true;
}

function storageGet(keys: string[]): Promise<Record<string, unknown>> {
  return new Promise((resolve) => {
    activeExtensionApi.storage.local.get(keys, (items) => resolve(items));
  });
}

function storageSet(items: Record<string, unknown>): Promise<void> {
  return new Promise((resolve) => {
    activeExtensionApi.storage.local.set(items, () => resolve());
  });
}

async function copyPreText(
  preElement: HTMLPreElement,
  button: HTMLButtonElement,
): Promise<void> {
  const originalText = button.textContent ?? "복사";
  try {
    await navigator.clipboard.writeText(preElement.textContent ?? "");
    button.textContent = "복사됨";
  } catch {
    button.textContent = "실패";
  }
  window.setTimeout(() => {
    button.textContent = originalText;
  }, 1200);
}

function hasActiveDebugSelection(): boolean {
  const selection = window.getSelection();
  if (selection === null || selection.isCollapsed || selection.rangeCount === 0) {
    return false;
  }
  const container = selection.getRangeAt(0).commonAncestorContainer;
  const element =
    container instanceof HTMLElement ? container : container.parentElement;
  return element !== null && debugRoot.contains(element);
}
}

function getChromeExtensionApi(): ChromeExtensionApi | null {
  const candidate = (globalThis as typeof globalThis & {
    chrome?: Partial<ChromeExtensionApi>;
  }).chrome;
  return candidate?.storage?.local != null
    ? (candidate as ChromeExtensionApi)
    : null;
}

function getElement<T extends HTMLElement>(
  id: string,
  elementType: { new (...args: never[]): T },
): T {
  const element = document.getElementById(id);
  if (!(element instanceof elementType)) {
    throw new Error(`collector debug element를 찾지 못했습니다: ${id}`);
  }
  return element;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isTypingSegment(value: unknown): value is TypingSegmentPayload {
  return (
    isRecord(value) &&
    value.schema_version === "typing_segment.v1" &&
    typeof value.segment_id === "string"
  );
}

function isTypingSegmentArray(value: unknown): value is TypingSegmentPayload[] {
  return Array.isArray(value) && value.every(isTypingSegment);
}

function formatAgentError(error: unknown): Record<string, unknown> {
  if (error instanceof Error) {
    const candidate = error as AgentApiError;
    return {
      error: candidate.message,
      status: candidate.status,
      kind: candidate.kind,
    };
  }
  return { error: String(error) };
}
