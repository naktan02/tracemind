import { getAgentApiBaseUrl } from "../common/agentClient";
import type { TypingSegmentPayload } from "../contracts/generated";
import {
  COLLECTOR_DEBUG_ENABLED_STORAGE_KEY,
  COLLECTOR_STATUS_STORAGE_KEY,
  LAST_TYPING_SEGMENT_STORAGE_KEY,
  PENDING_TYPING_SEGMENTS_STORAGE_KEY,
} from "../extension/storageKeys";

type ChromeExtensionApi = {
  runtime: {
    getURL: (path: string) => string;
  };
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
  throw new Error("popup root element를 찾지 못했습니다.");
}

const extensionApi = getChromeExtensionApi();

root.innerHTML = `
  <style>
    :root {
      color: #17211d;
      background: #f7faf8;
      font-family:
        "SUIT Variable", "Pretendard Variable", "Pretendard", "Noto Sans KR",
        system-ui, sans-serif;
    }

    html,
    body {
      width: 360px;
      min-height: 420px;
      margin: 0;
    }

    * {
      box-sizing: border-box;
    }

    .popup-shell {
      width: 360px;
      min-height: 420px;
      padding: 16px;
      background:
        linear-gradient(135deg, rgba(223, 245, 235, 0.9), rgba(255, 249, 237, 0.9));
    }

    .panel {
      border: 1px solid rgba(48, 66, 58, 0.14);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.86);
      padding: 14px;
      box-shadow: 0 12px 26px rgba(23, 33, 29, 0.08);
    }

    h1 {
      margin: 0;
      font-size: 18px;
      line-height: 1.25;
    }

    .stack {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .label {
      margin: 0 0 4px;
      color: #52645c;
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
    }

    .value {
      margin: 0;
      color: #17211d;
      font-size: 13px;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }

    .status-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }

    .status-cell {
      border-radius: 8px;
      background: #f8fafc;
      padding: 10px;
    }

    .button-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }

    button {
      min-height: 38px;
      border: 1px solid #1d7a6f;
      border-radius: 8px;
      background: #1d7a6f;
      color: #ffffff;
      font: inherit;
      font-size: 13px;
      font-weight: 800;
      cursor: pointer;
    }

    button.secondary {
      background: #ffffff;
      color: #1d7a6f;
    }
  </style>
  <div class="popup-shell">
    <div class="stack">
      <section class="panel">
        <h1>TraceMind Collector</h1>
        <p class="value">${getAgentApiBaseUrl()}</p>
      </section>
      <section class="panel">
        <p class="label">Collector</p>
        <div class="status-grid">
          <div class="status-cell">
            <p class="label">pending</p>
            <p id="pending-count" class="value">0</p>
          </div>
          <div class="status-cell">
            <p class="label">debug</p>
            <p id="debug-state" class="value">off</p>
          </div>
        </div>
      </section>
      <section class="panel">
        <p class="label">Last Error</p>
        <p id="last-error" class="value">없음</p>
      </section>
      <section class="panel">
        <p class="label">Last Segment</p>
        <p id="last-segment" class="value">없음</p>
      </section>
      <div class="button-row">
        <button id="toggle-debug" type="button">debug 켜기</button>
        <button id="open-debug" class="secondary" type="button">debug 열기</button>
      </div>
      <div class="button-row">
        <button id="clear-queue" class="secondary" type="button">queue 비우기</button>
        <button id="open-parent" class="secondary" type="button">부모 화면 열기</button>
      </div>
    </div>
  </div>
`;

const pendingCount = getElement("pending-count");
const debugState = getElement("debug-state");
const lastError = getElement("last-error");
const lastSegment = getElement("last-segment");
const toggleDebugButton = getElement("toggle-debug", HTMLButtonElement);
const openDebugButton = getElement("open-debug", HTMLButtonElement);
const openParentButton = getElement("open-parent", HTMLButtonElement);
const clearQueueButton = getElement("clear-queue", HTMLButtonElement);

toggleDebugButton.addEventListener("click", () => {
  void toggleDebug();
});
openDebugButton.addEventListener("click", () => {
  openExtensionPage("collector-debug.html");
});
openParentButton.addEventListener("click", () => {
  openExtensionPage("parent.html");
});
clearQueueButton.addEventListener("click", () => {
  void clearQueue();
});

void refreshPopup();

async function refreshPopup(): Promise<void> {
  if (extensionApi === null) {
    lastError.textContent = "확장 프로그램 컨텍스트가 아닙니다.";
    return;
  }

  const items = await storageGet([
    COLLECTOR_DEBUG_ENABLED_STORAGE_KEY,
    COLLECTOR_STATUS_STORAGE_KEY,
    LAST_TYPING_SEGMENT_STORAGE_KEY,
    PENDING_TYPING_SEGMENTS_STORAGE_KEY,
  ]);
  const pendingSegments = items[PENDING_TYPING_SEGMENTS_STORAGE_KEY];
  const pendingLength = Array.isArray(pendingSegments) ? pendingSegments.length : 0;
  const isDebugEnabled = items[COLLECTOR_DEBUG_ENABLED_STORAGE_KEY] === true;
  const status = isRecord(items[COLLECTOR_STATUS_STORAGE_KEY])
    ? items[COLLECTOR_STATUS_STORAGE_KEY]
    : {};
  const segment = items[LAST_TYPING_SEGMENT_STORAGE_KEY];

  pendingCount.textContent = String(pendingLength);
  debugState.textContent = isDebugEnabled ? "on" : "off";
  toggleDebugButton.textContent = isDebugEnabled ? "debug 끄기" : "debug 켜기";
  toggleDebugButton.className = isDebugEnabled ? "secondary" : "";
  lastError.textContent =
    typeof status.last_error === "string" ? status.last_error : "없음";
  lastSegment.textContent = isTypingSegment(segment)
    ? formatSegmentSummary(segment)
    : "없음";
}

async function toggleDebug(): Promise<void> {
  if (extensionApi === null) {
    return;
  }
  const items = await storageGet([COLLECTOR_DEBUG_ENABLED_STORAGE_KEY]);
  const nextEnabled = items[COLLECTOR_DEBUG_ENABLED_STORAGE_KEY] !== true;
  await storageSet({ [COLLECTOR_DEBUG_ENABLED_STORAGE_KEY]: nextEnabled });
  await refreshPopup();
}

async function clearQueue(): Promise<void> {
  if (extensionApi === null) {
    return;
  }
  await storageSet({
    [PENDING_TYPING_SEGMENTS_STORAGE_KEY]: [],
    [COLLECTOR_STATUS_STORAGE_KEY]: {
      pending_count: 0,
      last_error: null,
    },
  });
  await refreshPopup();
}

function openExtensionPage(path: string): void {
  if (extensionApi === null) {
    return;
  }
  window.open(extensionApi.runtime.getURL(path), "_blank", "noopener");
}

function storageGet(keys: string[]): Promise<Record<string, unknown>> {
  return new Promise((resolve) => {
    extensionApi?.storage.local.get(keys, (items) => resolve(items));
  });
}

function storageSet(items: Record<string, unknown>): Promise<void> {
  return new Promise((resolve) => {
    extensionApi?.storage.local.set(items, () => resolve());
  });
}

function getElement<T extends HTMLElement = HTMLElement>(
  id: string,
  elementType?: { new (...args: never[]): T },
): T {
  const element = document.getElementById(id);
  if (!(element instanceof HTMLElement)) {
    throw new Error(`popup element를 찾지 못했습니다: ${id}`);
  }
  if (elementType !== undefined && !(element instanceof elementType)) {
    throw new Error(`popup element 타입이 다릅니다: ${id}`);
  }
  return element as T;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isTypingSegment(value: unknown): value is TypingSegmentPayload {
  return (
    isRecord(value) &&
    value.schema_version === "typing_segment.v1" &&
    typeof value.surface_type === "string"
  );
}

function formatSegmentSummary(segment: TypingSegmentPayload): string {
  if (segment.page_url == null) {
    return segment.surface_type;
  }
  try {
    return `${segment.surface_type} · ${new URL(segment.page_url).hostname}`;
  } catch {
    return segment.surface_type;
  }
}

function getChromeExtensionApi(): ChromeExtensionApi | null {
  const candidate = (globalThis as typeof globalThis & {
    chrome?: Partial<ChromeExtensionApi>;
  }).chrome;
  return candidate?.runtime?.getURL != null && candidate.storage?.local != null
    ? (candidate as ChromeExtensionApi)
    : null;
}
