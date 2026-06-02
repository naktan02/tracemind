import type { TypingSegmentPayload } from "../contracts/generated";
import {
  COLLECTOR_DEBUG_ENABLED_STORAGE_KEY,
  COLLECTOR_STATUS_STORAGE_KEY,
  LAST_TYPING_SEGMENT_STORAGE_KEY,
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

    .panel h2 {
      margin: 0 0 10px;
      font-size: 18px;
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
    }
  </style>
  <div class="debug-shell">
    <div class="debug-header">
      <div>
        <h1>Collector Debug</h1>
        <p>
          debug 저장을 켠 뒤 fixture 페이지에서 입력하면 마지막 TypingSegment
          JSON이 여기에 표시됩니다.
        </p>
      </div>
      <div>
        <button id="toggle-debug" type="button">debug 켜기</button>
        <button id="refresh-debug" class="secondary" type="button">새로고침</button>
      </div>
    </div>
    <div class="debug-grid">
      <section class="panel">
        <h2>Collector Status</h2>
        <pre id="collector-status">{}</pre>
      </section>
      <section class="panel">
        <h2>Last TypingSegmentPayload</h2>
        <pre id="last-segment">아직 저장된 segment가 없습니다.</pre>
      </section>
    </div>
  </div>
`;

const toggleButton = getElement("toggle-debug", HTMLButtonElement);
const refreshButton = getElement("refresh-debug", HTMLButtonElement);
const statusPre = getElement("collector-status", HTMLPreElement);
const segmentPre = getElement("last-segment", HTMLPreElement);

toggleButton.addEventListener("click", () => {
  void toggleDebug();
});
refreshButton.addEventListener("click", () => {
  void refreshDebugView();
});

void refreshDebugView();
window.setInterval(() => {
  void refreshDebugView();
}, 1000);

async function toggleDebug(): Promise<void> {
  const enabled = await loadDebugEnabled();
  await storageSet({ [COLLECTOR_DEBUG_ENABLED_STORAGE_KEY]: !enabled });
  await refreshDebugView();
}

async function refreshDebugView(): Promise<void> {
  const items = await storageGet([
    COLLECTOR_DEBUG_ENABLED_STORAGE_KEY,
    LAST_TYPING_SEGMENT_STORAGE_KEY,
    COLLECTOR_STATUS_STORAGE_KEY,
  ]);
  const enabled = items[COLLECTOR_DEBUG_ENABLED_STORAGE_KEY] === true;
  toggleButton.textContent = enabled ? "debug 끄기" : "debug 켜기";
  toggleButton.className = enabled ? "secondary" : "";

  statusPre.textContent = JSON.stringify(
    {
      debug_enabled: enabled,
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
