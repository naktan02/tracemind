import {
  COLLECTOR_DEBUG_ENABLED_STORAGE_KEY,
  COLLECTOR_STATUS_STORAGE_KEY,
  LAST_TYPING_SEGMENT_STORAGE_KEY,
  PENDING_CAPTURED_TEXT_EVENTS_STORAGE_KEY,
  TYPING_SEGMENT_HISTORY_STORAGE_KEY,
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
      min-height: 360px;
      margin: 0;
    }

    * {
      box-sizing: border-box;
    }

    .popup-shell {
      width: 360px;
      min-height: 360px;
      padding: 16px;
      background: #f7f8f6;
    }

    .panel {
      border: 1px solid #d8ded9;
      border-radius: 6px;
      background: #ffffff;
      padding: 14px;
    }

    h1 {
      margin: 0;
      color: #008f89;
      font-size: 24px;
      line-height: 1.25;
    }

    .stack {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .value {
      margin: 8px 0 0;
      color: #17211d;
      font-size: 14px;
      line-height: 1.55;
    }

    .button-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }

    button {
      min-height: 42px;
      border: 1px solid #00756f;
      border-radius: 4px;
      background: #00756f;
      color: #ffffff;
      font: inherit;
      font-size: 13px;
      font-weight: 800;
      cursor: pointer;
    }

    button.secondary {
      background: #ffffff;
      color: #00756f;
    }

    .entry-button-row {
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
    }

    .tools-panel {
      margin-top: 18px;
      padding-top: 12px;
      border-top: 1px solid #d8ded9;
    }

    .tools-panel .button-row + .button-row {
      margin-top: 8px;
    }
  </style>
  <div class="popup-shell">
    <div class="stack">
      <section class="panel">
        <h1>TraceMind</h1>
        <p class="value">
          온라인에서 남긴 글의 흐름을 바탕으로 현재 마음 상태를 확인하고,
          필요한 도움을 이어주는 개인 보호 화면입니다.
        </p>
      </section>
      <section class="panel entry-button-row">
        <button id="open-self" type="button">본인 페이지</button>
        <button id="open-parent" class="secondary" type="button">부모 페이지</button>
      </section>
      <section class="tools-panel">
        <div class="button-row">
          <button id="toggle-debug" type="button">debug 켜기</button>
          <button id="open-debug" class="secondary" type="button">debug 열기</button>
        </div>
        <div class="button-row">
          <button id="clear-queue" class="secondary" type="button">queue 비우기</button>
        </div>
      </section>
    </div>
  </div>
`;

const toggleDebugButton = getElement("toggle-debug", HTMLButtonElement);
const openDebugButton = getElement("open-debug", HTMLButtonElement);
const openSelfButton = getElement("open-self", HTMLButtonElement);
const openParentButton = getElement("open-parent", HTMLButtonElement);
const clearQueueButton = getElement("clear-queue", HTMLButtonElement);

toggleDebugButton.addEventListener("click", () => {
  void toggleDebug();
});
openDebugButton.addEventListener("click", () => {
  openExtensionPage("collector-debug.html");
});
openSelfButton.addEventListener("click", () => {
  openExtensionPage("index.html#/child/unlock");
});
openParentButton.addEventListener("click", () => {
  openExtensionPage("parent.html#/parent/unlock");
});
clearQueueButton.addEventListener("click", () => {
  void clearQueue();
});

void refreshPopup();

async function refreshPopup(): Promise<void> {
  const items = await storageGet([COLLECTOR_DEBUG_ENABLED_STORAGE_KEY]);
  const isDebugEnabled = items[COLLECTOR_DEBUG_ENABLED_STORAGE_KEY] === true;
  toggleDebugButton.textContent = isDebugEnabled ? "debug 끄기" : "debug 켜기";
  toggleDebugButton.className = isDebugEnabled ? "secondary" : "";
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
    [PENDING_CAPTURED_TEXT_EVENTS_STORAGE_KEY]: [],
    [TYPING_SEGMENT_HISTORY_STORAGE_KEY]: [],
    [LAST_TYPING_SEGMENT_STORAGE_KEY]: null,
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

function getChromeExtensionApi(): ChromeExtensionApi | null {
  const candidate = (globalThis as typeof globalThis & {
    chrome?: Partial<ChromeExtensionApi>;
  }).chrome;
  return candidate?.runtime?.getURL != null && candidate.storage?.local != null
    ? (candidate as ChromeExtensionApi)
    : null;
}
