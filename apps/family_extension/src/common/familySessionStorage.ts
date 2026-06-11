import type { FamilyAccessRole } from "../contracts/generated";

export type StoredFamilySession = {
  role: FamilyAccessRole;
  sessionToken: string;
  sessionExpiresAt: string | null;
};

const FAMILY_SESSION_STORAGE_KEY = "tracemind.familyActiveSession";

type ChromeLocalStorage = {
  get: (
    keys: string[],
    callback: (items: Record<string, unknown>) => void,
  ) => void;
  set: (items: Record<string, unknown>, callback?: () => void) => void;
  remove: (keys: string[], callback?: () => void) => void;
};

type ChromeStorageApi = {
  storage?: {
    local?: ChromeLocalStorage;
  };
};

export function loadFamilySessionSync(): StoredFamilySession | null {
  try {
    return parseStoredFamilySession(
      globalThis.localStorage?.getItem(FAMILY_SESSION_STORAGE_KEY),
    );
  } catch {
    return null;
  }
}

export async function saveFamilySession(
  session: StoredFamilySession,
): Promise<void> {
  try {
    globalThis.localStorage?.setItem(
      FAMILY_SESSION_STORAGE_KEY,
      JSON.stringify(session),
    );
  } catch {
    // localStorage는 dev/browser context에 따라 막힐 수 있어 chrome storage를 계속 시도한다.
  }
  await chromeStorageSet({ [FAMILY_SESSION_STORAGE_KEY]: session });
}

export async function clearFamilySession(): Promise<void> {
  try {
    globalThis.localStorage?.removeItem(FAMILY_SESSION_STORAGE_KEY);
  } catch {
    // localStorage 제거 실패는 chrome storage 제거와 독립적으로 처리한다.
  }
  await chromeStorageRemove([FAMILY_SESSION_STORAGE_KEY]);
}

function parseStoredFamilySession(value: unknown): StoredFamilySession | null {
  const candidate = typeof value === "string" ? parseJson(value) : value;
  if (!isRecord(candidate)) {
    return null;
  }
  if (candidate.role !== "child" && candidate.role !== "parent") {
    return null;
  }
  if (typeof candidate.sessionToken !== "string") {
    return null;
  }
  if (
    candidate.sessionExpiresAt !== null &&
    typeof candidate.sessionExpiresAt !== "string"
  ) {
    return null;
  }
  return {
    role: candidate.role,
    sessionToken: candidate.sessionToken,
    sessionExpiresAt: candidate.sessionExpiresAt,
  };
}

function parseJson(value: string | null): unknown {
  if (value == null) {
    return null;
  }
  try {
    return JSON.parse(value) as unknown;
  } catch {
    return null;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function getChromeStorage(): ChromeLocalStorage | null {
  const candidate = (globalThis as typeof globalThis & {
    chrome?: ChromeStorageApi;
  }).chrome;
  return candidate?.storage?.local ?? null;
}

function chromeStorageSet(items: Record<string, unknown>): Promise<void> {
  return new Promise((resolve) => {
    const storage = getChromeStorage();
    if (storage == null) {
      resolve();
      return;
    }
    storage.set(items, () => resolve());
  });
}

function chromeStorageRemove(keys: string[]): Promise<void> {
  return new Promise((resolve) => {
    const storage = getChromeStorage();
    if (storage == null) {
      resolve();
      return;
    }
    storage.remove(keys, () => resolve());
  });
}
