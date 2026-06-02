import type {
  TypingCaptureConfidence,
  TypingSurfaceType,
} from "../contracts/generated";

export type TextSurfaceSnapshot = {
  text: string;
  surfaceType: TypingSurfaceType;
  captureConfidence: TypingCaptureConfidence;
  fieldHint: string | null;
};

export type TextSurfaceAdapter = {
  matches: (target: EventTarget | null) => target is HTMLElement;
  read: (element: HTMLElement) => TextSurfaceSnapshot | null;
};

const INPUT_TEXT_TYPES = new Set([
  "",
  "email",
  "search",
  "tel",
  "text",
  "url",
]);

const inputSurfaceAdapter: TextSurfaceAdapter = {
  matches(target): target is HTMLInputElement {
    return target instanceof HTMLInputElement && INPUT_TEXT_TYPES.has(target.type);
  },
  read(element) {
    if (!(element instanceof HTMLInputElement)) {
      return null;
    }
    return {
      text: element.value,
      surfaceType: "input",
      captureConfidence: "high",
      fieldHint: readFieldHint(element),
    };
  },
};

const textareaSurfaceAdapter: TextSurfaceAdapter = {
  matches(target): target is HTMLTextAreaElement {
    return target instanceof HTMLTextAreaElement;
  },
  read(element) {
    if (!(element instanceof HTMLTextAreaElement)) {
      return null;
    }
    return {
      text: element.value,
      surfaceType: "textarea",
      captureConfidence: "high",
      fieldHint: readFieldHint(element),
    };
  },
};

const contentEditableSurfaceAdapter: TextSurfaceAdapter = {
  matches(target): target is HTMLElement {
    return target instanceof HTMLElement && findContentEditableRoot(target) !== null;
  },
  read(element) {
    const root = findContentEditableRoot(element);
    if (root === null) {
      return null;
    }
    return {
      text: root.innerText.trim(),
      surfaceType: "contenteditable",
      captureConfidence: "medium",
      fieldHint: readFieldHint(root),
    };
  },
};

const DEFAULT_SURFACE_ADAPTERS: TextSurfaceAdapter[] = [
  inputSurfaceAdapter,
  textareaSurfaceAdapter,
  contentEditableSurfaceAdapter,
];

export function readTextSurfaceSnapshot(
  target: EventTarget | null,
  adapters: TextSurfaceAdapter[] = DEFAULT_SURFACE_ADAPTERS,
): { element: HTMLElement; snapshot: TextSurfaceSnapshot } | null {
  for (const adapter of adapters) {
    if (!adapter.matches(target)) {
      continue;
    }
    const element = normalizeSurfaceElement(target, adapter);
    const snapshot = adapter.read(element);
    if (snapshot === null) {
      continue;
    }
    return { element, snapshot };
  }
  return null;
}

function normalizeSurfaceElement(
  target: HTMLElement,
  adapter: TextSurfaceAdapter,
): HTMLElement {
  if (adapter === contentEditableSurfaceAdapter) {
    return findContentEditableRoot(target) ?? target;
  }
  return target;
}

function findContentEditableRoot(target: HTMLElement): HTMLElement | null {
  const root = target.closest("[contenteditable='true'], [contenteditable='']");
  return root instanceof HTMLElement ? root : null;
}

function readFieldHint(element: HTMLElement): string | null {
  const candidates = [
    element.getAttribute("aria-label"),
    element.getAttribute("placeholder"),
    element.getAttribute("name"),
    element.getAttribute("role"),
  ];
  const hint = candidates.find((value) => value !== null && value.trim() !== "");
  return hint?.trim().slice(0, 256) ?? null;
}
