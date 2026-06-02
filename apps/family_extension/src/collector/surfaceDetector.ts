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
  "search",
  "tel",
  "text",
  "url",
]);

const inputSurfaceAdapter: TextSurfaceAdapter = {
  matches(target): target is HTMLInputElement {
    return (
      target instanceof HTMLInputElement &&
      INPUT_TEXT_TYPES.has(target.type) &&
      !isSensitiveTextSurface(target)
    );
  },
  read(element) {
    if (!(element instanceof HTMLInputElement) || isSensitiveTextSurface(element)) {
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
    return target instanceof HTMLTextAreaElement && !isSensitiveTextSurface(target);
  },
  read(element) {
    if (!(element instanceof HTMLTextAreaElement) || isSensitiveTextSurface(element)) {
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

const richEditorSurfaceAdapter: TextSurfaceAdapter = {
  matches(target): target is HTMLElement {
    const element = resolveEventTargetElement(target);
    return element !== null && findRichEditorRoot(element) !== null;
  },
  read(element) {
    const root = findRichEditorRoot(element);
    if (root === null || isSensitiveTextSurface(root)) {
      return null;
    }
    return {
      text: readElementText(root),
      surfaceType: "rich_editor",
      captureConfidence: "medium",
      fieldHint: readFieldHint(root),
    };
  },
};

const contentEditableSurfaceAdapter: TextSurfaceAdapter = {
  matches(target): target is HTMLElement {
    const element = resolveEventTargetElement(target);
    return element !== null && findContentEditableRoot(element) !== null;
  },
  read(element) {
    const root = findContentEditableRoot(element);
    if (root === null || isSensitiveTextSurface(root)) {
      return null;
    }
    return {
      text: readElementText(root),
      surfaceType: "contenteditable",
      captureConfidence: "medium",
      fieldHint: readFieldHint(root),
    };
  },
};

const DEFAULT_SURFACE_ADAPTERS: TextSurfaceAdapter[] = [
  inputSurfaceAdapter,
  textareaSurfaceAdapter,
];

export function readTextSurfaceSnapshot(
  target: EventTarget | null,
  adapters: TextSurfaceAdapter[] = DEFAULT_SURFACE_ADAPTERS,
): { element: HTMLElement; snapshot: TextSurfaceSnapshot } | null {
  const element = resolveEventTargetElement(target);
  if (element === null) {
    return null;
  }
  for (const adapter of adapters) {
    if (!adapter.matches(element)) {
      continue;
    }
    const normalizedElement = normalizeSurfaceElement(element, adapter);
    const snapshot = adapter.read(normalizedElement);
    if (snapshot === null) {
      continue;
    }
    return { element: normalizedElement, snapshot };
  }
  return null;
}

function resolveEventTargetElement(target: EventTarget | null): HTMLElement | null {
  if (target instanceof HTMLElement) {
    return target;
  }
  if (target instanceof Node && target.parentElement instanceof HTMLElement) {
    return target.parentElement;
  }
  return null;
}

function normalizeSurfaceElement(
  target: HTMLElement,
  adapter: TextSurfaceAdapter,
): HTMLElement {
  if (adapter === richEditorSurfaceAdapter) {
    return findRichEditorRoot(target) ?? target;
  }
  if (adapter === contentEditableSurfaceAdapter) {
    return findContentEditableRoot(target) ?? target;
  }
  return target;
}

const RICH_EDITOR_ROOT_SELECTOR = [
  "[data-lexical-editor='true']",
  "[data-slate-editor='true']",
  "[data-contents='true']",
  ".ProseMirror",
  ".ql-editor",
  ".DraftEditor-root",
  ".codex-editor__redactor",
  ".se-main-container",
  ".se-section-document",
  ".se-component-content",
  ".se-module-text",
].join(",");

function findRichEditorRoot(target: HTMLElement): HTMLElement | null {
  const candidates: HTMLElement[] = [];
  let current: HTMLElement | null = target;
  while (current !== null && current !== current.ownerDocument.body) {
    if (current.matches(RICH_EDITOR_ROOT_SELECTOR)) {
      candidates.push(current);
    }
    current = current.parentElement;
  }
  return candidates.length > 0 ? candidates[candidates.length - 1] : null;
}

function findContentEditableRoot(target: HTMLElement): HTMLElement | null {
  const editableDocumentRoot = findEditableDocumentRoot(target);
  if (editableDocumentRoot !== null) {
    return editableDocumentRoot;
  }
  if (!isEditableElement(target)) {
    return null;
  }
  let root = target;
  let parent = target.parentElement;
  while (parent !== null && isEditableElement(parent)) {
    root = parent;
    parent = parent.parentElement;
  }
  return root;
}

function findEditableDocumentRoot(target: HTMLElement): HTMLElement | null {
  const ownerDocument = target.ownerDocument;
  if (ownerDocument.designMode.toLowerCase() !== "on") {
    return null;
  }
  return ownerDocument.body instanceof HTMLElement ? ownerDocument.body : null;
}

function isEditableElement(element: HTMLElement): boolean {
  if (element.isContentEditable) {
    return true;
  }
  const contentEditable = element.getAttribute("contenteditable");
  return (
    contentEditable !== null && contentEditable.trim().toLowerCase() !== "false"
  );
}

function readElementText(element: HTMLElement): string {
  const innerText = element.innerText?.trim();
  if (innerText) {
    return innerText;
  }
  return (element.textContent ?? "").trim();
}

const SENSITIVE_FIELD_KEYWORDS = [
  "account",
  "email",
  "e-mail",
  "login",
  "otp",
  "pass",
  "password",
  "token",
  "user",
  "username",
  "userid",
  "계정",
  "로그인",
  "비밀번호",
  "아이디",
  "이메일",
  "인증",
];

const SENSITIVE_AUTOCOMPLETE_TOKENS = new Set([
  "current-password",
  "email",
  "new-password",
  "one-time-code",
  "username",
]);

function isSensitiveTextSurface(element: HTMLElement): boolean {
  if (element instanceof HTMLInputElement && element.type === "password") {
    return true;
  }
  if (element instanceof HTMLInputElement && element.type === "email") {
    return true;
  }
  if (hasSensitiveAutocomplete(element)) {
    return true;
  }
  if (isInsidePasswordForm(element)) {
    return true;
  }
  return SENSITIVE_FIELD_KEYWORDS.some((keyword) =>
    readSensitiveFieldText(element).includes(keyword),
  );
}

function hasSensitiveAutocomplete(element: HTMLElement): boolean {
  const autocomplete = element.getAttribute("autocomplete");
  if (autocomplete === null) {
    return false;
  }
  return autocomplete
    .toLowerCase()
    .split(/\s+/)
    .some((token) => SENSITIVE_AUTOCOMPLETE_TOKENS.has(token));
}

function isInsidePasswordForm(element: HTMLElement): boolean {
  const form = element.closest("form");
  return form !== null && form.querySelector("input[type='password']") !== null;
}

function readSensitiveFieldText(element: HTMLElement): string {
  return [
    element.id,
    element.getAttribute("aria-label"),
    element.getAttribute("autocomplete"),
    element.getAttribute("name"),
    element.getAttribute("placeholder"),
    element.getAttribute("title"),
  ]
    .filter((value): value is string => value !== null && value.trim() !== "")
    .join(" ")
    .toLowerCase();
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
