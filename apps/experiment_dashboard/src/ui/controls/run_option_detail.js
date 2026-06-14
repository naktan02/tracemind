import { escapeHtml } from "../../shared/formatting/html.js";

export function renderRunOptionDetail(detail, peerDetails) {
  const parts = splitDetail(detail);
  const peerParts = peerDetails.map(splitDetail);
  return parts
    .map((part, index) => {
      const content = renderPartContent(part, peerParts, index);
      return `<span class="run-option-detail-part">${content}</span>`;
    })
    .join(`<span class="run-option-detail-separator"> · </span>`);
}

function splitDetail(detail) {
  return String(detail ?? "")
    .split(" · ")
    .map((part) => part.trim())
    .filter(Boolean);
}

function hasDifferentPeerValue(peerParts, index) {
  const values = new Set(
    peerParts
      .map((parts) => parts[index])
      .filter((part) => part !== undefined && part !== ""),
  );
  return values.size > 1;
}

function renderPartContent(part, peerParts, index) {
  if (!hasDifferentPeerValue(peerParts, index)) {
    return escapeHtml(part);
  }
  const parsed = parseKeyValue(part);
  if (parsed === null) {
    return `<span class="run-option-detail-diff">${escapeHtml(part)}</span>`;
  }
  const value = `<span class="run-option-detail-diff">${escapeHtml(parsed.value)}</span>`;
  return `${escapeHtml(parsed.key)}=${value}`;
}

function parseKeyValue(part) {
  const separatorIndex = part.indexOf("=");
  if (separatorIndex <= 0 || separatorIndex === part.length - 1) return null;
  return {
    key: part.slice(0, separatorIndex),
    value: part.slice(separatorIndex + 1),
  };
}
