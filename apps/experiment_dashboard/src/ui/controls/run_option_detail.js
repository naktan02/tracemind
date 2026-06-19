import { escapeHtml } from "../../shared/formatting/html.js";

export function renderRunOptionDetail(detail, peerDetails) {
  const entries = splitDetail(detail).map(parseDetailEntry);
  const peerEntries = peerDetails.map((peerDetail) =>
    splitDetail(peerDetail).map(parseDetailEntry),
  );
  return entries
    .map((entry) => {
      const content = renderPartContent(entry, peerEntries);
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

function renderPartContent(entry, peerEntries) {
  if (!hasDifferentPeerValue(entry, peerEntries)) {
    return escapeHtml(entry.raw);
  }
  if (entry.key === null) {
    return `<span class="run-option-detail-diff">${escapeHtml(entry.raw)}</span>`;
  }
  const value = `<span class="run-option-detail-diff">${escapeHtml(entry.value)}</span>`;
  return `${escapeHtml(entry.key)}=${value}`;
}

function parseDetailEntry(part) {
  const separatorIndex = part.indexOf("=");
  if (separatorIndex <= 0 || separatorIndex === part.length - 1) {
    return {
      key: null,
      raw: part,
      value: part,
    };
  }
  return {
    key: part.slice(0, separatorIndex),
    raw: part,
    value: part.slice(separatorIndex + 1),
  };
}

function hasDifferentPeerValue(entry, peerEntries) {
  if (entry.key === null) {
    return !peerEntries.every((entries) =>
      entries.some((peerEntry) => peerEntry.key === null && peerEntry.raw === entry.raw),
    );
  }

  const values = new Set();
  for (const entries of peerEntries) {
    const peerEntry = entries.find((candidate) => candidate.key === entry.key);
    values.add(peerEntry ? peerEntry.value : "__missing__");
  }
  return values.size > 1;
}
