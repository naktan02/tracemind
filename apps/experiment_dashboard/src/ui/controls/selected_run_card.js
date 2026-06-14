import { escapeHtml } from "../../shared/formatting/html.js";

export function renderSelectedRunCard({
  id,
  label,
  detail,
  aliasValue,
  aliasPlaceholder,
  aliasDataAttribute,
  aliasAriaLabel,
  removeDataAttribute,
  removeAriaLabel,
}) {
  return `
    <article class="selected-run-card alias-run-card">
      <strong>${escapeHtml(label)}</strong>
      <input
        type="text"
        ${renderDataAttribute(aliasDataAttribute, id)}
        value="${escapeHtml(aliasValue ?? "")}"
        placeholder="${escapeHtml(aliasPlaceholder)}"
        aria-label="${escapeHtml(aliasAriaLabel)}"
      />
      <button
        type="button"
        ${renderDataAttribute(removeDataAttribute, id)}
        aria-label="${escapeHtml(removeAriaLabel)}"
      >x</button>
      <span class="selected-run-detail" aria-hidden="true">${escapeHtml(detail)}</span>
    </article>
  `;
}

function renderDataAttribute(name, value) {
  return `data-${name}="${escapeHtml(value)}"`;
}
