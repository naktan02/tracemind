import { escapeHtml } from "../../../shared/formatting/html.js";
import { fillSelect } from "../../../ui/controls/form_controls.js";
import { renderRunOptionDetail } from "../../../ui/controls/run_option_detail.js";
import {
  algorithmName,
  dataSourceLabel,
  labelBudgetLabel,
  runDescriptor,
  runHoverDetail,
  runId,
} from "../logic/labels.js";
import { flProjectionEvalSets, flRowsWithProjection } from "../logic/selectors.js";

export function normalizeFlProjectionSelection(bundle, rows, state) {
  const evalSets = flProjectionEvalSets(bundle, rows);
  if (!evalSets.includes(state.projectionEvalSet)) {
    state.projectionEvalSet = evalSets.includes("test") ? "test" : evalSets[0] ?? "test";
    state.projectionRunIds = [];
  }
}

export function renderFlProjectionPage(elements, rows, state, bundle, selectionRows = rows) {
  const evalSets = flProjectionEvalSets(bundle, selectionRows);
  fillSelect(elements.flProjectionEvalFilter, evalSets, state.projectionEvalSet, "eval 없음");
  const candidateRows = flRowsWithProjection(bundle, rows, state.projectionEvalSet);
  const selectedRunIds = new Set(state.projectionRunIds);
  const peerDetails = candidateRows.map(runHoverDetail);
  elements.flProjectionRunCheckboxes.innerHTML =
    candidateRows.length === 0
      ? `<p class="empty">projection 이미지가 있는 FL run이 없습니다.</p>`
      : candidateRows
          .map((row) => {
            const id = runId(row);
            const detail = runHoverDetail(row);
            return `
              <label class="run-option">
                <input
                  type="checkbox"
                  data-fl-projection-run-id="${escapeHtml(id)}"
                  ${selectedRunIds.has(id) ? "checked" : ""}
                />
                <span>
                  <strong>${escapeHtml(algorithmName(row))}</strong>
                  <small>${escapeHtml(runDescriptor(row))}</small>
                </span>
                <span class="run-option-detail" aria-hidden="true">${renderRunOptionDetail(detail, peerDetails)}</span>
              </label>
            `;
          })
          .join("");
  renderGallery(elements, selectionRows, state, bundle);
}

function renderGallery(elements, rows, state, bundle) {
  const runById = new Map(rows.map((row) => [runId(row), row]));
  const selectedRunIds = new Set(state.projectionRunIds);
  const images = (bundle.projection_images ?? []).filter(
    (image) => image.eval_set === state.projectionEvalSet && selectedRunIds.has(image.run_id),
  );
  elements.flProjectionGallery.innerHTML =
    images.length === 0
      ? `<p class="empty">선택한 FL run/eval set의 projection image가 없습니다.</p>`
      : images
          .map((image) => {
            const row = runById.get(image.run_id);
            return `
              <figure>
                <button
                  class="projection-remove"
                  type="button"
                  data-remove-fl-projection-run-id="${escapeHtml(image.run_id)}"
                  aria-label="FL projection 제거"
                >x</button>
                <img src="${escapeHtml(image.image_src)}" alt="${escapeHtml(image.run_id)} ${escapeHtml(image.eval_set)} projection" loading="lazy" />
                <figcaption>
                  <strong>${escapeHtml(row ? algorithmName(row) : image.run_id)}</strong>
                  <span>${row ? escapeHtml([dataSourceLabel(row), labelBudgetLabel(row), `eval=${image.eval_set}`].join(" · ")) : escapeHtml(image.eval_set)}</span>
                  <span>${escapeHtml(image.reducer ?? "projection")}${image.fallback_reason ? ` · ${escapeHtml(image.fallback_reason)}` : ""}</span>
                </figcaption>
              </figure>
            `;
          })
          .join("");
}
