import { escapeHtml } from "../../../shared/formatting/html.js";
import { fillSelect } from "../../../ui/controls/form_controls.js";
import { renderRunOptionDetail } from "../../../ui/controls/run_option_detail.js";
import {
  algorithmName,
  centralEvalSetLabel,
  runDescriptor,
  runDetail,
} from "../logic/labels.js";
import { centralAlgorithms, rowsForAlgorithms, rowsWithProjection } from "../logic/selectors.js";

export function normalizeProjectionSelection(bundle, rows, state) {
  const projectionRows = rowsWithProjection(bundle, rows, state.projectionEvalSet);
  const algorithms = centralAlgorithms(projectionRows);
  if (state.projectionAlgorithm && !algorithms.includes(state.projectionAlgorithm)) {
    state.projectionAlgorithm = null;
    state.projectionRunIds = [];
  }
  const visibleRunIds = new Set(projectionRows.map((row) => row.run_id));
  state.projectionRunIds = state.projectionRunIds.filter((runId) =>
    visibleRunIds.has(runId),
  );
}

export function renderProjectionPage(elements, rows, state, bundle) {
  const projectionRows = rowsWithProjection(bundle, rows, state.projectionEvalSet);
  const algorithms = centralAlgorithms(projectionRows);
  fillSelect(elements.projectionMethodFilter, algorithms, state.projectionAlgorithm, "algorithm 없음");
  const candidateRows = state.projectionAlgorithm
    ? rowsForAlgorithms(projectionRows, [state.projectionAlgorithm])
    : projectionRows;
  const selectedRunIds = new Set(state.projectionRunIds);
  const peerDetails = candidateRows.map(runDetail);
  elements.projectionRunCheckboxes.innerHTML =
    candidateRows.length === 0
      ? `<p class="empty">projection 이미지가 있는 algorithm/run이 없습니다.</p>`
      : candidateRows
          .map((row) => {
            const detail = runDetail(row);
            return `
              <label class="run-option">
                <input
                  type="checkbox"
                  data-projection-run-id="${escapeHtml(row.run_id)}"
                  ${selectedRunIds.has(row.run_id) ? "checked" : ""}
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
  renderProjectionGallery(elements, state, bundle);
}

function renderProjectionGallery(elements, state, bundle) {
  const selectedRunIds = new Set(state.projectionRunIds);
  const images = (bundle.projection_images ?? []).filter(
    (image) =>
      image.eval_set === state.projectionEvalSet && selectedRunIds.has(image.run_id),
  );
  elements.projectionGallery.innerHTML =
    images.length === 0
      ? `<p class="empty">선택한 run/eval set의 projection image가 없습니다.</p>`
      : images
          .map(
            (image) => `
              <figure>
                <button
                  class="projection-remove"
                  type="button"
                  data-remove-projection-run-id="${escapeHtml(image.run_id)}"
                  aria-label="projection 제거"
                >x</button>
                <img src="${escapeHtml(image.image_src)}" alt="${escapeHtml(image.run_id)} ${escapeHtml(image.eval_set)} projection" loading="lazy" />
                <figcaption>
                  <strong>${escapeHtml(image.run_id)}</strong>
                  <span>${escapeHtml(centralEvalSetLabel(image.eval_set))} · ${escapeHtml(image.reducer ?? "projection")}</span>
                </figcaption>
              </figure>
            `,
          )
          .join("");
}
