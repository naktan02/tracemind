import { useEffect, useMemo, useState } from "react";

import {
  formatDateTime,
  formatEntrypointName,
  formatMetricKey,
  formatMetricValue,
  formatRunStatus,
  formatSectionName,
  formatTrackName,
} from "../lib/formatters";
import type {
  ExperimentRunPayload,
  SavedWorkspaceSelectionPreviewPayload,
  SavedWorkspaceSummaryPayload,
} from "../types";

interface CompareRow {
  workspaceId: string;
  manifestId: string;
  trackName: string;
  entrypointName: string;
  updatedAt: string;
  latestRun: ExperimentRunPayload | null;
  selectionPreviews: SavedWorkspaceSelectionPreviewPayload[];
}

interface CompareGroup {
  groupKey: string;
  heading: string;
  baseChips: string[];
  rows: CompareRow[];
}

const PRIORITY_METRIC_KEYS = [
  "validation.accuracy_top_1",
  "validation.macro_f1",
  "test.accuracy_top_1",
  "test.macro_f1",
  "selection.accuracy_top_1",
  "teacher.accepted_ratio",
  "teacher.accepted_hidden_label_accuracy",
];

const METHOD_SECTION_KEYWORDS = [
  "method",
  "algorithm",
  "builder",
  "backend",
  "family",
  "aggregation",
  "privacy",
  "augmenter",
  "scoring",
];

export function WorkspaceCompareBoard(props: {
  currentWorkspaceId: string | null;
  savedWorkspaces: SavedWorkspaceSummaryPayload[];
  runs: ExperimentRunPayload[];
  rerunningWorkspaceId: string | null;
  deletingWorkspaceId: string | null;
  onRefreshBoard: () => void;
  onLoadWorkspace: (workspaceId: string) => void;
  onCloneWorkspace: (workspaceId: string) => void;
  onRelaunchWorkspace: (workspaceId: string) => void;
  onDeleteWorkspace: (workspaceId: string) => void;
}) {
  const compareRows = useMemo<CompareRow[]>(() => {
    const runById = new Map(props.runs.map((run) => [run.run_id, run]));
    return [...props.savedWorkspaces]
      .sort((left, right) => right.updated_at.localeCompare(left.updated_at))
      .map((workspace) => ({
        workspaceId: workspace.workspace_id,
        manifestId: workspace.manifest_id,
        trackName: workspace.track_name,
        entrypointName: workspace.entrypoint_name,
        updatedAt: workspace.updated_at,
        latestRun: workspace.latest_run_id
          ? (runById.get(workspace.latest_run_id) ?? null)
          : null,
        selectionPreviews: workspace.selection_previews,
      }));
  }, [props.runs, props.savedWorkspaces]);

  const groupedRows = useMemo<CompareGroup[]>(() => {
    const groups = new Map<string, CompareGroup>();

    for (const row of compareRows) {
      const basePreviews = row.selectionPreviews.filter(
        (preview) => !isMethodLikeSelection(preview),
      );
      const baseChips =
        basePreviews.length > 0
          ? basePreviews.map(formatSelectionChip)
          : [
              `${formatTrackName(row.trackName)} / ${formatEntrypointName(
                row.entrypointName,
              )}`,
            ];
      const groupKey = baseChips.join("|");
      if (!groups.has(groupKey)) {
        groups.set(groupKey, {
          groupKey,
          heading:
            basePreviews.length > 0
              ? "같은 기본 조합 비교"
              : "저장된 실험 비교",
          baseChips,
          rows: [],
        });
      }
      groups.get(groupKey)?.rows.push(row);
    }

    return [...groups.values()];
  }, [compareRows]);

  const availableMetricKeys = useMemo(() => {
    const keys = new Set<string>();
    for (const row of compareRows) {
      for (const metric of row.latestRun?.result_summary?.metrics ?? []) {
        keys.add(metric.metric_key);
      }
    }
    return [...keys].sort();
  }, [compareRows]);

  const [selectedMetricKeys, setSelectedMetricKeys] = useState<string[]>([]);

  useEffect(() => {
    setSelectedMetricKeys((current) => {
      const retained = current.filter((key) => availableMetricKeys.includes(key));
      if (retained.length > 0) {
        return retained;
      }
      const prioritized = PRIORITY_METRIC_KEYS.filter((key) =>
        availableMetricKeys.includes(key),
      );
      if (prioritized.length > 0) {
        return prioritized.slice(0, 4);
      }
      return availableMetricKeys.slice(0, 4);
    });
  }, [availableMetricKeys]);

  function toggleMetricKey(metricKey: string) {
    setSelectedMetricKeys((current) =>
      current.includes(metricKey)
        ? current.filter((candidate) => candidate !== metricKey)
        : [...current, metricKey],
    );
  }

  return (
    <div className="compare-board">
      <div className="panel-header panel-header--compact">
        <div>
          <p className="panel-kicker">저장된 실험 비교</p>
          <h3>같은 조합에서 방법론이나 파라미터를 바꿔 비교합니다</h3>
          <p className="hint-text">
            먼저 저장된 조합을 보고, 그대로 재실행하거나 복제 후 방법론만 바꿔 새
            실험으로 쌓을 수 있습니다.
          </p>
        </div>
        <div className="compare-board__meta">
          <span className="compare-count">{compareRows.length}개 저장됨</span>
          <button
            type="button"
            className="ghost-button ghost-button--small"
            onClick={props.onRefreshBoard}
          >
            새로고침
          </button>
        </div>
      </div>

      {availableMetricKeys.length > 0 ? (
        <div className="metric-pill-row">
          {availableMetricKeys.map((metricKey) => (
            <button
              key={metricKey}
              type="button"
              className={
                selectedMetricKeys.includes(metricKey)
                  ? "metric-pill metric-pill--active"
                  : "metric-pill"
              }
              onClick={() => toggleMetricKey(metricKey)}
            >
              {formatMetricKey(metricKey)}
            </button>
          ))}
        </div>
      ) : (
        <p className="hint-text">
          아직 완료된 결과가 없어 비교 지표를 고를 수 없습니다.
        </p>
      )}

      {groupedRows.length > 0 ? (
        <div className="compare-group-list">
          {groupedRows.map((group) => (
            <section className="compare-group" key={group.groupKey}>
              <div className="compare-group__header">
                <div>
                  <strong>{group.heading}</strong>
                  <div className="selection-chip-list">
                    {group.baseChips.map((chip) => (
                      <span className="selection-chip" key={chip}>
                        {chip}
                      </span>
                    ))}
                  </div>
                </div>
                <span className="compare-count">{group.rows.length}개 조합</span>
              </div>

              <div className="compare-table-wrap">
                <table className="compare-table">
                  <thead>
                    <tr>
                      <th>실험</th>
                      <th>방법론 / 변경점</th>
                      <th>최근 갱신</th>
                      <th>상태</th>
                      {selectedMetricKeys.map((metricKey) => (
                        <th key={metricKey}>{formatMetricKey(metricKey)}</th>
                      ))}
                      <th>동작</th>
                    </tr>
                  </thead>
                  <tbody>
                    {group.rows.map((row) => {
                      const metricMap = new Map(
                        (row.latestRun?.result_summary?.metrics ?? []).map((metric) => [
                          metric.metric_key,
                          metric.value,
                        ]),
                      );
                      const methodChips = buildMethodChips(row.selectionPreviews);
                      return (
                        <tr
                          key={row.workspaceId}
                          className={
                            row.workspaceId === props.currentWorkspaceId
                              ? "compare-row compare-row--active"
                              : "compare-row"
                          }
                        >
                          <td>
                            <div className="compare-workspace-cell">
                              <strong>{row.workspaceId}</strong>
                              <span>
                                {formatTrackName(row.trackName)} /{" "}
                                {formatEntrypointName(row.entrypointName)}
                              </span>
                              <span>manifest: {row.manifestId}</span>
                            </div>
                          </td>
                          <td>
                            <div className="compare-method-cell">
                              {methodChips.length > 0 ? (
                                methodChips.map((chip) => (
                                  <span className="selection-chip" key={chip}>
                                    {chip}
                                  </span>
                                ))
                              ) : (
                                <span className="status-inline status-inline--muted">
                                  별도 방법론 변경 없음
                                </span>
                              )}
                            </div>
                          </td>
                          <td>
                            <div className="compare-date-cell">
                              <span>저장: {formatDateTime(row.updatedAt)}</span>
                              {row.latestRun ? (
                                <span>
                                  실행:{" "}
                                  {formatDateTime(
                                    row.latestRun.finished_at ??
                                      row.latestRun.started_at,
                                  )}
                                </span>
                              ) : (
                                <span className="status-inline status-inline--muted">
                                  실행 기록 없음
                                </span>
                              )}
                            </div>
                          </td>
                          <td>
                            {row.latestRun ? (
                              <span
                                className={`run-status run-status--${row.latestRun.status}`}
                              >
                                {formatRunStatus(row.latestRun.status)}
                              </span>
                            ) : (
                              <span className="status-inline status-inline--muted">
                                미실행
                              </span>
                            )}
                          </td>
                          {selectedMetricKeys.map((metricKey) => (
                            <td key={metricKey}>
                              {metricMap.has(metricKey) ? (
                                <strong>
                                  {formatMetricValue(metricMap.get(metricKey) ?? 0)}
                                </strong>
                              ) : (
                                <span className="status-inline status-inline--muted">
                                  -
                                </span>
                              )}
                            </td>
                          ))}
                          <td>
                            <div className="compare-action-row">
                              <button
                                type="button"
                                className="ghost-button ghost-button--small"
                                onClick={() => props.onLoadWorkspace(row.workspaceId)}
                              >
                                불러와 수정
                              </button>
                              <button
                                type="button"
                                className="ghost-button ghost-button--small"
                                onClick={() => props.onCloneWorkspace(row.workspaceId)}
                              >
                                복제 후 수정
                              </button>
                              <button
                                type="button"
                                className="primary-button primary-button--small"
                                onClick={() =>
                                  props.onRelaunchWorkspace(row.workspaceId)
                                }
                                disabled={props.rerunningWorkspaceId === row.workspaceId}
                              >
                                {props.rerunningWorkspaceId === row.workspaceId
                                  ? "재실행 중..."
                                  : "그대로 재실행"}
                              </button>
                              <button
                                type="button"
                                className="ghost-button ghost-button--small ghost-button--danger"
                                onClick={() => {
                                  if (
                                    window.confirm(
                                      `${row.workspaceId}를 비교 목록에서 삭제하시겠습니까?`,
                                    )
                                  ) {
                                    props.onDeleteWorkspace(row.workspaceId);
                                  }
                                }}
                                disabled={props.deletingWorkspaceId === row.workspaceId}
                              >
                                {props.deletingWorkspaceId === row.workspaceId
                                  ? "삭제 중..."
                                  : "삭제"}
                              </button>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </section>
          ))}
        </div>
      ) : (
        <div className="message-block">
          <h3>아직 저장된 실험이 없습니다</h3>
          <p>
            먼저 조합을 하나 저장하면, 이 영역에서 날짜별 기록과 결과 지표를
            비교할 수 있습니다.
          </p>
        </div>
      )}
    </div>
  );
}

function isMethodLikeSelection(
  selection: SavedWorkspaceSelectionPreviewPayload,
): boolean {
  return METHOD_SECTION_KEYWORDS.some((keyword) =>
    selection.section_name.includes(keyword),
  );
}

function formatSelectionChip(
  selection: SavedWorkspaceSelectionPreviewPayload,
): string {
  return `${formatSectionName(selection.section_name)}: ${selection.variant_profile_name}`;
}

function buildMethodChips(
  selections: SavedWorkspaceSelectionPreviewPayload[],
): string[] {
  const methodChips = selections
    .filter(isMethodLikeSelection)
    .map(formatSelectionChip);
  const overrideChips = selections
    .filter((selection) => selection.override_keys.length > 0)
    .map(
      (selection) =>
        `${formatSectionName(selection.section_name)} 값 조정 ` +
        `(${selection.override_keys.join(", ")})`,
    );
  return [...methodChips, ...overrideChips];
}
