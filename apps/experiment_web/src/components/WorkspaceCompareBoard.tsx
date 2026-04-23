import { useEffect, useMemo, useState } from "react";

import {
  formatDateTime,
  formatMetricKey,
  formatMetricValue,
} from "../lib/formatters";
import type {
  ExperimentRunPayload,
  SavedWorkspaceSummaryPayload,
} from "../types";

interface CompareRow {
  workspaceId: string;
  manifestId: string;
  trackName: string;
  entrypointName: string;
  updatedAt: string;
  latestRun: ExperimentRunPayload | null;
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

export function WorkspaceCompareBoard(props: {
  currentWorkspaceId: string | null;
  savedWorkspaces: SavedWorkspaceSummaryPayload[];
  runs: ExperimentRunPayload[];
  rerunningWorkspaceId: string | null;
  onLoadWorkspace: (workspaceId: string) => void;
  onRelaunchWorkspace: (workspaceId: string) => void;
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
      }));
  }, [props.runs, props.savedWorkspaces]);

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
          <p className="panel-kicker">Compare</p>
          <h3>Latest workspace results</h3>
        </div>
        <span className="compare-count">{compareRows.length} workspaces</span>
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
          완료된 run 결과가 아직 없어 비교 metric을 고를 수 없습니다.
        </p>
      )}

      <div className="compare-table-wrap">
        <table className="compare-table">
          <thead>
            <tr>
              <th>Workspace</th>
              <th>Status</th>
              {selectedMetricKeys.map((metricKey) => (
                <th key={metricKey}>{formatMetricKey(metricKey)}</th>
              ))}
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {compareRows.length > 0 ? (
              compareRows.map((row) => {
                const metricMap = new Map(
                  (row.latestRun?.result_summary?.metrics ?? []).map((metric) => [
                    metric.metric_key,
                    metric.value,
                  ]),
                );
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
                          {row.trackName} / {row.entrypointName}
                        </span>
                        <span>updated: {formatDateTime(row.updatedAt)}</span>
                        <code>manifest: {row.manifestId}</code>
                      </div>
                    </td>
                    <td>
                      {row.latestRun ? (
                        <span
                          className={`run-status run-status--${row.latestRun.status}`}
                        >
                          {row.latestRun.status}
                        </span>
                      ) : (
                        <span className="status-inline status-inline--muted">
                          no runs
                        </span>
                      )}
                    </td>
                    {selectedMetricKeys.map((metricKey) => (
                      <td key={metricKey}>
                        {metricMap.has(metricKey) ? (
                          <strong>{formatMetricValue(metricMap.get(metricKey) ?? 0)}</strong>
                        ) : (
                          <span className="status-inline status-inline--muted">-</span>
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
                          Open
                        </button>
                        <button
                          type="button"
                          className="primary-button primary-button--small"
                          onClick={() => props.onRelaunchWorkspace(row.workspaceId)}
                          disabled={props.rerunningWorkspaceId === row.workspaceId}
                        >
                          {props.rerunningWorkspaceId === row.workspaceId
                            ? "Rerunning..."
                            : "Rerun"}
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })
            ) : (
              <tr>
                <td colSpan={selectedMetricKeys.length + 3}>
                  <p className="hint-text">
                    저장된 workspace가 아직 없어 latest compare board를 만들 수
                    없습니다.
                  </p>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
