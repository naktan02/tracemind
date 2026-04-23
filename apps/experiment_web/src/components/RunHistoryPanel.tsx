import { buildExperimentRunLogUrl } from "../api";
import {
  formatDateTime,
  formatEntrypointName,
  formatMetricKey,
  formatMetricValue,
  formatRunStatus,
  formatTrackName,
} from "../lib/formatters";
import type { ExperimentRunPayload } from "../types";

export function RunHistoryPanel(props: {
  apiBaseUrl: string;
  runs: ExperimentRunPayload[];
  runsError: string | null;
  isRunsLoading: boolean;
  onRefresh: () => void;
}) {
  return (
    <div className="subpanel subpanel--runs">
      <div className="subpanel__header">
        <div>
          <p className="panel-kicker">실행 기록</p>
          <h3>최근 실행</h3>
        </div>
        <button
          type="button"
          className="ghost-button ghost-button--small"
          onClick={props.onRefresh}
          disabled={props.isRunsLoading}
        >
          {props.isRunsLoading ? "새로고침 중..." : "새로고침"}
        </button>
      </div>

      {props.runsError ? (
        <div className="message-block message-block--error">
          <h3>실행 목록을 불러오지 못했습니다</h3>
          <p>{props.runsError}</p>
        </div>
      ) : null}

      <div className="run-list">
        {props.runs.length > 0 ? (
          props.runs.map((run) => (
            <article className="run-card" key={run.run_id}>
              <div className="run-card__header">
                <div>
                  <strong>{run.run_id}</strong>
                  <p>
                    {formatTrackName(run.track_name)} /{" "}
                    {formatEntrypointName(run.entrypoint_name)}
                  </p>
                </div>
                <span className={`run-status run-status--${run.status}`}>
                  {formatRunStatus(run.status)}
                </span>
              </div>
              <div className="run-card__body">
                <span>시작: {formatDateTime(run.started_at)}</span>
                {run.finished_at ? (
                  <span>종료: {formatDateTime(run.finished_at)}</span>
                ) : null}
                {run.workspace_id ? (
                  <code>저장된 조합: {run.workspace_id}</code>
                ) : (
                  <span>저장 전 초안에서 바로 실행</span>
                )}
                <code>산출물 경로: {run.artifact_root_path}</code>
                {run.error_message ? (
                  <p className="run-card__error">{run.error_message}</p>
                ) : null}
              </div>
              {run.result_summary && run.result_summary.metrics.length > 0 ? (
                <div className="run-metric-row">
                  {run.result_summary.metrics.slice(0, 3).map((metric) => (
                    <span className="run-metric-chip" key={metric.metric_key}>
                      {formatMetricKey(metric.metric_key)}{" "}
                      {formatMetricValue(metric.value)}
                    </span>
                  ))}
                </div>
              ) : null}
              <div className="run-link-row">
                <a
                  href={buildExperimentRunLogUrl(
                    props.apiBaseUrl,
                    run.run_id,
                    "stdout",
                  )}
                  target="_blank"
                  rel="noreferrer"
                >
                  stdout 로그
                </a>
                <a
                  href={buildExperimentRunLogUrl(
                    props.apiBaseUrl,
                    run.run_id,
                    "stderr",
                  )}
                  target="_blank"
                  rel="noreferrer"
                >
                  stderr 로그
                </a>
              </div>
            </article>
          ))
        ) : (
          <p className="hint-text">아직 실행한 실험이 없습니다.</p>
        )}
      </div>
    </div>
  );
}
