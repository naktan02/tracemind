import type { WellbeingSignalSummaryPayload } from "../../contracts/generated";
import {
  formatConfidenceLabel,
  formatSignalLevelLabel,
  formatTrendLabel,
} from "../lib/formatters";

type ParentWellbeingSummaryCardProps = {
  summary: WellbeingSignalSummaryPayload;
  statusItems?: string[];
};

export function ParentWellbeingSummaryCard({
  summary,
  statusItems = [],
}: ParentWellbeingSummaryCardProps) {
  return (
    <section className="hero-card parent-hero">
      <div className="parent-hero-main">
        <p className="eyebrow">보호자 안내</p>
        <h2>{summary.signal_label}</h2>
        <p className="section-copy">{summary.summary}</p>
        <div className="pill-row">
          <span className="info-pill strong">
            {formatSignalLevelLabel(summary.signal_level)}
          </span>
          {summary.trend !== "unknown" && (
            <span className="info-pill">{formatTrendLabel(summary.trend)}</span>
          )}
          {summary.confidence !== "low" && (
            <span className="info-pill">
              {formatConfidenceLabel(summary.confidence)}
            </span>
          )}
          {summary.low_data && <span className="info-pill">데이터가 아직 적어요</span>}
        </div>
      </div>
      {statusItems.length > 0 && (
        <aside className="parent-hero-status" aria-label="부모용 확인 상태">
          <p className="card-label">확인 상태</p>
          <ul className="compact-status-list">
            {statusItems.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </aside>
      )}
    </section>
  );
}
