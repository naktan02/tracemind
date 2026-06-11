import type { WellbeingSignalSummaryPayload } from "../../contracts/generated";
import {
  formatComputedAtLabel,
  formatConfidenceLabel,
  formatSignalLevelLabel,
  formatTrendLabel,
} from "../lib/formatters";

type ParentWellbeingSummaryCardProps = {
  summary: WellbeingSignalSummaryPayload;
};

export function ParentWellbeingSummaryCard({
  summary,
}: ParentWellbeingSummaryCardProps) {
  return (
    <section className="hero-card parent-hero">
      <div>
        <p className="eyebrow">보호자 안내</p>
        <h2>{summary.signal_label}</h2>
        <p className="section-copy">{summary.summary}</p>
        <div className="pill-row">
          <span className="info-pill strong">
            {formatSignalLevelLabel(summary.signal_level)}
          </span>
          <span className="info-pill">{formatTrendLabel(summary.trend)}</span>
          <span className="info-pill">
            {formatConfidenceLabel(summary.confidence)}
          </span>
          {summary.low_data && <span className="info-pill">데이터가 아직 적어요</span>}
        </div>
      </div>

      <div className="hero-meter">
        <span className="hero-meter-label">마지막 업데이트</span>
        <strong>{formatComputedAtLabel(summary.computed_at)}</strong>
        <span className="status-hint">
          부모 페이지는 필요한 대응 방향만 간단히 안내합니다.
        </span>
      </div>
    </section>
  );
}
