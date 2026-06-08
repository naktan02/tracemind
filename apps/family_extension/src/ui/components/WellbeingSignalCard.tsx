import type { WellbeingSignalSummaryPayload } from "../../contracts/generated";
import {
  formatComputedAtLabel,
  formatConfidenceLabel,
  formatSignalLevelLabel,
  formatTrendLabel,
} from "../lib/formatters";

type WellbeingSignalCardProps = {
  summary: WellbeingSignalSummaryPayload;
};

export function WellbeingSignalCard({ summary }: WellbeingSignalCardProps) {
  return (
    <section className="hero-card child-hero">
      <div>
        <p className="eyebrow">내 마음 상태</p>
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
        <span className="hero-meter-label">현재 위험도</span>
        <strong>{Math.round(summary.signal_score)}</strong>
        <span className="status-hint">
          업데이트 {formatComputedAtLabel(summary.computed_at)}
        </span>
      </div>
    </section>
  );
}
