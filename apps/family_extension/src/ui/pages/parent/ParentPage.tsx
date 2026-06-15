import { ParentWellbeingSummaryCard } from "../../components/ParentWellbeingSummaryCard";
import { useWellbeingSummary } from "../../hooks/useWellbeingSummary";
import {
  formatComputedAtLabel,
  formatConfidenceLabel,
} from "../../lib/formatters";

type ParentPageProps = {
  activeSessionExpiresAt: string | null;
};

export function ParentPage({ activeSessionExpiresAt }: ParentPageProps) {
  const summaryState = useWellbeingSummary({ enabled: true });

  return (
    <div className="page-stack">
      {summaryState.status === "loaded" && (
        <ParentWellbeingSummaryCard summary={summaryState.summary} />
      )}
      {summaryState.status === "loading" && (
        <section className="hero-card parent-hero">
          <div>
            <p className="eyebrow">보호자 안내</p>
            <h2>부모용 현재 상태를 불러오는 중</h2>
            <p className="section-copy">
              지금 필요한 관심 수준과 대응 방향을 확인하고 있습니다.
            </p>
            {activeSessionExpiresAt != null && (
              <p className="section-copy">
                현재 부모 세션은{" "}
                {formatComputedAtLabel(activeSessionExpiresAt)}까지 유지됩니다.
              </p>
            )}
          </div>
          <div className="hero-meter">
            <span className="hero-meter-label">상태 확인</span>
            <strong>...</strong>
          </div>
        </section>
      )}
      {summaryState.status === "error" && (
        <section className="hero-card parent-hero">
          <div>
            <p className="eyebrow">보호자 안내</p>
            <h2>부모용 현재 상태를 아직 불러오지 못했습니다</h2>
            <p className="section-copy">{summaryState.errorMessage}</p>
          </div>
          <div className="hero-meter">
            <span className="hero-meter-label">요약 상태</span>
            <strong>요청 실패</strong>
          </div>
        </section>
      )}

      {summaryState.status === "loaded" && (
        <section className="parent-guidance">
          <article className="surface-card">
            <p className="card-label">지금 필요한 대응</p>
            <p className="section-copy">
              {summaryState.summary.parent_guidance.response_priority}
            </p>
          </article>
          <article className="surface-card">
            <p className="card-label">대화 시작</p>
            <p className="section-copy">
              {summaryState.summary.parent_guidance.conversation_starter}
            </p>
          </article>
          <article className="surface-card">
            <p className="card-label">주의할 점</p>
            <p className="section-copy">
              {summaryState.summary.parent_guidance.caution_note}
            </p>
          </article>
          <article className="surface-card">
            <p className="card-label">확인 상태</p>
            <ul className="bullet-list">
              <li>
                보호자 세션 만료:{" "}
                {activeSessionExpiresAt == null
                  ? "확인할 수 없음"
                  : formatComputedAtLabel(activeSessionExpiresAt)}
              </li>
              <li>
                데이터 상태:{" "}
                {summaryState.summary.low_data
                  ? "데이터가 아직 적습니다"
                  : "기본 상태 해석 가능"}
              </li>
              <li>
                신뢰도: {formatConfidenceLabel(summaryState.summary.confidence)}
              </li>
            </ul>
          </article>
        </section>
      )}
    </div>
  );
}
