import { ParentWellbeingSummaryCard } from "../../components/ParentWellbeingSummaryCard";
import { WellbeingDataNotice } from "../../components/WellbeingDataNotice";
import { useWellbeingSummary } from "../../hooks/useWellbeingSummary";
import { formatComputedAtLabel } from "../../lib/formatters";

type ParentPageProps = {
  activeSessionExpiresAt: string | null;
  onMoveToChildUnlock: () => void;
  onMoveToGate: () => void;
};

export function ParentPage({
  activeSessionExpiresAt,
  onMoveToChildUnlock,
  onMoveToGate,
}: ParentPageProps) {
  const summaryState = useWellbeingSummary({ enabled: true });

  return (
    <div className="page-stack">
      {summaryState.status === "loaded" && (
        <ParentWellbeingSummaryCard summary={summaryState.summary} />
      )}
      {summaryState.status === "loaded" && (
        <WellbeingDataNotice summary={summaryState.summary} />
      )}
      {summaryState.status === "loading" && (
        <section className="hero-card parent-hero">
          <div>
            <p className="eyebrow">보호자 안내</p>
            <h2>부모용 현재 상태를 불러오는 중</h2>
            <p className="section-copy">
              아이에게 필요한 관심 수준과 대응 방향을 확인하고 있습니다.
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
            <p className="section-copy">{summaryState.summary.action_tip}</p>
          </article>
          <article className="surface-card">
            <p className="card-label">대화 방향</p>
            <p className="section-copy">
              아이가 쓴 문장을 캐묻기보다, 오늘 가장 힘들었던 순간과 지금 필요한
              도움을 먼저 물어보세요.
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
            </ul>
          </article>
        </section>
      )}

      <div className="button-row">
        <button className="ghost-button" type="button" onClick={onMoveToGate}>
          선택 화면으로 돌아가기
        </button>
        <button
          className="ghost-button"
          type="button"
          onClick={onMoveToChildUnlock}
        >
          아이용 화면으로 전환
        </button>
      </div>
    </div>
  );
}
