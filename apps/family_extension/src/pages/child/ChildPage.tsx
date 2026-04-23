import { WellbeingSignalCard } from "../../components/WellbeingSignalCard";
import { useWellbeingSummary } from "../../hooks/useWellbeingSummary";

type ChildPageProps = {
  onMoveToUnlock: () => void;
  onMoveToParent: () => void;
};

export function ChildPage({
  onMoveToUnlock,
  onMoveToParent,
}: ChildPageProps) {
  const summaryState = useWellbeingSummary();

  return (
    <div className="page-stack">
      {summaryState.status === "loaded" && (
        <WellbeingSignalCard summary={summaryState.summary} />
      )}
      {summaryState.status === "loading" && (
        <section className="hero-card child-hero">
          <div>
            <p className="eyebrow">Child View</p>
            <h2>현재 상태를 불러오는 중</h2>
            <p className="section-copy">
              아이용 화면은 wellbeing summary 한 건만 먼저 읽습니다. 그래프나 상세
              이유는 아직 보여주지 않습니다.
            </p>
          </div>
          <div className="hero-meter">
            <span className="hero-meter-label">wellbeing signal</span>
            <strong>...</strong>
          </div>
        </section>
      )}
      {summaryState.status === "error" && (
        <section className="hero-card child-hero">
          <div>
            <p className="eyebrow">Child View</p>
            <h2>현재 상태를 아직 불러오지 못했어요</h2>
            <p className="section-copy">
              로컬 프로그램이 꺼져 있거나, 브라우저에서 agent API 접근이 아직
              허용되지 않았을 수 있습니다.
            </p>
          </div>
          <div className="hero-meter">
            <span className="hero-meter-label">연결 상태</span>
            <strong>{summaryState.errorMessage}</strong>
          </div>
        </section>
      )}

      <section className="card-grid">
        <article className="surface-card">
          <p className="card-label">아이에게 지금 보이는 정보</p>
          <ul className="bullet-list">
            <li>현재 상태 한 줄</li>
            <li>짧은 요약 문구</li>
            <li>짧은 행동 제안 1개</li>
            <li>마지막 업데이트 시각</li>
          </ul>
        </article>
        <article className="surface-card">
          <p className="card-label">이 단계에서 여전히 하지 않는 것</p>
          <ul className="bullet-list">
            <li>카테고리별 점수 공개</li>
            <li>복잡한 상세 이유 노출</li>
            <li>부모용 차트/기록 노출</li>
          </ul>
        </article>
      </section>

      {summaryState.status === "loaded" && (
        <section className="card-grid">
          <article className="surface-card">
            <p className="card-label">오늘 한 줄 요약</p>
            <p className="section-copy">{summaryState.summary.summary}</p>
          </article>
          <article className="surface-card">
            <p className="card-label">지금 해보면 좋은 것</p>
            <p className="section-copy">{summaryState.summary.action_tip}</p>
          </article>
        </section>
      )}

      <div className="button-row">
        <button className="primary-button" type="button" onClick={onMoveToUnlock}>
          부모용 PIN 화면 보기
        </button>
        <button className="ghost-button" type="button" onClick={onMoveToParent}>
          부모 상세 shell 미리 보기
        </button>
      </div>
    </div>
  );
}
