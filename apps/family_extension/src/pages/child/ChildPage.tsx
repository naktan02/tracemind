type ChildPageProps = {
  onMoveToUnlock: () => void;
  onMoveToParent: () => void;
};

export function ChildPage({
  onMoveToUnlock,
  onMoveToParent,
}: ChildPageProps) {
  return (
    <div className="page-stack">
      <section className="hero-card child-hero">
        <div>
          <p className="eyebrow">Popup Entry</p>
          <h2>아이용 상태 화면 shell</h2>
          <p className="section-copy">
            다음 단계에서 현재 상태, 짧은 설명, 행동 제안이 이 화면에 연결됩니다.
            지금은 아이용 flow와 부모용 보호 화면 진입 경계를 먼저 고정합니다.
          </p>
        </div>
        <div className="hero-meter">
          <span className="hero-meter-label">future wellbeing signal</span>
          <strong>--</strong>
        </div>
      </section>

      <section className="card-grid">
        <article className="surface-card">
          <p className="card-label">아이에게 보이는 정보</p>
          <ul className="bullet-list">
            <li>현재 상태 한 줄</li>
            <li>짧은 행동 제안 1개</li>
            <li>마지막 업데이트 시각</li>
          </ul>
        </article>
        <article className="surface-card">
          <p className="card-label">이 단계에서 하지 않는 것</p>
          <ul className="bullet-list">
            <li>카테고리별 점수 공개</li>
            <li>복잡한 상세 이유 노출</li>
            <li>부모용 차트/기록 노출</li>
          </ul>
        </article>
      </section>

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
