type GatePageProps = {
  onMoveToChildUnlock: () => void;
  onMoveToParentUnlock: () => void;
};

export function GatePage({
  onMoveToChildUnlock,
  onMoveToParentUnlock,
}: GatePageProps) {
  return (
    <div className="page-stack">
      <section className="hero-card gate-hero">
        <div>
          <p className="eyebrow">화면 선택</p>
          <h2>누가 들어가나요?</h2>
          <p className="section-copy">
            아이와 보호자는 서로 다른 화면을 사용합니다. 각 화면은 PIN 확인 후
            열립니다.
          </p>
        </div>
        <div className="hero-meter">
          <span className="hero-meter-label">개인정보 보호</span>
          <strong>이 기기에서만 확인</strong>
        </div>
      </section>

      <section className="card-grid">
        <article className="surface-card role-card">
          <p className="card-label">아이용 화면</p>
          <p className="section-copy">
            현재 상태, 위험도 변화 그래프, AI 마음 도움을 확인합니다.
          </p>
          <button
            className="primary-button"
            type="button"
            onClick={onMoveToChildUnlock}
          >
            아이용 PIN 입력
          </button>
        </article>
        <article className="surface-card role-card">
          <p className="card-label">부모용 화면</p>
          <p className="section-copy">
            아이에게 필요한 관심 수준과 대응 방향만 확인합니다.
          </p>
          <button
            className="primary-button"
            type="button"
            onClick={onMoveToParentUnlock}
          >
            부모용 PIN 입력
          </button>
        </article>
      </section>
    </div>
  );
}
