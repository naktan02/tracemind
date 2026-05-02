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
          <p className="eyebrow">Family Gate</p>
          <h2>누가 들어가나요?</h2>
          <p className="section-copy">
            이 확장 프로그램은 이 PC의 로컬 agent만 사용합니다. 아이용 화면과
            부모용 화면은 둘 다 잠금 경계를 거쳐야 하고, 내부 wellbeing 결과는
            role에 맞는 surface에서만 보입니다.
          </p>
        </div>
        <div className="hero-meter">
          <span className="hero-meter-label">연결 모드</span>
          <strong>이 PC의 로컬 agent만 사용</strong>
        </div>
      </section>

      <section className="card-grid">
        <article className="surface-card role-card">
          <p className="card-label">아이용 화면</p>
          <p className="section-copy">
            현재 상태, 짧은 요약, 행동 제안 한 줄만 보여줍니다.
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
            현재 상태, 최근 추이, 권장 행동을 보호된 상세 화면으로 보여줍니다.
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
