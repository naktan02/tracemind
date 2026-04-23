type ParentPageProps = {
  onMoveToChild: () => void;
  onMoveToUnlock: () => void;
};

export function ParentPage({
  onMoveToChild,
  onMoveToUnlock,
}: ParentPageProps) {
  return (
    <div className="page-stack">
      <section className="hero-card parent-hero">
        <div>
          <p className="eyebrow">Parent Detail Entry</p>
          <h2>부모용 상세 shell</h2>
          <p className="section-copy">
            다음 단계에서 전체 wellbeing signal, 최근 추이 그래프, 권장 행동이 이
            화면에 연결됩니다. 현재는 보호자용 정보 밀도와 child 화면 분리를
            먼저 고정합니다.
          </p>
        </div>
        <div className="hero-meter hero-meter-wide">
          <span className="hero-meter-label">future 7d / 14d / 30d trend</span>
          <strong>라인 차트 자리</strong>
        </div>
      </section>

      <section className="card-grid">
        <article className="surface-card">
          <p className="card-label">여기에 올라갈 정보</p>
          <ul className="bullet-list">
            <li>전체 상태 카드</li>
            <li>단일 축 signal 추이 그래프</li>
            <li>권장 행동과 데이터 부족 여부</li>
          </ul>
        </article>
        <article className="surface-card">
          <p className="card-label">여기서 숨길 정보</p>
          <ul className="bullet-list">
            <li>내부 카테고리별 세부 점수</li>
            <li>실험용 방법론 이름</li>
            <li>모델/학습 metadata</li>
          </ul>
        </article>
      </section>

      <div className="button-row">
        <button className="ghost-button" type="button" onClick={onMoveToUnlock}>
          PIN shell로 돌아가기
        </button>
        <button className="ghost-button" type="button" onClick={onMoveToChild}>
          아이용 화면 보기
        </button>
      </div>
    </div>
  );
}
