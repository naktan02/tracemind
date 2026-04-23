import { PinPad } from "../../components/PinPad";

type UnlockPageProps = {
  pin: string;
  onPinChange: (nextValue: string) => void;
  onMoveToChild: () => void;
  onMoveToParent: () => void;
};

export function UnlockPage({
  pin,
  onPinChange,
  onMoveToChild,
  onMoveToParent,
}: UnlockPageProps) {
  return (
    <div className="page-stack">
      <section className="hero-card unlock-hero">
        <div>
          <p className="eyebrow">Protected Entry</p>
          <h2>부모용 PIN shell</h2>
          <p className="section-copy">
            이 화면은 child surface와 parent surface 사이 권한 경계를 분리하는
            자리입니다. 지금은 입력 shell만 두고, 실제 검증은 다음 단계에서
            `/api/v1/parent/unlock`에 연결합니다.
          </p>
        </div>
      </section>

      <section className="surface-card">
        <PinPad value={pin} onChange={onPinChange} />
      </section>

      <section className="card-grid">
        <article className="surface-card">
          <p className="card-label">다음 단계 연결 예정</p>
          <ul className="bullet-list">
            <li>PIN 검증 요청</li>
            <li>실패 횟수/잠금 상태 표시</li>
            <li>부모 세션 만료 시각 처리</li>
          </ul>
        </article>
        <article className="surface-card">
          <p className="card-label">현재 shell 목적</p>
          <ul className="bullet-list">
            <li>route 분리</li>
            <li>입력 컴포넌트 경계 고정</li>
            <li>popup과 parent detail 흐름 연결</li>
          </ul>
        </article>
      </section>

      <div className="button-row">
        <button className="ghost-button" type="button" onClick={onMoveToChild}>
          아이용 화면으로
        </button>
        <button className="primary-button" type="button" onClick={onMoveToParent}>
          부모 상세 shell 보기
        </button>
      </div>
    </div>
  );
}
