import { PinPad } from "../../components/PinPad";
import { formatComputedAtLabel } from "../../lib/formatters";
import type { ParentUnlockState } from "../../hooks/useParentUnlock";

type UnlockPageProps = {
  pin: string;
  unlockState: ParentUnlockState;
  onPinChange: (nextValue: string) => void;
  onSubmitUnlock: () => void;
  onMoveToChild: () => void;
  onMoveToParent: () => void;
};

export function UnlockPage({
  pin,
  unlockState,
  onPinChange,
  onSubmitUnlock,
  onMoveToChild,
  onMoveToParent,
}: UnlockPageProps) {
  const isSubmitting = unlockState.phase === "submitting";
  const isGranted = unlockState.phase === "granted";
  const canSubmit = pin.length >= 4 && !isSubmitting;

  return (
    <div className="page-stack">
      <section className="hero-card unlock-hero">
        <div>
          <p className="eyebrow">Protected Entry</p>
          <h2>부모용 PIN 검증</h2>
          <p className="section-copy">
            이 화면은 child surface와 parent surface 사이 권한 경계를 분리하는
            실제 진입 화면입니다. PIN 검증 결과와 잠금 상태를 여기서 먼저
            확인합니다.
          </p>
        </div>
        <div className="hero-meter">
          <span className="hero-meter-label">현재 unlock 상태</span>
          <strong>
            {unlockState.phase === "idle" && "입력 대기"}
            {unlockState.phase === "submitting" && "확인 중"}
            {unlockState.phase === "granted" && "접근 허용"}
            {unlockState.phase === "rejected" && "PIN 불일치"}
            {unlockState.phase === "locked" && "일시 잠금"}
            {unlockState.phase === "error" && "요청 실패"}
          </strong>
        </div>
      </section>

      <section className="surface-card">
        <PinPad value={pin} onChange={onPinChange} />
        <div className="button-row">
          <button
            className="primary-button"
            disabled={!canSubmit}
            type="button"
            onClick={onSubmitUnlock}
          >
            {isSubmitting ? "확인 중..." : "PIN 확인하기"}
          </button>
          {isGranted && (
            <button
              className="ghost-button"
              type="button"
              onClick={onMoveToParent}
            >
              부모 상세로 이동
            </button>
          )}
        </div>
      </section>

      <section className="card-grid">
        <article className="surface-card">
          <p className="card-label">검증 결과</p>
          {unlockState.phase === "idle" && (
            <p className="section-copy">
              4~6자리 PIN을 입력하면 부모 상세 화면 진입 가능 여부를 확인합니다.
            </p>
          )}
          {unlockState.phase === "submitting" && (
            <p className="section-copy">
              로컬 프로그램에 PIN 검증을 요청하고 있습니다.
            </p>
          )}
          {unlockState.phase === "granted" &&
            unlockState.response?.session_expires_at != null && (
              <p className="section-copy">
                부모 세션이 열렸습니다.{" "}
                {formatComputedAtLabel(unlockState.response.session_expires_at)}까지
                같은 세션을 사용할 수 있습니다.
              </p>
            )}
          {unlockState.phase === "rejected" && (
            <p className="section-copy">
              PIN이 맞지 않습니다. 남은 시도 횟수:{" "}
              {unlockState.response?.remaining_attempts ?? "-"}
            </p>
          )}
          {unlockState.phase === "locked" &&
            unlockState.response?.locked_until != null && (
              <p className="section-copy">
                너무 많은 시도로 잠시 잠겼습니다.{" "}
                {formatComputedAtLabel(unlockState.response.locked_until)} 이후에 다시
                시도해 주세요.
              </p>
            )}
          {unlockState.phase === "error" && (
            <p className="section-copy">{unlockState.errorMessage}</p>
          )}
        </article>
        <article className="surface-card">
          <p className="card-label">이 단계에서 확보한 것</p>
          <ul className="bullet-list">
            <li>실제 unlock API 연결</li>
            <li>실패 횟수와 잠금 상태 해석</li>
            <li>부모 세션 응답 기반 route guard 준비</li>
          </ul>
        </article>
      </section>

      <div className="button-row">
        <button className="ghost-button" type="button" onClick={onMoveToChild}>
          아이용 화면으로
        </button>
        <button
          className="ghost-button"
          disabled={!isGranted}
          type="button"
          onClick={onMoveToParent}
        >
          부모 상세로 이동
        </button>
      </div>
    </div>
  );
}
