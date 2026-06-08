import { PinPad } from "../../components/PinPad";
import type { FamilyAccessRole } from "../../../contracts/generated";
import { formatComputedAtLabel } from "../../lib/formatters";
import type { FamilyUnlockState } from "../../hooks/useFamilyAccess";

type UnlockPageProps = {
  pin: string;
  pinLabel: string;
  role: FamilyAccessRole;
  unlockState: FamilyUnlockState;
  onPinChange: (nextValue: string) => void;
  onSubmitUnlock: () => void;
  onMoveToRoleSurface: () => void;
};

export function UnlockPage({
  pin,
  pinLabel,
  role,
  unlockState,
  onPinChange,
  onSubmitUnlock,
  onMoveToRoleSurface,
}: UnlockPageProps) {
  const isSubmitting = unlockState.phase === "submitting";
  const isGranted = unlockState.phase === "granted";
  const canSubmit = pin.length >= 4 && !isSubmitting;

  return (
    <div className="page-stack">
      <section className="hero-card unlock-hero">
        <div>
          <p className="eyebrow">PIN 확인</p>
          <h2>{pinLabel} 검증</h2>
          <p className="section-copy">
            선택한 화면으로 들어가기 위해 PIN을 확인합니다.
          </p>
        </div>
        <div className="hero-meter">
          <span className="hero-meter-label">현재 상태</span>
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
        <PinPad
          helpText={`확인 후 ${role === "child" ? "본인" : "부모"} 페이지로 이동합니다.`}
          inputId={`${role}-pin`}
          label={pinLabel}
          value={pin}
          onChange={onPinChange}
        />
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
              onClick={onMoveToRoleSurface}
            >
              {role === "child" ? "본인 페이지로 이동" : "보호자 안내로 이동"}
            </button>
          )}
        </div>
      </section>

      <section className="card-grid">
        <article className="surface-card">
          <p className="card-label">검증 결과</p>
          {unlockState.phase === "idle" && (
            <p className="section-copy">
              4~6자리 PIN을 입력하면 선택한 화면을 열 수 있는지 확인합니다.
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
                {role === "child" ? "본인" : "부모"} 세션이 열렸습니다.{" "}
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
          <p className="card-label">안내</p>
          <ul className="bullet-list">
            <li>PIN이 여러 번 틀리면 잠시 잠깁니다.</li>
            <li>자리를 비울 때는 화면을 닫아 주세요.</li>
            <li>부모 페이지에는 원문과 그래프가 표시되지 않습니다.</li>
          </ul>
        </article>
      </section>

      <div className="button-row">
        <button
          className="ghost-button"
          disabled={!isGranted}
          type="button"
          onClick={onMoveToRoleSurface}
        >
          {role === "child" ? "본인 페이지로 이동" : "보호자 안내로 이동"}
        </button>
      </div>
    </div>
  );
}
