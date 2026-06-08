import { useMemo, useState } from "react";

type SetupPageProps = {
  isSubmitting: boolean;
  errorMessage: string | null;
  onSubmitSetup: (childPin: string, parentPin: string) => Promise<void> | void;
};

export function SetupPage({
  isSubmitting,
  errorMessage,
  onSubmitSetup,
}: SetupPageProps) {
  const [childPin, setChildPin] = useState("");
  const [childPinConfirm, setChildPinConfirm] = useState("");
  const [parentPin, setParentPin] = useState("");
  const [parentPinConfirm, setParentPinConfirm] = useState("");
  const [localError, setLocalError] = useState<string | null>(null);

  const canSubmit = useMemo(
    () =>
      childPin.length >= 4 &&
      childPin === childPinConfirm &&
      parentPin.length >= 4 &&
      parentPin === parentPinConfirm &&
      !isSubmitting,
    [childPin, childPinConfirm, isSubmitting, parentPin, parentPinConfirm],
  );

  async function handleSubmit() {
    if (childPin !== childPinConfirm) {
      setLocalError("본인 PIN 확인이 일치하지 않습니다.");
      return;
    }
    if (parentPin !== parentPinConfirm) {
      setLocalError("부모용 PIN 확인이 일치하지 않습니다.");
      return;
    }
    setLocalError(null);
    await onSubmitSetup(childPin, parentPin);
  }

  return (
    <div className="page-stack">
      <section className="hero-card setup-hero">
        <div>
          <p className="eyebrow">처음 설정</p>
          <h2>처음 사용할 PIN을 설정합니다</h2>
          <p className="section-copy">
            본인 페이지와 부모 페이지는 서로 다른 PIN으로 열립니다.
          </p>
        </div>
        <div className="hero-meter">
          <span className="hero-meter-label">보호 방식</span>
          <strong>화면별 PIN 분리</strong>
        </div>
      </section>

      <section className="card-grid">
        <article className="surface-card setup-card">
          <p className="card-label">본인 PIN</p>
          <PinInput
            id="child-pin"
            label="본인 PIN"
            value={childPin}
            onChange={setChildPin}
          />
          <PinInput
            id="child-pin-confirm"
            label="본인 PIN 확인"
            value={childPinConfirm}
            onChange={setChildPinConfirm}
          />
        </article>
        <article className="surface-card setup-card">
          <p className="card-label">부모용 PIN</p>
          <PinInput
            id="parent-pin"
            label="부모용 PIN"
            value={parentPin}
            onChange={setParentPin}
          />
          <PinInput
            id="parent-pin-confirm"
            label="부모용 PIN 확인"
            value={parentPinConfirm}
            onChange={setParentPinConfirm}
          />
        </article>
      </section>

      {(localError != null || errorMessage != null) && (
        <section className="status-banner warning">
          <div>
            <p className="card-label">설정 상태</p>
            <p className="section-copy">{localError ?? errorMessage}</p>
          </div>
        </section>
      )}

      <div className="button-row">
        <button
          className="primary-button"
          disabled={!canSubmit}
          type="button"
          onClick={handleSubmit}
        >
          {isSubmitting ? "저장 중..." : "초기 설정 저장"}
        </button>
      </div>
    </div>
  );
}

type PinInputProps = {
  id: string;
  label: string;
  value: string;
  onChange: (nextValue: string) => void;
};

function PinInput({ id, label, value, onChange }: PinInputProps) {
  return (
    <label className="pin-shell" htmlFor={id}>
      <span className="pin-label">{label}</span>
      <input
        id={id}
        className="pin-input"
        inputMode="numeric"
        maxLength={6}
        minLength={4}
        pattern="[0-9]*"
        placeholder="4~6자리 숫자"
        value={value}
        onChange={(event) =>
          onChange(event.target.value.replace(/[^0-9]/g, "").slice(0, 6))
        }
      />
      <span className="pin-help">4~6자리 숫자만 사용할 수 있습니다.</span>
    </label>
  );
}
