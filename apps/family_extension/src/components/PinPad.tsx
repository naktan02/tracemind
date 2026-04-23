type PinPadProps = {
  value: string;
  onChange: (nextValue: string) => void;
};

export function PinPad({ value, onChange }: PinPadProps) {
  return (
    <div className="pin-shell">
      <label className="pin-label" htmlFor="parent-pin">
        부모용 PIN
      </label>
      <input
        id="parent-pin"
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
      <p className="pin-help">
        shell 단계에서는 입력만 확인합니다. 실제 unlock API 연결은 다음 단계에서
        붙입니다.
      </p>
    </div>
  );
}
