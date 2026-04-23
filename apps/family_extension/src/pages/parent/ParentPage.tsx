import { useState } from "react";

import { ParentWellbeingSummaryCard } from "../../components/ParentWellbeingSummaryCard";
import { WellbeingSignalTrendChart } from "../../components/WellbeingSignalTrendChart";
import { useWellbeingSummary } from "../../hooks/useWellbeingSummary";
import { useWellbeingTimeseries } from "../../hooks/useWellbeingTimeseries";
import { formatComputedAtLabel } from "../../lib/formatters";

type ParentPageProps = {
  activeSessionExpiresAt: string | null;
  hasActiveParentSession: boolean;
  onMoveToChild: () => void;
  onMoveToUnlock: () => void;
};

export function ParentPage({
  activeSessionExpiresAt,
  hasActiveParentSession,
  onMoveToChild,
  onMoveToUnlock,
}: ParentPageProps) {
  const [selectedRange, setSelectedRange] = useState<"7d" | "14d" | "30d">("7d");
  const summaryState = useWellbeingSummary({ enabled: hasActiveParentSession });
  const timeseriesState = useWellbeingTimeseries({
    enabled: hasActiveParentSession,
    requestedRange: selectedRange,
  });

  if (!hasActiveParentSession) {
    return (
      <div className="page-stack">
        <section className="hero-card parent-hero">
          <div>
            <p className="eyebrow">Protected Parent Detail</p>
            <h2>부모용 상세 화면은 아직 잠겨 있습니다</h2>
            <p className="section-copy">
              이 화면은 PIN 검증을 통과한 세션에서만 열립니다. 먼저 부모용 PIN
              화면에서 인증을 완료해 주세요.
            </p>
          </div>
          <div className="hero-meter hero-meter-wide">
            <span className="hero-meter-label">access state</span>
            <strong>잠금 상태</strong>
          </div>
        </section>

        <section className="card-grid">
          <article className="surface-card">
            <p className="card-label">왜 막아두는가</p>
            <ul className="bullet-list">
              <li>부모용 화면은 아이용보다 더 많은 설명과 추이를 보여줍니다.</li>
              <li>확장 프로그램 안에서도 권한 경계를 먼저 분명히 유지합니다.</li>
            </ul>
          </article>
          <article className="surface-card">
            <p className="card-label">다음 단계에서 여기에 붙는 것</p>
            <ul className="bullet-list">
              <li>현재 상태 카드</li>
              <li>7d / 14d / 30d 추이 그래프</li>
              <li>권장 행동과 데이터 부족 표시</li>
            </ul>
          </article>
        </section>

        <div className="button-row">
          <button
            className="primary-button"
            type="button"
            onClick={onMoveToUnlock}
          >
            부모용 PIN 화면으로
          </button>
          <button className="ghost-button" type="button" onClick={onMoveToChild}>
            아이용 화면 보기
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="page-stack">
      {summaryState.status === "loaded" && (
        <ParentWellbeingSummaryCard summary={summaryState.summary} />
      )}
      {summaryState.status === "loading" && (
        <section className="hero-card parent-hero">
          <div>
            <p className="eyebrow">Parent Detail</p>
            <h2>부모용 현재 상태를 불러오는 중</h2>
            <p className="section-copy">
              현재 전체 상태와 권장 행동을 로컬 프로그램에서 읽고 있습니다.
            </p>
            {activeSessionExpiresAt != null && (
              <p className="section-copy">
                현재 부모 세션은{" "}
                {formatComputedAtLabel(activeSessionExpiresAt)}까지 유지됩니다.
              </p>
            )}
          </div>
          <div className="hero-meter">
            <span className="hero-meter-label">current summary</span>
            <strong>...</strong>
          </div>
        </section>
      )}
      {summaryState.status === "error" && (
        <section className="hero-card parent-hero">
          <div>
            <p className="eyebrow">Parent Detail</p>
            <h2>부모용 현재 상태를 아직 불러오지 못했습니다</h2>
            <p className="section-copy">{summaryState.errorMessage}</p>
          </div>
          <div className="hero-meter">
            <span className="hero-meter-label">summary 상태</span>
            <strong>요청 실패</strong>
          </div>
        </section>
      )}

      {timeseriesState.status === "loaded" && (
        <WellbeingSignalTrendChart
          activeRange={selectedRange}
          onRangeChange={setSelectedRange}
          timeseries={timeseriesState.timeseries}
        />
      )}
      {timeseriesState.status === "loading" && (
        <section className="surface-card trend-card">
          <div className="trend-card-header">
            <div>
              <p className="card-label">최근 변화</p>
              <p className="section-copy">
                최근 wellbeing signal 추이를 불러오는 중입니다.
              </p>
            </div>
          </div>
          <div className="trend-chart-loading">그래프 데이터를 준비하고 있습니다.</div>
        </section>
      )}
      {timeseriesState.status === "error" && (
        <section className="surface-card trend-card">
          <div className="trend-card-header">
            <div>
              <p className="card-label">최근 변화</p>
              <p className="section-copy">{timeseriesState.errorMessage}</p>
            </div>
          </div>
        </section>
      )}

      {summaryState.status === "loaded" && (
        <section className="card-grid">
          <article className="surface-card">
            <p className="card-label">권장 행동</p>
            <p className="section-copy">{summaryState.summary.action_tip}</p>
          </article>
          <article className="surface-card">
            <p className="card-label">현재 부모 세션</p>
            <ul className="bullet-list">
              <li>
                만료 시각:{" "}
                {activeSessionExpiresAt == null
                  ? "확인할 수 없음"
                  : formatComputedAtLabel(activeSessionExpiresAt)}
              </li>
              <li>
                데이터 상태:{" "}
                {summaryState.summary.low_data
                  ? "데이터가 아직 적습니다"
                  : "기본 상태 해석 가능"}
              </li>
            </ul>
          </article>
        </section>
      )}

      <div className="button-row">
        <button className="ghost-button" type="button" onClick={onMoveToUnlock}>
          PIN 화면으로 돌아가기
        </button>
        <button className="ghost-button" type="button" onClick={onMoveToChild}>
          아이용 화면 보기
        </button>
      </div>
    </div>
  );
}
