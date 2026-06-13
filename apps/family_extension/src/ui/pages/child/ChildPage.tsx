import { useEffect, useState } from "react";

import type {
  ChildSupportProactivePromptPayload,
  WellbeingSignalRange,
} from "../../../contracts/generated";
import { getChildSupportProactivePrompt } from "../../api/childSupport";
import { ChildSupportCoachPanel } from "../../components/ChildSupportCoachPanel";
import { WellbeingSpaceWebGraph } from "../../components/WellbeingSpaceWebGraph";
import { WellbeingSignalCard } from "../../components/WellbeingSignalCard";
import { WellbeingSignalTrendChart } from "../../components/WellbeingSignalTrendChart";
import { useWellbeingSpaceWeb } from "../../hooks/useWellbeingSpaceWeb";
import { useWellbeingSummary } from "../../hooks/useWellbeingSummary";
import { useWellbeingTimeseries } from "../../hooks/useWellbeingTimeseries";

type ChildPageProps = {
  activeTab: ChildTab;
  onOpenCoach?: () => void;
};

export type ChildTab = "ai" | "analysis" | "checkin";
const PROACTIVE_PROMPT_SCORE_THRESHOLD = 35;

export function ChildPage({ activeTab, onOpenCoach }: ChildPageProps) {
  const [selectedRange, setSelectedRange] = useState<WellbeingSignalRange>("7d");
  const [proactivePrompt, setProactivePrompt] =
    useState<ChildSupportProactivePromptPayload | null>(null);
  const [isProactivePromptDismissed, setIsProactivePromptDismissed] =
    useState(false);
  const summaryState = useWellbeingSummary();
  const timeseriesState = useWellbeingTimeseries({
    enabled: activeTab === "analysis",
    requestedRange: selectedRange,
  });
  const spaceWebState = useWellbeingSpaceWeb({
    enabled: activeTab === "analysis",
    requestedRange: selectedRange,
  });

  useEffect(() => {
    let cancelled = false;

    async function loadPrompt() {
      if (
        activeTab !== "analysis" ||
        isProactivePromptDismissed ||
        proactivePrompt !== null ||
        summaryState.status !== "loaded" ||
        summaryState.summary.low_data ||
        summaryState.summary.signal_score < PROACTIVE_PROMPT_SCORE_THRESHOLD
      ) {
        return;
      }
      try {
        const prompt = await getChildSupportProactivePrompt();
        if (
          cancelled ||
          !prompt.should_prompt ||
          prompt.prompt_text === null
        ) {
          return;
        }
        setProactivePrompt(prompt);
      } catch {
        // 선제 질문은 실패해도 분석 화면을 막지 않는다.
      }
    }

    void loadPrompt();
    return () => {
      cancelled = true;
    };
  }, [activeTab, isProactivePromptDismissed, proactivePrompt, summaryState]);

  return (
    <div className="page-stack">
      {activeTab === "analysis" &&
        proactivePrompt?.prompt_text != null &&
        !isProactivePromptDismissed && (
          <section
            className="proactive-support-popup"
            aria-label="AI 마음 도움 선제 질문"
          >
            <div>
              <p className="card-label">AI 마음 도움</p>
              <p>{proactivePrompt.prompt_text}</p>
            </div>
            <div className="button-row">
              <button
                className="primary-button"
                type="button"
                onClick={() => {
                  setIsProactivePromptDismissed(true);
                  onOpenCoach?.();
                }}
              >
                지금 답해보기
              </button>
              <button
                className="ghost-button"
                type="button"
                onClick={() => setIsProactivePromptDismissed(true)}
              >
                나중에
              </button>
            </div>
          </section>
        )}

      {summaryState.status === "loaded" && (
        <WellbeingSignalCard summary={summaryState.summary} />
      )}
      {summaryState.status === "loading" && (
        <section className="hero-card child-hero">
          <div>
            <p className="eyebrow">내 마음 상태</p>
            <h2>현재 상태를 불러오는 중</h2>
            <p className="section-copy">
              최근 상태 요약과 필요한 도움을 준비하고 있습니다.
            </p>
          </div>
          <div className="hero-meter">
            <span className="hero-meter-label">상태 확인</span>
            <strong>...</strong>
          </div>
        </section>
      )}
      {summaryState.status === "error" && (
        <section className="hero-card child-hero">
          <div>
            <p className="eyebrow">내 마음 상태</p>
            <h2>현재 상태를 아직 불러오지 못했어요</h2>
            <p className="section-copy">
              마음 상태를 확인하는 프로그램과 연결되지 않았습니다.
            </p>
          </div>
          <div className="hero-meter">
            <span className="hero-meter-label">연결 상태</span>
            <strong>{summaryState.errorMessage}</strong>
          </div>
        </section>
      )}

      {activeTab === "ai" && <ChildSupportCoachPanel />}

      {activeTab === "analysis" && (
        <section className="tab-panel">
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
                    시간에 따른 위험도 변화를 준비하고 있습니다.
                  </p>
                </div>
              </div>
              <div className="trend-chart-loading">
                그래프 데이터를 불러오는 중입니다.
              </div>
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

          {spaceWebState.status === "loaded" && (
            <WellbeingSpaceWebGraph spaceWeb={spaceWebState.spaceWeb} />
          )}
          {spaceWebState.status === "loading" && (
            <section className="surface-card trend-card">
              <div className="trend-card-header">
                <div>
                  <p className="card-label">내 공간웹</p>
                  <p className="section-copy">
                    공간웹 데이터를 준비하고 있습니다.
                  </p>
                </div>
              </div>
              <div className="trend-chart-loading">
                공간웹 데이터를 불러오는 중입니다.
              </div>
            </section>
          )}
          {spaceWebState.status === "error" && (
            <section className="surface-card trend-card">
              <div className="trend-card-header">
                <div>
                  <p className="card-label">내 공간웹</p>
                  <p className="section-copy">{spaceWebState.errorMessage}</p>
                </div>
              </div>
            </section>
          )}

          {summaryState.status === "loaded" && (
            <section className="card-grid">
              <article className="surface-card">
                <p className="card-label">오늘 한 줄 요약</p>
                <p className="section-copy">{summaryState.summary.summary}</p>
              </article>
              <article className="surface-card">
                <p className="card-label">지금 해보면 좋은 것</p>
                <p className="section-copy">{summaryState.summary.action_tip}</p>
              </article>
            </section>
          )}
        </section>
      )}

      {activeTab === "checkin" && (
        <section className="checkin-panel">
          <article className="surface-card">
            <p className="card-label">자기 점검 질문</p>
            <ul className="checkin-list">
              <li>오늘 나를 가장 지치게 한 일은 무엇이었나요?</li>
              <li>지금 혼자 해결하기 어렵다고 느끼는 부분이 있나요?</li>
              <li>믿을 수 있는 어른에게 말한다면 어떤 말부터 꺼낼 수 있을까요?</li>
            </ul>
          </article>
          <article className="surface-card">
            <p className="card-label">도움 받을 수 있는 곳</p>
            <p className="section-copy">
              위급하거나 혼자 버티기 어렵다면 보호자, 학교 상담실, 청소년상담
              1388, 자살예방 상담전화 109에 바로 도움을 요청하세요.
            </p>
          </article>
        </section>
      )}
    </div>
  );
}
