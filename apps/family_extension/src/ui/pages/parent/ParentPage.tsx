import { ParentWellbeingSummaryCard } from "../../components/ParentWellbeingSummaryCard";
import { useWellbeingSummary } from "../../hooks/useWellbeingSummary";
import { useWellbeingTimeseries } from "../../hooks/useWellbeingTimeseries";
import { useWellbeingSpaceWeb } from "../../hooks/useWellbeingSpaceWeb";
import {
  formatComputedAtLabel,
  formatConfidenceLabel,
  formatSignalLevelLabel,
  formatTrendLabel,
} from "../../lib/formatters";
import type {
  WellbeingSignalSummaryPayload,
  WellbeingSignalTimeseriesPayload,
  WellbeingSpaceWebNodePayload,
} from "../../../contracts/generated";

type ParentPageProps = {
  activeSessionExpiresAt: string | null;
};

export function ParentPage({ activeSessionExpiresAt }: ParentPageProps) {
  const summaryState = useWellbeingSummary({ enabled: true });
  const timeseriesState = useWellbeingTimeseries({
    enabled: true,
    requestedRange: "7d",
  });
  const spaceWebState = useWellbeingSpaceWeb({
    enabled: true,
    requestedRange: "7d",
  });
  const timeseriesSnapshot =
    timeseriesState.status === "loaded"
      ? buildParentTimeseriesSnapshot(timeseriesState.timeseries)
      : null;
  const topSpaceNodes =
    spaceWebState.status === "loaded"
      ? [...spaceWebState.spaceWeb.nodes]
          .sort((left, right) => right.intensity - left.intensity)
          .slice(0, 3)
      : [];
  const parentCopy =
    summaryState.status === "loaded"
      ? buildParentGuidanceCopy({
          summary: summaryState.summary,
          timeseriesSnapshot,
          topSpaceNodes,
        })
      : null;

  return (
    <div className="page-stack">
      {summaryState.status === "loaded" && (
        <ParentWellbeingSummaryCard
          summary={summaryState.summary}
          statusItems={parentCopy?.statusItems ?? []}
        />
      )}
      {summaryState.status === "loading" && (
        <section className="hero-card parent-hero">
          <div>
            <p className="eyebrow">보호자 안내</p>
            <h2>부모용 현재 상태를 불러오는 중</h2>
            <p className="section-copy">
              지금 필요한 관심 수준과 대응 방향을 확인하고 있습니다.
            </p>
            {activeSessionExpiresAt != null && (
              <p className="section-copy">
                현재 부모 세션은{" "}
                {formatComputedAtLabel(activeSessionExpiresAt)}까지 유지됩니다.
              </p>
            )}
          </div>
          <div className="hero-meter">
            <span className="hero-meter-label">상태 확인</span>
            <strong>...</strong>
          </div>
        </section>
      )}
      {summaryState.status === "error" && (
        <section className="hero-card parent-hero">
          <div>
            <p className="eyebrow">보호자 안내</p>
            <h2>부모용 현재 상태를 아직 불러오지 못했습니다</h2>
            <p className="section-copy">{summaryState.errorMessage}</p>
          </div>
          <div className="hero-meter">
            <span className="hero-meter-label">요약 상태</span>
            <strong>요청 실패</strong>
          </div>
        </section>
      )}

      {summaryState.status === "loaded" && (
        <section className="parent-guidance">
          <article className="surface-card">
            <p className="card-label">지금 필요한 대응</p>
            <p className="section-copy">{parentCopy?.responsePriority}</p>
          </article>
          <article className="surface-card">
            <p className="card-label">대화 시작</p>
            <p className="section-copy">{parentCopy?.conversationStarter}</p>
          </article>
          <article className="surface-card">
            <p className="card-label">주의할 점</p>
            <p className="section-copy">{parentCopy?.cautionNote}</p>
          </article>
          <article className="surface-card">
            <p className="card-label">지금 확인할 것</p>
            <ul className="bullet-list">
              {parentCopy?.checkItems.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </article>
        </section>
      )}
    </div>
  );
}

type ParentTimeseriesSnapshot = {
  latestScore: number;
  summary: string;
  rangeLabel: string;
  trendKind: "rising" | "falling" | "steady" | "unknown";
};

type ParentGuidanceCopy = {
  responsePriority: string;
  conversationStarter: string;
  cautionNote: string;
  statusItems: string[];
  checkItems: string[];
};

function buildParentTimeseriesSnapshot(
  timeseries: WellbeingSignalTimeseriesPayload,
): ParentTimeseriesSnapshot | null {
  if (timeseries.points.length === 0) {
    return null;
  }
  const first = timeseries.points[0];
  const latest = timeseries.points[timeseries.points.length - 1];
  const delta = latest.signal_score - first.signal_score;
  const roundedDelta = Math.round(delta);
  const trendKind =
    roundedDelta >= 8 ? "rising" : roundedDelta <= -8 ? "falling" : "steady";
  const direction =
    trendKind === "rising"
      ? "높아졌습니다"
      : trendKind === "falling"
        ? "낮아졌습니다"
        : "큰 변화 없이 유지되고 있습니다";
  return {
    latestScore: Math.round(latest.signal_score),
    summary: `최근 7일 기준 ${Math.abs(roundedDelta)}점 ${direction}.`,
    rangeLabel: `${formatComputedAtLabel(first.ts)} - ${formatComputedAtLabel(
      latest.ts,
    )}`,
    trendKind,
  };
}

function buildParentGuidanceCopy({
  summary,
  timeseriesSnapshot,
  topSpaceNodes,
}: {
  summary: WellbeingSignalSummaryPayload;
  timeseriesSnapshot: ParentTimeseriesSnapshot | null;
  topSpaceNodes: WellbeingSpaceWebNodePayload[];
}): ParentGuidanceCopy {
  const highestNode = topSpaceNodes[0] ?? null;
  const highNodes = topSpaceNodes.filter(
    (node) => node.level === "high" || node.level === "very_high",
  );
  const trendKind = timeseriesSnapshot?.trendKind ?? "unknown";
  const trendPhrase =
    timeseriesSnapshot?.summary ??
    (summary.trend === "unknown" ? null : formatTrendLabel(summary.trend));
  const topAxisPhrase =
    highestNode == null
      ? "뚜렷하게 도드라진 관찰 축은 아직 없습니다"
      : `가장 두드러진 축은 ${highestNode.label}(${formatSignalLevelLabel(
          highestNode.level,
        )}, ${formatTrendLabel(highestNode.trend)})입니다`;
  const responsePriority = buildResponsePriorityCopy({
    summary,
    trendKind,
    highNodes,
    topAxisPhrase,
    trendPhrase,
  });
  const conversationStarter = buildConversationStarterCopy({
    summary,
    highestNode,
    trendKind,
  });
  const cautionNote = buildCautionNoteCopy({
    summary,
    highNodes,
    trendKind,
  });
  return {
    responsePriority,
    conversationStarter,
    cautionNote,
    statusItems: buildParentStatusItems({
      summary,
      timeseriesSnapshot,
      topSpaceNodes,
    }),
    checkItems: buildParentCheckItems({
      lowData: summary.low_data,
      signalScore: summary.signal_score,
      trendSummary: trendKind,
      hasHighSpaceNode: highNodes.length > 0,
    }),
  };
}

function buildResponsePriorityCopy({
  summary,
  trendKind,
  highNodes,
  topAxisPhrase,
  trendPhrase,
}: {
  summary: WellbeingSignalSummaryPayload;
  trendKind: ParentTimeseriesSnapshot["trendKind"];
  highNodes: WellbeingSpaceWebNodePayload[];
  topAxisPhrase: string;
  trendPhrase: string | null;
}): string {
  if (summary.signal_level === "very_high" || highNodes.length > 0) {
    return `${summary.parent_guidance.response_priority} 현재 점수는 ${Math.round(
      summary.signal_score,
    )}점이고, ${topAxisPhrase}. 오늘은 혼자 두지 말고 안전 여부를 먼저 확인하세요.`;
  }
  if (summary.signal_level === "high" || trendKind === "rising") {
    return `${summary.parent_guidance.response_priority} ${
      trendPhrase == null ? "" : `${trendPhrase} `
    }${topAxisPhrase}. 오늘 안에 짧게 상태를 확인하는 쪽이 좋습니다.`;
  }
  if (summary.low_data) {
    return `아직 관측 데이터가 적습니다. ${summary.parent_guidance.response_priority}`;
  }
  return `${summary.parent_guidance.response_priority} 현재 점수는 ${Math.round(
    summary.signal_score,
  )}점입니다${
    trendPhrase == null ? "" : ` ${trendPhrase}`
  }`;
}

function buildConversationStarterCopy({
  summary,
  highestNode,
  trendKind,
}: {
  summary: WellbeingSignalSummaryPayload;
  highestNode: WellbeingSpaceWebNodePayload | null;
  trendKind: ParentTimeseriesSnapshot["trendKind"];
}): string {
  if (highestNode != null) {
    return `${highestNode.label} 쪽 신호가 상대적으로 두드러집니다. "${highestNode.label} 때문에 요즘 마음이 더 무거운 순간이 있었어?"처럼 한 가지 축만 짧게 물어보세요.`;
  }
  if (trendKind === "rising") {
    return "최근 며칠 사이 부담이 올라간 흐름이 보입니다. \"요즘 갑자기 더 힘들어진 일이 있었어?\"처럼 이유를 단정하지 않고 물어보세요.";
  }
  if (summary.low_data) {
    return "아직 근거가 적습니다. \"오늘 괜찮았던 일 하나랑 힘들었던 일 하나만 말해줄래?\"처럼 넓게 시작하세요.";
  }
  return summary.parent_guidance.conversation_starter;
}

function buildCautionNoteCopy({
  summary,
  highNodes,
  trendKind,
}: {
  summary: WellbeingSignalSummaryPayload;
  highNodes: WellbeingSpaceWebNodePayload[];
  trendKind: ParentTimeseriesSnapshot["trendKind"];
}): string {
  if (summary.signal_level === "very_high" || highNodes.length > 0) {
    return "아이를 추궁하거나 기록을 봤다고 길게 설명하지 말고, 지금 안전한지와 곁에 있을 수 있는지만 먼저 확인하세요.";
  }
  if (trendKind === "rising") {
    return "최근 상승 흐름이 있어도 원인을 단정하지 마세요. 답이 짧으면 멈추고, 저녁에 한 번 더 확인하세요.";
  }
  if (summary.low_data) {
    return summary.parent_guidance.caution_note;
  }
  return "상태가 안정적으로 보여도 감정을 가볍게 넘기지 말고, 아이가 말한 표현을 그대로 받아 주세요.";
}

function buildParentStatusItems({
  summary,
  timeseriesSnapshot,
  topSpaceNodes,
}: {
  summary: WellbeingSignalSummaryPayload;
  timeseriesSnapshot: ParentTimeseriesSnapshot | null;
  topSpaceNodes: WellbeingSpaceWebNodePayload[];
}): string[] {
  const currentStatus = [
    `현재 수준: ${formatSignalLevelLabel(summary.signal_level)}`,
    `${Math.round(summary.signal_score)}점`,
  ];
  if (summary.trend !== "unknown") {
    currentStatus.push(formatTrendLabel(summary.trend));
  }
  const dataStatus = [summary.low_data ? "데이터가 아직 적습니다" : "기본 상태 해석 가능"];
  if (summary.confidence !== "low") {
    dataStatus.unshift(`신뢰도: ${formatConfidenceLabel(summary.confidence)}`);
  }
  const items = [currentStatus.join(" · "), dataStatus.join(" · ")];
  if (timeseriesSnapshot != null) {
    items.push(`최근 7일: ${timeseriesSnapshot.summary}`);
  }
  if (topSpaceNodes.length > 0) {
    items.push(
      `주요 축: ${topSpaceNodes
        .map((node) => `${node.label} ${formatSignalLevelLabel(node.level)}`)
        .join(", ")}`,
    );
  }
  return items;
}

function buildParentCheckItems({
  lowData,
  signalScore,
  trendSummary,
  hasHighSpaceNode,
}: {
  lowData: boolean;
  signalScore: number;
  trendSummary: ParentTimeseriesSnapshot["trendKind"];
  hasHighSpaceNode: boolean;
}): string[] {
  const items: string[] = [];
  if (lowData) {
    items.push("아직 데이터가 적으니 단정하지 말고 짧게 안부를 확인하세요.");
  }
  if (signalScore >= 70 || hasHighSpaceNode) {
    items.push("오늘 안전한 장소에 있는지와 곁에 도움을 청할 어른이 있는지 확인하세요.");
  }
  if (trendSummary === "rising") {
    items.push("최근 며칠 사이 부담이 커진 이유가 있는지 부드럽게 물어보세요.");
  }
  if (trendSummary === "falling") {
    items.push("나아진 흐름을 인정하되, 부담스러운 확인 질문은 줄이세요.");
  }
  if (items.length === 0) {
    items.push("평소처럼 대화를 열어 두고, 아이가 먼저 말할 시간을 주세요.");
  }
  return items;
}
