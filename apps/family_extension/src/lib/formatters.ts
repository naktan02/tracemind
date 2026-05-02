import type {
  WellbeingSignalConfidence,
  WellbeingSignalLevel,
  WellbeingSignalRange,
  WellbeingSignalTrend,
} from "../contracts/generated";

export function formatComputedAtLabel(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "업데이트 시각을 확인할 수 없음";
  }
  return new Intl.DateTimeFormat("ko-KR", {
    month: "numeric",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

export function formatSignalLevelLabel(level: WellbeingSignalLevel): string {
  switch (level) {
    case "low":
      return "안정";
    case "moderate":
      return "관찰 필요";
    case "high":
      return "주의 필요";
    case "very_high":
      return "도움 필요";
  }
}

export function formatTrendLabel(trend: WellbeingSignalTrend): string {
  switch (trend) {
    case "rising":
      return "최근 올라가는 중";
    case "steady":
      return "최근 비슷한 흐름";
    case "falling":
      return "최근 낮아지는 중";
    case "volatile":
      return "최근 흔들림이 큼";
    case "unknown":
      return "최근 변화 정보 부족";
  }
}

export function formatConfidenceLabel(
  confidence: WellbeingSignalConfidence,
): string {
  switch (confidence) {
    case "low":
      return "참고 신호 낮음";
    case "medium":
      return "참고 신호 보통";
    case "high":
      return "참고 신호 높음";
  }
}

export function formatRangeLabel(range: WellbeingSignalRange): string {
  switch (range) {
    case "7d":
      return "7일";
    case "14d":
      return "14일";
    case "30d":
      return "30일";
  }
}
