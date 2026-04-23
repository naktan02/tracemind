import type { WellbeingSignalSummaryPayload } from "../contracts/generated";
import { getWellbeingFreshnessState } from "../lib/wellbeingFreshness";

type WellbeingDataNoticeProps = {
  summary: WellbeingSignalSummaryPayload;
};

export function WellbeingDataNotice({ summary }: WellbeingDataNoticeProps) {
  const freshness = getWellbeingFreshnessState(summary.computed_at);
  const messages: string[] = [];

  if (summary.low_data) {
    messages.push(
      "최근 데이터가 아직 적어서 현재 상태 해석이 충분히 안정적이지 않을 수 있습니다.",
    );
  }

  if (freshness === "stale") {
    messages.push("마지막 업데이트가 오래되어 현재 상태보다 늦을 수 있습니다.");
  }

  if (messages.length === 0) {
    return null;
  }

  return (
    <section className="status-banner notice">
      <div>
        <p className="card-label">데이터 안내</p>
        <ul className="bullet-list">
          {messages.map((message) => (
            <li key={message}>{message}</li>
          ))}
        </ul>
      </div>
    </section>
  );
}
