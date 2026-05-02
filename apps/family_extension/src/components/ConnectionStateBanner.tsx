import { getAgentApiBaseUrl } from "../api/client";
import type { LocalProgramHealthState } from "../hooks/useLocalProgramHealth";

type ConnectionStateBannerProps = {
  healthState: LocalProgramHealthState;
};

export function ConnectionStateBanner({
  healthState,
}: ConnectionStateBannerProps) {
  if (healthState === "connected") {
    return null;
  }

  return (
    <section
      className={
        healthState === "offline"
          ? "status-banner warning"
          : "status-banner neutral"
      }
    >
      <div>
        <p className="card-label">로컬 프로그램 상태</p>
        <p className="section-copy">
          {healthState === "checking" &&
            "로컬 프로그램 연결 상태를 다시 확인하고 있습니다."}
          {healthState === "offline" &&
            "로컬 프로그램에 아직 연결되지 않았습니다. wellbeing 화면은 마지막 저장값이나 fallback만 보여줄 수 있습니다."}
        </p>
      </div>
      <span className="status-hint">{getAgentApiBaseUrl()}</span>
    </section>
  );
}
