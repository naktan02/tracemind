import { useState } from "react";

import { CentralAdaptationWorkspacePage } from "./pages/CentralAdaptationWorkspacePage";
import { FederatedRuntimeWorkspacePage } from "./pages/FederatedRuntimeWorkspacePage";
import { WorkspaceRecordsPage } from "./pages/WorkspaceRecordsPage";
import { SeedWorkspacePage } from "./pages/SeedWorkspacePage";
import { useExperimentWorkspaceController } from "./hooks/useExperimentWorkspaceController";
import { formatTrackName } from "./lib/formatters";

function App() {
  const controller = useExperimentWorkspaceController();
  const [viewMode, setViewMode] = useState<"builder" | "records">("builder");

  return (
    <div className="app-shell">
      <header className="app-topbar">
        <div>
          <p className="eyebrow">TraceMind 개발자 실험 공간</p>
          <h1>실험 조합 워크스페이스</h1>
          <p className="app-topbar__text">
            저장된 실험을 비교하고, 같은 조합에서 방법론이나 파라미터만 바꿔
            다시 실행하는 개발자용 화면입니다.
          </p>
          <div className="topbar-guide">
            <span>1. 탭 선택</span>
            <span>2. 저장된 실험 비교</span>
            <span>3. 조합 수정 후 저장/실행</span>
          </div>
        </div>
        <div className="topbar-meta">
          <div className="topbar-chip">
            <span className="meta-label">API</span>
            <strong>{controller.apiBaseUrl}</strong>
          </div>
          <div className="topbar-chip">
            <span className="meta-label">현재 초안</span>
            <strong>{controller.currentWorkspaceId ?? "저장 전 초안"}</strong>
          </div>
          <div className="topbar-chip">
            <span className="meta-label">저장된 실험</span>
            <strong>{controller.savedWorkspaces.length}</strong>
          </div>
          <div className="topbar-chip">
            <span className="meta-label">최근 실행</span>
            <strong>{controller.runs.length}</strong>
          </div>
        </div>
      </header>

      <section className="page-mode-switcher" aria-label="실험 화면 모드">
        <button
          type="button"
          className={
            viewMode === "builder" ? "mode-tab mode-tab--active" : "mode-tab"
          }
          onClick={() => setViewMode("builder")}
        >
          <strong>조합 편집</strong>
          <small>새 조합을 만들고 값을 수정합니다.</small>
        </button>
        <button
          type="button"
          className={
            viewMode === "records" ? "mode-tab mode-tab--active" : "mode-tab"
          }
          onClick={() => setViewMode("records")}
        >
          <strong>기록/비교</strong>
          <small>저장된 실험과 결과를 비교합니다.</small>
        </button>
      </section>

      {controller.isCatalogLoading ? (
        <main className="status-panel">
          <h2>실험 목록을 불러오는 중입니다</h2>
          <p>현재 조합 가능한 실험 축과 실행 surface를 읽고 있습니다.</p>
        </main>
      ) : null}

      {controller.catalogError ? (
        <main className="status-panel status-panel--error">
          <h2>실험 목록을 불러오지 못했습니다</h2>
          <p>{controller.catalogError}</p>
        </main>
      ) : null}

      {controller.catalog ? (
        <section className="track-switcher">
          <div className="track-tabs" role="tablist" aria-label="실험 탭">
            {controller.catalog.tracks.map((track) => (
              <button
                key={track.track_name}
                type="button"
                className={
                  track.track_name === controller.activeTrack?.track_name
                    ? "track-tab track-tab--active"
                    : "track-tab"
                }
                onClick={() => controller.handleTrackChange(track)}
              >
                <span>{track.display_name}</span>
                <small>{track.description ?? formatTrackName(track.track_name)}</small>
              </button>
            ))}
          </div>
        </section>
      ) : null}

      {viewMode === "records" ? <WorkspaceRecordsPage controller={controller} /> : null}

      {viewMode === "builder" && controller.activeTrack?.track_name === "seed" ? (
        <SeedWorkspacePage controller={controller} />
      ) : null}
      {viewMode === "builder" &&
      controller.activeTrack?.track_name === "central_adaptation" ? (
        <CentralAdaptationWorkspacePage controller={controller} />
      ) : null}
      {viewMode === "builder" &&
      controller.activeTrack?.track_name === "federated_runtime" ? (
        <FederatedRuntimeWorkspacePage controller={controller} />
      ) : null}
    </div>
  );
}

export default App;
