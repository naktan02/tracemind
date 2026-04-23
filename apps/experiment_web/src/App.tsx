import { CentralAdaptationWorkspacePage } from "./pages/CentralAdaptationWorkspacePage";
import { FederatedRuntimeWorkspacePage } from "./pages/FederatedRuntimeWorkspacePage";
import { SeedWorkspacePage } from "./pages/SeedWorkspacePage";
import { useExperimentWorkspaceController } from "./hooks/useExperimentWorkspaceController";

function App() {
  const controller = useExperimentWorkspaceController();

  return (
    <div className="app-shell">
      <header className="app-topbar">
        <div>
          <p className="eyebrow">TraceMind Developer Workspace</p>
          <h1>Experiment Workspace</h1>
          <p className="app-topbar__text">
            lane별 preset을 조합하고, compile preview와 local run 결과를 같은
            도구 안에서 비교합니다.
          </p>
        </div>
        <div className="topbar-meta">
          <div className="topbar-chip">
            <span className="meta-label">API</span>
            <strong>{controller.apiBaseUrl}</strong>
          </div>
          <div className="topbar-chip">
            <span className="meta-label">Draft</span>
            <strong>{controller.currentWorkspaceId ?? "unsaved"}</strong>
          </div>
          <div className="topbar-chip">
            <span className="meta-label">Runs</span>
            <strong>{controller.runs.length}</strong>
          </div>
        </div>
      </header>

      {controller.isCatalogLoading ? (
        <main className="status-panel">
          <h2>Loading catalog</h2>
          <p>현재 experiment catalog와 compile surface를 읽는 중입니다.</p>
        </main>
      ) : null}

      {controller.catalogError ? (
        <main className="status-panel status-panel--error">
          <h2>Catalog request failed</h2>
          <p>{controller.catalogError}</p>
        </main>
      ) : null}

      {controller.catalog ? (
        <section className="track-switcher">
          <div className="track-tabs" role="tablist" aria-label="Experiment tracks">
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
                <small>{track.track_name}</small>
              </button>
            ))}
          </div>
        </section>
      ) : null}

      {controller.activeTrack?.track_name === "seed" ? (
        <SeedWorkspacePage controller={controller} />
      ) : null}
      {controller.activeTrack?.track_name === "central_adaptation" ? (
        <CentralAdaptationWorkspacePage controller={controller} />
      ) : null}
      {controller.activeTrack?.track_name === "federated_runtime" ? (
        <FederatedRuntimeWorkspacePage controller={controller} />
      ) : null}
    </div>
  );
}

export default App;
