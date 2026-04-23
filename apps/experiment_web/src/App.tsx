import { CentralAdaptationWorkspacePage } from "./pages/CentralAdaptationWorkspacePage";
import { FederatedRuntimeWorkspacePage } from "./pages/FederatedRuntimeWorkspacePage";
import { SeedWorkspacePage } from "./pages/SeedWorkspacePage";
import { useExperimentWorkspaceController } from "./hooks/useExperimentWorkspaceController";

function App() {
  const controller = useExperimentWorkspaceController();

  return (
    <div className="page-shell">
      <header className="hero">
        <div className="hero-copy">
          <p className="eyebrow">TraceMind Developer Workspace</p>
          <h1>Experiment lanes into local runs</h1>
          <p className="hero-text">
            Track, entrypoint, preset, readiness, compile result, workspace save,
            local run launch, log path, artifact root까지 한 화면에서 확인하는
            Phase 4 MVP입니다.
          </p>
        </div>
        <div className="hero-meta">
          <div className="meta-card">
            <span className="meta-label">API Base</span>
            <span className="meta-value">{controller.apiBaseUrl}</span>
          </div>
          <div className="meta-card">
            <span className="meta-label">Draft</span>
            <span className="meta-value">
              {controller.currentWorkspaceId
                ? controller.currentWorkspaceId
                : "unsaved workspace draft"}
            </span>
          </div>
          <div className="meta-card">
            <span className="meta-label">Run History</span>
            <span className="meta-value">{controller.runs.length} recent runs</span>
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
        <section className="panel panel--track-tabs">
          <div className="panel-header panel-header--compact">
            <div>
              <p className="panel-kicker">Tracks</p>
              <h2>Lane Pages</h2>
            </div>
          </div>
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
