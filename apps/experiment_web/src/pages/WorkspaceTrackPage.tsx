import { CatalogSectionCard } from "../components/CatalogSectionCard";
import { ResultBlock } from "../components/ResultBlock";
import { RunHistoryPanel } from "../components/RunHistoryPanel";
import { SavedWorkspacePanel } from "../components/SavedWorkspacePanel";
import { EMPTY_OVERRIDE_JSON } from "../lib/overridePatch";
import {
  buildWorkspaceManifestPreview,
  getEntrypointSection,
} from "../lib/workspaceManifest";
import type { ExperimentWorkspaceController } from "../hooks/useExperimentWorkspaceController";

export function WorkspaceTrackPage(props: {
  controller: ExperimentWorkspaceController;
}) {
  const {
    controller,
  } = props;

  if (!controller.activeTrack) {
    return null;
  }

  const entrypointSection = getEntrypointSection(controller.activeTrack);

  return (
    <main className="workspace-grid">
      <section className="panel lane-panel">
        <div className="panel-header">
          <div>
            <p className="panel-kicker">Lane</p>
            <h2>Track and entrypoint</h2>
          </div>
        </div>

        <div className="track-summary">
          <p>{controller.activeTrack.description}</p>
          <div className="pill-row">
            {controller.activeTrack.supported_runtime_paths.map((runtimePath) => (
              <span className="pill" key={runtimePath}>
                {runtimePath}
              </span>
            ))}
          </div>
        </div>

        <div className="entrypoint-list">
          {entrypointSection?.items.map((item) => (
              <button
                key={item.item_name}
                type="button"
                className={
                  controller.entrypointItem?.item_name === item.item_name
                    ? "entrypoint-card entrypoint-card--active"
                    : "entrypoint-card"
                }
                onClick={() => controller.handleEntrypointChange(item)}
              >
                <strong>{item.display_name}</strong>
                <span>{item.script_path}</span>
                <code>{item.source_of_truth}</code>
              </button>
            ))}
        </div>

        <SavedWorkspacePanel
          currentWorkspaceId={controller.currentWorkspaceId}
          savedWorkspaces={controller.savedWorkspaces}
          savedWorkspacesError={controller.savedWorkspacesError}
          isSavedWorkspacesLoading={controller.isSavedWorkspacesLoading}
          loadingWorkspaceId={controller.loadingWorkspaceId}
          onRefresh={() => void controller.refreshSavedWorkspaces()}
          onLoadWorkspace={(workspaceId) =>
            void controller.handleLoadSavedWorkspace(workspaceId)
          }
        />
      </section>

      <section className="panel catalog-panel">
        <div className="panel-header">
          <div>
            <p className="panel-kicker">Palette</p>
            <h2>Catalog sections</h2>
          </div>
          <button
            type="button"
            className="ghost-button"
            onClick={controller.handleResetLane}
          >
            Reset lane
          </button>
        </div>

        <div className="section-list">
          {controller.nonEntrypointSections.map((section) => (
            <CatalogSectionCard
              key={section.section_name}
              section={section}
              selectedItemName={
                controller.selectedItemNameBySection[section.section_name] ?? null
              }
              selectedOverrideText={
                controller.overrideTextBySection[section.section_name] ??
                EMPTY_OVERRIDE_JSON
              }
              selectedOverridePatch={
                controller.sectionOverrideParseBySection[section.section_name]
                  ?.value ?? {}
              }
              onItemToggle={(selectedSection, item) =>
                controller.handleSectionItemToggle(
                  selectedSection.section_name,
                  item.item_name,
                )
              }
              onOverrideTextChange={controller.handleSectionOverrideTextChange}
              onOverrideFieldChange={controller.handleSectionOverrideFieldChange}
            />
          ))}
        </div>
      </section>

      <section className="panel preview-panel">
        <div className="panel-header panel-header--stacked">
          <div>
            <p className="panel-kicker">Preview</p>
            <h2>Workspace manifest, compile result, and runs</h2>
          </div>
          <div className="action-row">
            <button
              type="button"
              className="ghost-button"
              onClick={() => void controller.handleCompilePreview()}
              disabled={controller.isCompiling}
            >
              {controller.isCompiling ? "Compiling..." : "Compile preview"}
            </button>
            <button
              type="button"
              className="ghost-button"
              onClick={() => void controller.handleSaveWorkspace()}
              disabled={controller.isWorkspaceSaving}
            >
              {controller.isWorkspaceSaving ? "Saving..." : "Save workspace"}
            </button>
            <button
              type="button"
              className="primary-button"
              onClick={() => void controller.handleLaunchRun()}
              disabled={controller.isRunLaunching}
            >
              {controller.isRunLaunching ? "Launching..." : "Launch run"}
            </button>
          </div>
        </div>

        <div className="draft-summary">
          <div className="draft-summary__card">
            <span className="meta-label">Manifest</span>
            <strong>{controller.manifestId ?? "not ready"}</strong>
          </div>
          <div className="draft-summary__card">
            <span className="meta-label">Workspace</span>
            <strong>{controller.currentWorkspaceId ?? "unsaved draft"}</strong>
          </div>
        </div>

        <div className="preview-block">
          <label htmlFor="global-override">Global override patch</label>
          <textarea
            id="global-override"
            value={controller.globalOverrideText}
            onChange={(event) =>
              controller.handleGlobalOverrideTextChange(event.target.value)
            }
            spellCheck={false}
          />
          <p className="hint-text">
            top-level Hydra override만 넣습니다. 예:{" "}
            {`{"train_batch_size": 32, "training_task.local_epochs": 2}`}
          </p>
        </div>

        {controller.actionNotice ? (
          <div
            className={
              controller.actionNotice.tone === "ok"
                ? "message-block message-block--ok"
                : "message-block message-block--error"
            }
          >
            <h3>{controller.actionNotice.title}</h3>
            <p>{controller.actionNotice.message}</p>
          </div>
        ) : null}

        <div className="preview-block">
          <h3>Workspace manifest</h3>
          <pre>
            {JSON.stringify(
              controller.workspaceManifest ??
                buildWorkspaceManifestPreview(
                  controller.manifestId,
                  controller.activeTrack.track_name,
                  controller.entrypointItem?.item_name ?? null,
                  controller.nonEntrypointSections,
                  controller.selectedItemNameBySection,
                  controller.sectionOverrideValueBySection,
                  controller.globalOverrideParse.value,
                ),
              null,
              2,
            )}
          </pre>
        </div>

        {controller.localParseErrors.length > 0 ? (
          <div className="message-block message-block--error">
            <h3>Local parse error</h3>
            <ul>
              {controller.localParseErrors.map((error) => (
                <li key={error}>{error}</li>
              ))}
            </ul>
          </div>
        ) : null}

        {controller.compileError ? (
          <div className="message-block message-block--error">
            <h3>Compile error</h3>
            <p>{controller.compileError}</p>
          </div>
        ) : null}

        {controller.compilePlan ? (
          <div className="compile-result">
            <div className="message-block message-block--ok">
              <h3>Compile result</h3>
              <p>
                {controller.compilePlan.track_name} /{" "}
                {controller.compilePlan.entrypoint_name}
              </p>
            </div>

            {controller.compilePlan.warnings.length > 0 ? (
              <div className="message-block message-block--warning">
                <h3>Warnings</h3>
                <ul>
                  {controller.compilePlan.warnings.map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
              </div>
            ) : null}

            <div className="result-stack">
              <ResultBlock
                title="Script"
                lines={[
                  controller.compilePlan.script_path,
                  controller.compilePlan.job_config_path,
                ]}
              />
              <ResultBlock
                title="Default groups"
                lines={[
                  ...controller.compilePlan.base_default_groups,
                  ...controller.compilePlan.selection_default_groups,
                ]}
              />
              <ResultBlock
                title="Hydra overrides"
                lines={controller.compilePlan.hydra_overrides}
              />
              <ResultBlock
                title="Command args"
                lines={controller.compilePlan.command_args}
              />
            </div>
          </div>
        ) : null}

        <RunHistoryPanel
          apiBaseUrl={controller.apiBaseUrl}
          runs={controller.runs}
          runsError={controller.runsError}
          isRunsLoading={controller.isRunsLoading}
          onRefresh={() => void controller.refreshRuns()}
        />
      </section>
    </main>
  );
}
