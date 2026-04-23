import { useEffect, useMemo, useState } from "react";

import { CatalogSectionCard } from "../components/CatalogSectionCard";
import { ResultBlock } from "../components/ResultBlock";
import { RunHistoryPanel } from "../components/RunHistoryPanel";
import { SavedWorkspacePanel } from "../components/SavedWorkspacePanel";
import { WorkspaceCompareBoard } from "../components/WorkspaceCompareBoard";
import {
  WorkspaceStepRail,
  type WorkspaceStepItem,
} from "../components/WorkspaceStepRail";
import { EMPTY_OVERRIDE_JSON } from "../lib/overridePatch";
import {
  buildWorkspaceManifestPreview,
  getEntrypointSection,
} from "../lib/workspaceManifest";
import type { ExperimentWorkspaceController } from "../hooks/useExperimentWorkspaceController";

const REVIEW_STEP_ID = "__review__";
const ENTRYPOINT_STEP_ID = "__entrypoint__";

export function WorkspaceTrackPage(props: {
  controller: ExperimentWorkspaceController;
}) {
  const { controller } = props;

  if (!controller.activeTrack) {
    return null;
  }

  const entrypointSection = getEntrypointSection(controller.activeTrack);
  const [activeStepId, setActiveStepId] = useState<string>(ENTRYPOINT_STEP_ID);

  useEffect(() => {
    setActiveStepId(ENTRYPOINT_STEP_ID);
  }, [controller.activeTrack.track_name]);

  const sectionSelections = useMemo(
    () =>
      controller.nonEntrypointSections.map((section) => {
        const selectedItem =
          section.items.find(
            (item) =>
              item.item_name ===
              controller.selectedItemNameBySection[section.section_name],
          ) ?? null;
        return {
          section,
          selectedItem,
        };
      }),
    [controller.nonEntrypointSections, controller.selectedItemNameBySection],
  );

  const configuredSections = sectionSelections.filter(
    (selection) => selection.selectedItem !== null,
  );

  const steps = useMemo<WorkspaceStepItem[]>(() => {
    const selectionSteps = sectionSelections.map(({ section, selectedItem }) => ({
      stepId: section.section_name,
      label: section.display_name,
      detail:
        selectedItem?.display_name ??
        (section.selection_mode === "single_required"
          ? "required"
          : "optional"),
      tone: (selectedItem ? "complete" : "pending") as
        | "complete"
        | "active"
        | "pending",
    }));

    return [
      {
        stepId: ENTRYPOINT_STEP_ID,
        label: "Entrypoint",
        detail: controller.entrypointItem?.display_name ?? "choose a job",
        tone: controller.entrypointItem ? "complete" : "pending",
      },
      ...selectionSteps,
      {
        stepId: REVIEW_STEP_ID,
        label: "Review & Run",
        detail: controller.compilePlan ? "compiled" : "preview before launch",
        tone: controller.compilePlan ? "complete" : "pending",
      },
    ];
  }, [
    controller.compilePlan,
    controller.entrypointItem,
    sectionSelections,
  ]);

  const currentStepIndex = steps.findIndex((step) => step.stepId === activeStepId);
  const previousStep = currentStepIndex > 0 ? steps[currentStepIndex - 1] : null;
  const nextStep =
    currentStepIndex >= 0 && currentStepIndex < steps.length - 1
      ? steps[currentStepIndex + 1]
      : null;
  const activeSection =
    activeStepId === ENTRYPOINT_STEP_ID || activeStepId === REVIEW_STEP_ID
      ? null
      : controller.nonEntrypointSections.find(
          (section) => section.section_name === activeStepId,
        ) ?? null;

  const workspacePreview =
    controller.workspaceManifest ??
    buildWorkspaceManifestPreview(
      controller.manifestId,
      controller.activeTrack.track_name,
      controller.entrypointItem?.item_name ?? null,
      controller.nonEntrypointSections,
      controller.selectedItemNameBySection,
      controller.sectionOverrideValueBySection,
      controller.globalOverrideParse.value,
    );

  return (
    <main className="workspace-dashboard">
      <aside className="panel workflow-rail">
        <div className="panel-header panel-header--compact">
          <div>
            <p className="panel-kicker">Lane</p>
            <h2>{controller.activeTrack.display_name}</h2>
          </div>
        </div>

        <p className="workflow-rail__description">
          {controller.activeTrack.description}
        </p>

        <div className="pill-row">
          {controller.activeTrack.supported_runtime_paths.map((runtimePath) => (
            <span className="pill" key={runtimePath}>
              {runtimePath}
            </span>
          ))}
        </div>

        <div className="workflow-progress">
          <strong>
            {configuredSections.length} / {controller.nonEntrypointSections.length}
          </strong>
          <span>preset sections configured</span>
        </div>

        <WorkspaceStepRail
          activeStepId={activeStepId}
          steps={steps.map((step) => ({
            ...step,
            tone: step.stepId === activeStepId ? "active" : step.tone,
          }))}
          onStepChange={setActiveStepId}
        />

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
      </aside>

      <section className="panel workspace-main">
        {activeStepId === ENTRYPOINT_STEP_ID ? (
          <>
            <div className="panel-header">
              <div>
                <p className="panel-kicker">Step 1</p>
                <h2>Choose the entrypoint</h2>
              </div>
              <button
                type="button"
                className="ghost-button"
                onClick={controller.handleResetLane}
              >
                Reset lane
              </button>
            </div>

            <p className="step-intro">
              먼저 실행할 실험 job을 고릅니다. 여기서 정한 entrypoint가 이후
              preset section의 compile 의미를 결정합니다.
            </p>

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
          </>
        ) : null}

        {activeSection ? (
          <>
            <div className="panel-header">
              <div>
                <p className="panel-kicker">Preset step</p>
                <h2>{activeSection.display_name}</h2>
              </div>
              <button
                type="button"
                className="ghost-button"
                onClick={() => setActiveStepId(REVIEW_STEP_ID)}
              >
                Skip to review
              </button>
            </div>

            <p className="step-intro">
              {activeSection.description ??
                "현재 step에서 이 축의 preset과 override를 정합니다."}
            </p>

            <CatalogSectionCard
              section={activeSection}
              selectedItemName={
                controller.selectedItemNameBySection[activeSection.section_name] ??
                null
              }
              selectedOverrideText={
                controller.overrideTextBySection[activeSection.section_name] ??
                EMPTY_OVERRIDE_JSON
              }
              selectedOverridePatch={
                controller.sectionOverrideParseBySection[activeSection.section_name]
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
          </>
        ) : null}

        {activeStepId === REVIEW_STEP_ID ? (
          <>
            <div className="panel-header">
              <div>
                <p className="panel-kicker">Final step</p>
                <h2>Review the draft</h2>
              </div>
            </div>

            <p className="step-intro">
              여기서는 top-level override와 manifest preview를 확인합니다. raw
              JSON은 기본 숨김으로 두고, 필요할 때만 advanced block을 엽니다.
            </p>

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

            <div className="selection-summary-grid">
              {configuredSections.length > 0 ? (
                configuredSections.map(({ section, selectedItem }) => (
                  <article className="selection-summary-card" key={section.section_name}>
                    <span className="meta-label">{section.display_name}</span>
                    <strong>{selectedItem?.display_name}</strong>
                    <span>{selectedItem?.source_of_truth}</span>
                  </article>
                ))
              ) : (
                <div className="message-block">
                  <h3>No preset selected yet</h3>
                  <p>optional section은 비워둘 수 있지만, 현재는 추가 선택이 없습니다.</p>
                </div>
              )}
            </div>

            <details className="advanced-panel">
              <summary>Advanced manifest preview</summary>
              <pre>{JSON.stringify(workspacePreview, null, 2)}</pre>
            </details>

            {controller.compilePlan ? (
              <details className="advanced-panel" open>
                <summary>Compiled command preview</summary>
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
                </div>
              </details>
            ) : null}
          </>
        ) : null}

        <div className="step-footer">
          <button
            type="button"
            className="ghost-button"
            disabled={previousStep === null}
            onClick={() => previousStep && setActiveStepId(previousStep.stepId)}
          >
            Previous
          </button>
          <button
            type="button"
            className="primary-button"
            disabled={nextStep === null}
            onClick={() => nextStep && setActiveStepId(nextStep.stepId)}
          >
            {nextStep ? `Next: ${nextStep.label}` : "Done"}
          </button>
        </div>
      </section>

      <aside className="summary-rail">
        <section className="panel summary-panel">
          <div className="panel-header panel-header--compact">
            <div>
              <p className="panel-kicker">Run summary</p>
              <h2>Compile, save, and launch</h2>
            </div>
          </div>

          <div className="draft-summary">
            <div className="draft-summary__card">
              <span className="meta-label">Entrypoint</span>
              <strong>
                {controller.entrypointItem?.display_name ?? "not selected"}
              </strong>
            </div>
            <div className="draft-summary__card">
              <span className="meta-label">Workspace</span>
              <strong>{controller.currentWorkspaceId ?? "unsaved draft"}</strong>
            </div>
            <div className="draft-summary__card">
              <span className="meta-label">Manifest</span>
              <strong>{controller.manifestId ?? "not ready"}</strong>
            </div>
          </div>

          <div className="selection-chip-list">
            {configuredSections.map(({ section, selectedItem }) => (
              <span className="selection-chip" key={section.section_name}>
                {section.display_name}: {selectedItem?.display_name}
              </span>
            ))}
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
            <div className="message-block message-block--ok">
              <h3>Compiled</h3>
              <p>
                {controller.compilePlan.track_name} /{" "}
                {controller.compilePlan.entrypoint_name}
              </p>
              {controller.compilePlan.warnings.length > 0 ? (
                <ul>
                  {controller.compilePlan.warnings.map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
              ) : (
                <p className="hint-text">현재 compile warning이 없습니다.</p>
              )}
            </div>
          ) : null}
        </section>

        <section className="panel compare-panel">
          <WorkspaceCompareBoard
            currentWorkspaceId={controller.currentWorkspaceId}
            savedWorkspaces={controller.savedWorkspaces}
            runs={controller.runs}
            rerunningWorkspaceId={controller.rerunningWorkspaceId}
            onLoadWorkspace={(workspaceId) =>
              void controller.handleLoadSavedWorkspace(workspaceId)
            }
            onRelaunchWorkspace={(workspaceId) =>
              void controller.handleRelaunchWorkspace(workspaceId)
            }
          />
        </section>

        <section className="panel">
          <RunHistoryPanel
            apiBaseUrl={controller.apiBaseUrl}
            runs={controller.runs}
            runsError={controller.runsError}
            isRunsLoading={controller.isRunsLoading}
            onRefresh={() => void controller.refreshRuns()}
          />
        </section>
      </aside>
    </main>
  );
}
