import { useEffect, useMemo, useState } from "react";

import { CatalogSectionCard } from "../components/CatalogSectionCard";
import { ResultBlock } from "../components/ResultBlock";
import {
  WorkspaceStepRail,
  type WorkspaceStepItem,
} from "../components/WorkspaceStepRail";
import type { ExperimentWorkspaceController } from "../hooks/useExperimentWorkspaceController";
import {
  formatDateTime,
  formatEntrypointName,
  formatRunStatus,
  formatTrackName,
} from "../lib/formatters";
import { EMPTY_OVERRIDE_JSON } from "../lib/overridePatch";
import {
  getCentralMethodOptions,
  getEntrypointGuide,
  getSectionPresentation,
  getSectionDisplayCopy,
  getVisibleWorkspaceSections,
  resolveSelectedCentralMethod,
} from "../lib/workspaceSections";
import {
  buildWorkspaceManifestPreview,
  getEntrypointSection,
} from "../lib/workspaceManifest";

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
  const isCentralAdaptation =
    controller.activeTrack.track_name === "central_adaptation";
  const centralMethodOptions = useMemo(() => getCentralMethodOptions(), []);
  const visibleSections = useMemo(
    () =>
      getVisibleWorkspaceSections(
        controller.entrypointItem?.item_name ?? null,
        controller.nonEntrypointSections,
      ),
    [controller.entrypointItem, controller.nonEntrypointSections],
  );

  useEffect(() => {
    setActiveStepId(ENTRYPOINT_STEP_ID);
  }, [controller.activeTrack.track_name]);

  useEffect(() => {
    if (
      activeStepId === ENTRYPOINT_STEP_ID ||
      activeStepId === REVIEW_STEP_ID ||
      visibleSections.some((section) => section.section_name === activeStepId)
    ) {
      return;
    }
    setActiveStepId(ENTRYPOINT_STEP_ID);
  }, [activeStepId, visibleSections]);

  const selectedCentralMethod = resolveSelectedCentralMethod({
    entrypointName: controller.entrypointItem?.item_name ?? null,
    selectedItemNameBySection: controller.selectedItemNameBySection,
  });

  const latestRun =
    controller.runs.find(
      (run) =>
        run.track_name === controller.activeTrack?.track_name &&
        (controller.currentWorkspaceId === null ||
          run.workspace_id === controller.currentWorkspaceId),
    ) ??
    controller.runs.find(
      (run) => run.track_name === controller.activeTrack?.track_name,
    ) ??
    null;

  const sectionSelections = useMemo(
    () =>
      visibleSections.map((section) => {
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
    [controller.selectedItemNameBySection, visibleSections],
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
          ? "필수 선택"
          : "선택 안 함"),
      tone: (selectedItem ? "complete" : "pending") as
        | "complete"
        | "active"
        | "pending",
    }));

    return [
      {
        stepId: ENTRYPOINT_STEP_ID,
        label: isCentralAdaptation ? "방법론" : "실행 작업",
        detail:
          isCentralAdaptation
            ? selectedCentralMethod?.displayName ?? "먼저 선택"
            : controller.entrypointItem?.display_name ??
              controller.entrypointItem?.item_name ??
              "먼저 선택",
        tone: controller.entrypointItem ? "complete" : "pending",
      },
      ...selectionSteps,
      {
        stepId: REVIEW_STEP_ID,
        label: "검토 및 실행",
        detail: controller.compilePlan ? "실행 준비 완료" : "마지막 확인",
        tone: controller.compilePlan ? "complete" : "pending",
      },
    ];
  }, [
    controller.compilePlan,
    controller.entrypointItem,
    isCentralAdaptation,
    sectionSelections,
    selectedCentralMethod,
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
      : visibleSections.find(
          (section) => section.section_name === activeStepId,
        ) ?? null;

  const datasetSection = visibleSections.find(
    (section) => section.section_name === "dataset_presets",
  );
  const selectedDatasetItem =
    datasetSection?.items.find(
      (item) =>
        item.item_name ===
        controller.selectedItemNameBySection[datasetSection.section_name],
    ) ?? null;
  const datasetAssetPaths = extractDatasetAssetPaths(selectedDatasetItem?.metadata);
  const currentEntrypointGuide = getEntrypointGuide(
    controller.entrypointItem?.item_name ?? null,
  );
  const activeSectionCopy = activeSection
    ? getSectionDisplayCopy(
        activeSection,
        controller.entrypointItem?.item_name ?? null,
      )
    : null;

  const workspacePreview =
    controller.workspaceManifest ??
    buildWorkspaceManifestPreview(
      controller.manifestId,
      controller.activeTrack.track_name,
      controller.entrypointItem?.item_name ?? null,
      visibleSections,
      controller.selectedItemNameBySection,
      controller.sectionOverrideValueBySection,
      controller.globalOverrideParse.value,
    );

  return (
    <main className="workspace-page">
      <section className="panel overview-guide">
        <div className="panel-header panel-header--compact">
          <div>
            <p className="panel-kicker">현재 탭 안내</p>
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

        <div className="workspace-guide-list">
          <article className="workspace-guide-card">
            <strong>이 탭에서 하는 일</strong>
            <p>
              {formatTrackName(controller.activeTrack.track_name)} 단계에서 어떤
              방법을 비교할지 정하고 저장합니다.
            </p>
          </article>
          {currentEntrypointGuide ? (
            <article className="workspace-guide-card">
              <strong>현재 선택한 방법 설명</strong>
              <p>{currentEntrypointGuide}</p>
            </article>
          ) : null}
          <article className="workspace-guide-card">
            <strong>기록은 어디서 보나</strong>
            <p>
              맨 위의 `기록/비교` 페이지에서 이미 저장된 실험과 결과를 비교하고,
              여기서는 새 조합 편집에만 집중합니다.
            </p>
          </article>
        </div>
      </section>

      <section className="workspace-dashboard">
        <aside className="panel workflow-rail">
          <div className="panel-header panel-header--compact">
            <div>
              <p className="panel-kicker">새 조합 만들기</p>
              <h2>단계별 편집</h2>
            </div>
          </div>

          <div className="workflow-progress">
            <strong>
              {configuredSections.length} / {visibleSections.length}
            </strong>
            <span>현재 선택한 블록 수</span>
          </div>

          <WorkspaceStepRail
            activeStepId={activeStepId}
            steps={steps.map((step) => ({
              ...step,
              tone: step.stepId === activeStepId ? "active" : step.tone,
            }))}
            onStepChange={setActiveStepId}
          />
        </aside>

        <section className="panel workspace-main">
          {activeStepId === ENTRYPOINT_STEP_ID ? (
            <>
              <div className="panel-header">
                <div>
                  <p className="panel-kicker">1단계</p>
                  <h2>
                    {isCentralAdaptation
                      ? "비교할 방법론을 고르세요"
                      : "이 탭에서 실행할 작업을 고르세요"}
                  </h2>
                </div>
                <button
                  type="button"
                  className="ghost-button"
                  onClick={controller.handleResetLane}
                >
                  현재 탭 초기화
                </button>
              </div>

              <p className="step-intro">
                {isCentralAdaptation
                  ? "Gold, Confidence, Margin, FixMatch 같은 중앙 적응 방법론을 먼저 고릅니다. 이 선택이 학생 데이터, 교사 데이터, 방법론 파라미터 블록을 결정합니다."
                  : "먼저 어떤 실행 작업을 할지 정합니다. 이 선택이 이후 블록의 의미와 실행 명령을 결정합니다."}
              </p>

              <div className="entrypoint-list">
                {isCentralAdaptation
                  ? centralMethodOptions.map((option) => (
                      <button
                        key={option.methodId}
                        type="button"
                        className={
                          selectedCentralMethod?.methodId === option.methodId
                            ? "entrypoint-card entrypoint-card--active"
                            : "entrypoint-card"
                        }
                        onClick={() =>
                          controller.handleCentralMethodChange({
                            entrypointName: option.entrypointName,
                            selectedItemsBySection: option.selectedItemsBySection,
                          })
                        }
                      >
                        <strong>{option.displayName}</strong>
                        <span>{option.description}</span>
                        <code>{formatEntrypointName(option.entrypointName)}</code>
                      </button>
                    ))
                  : entrypointSection?.items.map((item) => (
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
                        <strong>{formatEntrypointName(item.item_name)}</strong>
                        <span>{item.display_name}</span>
                        <code>{item.script_path}</code>
                      </button>
                    ))}
              </div>
            </>
          ) : null}

          {activeSection ? (
            <>
              <div className="panel-header">
                <div>
                  <p className="panel-kicker">블록 설정</p>
                  <h2>{activeSectionCopy?.displayName ?? activeSection.display_name}</h2>
                </div>
                <button
                  type="button"
                  className="ghost-button"
                  onClick={() => setActiveStepId(REVIEW_STEP_ID)}
                >
                  마지막 검토로 이동
                </button>
              </div>

              <p className="step-intro">
                {activeSectionCopy?.description ??
                  `${activeSection.display_name} 블록의 preset과 run-local 값을 조정합니다.`}
              </p>

              <CatalogSectionCard
                section={{
                  ...activeSection,
                  display_name: activeSectionCopy?.displayName ?? activeSection.display_name,
                  description: activeSectionCopy?.description ?? activeSection.description,
                }}
                presentation={getSectionPresentation(
                  activeSection,
                  controller.entrypointItem?.item_name ?? null,
                )}
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
                  <p className="panel-kicker">마지막 단계</p>
                  <h2>현재 초안을 검토하고 실행하세요</h2>
                </div>
              </div>

              <p className="step-intro">
                여기서는 현재 조합을 한 번 더 확인합니다. 전역 override와 실제
                실행 명령은 이 단계에서만 확인하면 됩니다.
              </p>

              <div className="preview-block">
                <label htmlFor="global-override">전역 override 패치</label>
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
                  configuredSections.map(({ section, selectedItem }) => {
                    const sectionCopy = getSectionDisplayCopy(
                      section,
                      controller.entrypointItem?.item_name ?? null,
                    );
                    return (
                      <article
                        className="selection-summary-card"
                        key={section.section_name}
                      >
                        <span className="meta-label">{sectionCopy.displayName}</span>
                        <strong>{selectedItem?.display_name}</strong>
                        <span>{selectedItem?.source_of_truth}</span>
                      </article>
                    );
                  })
                ) : (
                  <div className="message-block">
                    <h3>아직 선택한 블록이 없습니다</h3>
                    <p>
                      optional 블록은 비워둘 수 있지만, 현재는 추가로 고른 preset이
                      없습니다.
                    </p>
                  </div>
                )}
              </div>

              <details className="advanced-panel">
                <summary>고급 manifest 미리보기</summary>
                <pre>{JSON.stringify(workspacePreview, null, 2)}</pre>
              </details>

              {selectedDatasetItem ? (
                <div className="selection-summary-grid">
                  <article className="selection-summary-card">
                    <span className="meta-label">평가 데이터</span>
                    <strong>{selectedDatasetItem.display_name}</strong>
                    <span>
                      validation:{" "}
                      {datasetAssetPaths.validation_jsonl ?? "설정 없음"}
                    </span>
                    <span>test: {datasetAssetPaths.test_jsonl ?? "설정 없음"}</span>
                  </article>
                </div>
              ) : null}

              {controller.compilePlan ? (
                <details className="advanced-panel" open>
                  <summary>실행 명령 미리보기</summary>
                  <div className="result-stack">
                    <ResultBlock
                      title="실행 스크립트"
                      lines={[
                        controller.compilePlan.script_path,
                        controller.compilePlan.job_config_path,
                      ]}
                    />
                    <ResultBlock
                      title="기본 그룹"
                      lines={[
                        ...controller.compilePlan.base_default_groups,
                        ...controller.compilePlan.selection_default_groups,
                      ]}
                    />
                    <ResultBlock
                      title="Hydra override"
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
              이전 단계
            </button>
            <button
              type="button"
              className="primary-button"
              disabled={nextStep === null}
              onClick={() => nextStep && setActiveStepId(nextStep.stepId)}
            >
              {nextStep ? `다음: ${nextStep.label}` : "완료"}
            </button>
          </div>
        </section>

        <aside className="summary-rail">
          <section className="panel summary-panel">
            <div className="panel-header panel-header--compact">
              <div>
                <p className="panel-kicker">현재 초안</p>
                <h2>저장과 실행</h2>
              </div>
            </div>

              <div className="draft-summary">
                <div className="draft-summary__card">
                  <span className="meta-label">
                    {isCentralAdaptation ? "방법론" : "실행 작업"}
                  </span>
                  <strong>
                    {isCentralAdaptation
                      ? selectedCentralMethod?.displayName ?? "아직 선택 안 함"
                      : controller.entrypointItem
                        ? formatEntrypointName(controller.entrypointItem.item_name)
                        : "아직 선택 안 함"}
                  </strong>
                </div>
              <div className="draft-summary__card">
                <span className="meta-label">저장 상태</span>
                <strong>{controller.currentWorkspaceId ?? "저장 전 초안"}</strong>
              </div>
              <div className="draft-summary__card">
                <span className="meta-label">Manifest</span>
                <strong>{controller.manifestId ?? "아직 없음"}</strong>
              </div>
            </div>

            <div className="selection-chip-list">
              {configuredSections.length > 0 ? (
                configuredSections.map(({ section, selectedItem }) => {
                  const sectionCopy = getSectionDisplayCopy(
                    section,
                    controller.entrypointItem?.item_name ?? null,
                  );
                  return (
                    <span className="selection-chip" key={section.section_name}>
                      {sectionCopy.displayName}: {selectedItem?.display_name}
                    </span>
                  );
                })
              ) : (
                <span className="status-inline status-inline--muted">
                  아직 선택된 블록이 없습니다.
                </span>
              )}
            </div>

            {selectedDatasetItem ? (
              <div className="draft-summary">
                <div className="draft-summary__card">
                  <span className="meta-label">검증 데이터</span>
                  <strong>
                    {datasetAssetPaths.validation_jsonl ?? "dataset alias에 따름"}
                  </strong>
                </div>
                <div className="draft-summary__card">
                  <span className="meta-label">테스트 데이터</span>
                  <strong>
                    {datasetAssetPaths.test_jsonl ?? "dataset alias에 따름"}
                  </strong>
                </div>
              </div>
            ) : null}

            <div className="action-row">
              <button
                type="button"
                className="ghost-button"
                onClick={() => void controller.handleCompilePreview()}
                disabled={controller.isCompiling}
              >
                {controller.isCompiling
                  ? "미리보기 생성 중..."
                  : "실행 명령 미리보기"}
              </button>
              <button
                type="button"
                className="ghost-button"
                onClick={() => void controller.handleSaveWorkspace()}
                disabled={controller.isWorkspaceSaving}
              >
                {controller.isWorkspaceSaving ? "저장 중..." : "현재 조합 저장"}
              </button>
              <button
                type="button"
                className="primary-button"
                onClick={() => void controller.handleLaunchRun()}
                disabled={controller.isRunLaunching}
              >
                {controller.isRunLaunching ? "실행 시작 중..." : "이 조합 실행"}
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

            {controller.isRunLaunching ? (
              <div className="message-block message-block--warning">
                <h3>실행 요청을 보내는 중입니다</h3>
                <p>백엔드가 새 run을 만들고 있습니다. 잠시 뒤 상태가 갱신됩니다.</p>
              </div>
            ) : null}

            {controller.localParseErrors.length > 0 ? (
              <div className="message-block message-block--error">
                <h3>입력한 값에 문제가 있습니다</h3>
                <ul>
                  {controller.localParseErrors.map((error) => (
                    <li key={error}>{error}</li>
                  ))}
                </ul>
              </div>
            ) : null}

            {controller.compileError ? (
              <div className="message-block message-block--error">
                <h3>실행 미리보기를 만들지 못했습니다</h3>
                <p>{controller.compileError}</p>
              </div>
            ) : null}

            {controller.compilePlan ? (
              <div className="message-block message-block--ok">
                <h3>실행 가능 상태입니다</h3>
                <p>
                  {formatTrackName(controller.compilePlan.track_name)} /{" "}
                  {formatEntrypointName(controller.compilePlan.entrypoint_name)}
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

            <div className="message-block">
              <h3>최근 실행 상태</h3>
              {latestRun ? (
                <>
                  <p>
                    {formatEntrypointName(latestRun.entrypoint_name)} /{" "}
                    {formatRunStatus(latestRun.status)}
                  </p>
                  <p className="hint-text">run_id: {latestRun.run_id}</p>
                  {latestRun.status === "running" ? (
                    <p className="status-inline status-inline--running">
                      현재 이 화면에서 실행 중입니다.
                    </p>
                  ) : null}
                  <p className="hint-text">
                    시작 시각: {formatDateTime(latestRun.started_at)}
                  </p>
                  {latestRun.finished_at ? (
                    <p className="hint-text">
                      종료 시각: {formatDateTime(latestRun.finished_at)}
                    </p>
                  ) : (
                    <p className="hint-text">
                      아직 종료되지 않았습니다. `기록/비교` 페이지에서도 같은
                      상태를 볼 수 있습니다.
                    </p>
                  )}
                </>
              ) : (
                <p className="hint-text">
                  아직 실행 기록이 없습니다. `이 조합 실행`을 누르면 여기에 바로
                  상태가 보입니다.
                </p>
              )}
            </div>
          </section>
        </aside>
      </section>
    </main>
  );
}

function extractDatasetAssetPaths(metadata: Record<string, unknown> | undefined): {
  validation_jsonl: string | null;
  test_jsonl: string | null;
} {
  const assetPaths = metadata?.asset_paths;
  if (!assetPaths || typeof assetPaths !== "object") {
    return {
      validation_jsonl: null,
      test_jsonl: null,
    };
  }
  const assetPathRecord = assetPaths as Record<string, unknown>;
  return {
    validation_jsonl:
      typeof assetPathRecord.validation_jsonl === "string"
        ? assetPathRecord.validation_jsonl
        : null,
    test_jsonl:
      typeof assetPathRecord.test_jsonl === "string"
        ? assetPathRecord.test_jsonl
        : null,
  };
}
