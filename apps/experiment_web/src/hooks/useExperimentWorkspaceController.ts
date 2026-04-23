import { compileExperimentWorkspace } from "../api";
import { resolveApiBaseUrl } from "../api";
import { asErrorMessage } from "../lib/formatters";
import type { ObjectParseResult } from "../lib/overridePatch";
import {
  getEntrypointSection,
} from "../lib/workspaceManifest";
import { hydrateWorkspaceDraftFromSavedWorkspace } from "../lib/workspaceHydrator";
import { useState } from "react";
import { useExperimentCatalog } from "./useExperimentCatalog";
import { useExperimentRuns } from "./useExperimentRuns";
import { useSavedWorkspaces } from "./useSavedWorkspaces";
import { useWorkspaceDraft } from "./useWorkspaceDraft";
import type {
  CatalogItemPayload,
  CatalogOverrideFieldPayload,
  CatalogSectionPayload,
  CatalogTrackPayload,
  ExperimentCatalogPayload,
  ExperimentRunPayload,
  ResolvedExperimentPlanPayload,
  SavedWorkspaceSummaryPayload,
  WorkspaceConfigScalar,
  WorkspaceManifestPayload,
} from "../types";

export interface ActionNotice {
  tone: "ok" | "error";
  title: string;
  message: string;
}

export interface ExperimentWorkspaceController {
  apiBaseUrl: string;
  catalog: ExperimentCatalogPayload | null;
  catalogError: string | null;
  isCatalogLoading: boolean;
  manifestId: string | null;
  currentWorkspaceId: string | null;
  selectedTrackName: string | null;
  selectedEntrypointName: string | null;
  activeTrack: CatalogTrackPayload | null;
  entrypointItem: CatalogItemPayload | null;
  nonEntrypointSections: CatalogSectionPayload[];
  selectedItemNameBySection: Record<string, string | null>;
  overrideTextBySection: Record<string, string>;
  globalOverrideText: string;
  savedWorkspaces: SavedWorkspaceSummaryPayload[];
  savedWorkspacesError: string | null;
  isSavedWorkspacesLoading: boolean;
  loadingWorkspaceId: string | null;
  deletingWorkspaceId: string | null;
  runs: ExperimentRunPayload[];
  runsError: string | null;
  isRunsLoading: boolean;
  compilePlan: ResolvedExperimentPlanPayload | null;
  compileError: string | null;
  isCompiling: boolean;
  isWorkspaceSaving: boolean;
  isRunLaunching: boolean;
  rerunningWorkspaceId: string | null;
  actionNotice: ActionNotice | null;
  sectionOverrideParseBySection: Record<string, ObjectParseResult>;
  sectionOverrideValueBySection: Record<
    string,
    Record<string, WorkspaceConfigScalar>
  >;
  globalOverrideParse: ObjectParseResult;
  localParseErrors: string[];
  workspaceManifest: WorkspaceManifestPayload | null;
  refreshSavedWorkspaces: () => Promise<void>;
  refreshRuns: (options?: { silent?: boolean }) => Promise<void>;
  handleCompilePreview: () => Promise<void>;
  handleSaveWorkspace: () => Promise<void>;
  handleLoadSavedWorkspace: (workspaceId: string) => Promise<void>;
  handleCloneWorkspace: (workspaceId: string) => Promise<void>;
  handleDeleteWorkspace: (workspaceId: string) => Promise<void>;
  handleLaunchRun: () => Promise<void>;
  handleRelaunchWorkspace: (workspaceId: string) => Promise<void>;
  handleTrackChange: (track: CatalogTrackPayload) => void;
  handleEntrypointChange: (item: CatalogItemPayload) => void;
  handleCentralMethodChange: (params: {
    entrypointName: string;
    selectedItemsBySection: Record<string, string>;
  }) => void;
  handleSectionItemToggle: (sectionName: string, itemName: string) => void;
  handleResetLane: () => void;
  handleSectionOverrideTextChange: (
    sectionName: string,
    nextText: string,
  ) => void;
  handleSectionOverrideFieldChange: (
    sectionName: string,
    field: CatalogOverrideFieldPayload,
    nextValue: string | number | boolean | undefined,
  ) => void;
  handleGlobalOverrideTextChange: (nextText: string) => void;
}

export function useExperimentWorkspaceController(): ExperimentWorkspaceController {
  const apiBaseUrl = resolveApiBaseUrl();
  const catalogState = useExperimentCatalog(apiBaseUrl);
  const draftState = useWorkspaceDraft({
    activeTrack: catalogState.activeTrack,
    entrypointName: catalogState.entrypointItem?.item_name ?? null,
    sections: catalogState.nonEntrypointSections,
  });
  const savedWorkspacesState = useSavedWorkspaces(apiBaseUrl);
  const runsState = useExperimentRuns(apiBaseUrl);

  const [compilePlan, setCompilePlan] = useState<ResolvedExperimentPlanPayload | null>(
    null,
  );
  const [compileError, setCompileError] = useState<string | null>(null);
  const [isCompiling, setIsCompiling] = useState(false);
  const [isWorkspaceSaving, setIsWorkspaceSaving] = useState(false);
  const [isRunLaunching, setIsRunLaunching] = useState(false);
  const [rerunningWorkspaceId, setRerunningWorkspaceId] = useState<string | null>(
    null,
  );
  const [actionNotice, setActionNotice] = useState<ActionNotice | null>(null);

  function clearPreviewState() {
    setCompilePlan(null);
    setCompileError(null);
    setActionNotice(null);
  }

  async function handleCompilePreview() {
    if (!draftState.workspaceManifest) {
      setCompileError("먼저 탭과 실행 작업을 선택하세요.");
      setCompilePlan(null);
      return;
    }
    if (draftState.localParseErrors.length > 0) {
      setCompileError(draftState.localParseErrors[0]);
      setCompilePlan(null);
      return;
    }

    setIsCompiling(true);
    setCompileError(null);
    try {
      const plan = await compileExperimentWorkspace(
        apiBaseUrl,
        draftState.workspaceManifest,
      );
      setCompilePlan(plan);
    } catch (error) {
      setCompilePlan(null);
      setCompileError(asErrorMessage(error));
    } finally {
      setIsCompiling(false);
    }
  }

  async function handleSaveWorkspace() {
    if (!draftState.workspaceManifest) {
      setActionNotice({
        tone: "error",
        title: "저장에 실패했습니다",
        message: "먼저 탭과 실행 작업을 선택하세요.",
      });
      return;
    }
    if (draftState.localParseErrors.length > 0) {
      setActionNotice({
        tone: "error",
        title: "저장에 실패했습니다",
        message: draftState.localParseErrors[0],
      });
      return;
    }

    setIsWorkspaceSaving(true);
    setActionNotice(null);
    try {
      const savedWorkspace = await savedWorkspacesState.saveWorkspace(
        draftState.workspaceManifest,
      );
      draftState.markSavedWorkspace(
        savedWorkspace.workspace_id,
        savedWorkspace.manifest.manifest_id,
      );
      setCompilePlan(savedWorkspace.resolved_plan);
      setCompileError(null);
      await savedWorkspacesState.refreshSavedWorkspaces();
      setActionNotice({
        tone: "ok",
        title: "조합을 저장했습니다",
        message: `${savedWorkspace.workspace_id}로 저장했습니다.`,
      });
    } catch (error) {
      setActionNotice({
        tone: "error",
        title: "저장에 실패했습니다",
        message: asErrorMessage(error),
      });
    } finally {
      setIsWorkspaceSaving(false);
    }
  }

  async function handleLoadSavedWorkspace(workspaceId: string) {
    if (!catalogState.catalog) {
      return;
    }

    setActionNotice(null);
    try {
      const detail = await savedWorkspacesState.loadWorkspace(workspaceId);
      const hydrated = hydrateWorkspaceDraftFromSavedWorkspace(
        detail,
        catalogState.catalog,
      );
      catalogState.setInitialSelection(
        hydrated.trackName,
        hydrated.entrypointName,
      );
      draftState.applyHydratedWorkspace(detail.workspace_id, hydrated);
      setCompilePlan(hydrated.compilePlan);
      setCompileError(null);
      setActionNotice({
        tone: "ok",
        title: "저장된 조합을 불러왔습니다",
        message: `${detail.workspace_id}를 현재 편집 초안으로 불러왔습니다.`,
      });
    } catch (error) {
      setActionNotice({
        tone: "error",
        title: "불러오기에 실패했습니다",
        message: asErrorMessage(error),
      });
    }
  }

  async function handleCloneWorkspace(workspaceId: string) {
    if (!catalogState.catalog) {
      return;
    }

    setActionNotice(null);
    try {
      const detail = await savedWorkspacesState.loadWorkspace(workspaceId);
      const hydrated = hydrateWorkspaceDraftFromSavedWorkspace(
        detail,
        catalogState.catalog,
      );
      catalogState.setInitialSelection(
        hydrated.trackName,
        hydrated.entrypointName,
      );
      draftState.applyHydratedWorkspaceClone(hydrated.trackName, hydrated);
      setCompilePlan(hydrated.compilePlan);
      setCompileError(null);
      setActionNotice({
        tone: "ok",
        title: "복제용 초안을 만들었습니다",
        message:
          `${workspaceId}를 기반으로 새 초안을 만들었습니다. 방법론이나 ` +
          "파라미터를 바꾼 뒤 새 이름으로 저장해 실행할 수 있습니다.",
      });
    } catch (error) {
      setActionNotice({
        tone: "error",
        title: "복제용 초안 생성에 실패했습니다",
        message: asErrorMessage(error),
      });
    }
  }

  async function handleDeleteWorkspace(workspaceId: string) {
    setActionNotice(null);
    try {
      const deleted = await savedWorkspacesState.deleteWorkspace(workspaceId);
      if (
        draftState.currentWorkspaceId === workspaceId &&
        catalogState.activeTrack !== null
      ) {
        draftState.forkCurrentWorkspaceDraft(catalogState.activeTrack.track_name);
      }
      await savedWorkspacesState.refreshSavedWorkspaces();
      setActionNotice({
        tone: "ok",
        title: "저장된 조합을 삭제했습니다",
        message: `${deleted.workspace_id}를 비교판과 저장 목록에서 제거했습니다.`,
      });
    } catch (error) {
      setActionNotice({
        tone: "error",
        title: "삭제에 실패했습니다",
        message: asErrorMessage(error),
      });
    }
  }

  async function handleLaunchRun() {
    if (!draftState.workspaceManifest) {
      setActionNotice({
        tone: "error",
        title: "실행 시작에 실패했습니다",
        message: "먼저 탭과 실행 작업을 선택하세요.",
      });
      return;
    }
    if (draftState.localParseErrors.length > 0) {
      setActionNotice({
        tone: "error",
        title: "실행 시작에 실패했습니다",
        message: draftState.localParseErrors[0],
      });
      return;
    }

    setIsRunLaunching(true);
    setActionNotice(null);
    try {
      const launchedRun = await runsState.launchRun({
        manifest: draftState.workspaceManifest,
        workspace_id: draftState.currentWorkspaceId,
      });
      await runsState.refreshRuns();
      await savedWorkspacesState.refreshSavedWorkspaces();
      setActionNotice({
        tone: "ok",
        title: "실행을 시작했습니다",
        message: `${launchedRun.run_id}를 시작했습니다.`,
      });
    } catch (error) {
      setActionNotice({
        tone: "error",
        title: "실행 시작에 실패했습니다",
        message: asErrorMessage(error),
      });
    } finally {
      setIsRunLaunching(false);
    }
  }

  async function handleRelaunchWorkspace(workspaceId: string) {
    setRerunningWorkspaceId(workspaceId);
    setActionNotice(null);
    try {
      const detail = await savedWorkspacesState.loadWorkspace(workspaceId);
      const launchedRun = await runsState.launchRun({
        manifest: detail.manifest,
        workspace_id: detail.workspace_id,
      });
      await runsState.refreshRuns();
      await savedWorkspacesState.refreshSavedWorkspaces();
      setActionNotice({
        tone: "ok",
        title: "저장된 조합을 다시 실행했습니다",
        message: `${workspaceId}를 다시 실행해 ${launchedRun.run_id}를 시작했습니다.`,
      });
    } catch (error) {
      setActionNotice({
        tone: "error",
        title: "재실행에 실패했습니다",
        message: asErrorMessage(error),
      });
    } finally {
      setRerunningWorkspaceId(null);
    }
  }

  function handleTrackChange(track: CatalogTrackPayload) {
    catalogState.handleTrackChange(track);
    const nextEntrypoint = getEntrypointSection(track)?.items[0] ?? null;
    draftState.resetDraft(track, nextEntrypoint?.item_name ?? null);
    clearPreviewState();
  }

  function handleEntrypointChange(item: CatalogItemPayload) {
    if (catalogState.activeTrack) {
      draftState.forkCurrentWorkspaceDraft(catalogState.activeTrack.track_name);
    }
    catalogState.handleEntrypointChange(item);
    clearPreviewState();
  }

  function handleCentralMethodChange(params: {
    entrypointName: string;
    selectedItemsBySection: Record<string, string>;
  }) {
    if (!catalogState.activeTrack) {
      return;
    }
    const entrypointSection = getEntrypointSection(catalogState.activeTrack);
    const entrypointItem =
      entrypointSection?.items.find(
        (item) => item.item_name === params.entrypointName,
      ) ?? null;
    if (!entrypointItem) {
      return;
    }
    catalogState.handleEntrypointChange(entrypointItem);
    draftState.resetDraft(catalogState.activeTrack, entrypointItem.item_name);
    draftState.replaceSelectedItems(
      catalogState.activeTrack.track_name,
      Object.fromEntries(
        Object.entries(params.selectedItemsBySection).map(([key, value]) => [
          key,
          value,
        ]),
      ),
    );
    clearPreviewState();
  }

  function handleSectionItemToggle(sectionName: string, itemName: string) {
    draftState.handleSectionItemToggle(sectionName, itemName);
    clearPreviewState();
  }

  function handleResetLane() {
    const firstEntrypoint = catalogState.activeTrack
      ? getEntrypointSection(catalogState.activeTrack)?.items[0] ?? null
      : null;
    if (firstEntrypoint) {
      catalogState.handleEntrypointChange(firstEntrypoint);
    }
    draftState.resetDraft(catalogState.activeTrack, firstEntrypoint?.item_name ?? null);
    clearPreviewState();
  }

  function handleSectionOverrideTextChange(sectionName: string, nextText: string) {
    draftState.handleSectionOverrideTextChange(
      catalogState.activeTrack?.track_name ?? null,
      sectionName,
      nextText,
    );
    clearPreviewState();
  }

  function handleSectionOverrideFieldChange(
    sectionName: string,
    field: CatalogOverrideFieldPayload,
    nextValue: string | number | boolean | undefined,
  ) {
    draftState.handleSectionOverrideFieldChange(
      catalogState.activeTrack?.track_name ?? null,
      sectionName,
      field,
      nextValue,
    );
    clearPreviewState();
  }

  function handleGlobalOverrideTextChange(nextText: string) {
    draftState.handleGlobalOverrideTextChange(
      catalogState.activeTrack?.track_name ?? null,
      nextText,
    );
    clearPreviewState();
  }

  return {
    apiBaseUrl,
    catalog: catalogState.catalog,
    catalogError: catalogState.catalogError,
    isCatalogLoading: catalogState.isCatalogLoading,
    manifestId: draftState.manifestId,
    currentWorkspaceId: draftState.currentWorkspaceId,
    selectedTrackName: catalogState.selectedTrackName,
    selectedEntrypointName: catalogState.selectedEntrypointName,
    activeTrack: catalogState.activeTrack,
    entrypointItem: catalogState.entrypointItem,
    nonEntrypointSections: catalogState.nonEntrypointSections,
    selectedItemNameBySection: draftState.selectedItemNameBySection,
    overrideTextBySection: draftState.overrideTextBySection,
    globalOverrideText: draftState.globalOverrideText,
    savedWorkspaces: savedWorkspacesState.savedWorkspaces,
    savedWorkspacesError: savedWorkspacesState.savedWorkspacesError,
    isSavedWorkspacesLoading: savedWorkspacesState.isSavedWorkspacesLoading,
    loadingWorkspaceId: savedWorkspacesState.loadingWorkspaceId,
    deletingWorkspaceId: savedWorkspacesState.deletingWorkspaceId,
    runs: runsState.runs,
    runsError: runsState.runsError,
    isRunsLoading: runsState.isRunsLoading,
    compilePlan,
    compileError,
    isCompiling,
    isWorkspaceSaving,
    isRunLaunching,
    rerunningWorkspaceId,
    actionNotice,
    sectionOverrideParseBySection: draftState.sectionOverrideParseBySection,
    sectionOverrideValueBySection: draftState.sectionOverrideValueBySection,
    globalOverrideParse: draftState.globalOverrideParse,
    localParseErrors: draftState.localParseErrors,
    workspaceManifest: draftState.workspaceManifest,
    refreshSavedWorkspaces: savedWorkspacesState.refreshSavedWorkspaces,
    refreshRuns: runsState.refreshRuns,
    handleCompilePreview,
    handleSaveWorkspace,
    handleLoadSavedWorkspace,
    handleCloneWorkspace,
    handleDeleteWorkspace,
    handleLaunchRun,
    handleRelaunchWorkspace,
    handleTrackChange,
    handleEntrypointChange,
    handleCentralMethodChange,
    handleSectionItemToggle,
    handleResetLane,
    handleSectionOverrideTextChange,
    handleSectionOverrideFieldChange,
    handleGlobalOverrideTextChange,
  };
}
