import { useEffect, useState } from "react";

import {
  compileExperimentWorkspace,
  getSavedExperimentWorkspace,
  launchExperimentRun,
  listExperimentRuns,
  listSavedExperimentWorkspaces,
  loadExperimentCatalog,
  resolveApiBaseUrl,
  saveExperimentWorkspace,
} from "../api";
import { asErrorMessage } from "../lib/formatters";
import {
  EMPTY_OVERRIDE_JSON,
  buildSectionOverrideErrors,
  buildSectionOverrideParseBySection,
  buildSectionOverrideValueBySection,
  buildWorkspaceManifest,
  createManifestId,
  formatOverridePatch,
  getEntrypointSection,
  hydrateWorkspaceDraftFromSavedWorkspace,
  parseOverrideObject,
} from "../lib/workspaceDraft";
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
  runs: ExperimentRunPayload[];
  runsError: string | null;
  isRunsLoading: boolean;
  compilePlan: ResolvedExperimentPlanPayload | null;
  compileError: string | null;
  isCompiling: boolean;
  isWorkspaceSaving: boolean;
  isRunLaunching: boolean;
  actionNotice: ActionNotice | null;
  sectionOverrideParseBySection: Record<
    string,
    ReturnType<typeof parseOverrideObject>
  >;
  sectionOverrideValueBySection: Record<
    string,
    Record<string, WorkspaceConfigScalar>
  >;
  globalOverrideParse: ReturnType<typeof parseOverrideObject>;
  localParseErrors: string[];
  workspaceManifest: WorkspaceManifestPayload | null;
  refreshSavedWorkspaces: () => Promise<void>;
  refreshRuns: (options?: { silent?: boolean }) => Promise<void>;
  handleCompilePreview: () => Promise<void>;
  handleSaveWorkspace: () => Promise<void>;
  handleLoadSavedWorkspace: (workspaceId: string) => Promise<void>;
  handleLaunchRun: () => Promise<void>;
  handleTrackChange: (track: CatalogTrackPayload) => void;
  handleEntrypointChange: (item: CatalogItemPayload) => void;
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
  const [catalog, setCatalog] = useState<ExperimentCatalogPayload | null>(null);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [isCatalogLoading, setIsCatalogLoading] = useState(true);

  const [manifestId, setManifestId] = useState<string | null>(null);
  const [currentWorkspaceId, setCurrentWorkspaceId] = useState<string | null>(null);
  const [selectedTrackName, setSelectedTrackName] = useState<string | null>(null);
  const [selectedEntrypointName, setSelectedEntrypointName] = useState<string | null>(
    null,
  );
  const [selectedItemNameBySection, setSelectedItemNameBySection] = useState<
    Record<string, string | null>
  >({});
  const [overrideTextBySection, setOverrideTextBySection] = useState<
    Record<string, string>
  >({});
  const [globalOverrideText, setGlobalOverrideText] =
    useState<string>(EMPTY_OVERRIDE_JSON);

  const [savedWorkspaces, setSavedWorkspaces] = useState<
    SavedWorkspaceSummaryPayload[]
  >([]);
  const [savedWorkspacesError, setSavedWorkspacesError] = useState<string | null>(
    null,
  );
  const [isSavedWorkspacesLoading, setIsSavedWorkspacesLoading] = useState(false);
  const [loadingWorkspaceId, setLoadingWorkspaceId] = useState<string | null>(null);

  const [runs, setRuns] = useState<ExperimentRunPayload[]>([]);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [isRunsLoading, setIsRunsLoading] = useState(false);

  const [compilePlan, setCompilePlan] = useState<ResolvedExperimentPlanPayload | null>(
    null,
  );
  const [compileError, setCompileError] = useState<string | null>(null);
  const [isCompiling, setIsCompiling] = useState(false);
  const [isWorkspaceSaving, setIsWorkspaceSaving] = useState(false);
  const [isRunLaunching, setIsRunLaunching] = useState(false);
  const [actionNotice, setActionNotice] = useState<ActionNotice | null>(null);

  async function refreshSavedWorkspaces() {
    setIsSavedWorkspacesLoading(true);
    try {
      const payload = await listSavedExperimentWorkspaces(apiBaseUrl);
      setSavedWorkspaces(payload);
      setSavedWorkspacesError(null);
    } catch (error) {
      setSavedWorkspacesError(asErrorMessage(error));
    } finally {
      setIsSavedWorkspacesLoading(false);
    }
  }

  async function refreshRuns(options?: { silent?: boolean }) {
    if (!options?.silent) {
      setIsRunsLoading(true);
    }
    try {
      const payload = await listExperimentRuns(apiBaseUrl);
      setRuns(payload);
      setRunsError(null);
    } catch (error) {
      setRunsError(asErrorMessage(error));
    } finally {
      if (!options?.silent) {
        setIsRunsLoading(false);
      }
    }
  }

  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      setIsCatalogLoading(true);
      setCatalogError(null);

      try {
        const payload = await loadExperimentCatalog(apiBaseUrl);
        if (cancelled) {
          return;
        }
        setCatalog(payload);
        const firstTrack = payload.tracks[0] ?? null;
        const firstEntrypoint = firstTrack
          ? getEntrypointSection(firstTrack)?.items[0] ?? null
          : null;
        setSelectedTrackName(firstTrack?.track_name ?? null);
        setSelectedEntrypointName(firstEntrypoint?.item_name ?? null);
        setManifestId(
          firstTrack ? createManifestId(firstTrack.track_name) : null,
        );
        setCurrentWorkspaceId(null);
      } catch (error) {
        if (cancelled) {
          return;
        }
        setCatalogError(asErrorMessage(error));
      } finally {
        if (!cancelled) {
          setIsCatalogLoading(false);
        }
      }
    }

    void bootstrap();
    return () => {
      cancelled = true;
    };
  }, [apiBaseUrl]);

  useEffect(() => {
    void refreshSavedWorkspaces();
    void refreshRuns();
  }, [apiBaseUrl]);

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      void refreshRuns({ silent: true });
    }, 4000);
    return () => {
      window.clearInterval(intervalId);
    };
  }, [apiBaseUrl]);

  const activeTrack =
    catalog?.tracks.find((track) => track.track_name === selectedTrackName) ?? null;
  const entrypointSection = activeTrack ? getEntrypointSection(activeTrack) : null;
  const entrypointItem =
    entrypointSection?.items.find((item) => item.item_name === selectedEntrypointName) ??
    entrypointSection?.items[0] ??
    null;
  const nonEntrypointSections =
    activeTrack?.sections.filter(
      (section) => section.section_name !== entrypointSection?.section_name,
    ) ?? [];

  const sectionOverrideParseBySection = buildSectionOverrideParseBySection(
    nonEntrypointSections,
    overrideTextBySection,
  );
  const sectionOverrideValueBySection = buildSectionOverrideValueBySection(
    sectionOverrideParseBySection,
  );
  const globalOverrideParse = parseOverrideObject(globalOverrideText);
  const sectionOverrideErrors = buildSectionOverrideErrors(
    nonEntrypointSections,
    selectedItemNameBySection,
    sectionOverrideParseBySection,
  );
  const localParseErrors = [
    globalOverrideParse.error
      ? `global_override_patch: ${globalOverrideParse.error}`
      : null,
    ...sectionOverrideErrors,
  ].filter(Boolean) as string[];
  const workspaceManifest =
    activeTrack && entrypointItem && manifestId
      ? buildWorkspaceManifest(
          manifestId,
          activeTrack.track_name,
          entrypointItem.item_name,
          nonEntrypointSections,
          selectedItemNameBySection,
          sectionOverrideValueBySection,
          globalOverrideParse.value,
        )
      : null;

  function forkCurrentWorkspaceDraft(trackName: string) {
    if (currentWorkspaceId === null) {
      return;
    }
    setCurrentWorkspaceId(null);
    setManifestId(createManifestId(trackName));
  }

  async function handleCompilePreview() {
    if (!workspaceManifest) {
      setCompileError("먼저 track과 entrypoint를 선택하세요.");
      setCompilePlan(null);
      return;
    }
    if (localParseErrors.length > 0) {
      setCompileError(localParseErrors[0]);
      setCompilePlan(null);
      return;
    }

    setIsCompiling(true);
    setCompileError(null);
    try {
      const plan = await compileExperimentWorkspace(apiBaseUrl, workspaceManifest);
      setCompilePlan(plan);
    } catch (error) {
      setCompilePlan(null);
      setCompileError(asErrorMessage(error));
    } finally {
      setIsCompiling(false);
    }
  }

  async function handleSaveWorkspace() {
    if (!workspaceManifest) {
      setActionNotice({
        tone: "error",
        title: "Workspace save failed",
        message: "먼저 track과 entrypoint를 선택하세요.",
      });
      return;
    }
    if (localParseErrors.length > 0) {
      setActionNotice({
        tone: "error",
        title: "Workspace save failed",
        message: localParseErrors[0],
      });
      return;
    }

    setIsWorkspaceSaving(true);
    setActionNotice(null);
    try {
      const savedWorkspace = await saveExperimentWorkspace(
        apiBaseUrl,
        workspaceManifest,
      );
      setCurrentWorkspaceId(savedWorkspace.workspace_id);
      setManifestId(savedWorkspace.manifest.manifest_id);
      setCompilePlan(savedWorkspace.resolved_plan);
      setCompileError(null);
      await refreshSavedWorkspaces();
      setActionNotice({
        tone: "ok",
        title: "Workspace saved",
        message: `${savedWorkspace.workspace_id}로 저장했습니다.`,
      });
    } catch (error) {
      setActionNotice({
        tone: "error",
        title: "Workspace save failed",
        message: asErrorMessage(error),
      });
    } finally {
      setIsWorkspaceSaving(false);
    }
  }

  async function handleLoadSavedWorkspace(workspaceId: string) {
    if (!catalog) {
      return;
    }

    setLoadingWorkspaceId(workspaceId);
    setActionNotice(null);
    try {
      const detail = await getSavedExperimentWorkspace(apiBaseUrl, workspaceId);
      const hydrated = hydrateWorkspaceDraftFromSavedWorkspace(detail, catalog);
      setManifestId(hydrated.manifestId);
      setCurrentWorkspaceId(detail.workspace_id);
      setSelectedTrackName(hydrated.trackName);
      setSelectedEntrypointName(hydrated.entrypointName);
      setSelectedItemNameBySection(hydrated.selectedItemNameBySection);
      setOverrideTextBySection(hydrated.overrideTextBySection);
      setGlobalOverrideText(hydrated.globalOverrideText);
      setCompilePlan(hydrated.compilePlan);
      setCompileError(null);
      setActionNotice({
        tone: "ok",
        title: "Workspace loaded",
        message: `${detail.workspace_id}를 현재 draft로 불러왔습니다.`,
      });
    } catch (error) {
      setActionNotice({
        tone: "error",
        title: "Workspace load failed",
        message: asErrorMessage(error),
      });
    } finally {
      setLoadingWorkspaceId(null);
    }
  }

  async function handleLaunchRun() {
    if (!workspaceManifest) {
      setActionNotice({
        tone: "error",
        title: "Run launch failed",
        message: "먼저 track과 entrypoint를 선택하세요.",
      });
      return;
    }
    if (localParseErrors.length > 0) {
      setActionNotice({
        tone: "error",
        title: "Run launch failed",
        message: localParseErrors[0],
      });
      return;
    }

    setIsRunLaunching(true);
    setActionNotice(null);
    try {
      const launchedRun = await launchExperimentRun(apiBaseUrl, {
        manifest: workspaceManifest,
        workspace_id: currentWorkspaceId,
      });
      await refreshRuns();
      await refreshSavedWorkspaces();
      setActionNotice({
        tone: "ok",
        title: "Run launched",
        message: `${launchedRun.run_id}를 시작했습니다.`,
      });
    } catch (error) {
      setActionNotice({
        tone: "error",
        title: "Run launch failed",
        message: asErrorMessage(error),
      });
    } finally {
      setIsRunLaunching(false);
    }
  }

  function handleTrackChange(track: CatalogTrackPayload) {
    const nextEntrypoint = getEntrypointSection(track)?.items[0] ?? null;
    setSelectedTrackName(track.track_name);
    setSelectedEntrypointName(nextEntrypoint?.item_name ?? null);
    setManifestId(createManifestId(track.track_name));
    setCurrentWorkspaceId(null);
    setSelectedItemNameBySection({});
    setOverrideTextBySection({});
    setGlobalOverrideText(EMPTY_OVERRIDE_JSON);
    setCompilePlan(null);
    setCompileError(null);
    setActionNotice(null);
  }

  function handleEntrypointChange(item: CatalogItemPayload) {
    if (activeTrack) {
      forkCurrentWorkspaceDraft(activeTrack.track_name);
    }
    setSelectedEntrypointName(item.item_name);
    setCompilePlan(null);
    setCompileError(null);
    setActionNotice(null);
  }

  function handleSectionItemToggle(sectionName: string, itemName: string) {
    if (activeTrack) {
      forkCurrentWorkspaceDraft(activeTrack.track_name);
    }
    const nextValue =
      selectedItemNameBySection[sectionName] === itemName ? null : itemName;
    setSelectedItemNameBySection((current) => ({
      ...current,
      [sectionName]: nextValue,
    }));
    setCompilePlan(null);
    setCompileError(null);
    setActionNotice(null);
  }

  function handleResetLane() {
    if (!activeTrack) {
      return;
    }
    const firstEntrypoint = getEntrypointSection(activeTrack)?.items[0] ?? null;
    setManifestId(createManifestId(activeTrack.track_name));
    setCurrentWorkspaceId(null);
    setSelectedEntrypointName(firstEntrypoint?.item_name ?? null);
    setSelectedItemNameBySection({});
    setOverrideTextBySection({});
    setGlobalOverrideText(EMPTY_OVERRIDE_JSON);
    setCompilePlan(null);
    setCompileError(null);
    setActionNotice(null);
  }

  function handleSectionOverrideTextChange(sectionName: string, nextText: string) {
    if (activeTrack) {
      forkCurrentWorkspaceDraft(activeTrack.track_name);
    }
    setOverrideTextBySection((current) => ({
      ...current,
      [sectionName]: nextText,
    }));
    setCompilePlan(null);
    setCompileError(null);
    setActionNotice(null);
  }

  function handleSectionOverrideFieldChange(
    sectionName: string,
    field: CatalogOverrideFieldPayload,
    nextValue: string | number | boolean | undefined,
  ) {
    const currentPatch = sectionOverrideParseBySection[sectionName]?.value ?? {};
    const nextPatch = {
      ...currentPatch,
    };
    if (nextValue === undefined || nextValue === field.default_value) {
      delete nextPatch[field.field_name];
    } else {
      nextPatch[field.field_name] = nextValue;
    }
    handleSectionOverrideTextChange(sectionName, formatOverridePatch(nextPatch));
  }

  function handleGlobalOverrideTextChange(nextText: string) {
    if (activeTrack) {
      forkCurrentWorkspaceDraft(activeTrack.track_name);
    }
    setGlobalOverrideText(nextText);
    setCompilePlan(null);
    setCompileError(null);
    setActionNotice(null);
  }

  return {
    apiBaseUrl,
    catalog,
    catalogError,
    isCatalogLoading,
    manifestId,
    currentWorkspaceId,
    selectedTrackName,
    selectedEntrypointName,
    activeTrack,
    entrypointItem,
    nonEntrypointSections,
    selectedItemNameBySection,
    overrideTextBySection,
    globalOverrideText,
    savedWorkspaces,
    savedWorkspacesError,
    isSavedWorkspacesLoading,
    loadingWorkspaceId,
    runs,
    runsError,
    isRunsLoading,
    compilePlan,
    compileError,
    isCompiling,
    isWorkspaceSaving,
    isRunLaunching,
    actionNotice,
    sectionOverrideParseBySection,
    sectionOverrideValueBySection,
    globalOverrideParse,
    localParseErrors,
    workspaceManifest,
    refreshSavedWorkspaces,
    refreshRuns,
    handleCompilePreview,
    handleSaveWorkspace,
    handleLoadSavedWorkspace,
    handleLaunchRun,
    handleTrackChange,
    handleEntrypointChange,
    handleSectionItemToggle,
    handleResetLane,
    handleSectionOverrideTextChange,
    handleSectionOverrideFieldChange,
    handleGlobalOverrideTextChange,
  };
}
