import { useEffect, useState } from "react";

import {
  createManifestId,
  buildWorkspaceManifest,
} from "../lib/workspaceManifest";
import { getVisibleWorkspaceSections } from "../lib/workspaceSections";
import {
  EMPTY_OVERRIDE_JSON,
  buildSectionOverrideErrors,
  buildSectionOverrideParseBySection,
  buildSectionOverrideValueBySection,
  formatOverridePatch,
  parseOverrideObject,
} from "../lib/overridePatch";
import type { HydratedWorkspaceDraft } from "../lib/workspaceHydrator";
import type {
  CatalogOverrideFieldPayload,
  CatalogSectionPayload,
  CatalogTrackPayload,
  WorkspaceConfigScalar,
  WorkspaceManifestPayload,
} from "../types";

export interface WorkspaceDraftState {
  manifestId: string | null;
  currentWorkspaceId: string | null;
  selectedItemNameBySection: Record<string, string | null>;
  overrideTextBySection: Record<string, string>;
  globalOverrideText: string;
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
  initializeDraft: (
    trackName: string | null,
    entrypointName: string | null,
    sections: CatalogSectionPayload[],
  ) => void;
  markSavedWorkspace: (workspaceId: string, manifestId: string) => void;
  forkCurrentWorkspaceDraft: (trackName: string) => void;
  applyHydratedWorkspace: (workspaceId: string, draft: HydratedWorkspaceDraft) => void;
  applyHydratedWorkspaceClone: (
    trackName: string,
    draft: HydratedWorkspaceDraft,
  ) => void;
  resetDraft: (track: CatalogTrackPayload | null, entrypointName: string | null) => void;
  replaceSelectedItems: (
    trackName: string | null,
    nextSelectedItems: Record<string, string | null>,
  ) => void;
  handleSectionItemToggle: (sectionName: string, itemName: string) => void;
  handleSectionOverrideTextChange: (
    trackName: string | null,
    sectionName: string,
    nextText: string,
  ) => void;
  handleSectionOverrideFieldChange: (
    trackName: string | null,
    sectionName: string,
    field: CatalogOverrideFieldPayload,
    nextValue: string | number | boolean | undefined,
  ) => void;
  handleGlobalOverrideTextChange: (
    trackName: string | null,
    nextText: string,
  ) => void;
}

export function useWorkspaceDraft(params: {
  activeTrack: CatalogTrackPayload | null;
  entrypointName: string | null;
  sections: CatalogSectionPayload[];
}): WorkspaceDraftState {
  const { activeTrack, entrypointName, sections } = params;
  const [manifestId, setManifestId] = useState<string | null>(null);
  const [currentWorkspaceId, setCurrentWorkspaceId] = useState<string | null>(null);
  const [selectedItemNameBySection, setSelectedItemNameBySection] = useState<
    Record<string, string | null>
  >({});
  const [overrideTextBySection, setOverrideTextBySection] = useState<
    Record<string, string>
  >({});
  const [globalOverrideText, setGlobalOverrideText] =
    useState<string>(EMPTY_OVERRIDE_JSON);

  useEffect(() => {
    if (!activeTrack || !entrypointName) {
      return;
    }
    setManifestId((current) => current ?? createManifestId(activeTrack.track_name));
  }, [activeTrack, entrypointName]);

  const sectionOverrideParseBySection = buildSectionOverrideParseBySection(
    sections,
    overrideTextBySection,
  );
  const sectionOverrideValueBySection = buildSectionOverrideValueBySection(
    sectionOverrideParseBySection,
  );
  const globalOverrideParse = parseOverrideObject(globalOverrideText);
  const sectionOverrideErrors = buildSectionOverrideErrors(
    sections,
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
    activeTrack && entrypointName && manifestId
      ? buildWorkspaceManifest(
          manifestId,
          activeTrack.track_name,
          entrypointName,
          getVisibleWorkspaceSections(entrypointName, sections),
          selectedItemNameBySection,
          sectionOverrideValueBySection,
          globalOverrideParse.value,
        )
      : null;

  function initializeDraft(
    trackName: string | null,
    nextEntrypointName: string | null,
    _sections: CatalogSectionPayload[],
  ) {
    if (!trackName || !nextEntrypointName) {
      return;
    }
    setManifestId(createManifestId(trackName));
    setCurrentWorkspaceId(null);
    setSelectedItemNameBySection({});
    setOverrideTextBySection({});
    setGlobalOverrideText(EMPTY_OVERRIDE_JSON);
  }

  function markSavedWorkspace(workspaceId: string, nextManifestId: string) {
    setCurrentWorkspaceId(workspaceId);
    setManifestId(nextManifestId);
  }

  function forkCurrentWorkspaceDraft(trackName: string) {
    if (currentWorkspaceId === null) {
      return;
    }
    setCurrentWorkspaceId(null);
    setManifestId(createManifestId(trackName));
  }

  function applyHydratedWorkspace(
    workspaceId: string,
    draft: HydratedWorkspaceDraft,
  ) {
    setManifestId(draft.manifestId);
    setCurrentWorkspaceId(workspaceId);
    setSelectedItemNameBySection(draft.selectedItemNameBySection);
    setOverrideTextBySection(draft.overrideTextBySection);
    setGlobalOverrideText(draft.globalOverrideText);
  }

  function applyHydratedWorkspaceClone(
    trackName: string,
    draft: HydratedWorkspaceDraft,
  ) {
    setManifestId(createManifestId(trackName));
    setCurrentWorkspaceId(null);
    setSelectedItemNameBySection(draft.selectedItemNameBySection);
    setOverrideTextBySection(draft.overrideTextBySection);
    setGlobalOverrideText(draft.globalOverrideText);
  }

  function resetDraft(
    track: CatalogTrackPayload | null,
    nextEntrypointName: string | null,
  ) {
    initializeDraft(track?.track_name ?? null, nextEntrypointName, sections);
  }

  function replaceSelectedItems(
    trackName: string | null,
    nextSelectedItems: Record<string, string | null>,
  ) {
    if (trackName) {
      forkCurrentWorkspaceDraft(trackName);
    }
    setSelectedItemNameBySection(nextSelectedItems);
    setOverrideTextBySection({});
    setGlobalOverrideText(EMPTY_OVERRIDE_JSON);
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
  }

  function handleSectionOverrideTextChange(
    trackName: string | null,
    sectionName: string,
    nextText: string,
  ) {
    if (trackName) {
      forkCurrentWorkspaceDraft(trackName);
    }
    setOverrideTextBySection((current) => ({
      ...current,
      [sectionName]: nextText,
    }));
  }

  function handleSectionOverrideFieldChange(
    trackName: string | null,
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
    handleSectionOverrideTextChange(
      trackName,
      sectionName,
      formatOverridePatch(nextPatch),
    );
  }

  function handleGlobalOverrideTextChange(
    trackName: string | null,
    nextText: string,
  ) {
    if (trackName) {
      forkCurrentWorkspaceDraft(trackName);
    }
    setGlobalOverrideText(nextText);
  }

  return {
    manifestId,
    currentWorkspaceId,
    selectedItemNameBySection,
    overrideTextBySection,
    globalOverrideText,
    sectionOverrideParseBySection,
    sectionOverrideValueBySection,
    globalOverrideParse,
    localParseErrors,
    workspaceManifest,
    initializeDraft,
    markSavedWorkspace,
    forkCurrentWorkspaceDraft,
    applyHydratedWorkspace,
    applyHydratedWorkspaceClone,
    resetDraft,
    replaceSelectedItems,
    handleSectionItemToggle,
    handleSectionOverrideTextChange,
    handleSectionOverrideFieldChange,
    handleGlobalOverrideTextChange,
  };
}
