import { formatOverridePatch } from "./overridePatch";
import { getEntrypointSection } from "./workspaceManifest";
import type {
  CatalogItemPayload,
  CatalogSectionPayload,
  ExperimentCatalogPayload,
  ResolvedExperimentPlanPayload,
  SavedWorkspaceDetailPayload,
  WorkspaceSelectionPayload,
} from "../types";

export interface HydratedWorkspaceDraft {
  manifestId: string;
  trackName: string;
  entrypointName: string;
  selectedItemNameBySection: Record<string, string | null>;
  overrideTextBySection: Record<string, string>;
  globalOverrideText: string;
  compilePlan: ResolvedExperimentPlanPayload | null;
}

export function hydrateWorkspaceDraftFromSavedWorkspace(
  detail: SavedWorkspaceDetailPayload,
  catalog: ExperimentCatalogPayload,
): HydratedWorkspaceDraft {
  const track = catalog.tracks.find(
    (candidate) => candidate.track_name === detail.manifest.track_name,
  );
  if (!track) {
    throw new Error(
      `저장된 workspace track을 catalog에서 찾을 수 없습니다: ${detail.manifest.track_name}`,
    );
  }

  const entrypointSection = getEntrypointSection(track);
  const entrypointItem = entrypointSection?.items.find(
    (candidate) => candidate.item_name === detail.manifest.entrypoint_name,
  );
  if (!entrypointItem) {
    throw new Error(
      `저장된 workspace entrypoint를 catalog에서 찾을 수 없습니다: ${detail.manifest.entrypoint_name}`,
    );
  }

  const selectedItemNameBySection: Record<string, string | null> = {};
  const overrideTextBySection: Record<string, string> = {};

  for (const selection of detail.manifest.selections) {
    const section = track.sections.find(
      (candidate) => candidate.section_name === selection.section_name,
    );
    if (!section) {
      throw new Error(
        `저장된 workspace section을 catalog에서 찾을 수 없습니다: ${selection.section_name}`,
      );
    }

    const item = findCatalogItemForSelection(section, selection);
    if (!item) {
      throw new Error(
        `저장된 workspace selection을 catalog에서 찾을 수 없습니다: ${selection.variant_profile_name}`,
      );
    }

    selectedItemNameBySection[section.section_name] = item.item_name;
    overrideTextBySection[section.section_name] = formatOverridePatch(
      selection.override_patch,
    );
  }

  return {
    manifestId: detail.manifest.manifest_id,
    trackName: detail.manifest.track_name,
    entrypointName: detail.manifest.entrypoint_name,
    selectedItemNameBySection,
    overrideTextBySection,
    globalOverrideText: formatOverridePatch(detail.manifest.global_override_patch),
    compilePlan: detail.resolved_plan,
  };
}

function findCatalogItemForSelection(
  section: CatalogSectionPayload,
  selection: WorkspaceSelectionPayload,
): CatalogItemPayload | null {
  return (
    section.items.find(
      (item) =>
        item.compile_support === "preset_selector" &&
        (item.variant_profile_name ?? item.item_name) ===
          selection.variant_profile_name &&
        (selection.family_name === null ||
          item.family_name === selection.family_name ||
          item.family_name === null) &&
        (selection.core_method_name === null ||
          item.core_method_name === selection.core_method_name ||
          item.core_method_name === null),
    ) ?? null
  );
}
