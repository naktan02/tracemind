import type {
  CatalogSectionPayload,
  CatalogTrackPayload,
  WorkspaceConfigScalar,
  WorkspaceManifestPayload,
  WorkspaceSelectionPayload,
} from "../types";

export function getEntrypointSection(
  track: CatalogTrackPayload,
): CatalogSectionPayload | undefined {
  if (track.entrypoint_section_name) {
    return track.sections.find(
      (section) => section.section_name === track.entrypoint_section_name,
    );
  }
  return track.sections.find(
    (section) => section.item_kind === "experiment_entrypoint",
  );
}

export function createManifestId(trackName: string): string {
  return `${trackName}_${Date.now()}`;
}

export function buildWorkspaceManifest(
  manifestId: string,
  trackName: string,
  entrypointName: string,
  sections: CatalogSectionPayload[],
  selectedItemNameBySection: Record<string, string | null>,
  overridePatchBySection: Record<string, Record<string, WorkspaceConfigScalar>>,
  globalOverridePatch: Record<string, WorkspaceConfigScalar>,
): WorkspaceManifestPayload {
  return {
    schema_version: "workspace_manifest.v1",
    manifest_id: manifestId,
    track_name: trackName,
    entrypoint_name: entrypointName,
    selections: buildWorkspaceSelections(
      sections,
      selectedItemNameBySection,
      overridePatchBySection,
    ),
    global_override_patch: globalOverridePatch,
    notes: null,
  };
}

export function buildWorkspaceManifestPreview(
  manifestId: string | null,
  trackName: string,
  entrypointName: string | null,
  sections: CatalogSectionPayload[],
  selectedItemNameBySection: Record<string, string | null>,
  overridePatchBySection: Record<string, Record<string, WorkspaceConfigScalar>>,
  globalOverridePatch: Record<string, WorkspaceConfigScalar>,
) {
  return {
    schema_version: "workspace_manifest.v1",
    manifest_id: manifestId,
    track_name: trackName,
    entrypoint_name: entrypointName,
    selections: buildWorkspaceSelections(
      sections,
      selectedItemNameBySection,
      overridePatchBySection,
    ),
    global_override_patch: globalOverridePatch,
    notes: null,
  };
}

export function buildWorkspaceSelections(
  sections: CatalogSectionPayload[],
  selectedItemNameBySection: Record<string, string | null>,
  overridePatchBySection: Record<string, Record<string, WorkspaceConfigScalar>>,
): WorkspaceSelectionPayload[] {
  return sections.flatMap((section) => {
    const selectedItemName = selectedItemNameBySection[section.section_name];
    if (!selectedItemName) {
      return [];
    }
    const item = section.items.find(
      (candidate) => candidate.item_name === selectedItemName,
    );
    if (!item || item.compile_support !== "preset_selector") {
      return [];
    }

    return [
      {
        slot_name: section.default_slot_name ?? section.section_name,
        section_name: section.section_name,
        variant_profile_name: item.variant_profile_name ?? item.item_name,
        core_method_name: item.core_method_name ?? null,
        family_name: item.family_name ?? null,
        override_patch: overridePatchBySection[section.section_name] ?? {},
      },
    ];
  });
}
