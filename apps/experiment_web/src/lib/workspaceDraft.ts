import type {
  CatalogItemPayload,
  CatalogSectionPayload,
  CatalogTrackPayload,
  ExperimentCatalogPayload,
  ResolvedExperimentPlanPayload,
  SavedWorkspaceDetailPayload,
  WorkspaceConfigScalar,
  WorkspaceManifestPayload,
  WorkspaceSelectionPayload,
} from "../types";

export const EMPTY_OVERRIDE_JSON = "{}";

export interface ObjectParseResult {
  value: Record<string, WorkspaceConfigScalar>;
  error: string | null;
}

export interface HydratedWorkspaceDraft {
  manifestId: string;
  trackName: string;
  entrypointName: string;
  selectedItemNameBySection: Record<string, string | null>;
  overrideTextBySection: Record<string, string>;
  globalOverrideText: string;
  compilePlan: ResolvedExperimentPlanPayload | null;
}

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
    const item = section.items.find((candidate) => candidate.item_name === selectedItemName);
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

export function buildSectionOverrideParseBySection(
  sections: CatalogSectionPayload[],
  overrideTextBySection: Record<string, string>,
): Record<string, ObjectParseResult> {
  const parseBySection: Record<string, ObjectParseResult> = {};
  for (const section of sections) {
    parseBySection[section.section_name] = parseOverrideObject(
      overrideTextBySection[section.section_name] ?? EMPTY_OVERRIDE_JSON,
    );
  }
  return parseBySection;
}

export function buildSectionOverrideValueBySection(
  overrideParseBySection: Record<string, ObjectParseResult>,
): Record<string, Record<string, WorkspaceConfigScalar>> {
  return Object.fromEntries(
    Object.entries(overrideParseBySection).map(([sectionName, parseResult]) => [
      sectionName,
      parseResult.value,
    ]),
  );
}

export function buildSectionOverrideErrors(
  sections: CatalogSectionPayload[],
  selectedItemNameBySection: Record<string, string | null>,
  overrideParseBySection: Record<string, ObjectParseResult>,
): string[] {
  const errors: string[] = [];

  for (const section of sections) {
    if (!selectedItemNameBySection[section.section_name]) {
      continue;
    }
    const parseResult =
      overrideParseBySection[section.section_name] ?? parseOverrideObject("{}");
    if (parseResult.error) {
      errors.push(`${section.display_name}: ${parseResult.error}`);
    }
  }

  return errors;
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

export function formatOverridePatch(
  patch: Record<string, WorkspaceConfigScalar>,
): string {
  const normalized = Object.fromEntries(
    Object.entries(patch).sort(([left], [right]) => left.localeCompare(right)),
  );
  return JSON.stringify(normalized, null, 2);
}

export function parseOverrideObject(input: string): ObjectParseResult {
  const normalized = input.trim();
  if (!normalized) {
    return { value: {}, error: null };
  }

  try {
    const parsed = JSON.parse(normalized) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {
        value: {},
        error: "JSON object여야 합니다.",
      };
    }

    const validated: Record<string, WorkspaceConfigScalar> = {};
    for (const [key, value] of Object.entries(parsed)) {
      if (!isScalarOverrideValue(value)) {
        return {
          value: {},
          error: `${key} 값은 string/number/boolean/null만 허용합니다.`,
        };
      }
      validated[key] = value;
    }
    return { value: validated, error: null };
  } catch (error) {
    return {
      value: {},
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

function findCatalogItemForSelection(
  section: CatalogSectionPayload,
  selection: WorkspaceSelectionPayload,
): CatalogItemPayload | null {
  return (
    section.items.find(
      (item) =>
        item.compile_support === "preset_selector" &&
        (item.variant_profile_name ?? item.item_name) === selection.variant_profile_name &&
        (selection.family_name === null ||
          item.family_name === selection.family_name ||
          item.family_name === null) &&
        (selection.core_method_name === null ||
          item.core_method_name === selection.core_method_name ||
          item.core_method_name === null),
    ) ?? null
  );
}

function isScalarOverrideValue(value: unknown): value is WorkspaceConfigScalar {
  return (
    value === null ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  );
}
