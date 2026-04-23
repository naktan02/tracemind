export type CatalogItemCompileSupport =
  | "entrypoint"
  | "preset_selector"
  | "metadata_only";
export type CatalogSectionSelectionMode =
  | "single_required"
  | "single_optional"
  | "multi_optional";
export type CatalogOverrideFieldValueKind =
  | "string"
  | "integer"
  | "number"
  | "boolean";

export interface CatalogOverrideFieldPayload {
  field_name: string;
  value_kind: CatalogOverrideFieldValueKind;
  default_value: string | number | boolean;
}

export interface CatalogItemPayload {
  item_name: string;
  display_name: string;
  item_kind: string;
  family_name?: string | null;
  core_method_name?: string | null;
  variant_profile_name?: string | null;
  preset_group?: string | null;
  description?: string | null;
  source_of_truth: string;
  source_kind: string;
  compile_support: CatalogItemCompileSupport;
  compile_blocker_reason?: string | null;
  script_path?: string | null;
  supported_adapter_kinds: string[];
  supported_runtime_paths: string[];
  accepted_payload_formats: string[];
  default_groups: string[];
  declared_fields: string[];
  override_fields: CatalogOverrideFieldPayload[];
  tags: string[];
  metadata: Record<string, unknown>;
}

export interface CatalogSectionPayload {
  section_name: string;
  display_name: string;
  item_kind: string;
  description?: string | null;
  source_of_truth: string;
  source_kind: string;
  selection_mode: CatalogSectionSelectionMode;
  default_slot_name?: string | null;
  items: CatalogItemPayload[];
}

export interface CatalogTrackPayload {
  track_name: string;
  display_name: string;
  description?: string | null;
  entrypoint_section_name?: string | null;
  supported_runtime_paths: string[];
  sections: CatalogSectionPayload[];
}

export interface ExperimentCatalogPayload {
  schema_version: string;
  generated_at: string;
  source_root: string;
  tracks: CatalogTrackPayload[];
}

export type WorkspaceConfigScalar = string | number | boolean | null;

export interface WorkspaceSelectionPayload {
  slot_name: string;
  section_name: string;
  variant_profile_name: string;
  core_method_name?: string | null;
  family_name?: string | null;
  override_patch: Record<string, WorkspaceConfigScalar>;
}

export interface WorkspaceManifestPayload {
  schema_version: string;
  manifest_id: string;
  track_name: string;
  entrypoint_name: string;
  selections: WorkspaceSelectionPayload[];
  global_override_patch: Record<string, WorkspaceConfigScalar>;
  notes?: string | null;
}

export interface ResolvedWorkspaceSelectionPayload {
  slot_name: string;
  section_name: string;
  family_name?: string | null;
  core_method_name?: string | null;
  variant_profile_name: string;
  source_of_truth: string;
  preset_group?: string | null;
  compiled_selector?: string | null;
  compiled_overrides: string[];
}

export interface ResolvedExperimentPlanPayload {
  schema_version: string;
  manifest_id: string;
  track_name: string;
  entrypoint_name: string;
  job_config_path: string;
  script_path: string;
  base_default_groups: string[];
  selection_default_groups: string[];
  hydra_overrides: string[];
  command_args: string[];
  resolved_selections: ResolvedWorkspaceSelectionPayload[];
  warnings: string[];
}
