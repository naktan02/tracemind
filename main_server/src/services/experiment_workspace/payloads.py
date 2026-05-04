"""개발자용 실험 catalog API payload."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from shared.src.contracts.workspace_manifest_contracts import (
    ResolvedExperimentPlanPayload,
    WorkspaceConfigScalar,
    WorkspaceManifestPayload,
)

CatalogItemCompileSupport = Literal[
    "entrypoint",
    "preset_selector",
    "metadata_only",
]
CatalogOverrideFieldValueKind = Literal[
    "string",
    "integer",
    "number",
    "boolean",
]
CatalogSectionSelectionMode = Literal[
    "single_required",
    "single_optional",
    "multi_optional",
]
ExperimentRunStatus = Literal[
    "running",
    "succeeded",
    "failed",
    "interrupted",
]


class ExperimentRunMetricPayload(BaseModel):
    """실험 결과 비교 표에 노출할 scalar metric."""

    model_config = ConfigDict(extra="forbid")

    metric_key: str
    value: float


class ExperimentRunResultSummaryPayload(BaseModel):
    """run artifact에서 추출한 결과 요약."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "experiment_run_result_summary.v1"
    source_kind: str
    source_paths: tuple[str, ...] = ()
    metrics: tuple[ExperimentRunMetricPayload, ...] = ()


class CatalogOverrideFieldPayload(BaseModel):
    """Preset 위에 덧씌울 수 있는 typed scalar override field."""

    model_config = ConfigDict(extra="forbid")

    field_name: str
    value_kind: CatalogOverrideFieldValueKind
    default_value: str | int | float | bool


class CatalogItemPayload(BaseModel):
    """Catalog section 안의 개별 항목."""

    model_config = ConfigDict(extra="forbid")

    item_name: str
    display_name: str
    item_kind: str
    family_name: str | None = None
    core_method_name: str | None = None
    variant_profile_name: str | None = None
    compiled_selector_name: str | None = None
    preset_group: str | None = None
    description: str | None = None
    source_of_truth: str
    source_kind: str
    compile_support: CatalogItemCompileSupport
    compile_blocker_reason: str | None = None
    script_path: str | None = None
    supported_adapter_kinds: tuple[str, ...] = ()
    supported_runtime_paths: tuple[str, ...] = ()
    accepted_payload_formats: tuple[str, ...] = ()
    default_groups: tuple[str, ...] = ()
    declared_fields: tuple[str, ...] = ()
    override_fields: tuple[CatalogOverrideFieldPayload, ...] = ()
    default_override_patch: dict[str, WorkspaceConfigScalar] = Field(
        default_factory=dict
    )
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_compile_surface(self) -> CatalogItemPayload:
        if self.compile_support == "entrypoint":
            if self.script_path is None:
                raise ValueError("Entrypoint catalog item must declare script_path.")
            if self.preset_group is not None:
                raise ValueError(
                    "Entrypoint catalog item must not declare preset_group."
                )
            return self
        if self.compile_support == "preset_selector":
            if self.preset_group is None:
                raise ValueError(
                    "Preset-selector catalog item must declare preset_group."
                )
            if self.script_path is not None:
                raise ValueError(
                    "Preset-selector catalog item must not declare script_path."
                )
            return self
        if self.compile_blocker_reason is None:
            raise ValueError(
                "Metadata-only catalog item must explain why it is not compileable."
            )
        if self.preset_group is not None or self.script_path is not None:
            raise ValueError(
                "Metadata-only catalog item must not declare preset_group/script_path."
            )
        return self


class CatalogSectionPayload(BaseModel):
    """Track 내부의 한 전략 축 section."""

    model_config = ConfigDict(extra="forbid")

    section_name: str
    display_name: str
    item_kind: str
    description: str | None = None
    source_of_truth: str
    source_kind: str
    selection_mode: CatalogSectionSelectionMode = "single_optional"
    default_slot_name: str | None = None
    items: tuple[CatalogItemPayload, ...] = ()

    @model_validator(mode="after")
    def _normalize_selection_surface(self) -> CatalogSectionPayload:
        if self.default_slot_name is None and self.item_kind != "experiment_entrypoint":
            self.default_slot_name = self.section_name
        return self


class CatalogTrackPayload(BaseModel):
    """실험 workspace track 하나의 inventory."""

    model_config = ConfigDict(extra="forbid")

    track_name: str
    display_name: str
    description: str | None = None
    entrypoint_section_name: str | None = None
    supported_runtime_paths: tuple[str, ...] = ()
    sections: tuple[CatalogSectionPayload, ...] = ()

    @model_validator(mode="after")
    def _validate_entrypoint_section_name(self) -> CatalogTrackPayload:
        if self.entrypoint_section_name is None:
            return self
        matching_section = next(
            (
                section
                for section in self.sections
                if section.section_name == self.entrypoint_section_name
            ),
            None,
        )
        if matching_section is None:
            raise ValueError(
                "CatalogTrackPayload.entrypoint_section_name must point to an "
                "existing section."
            )
        if matching_section.item_kind != "experiment_entrypoint":
            raise ValueError(
                "CatalogTrackPayload.entrypoint_section_name must point to an "
                "experiment_entrypoint section."
            )
        return self


class ExperimentCatalogPayload(BaseModel):
    """웹/CLI가 읽는 read-only 실험 catalog."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "experiment_catalog.v1"
    generated_at: datetime
    source_root: str
    tracks: tuple[CatalogTrackPayload, ...]


class SavedWorkspaceSelectionPreviewPayload(BaseModel):
    """비교 표와 저장된 실험 카드에서 보여줄 선택 요약."""

    model_config = ConfigDict(extra="forbid")

    slot_name: str
    section_name: str
    variant_profile_name: str
    core_method_name: str | None = None
    family_name: str | None = None
    override_keys: tuple[str, ...] = ()


class SavedWorkspaceSummaryPayload(BaseModel):
    """저장된 workspace 목록 요약."""

    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    manifest_id: str
    track_name: str
    entrypoint_name: str
    created_at: datetime
    updated_at: datetime
    latest_run_id: str | None = None
    selection_previews: tuple[SavedWorkspaceSelectionPreviewPayload, ...] = ()


class SavedWorkspaceDetailPayload(SavedWorkspaceSummaryPayload):
    """재열기 가능한 saved workspace detail."""

    manifest: WorkspaceManifestPayload
    resolved_plan: ResolvedExperimentPlanPayload | None = None


class LaunchExperimentRunRequestPayload(BaseModel):
    """local experiment run launch request."""

    model_config = ConfigDict(extra="forbid")

    manifest: WorkspaceManifestPayload
    workspace_id: str | None = None


class ExperimentRunPayload(BaseModel):
    """실행 중이거나 완료된 local experiment run 상태."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    workspace_id: str | None = None
    manifest_id: str
    track_name: str
    entrypoint_name: str
    status: ExperimentRunStatus
    created_at: datetime
    started_at: datetime
    finished_at: datetime | None = None
    script_path: str
    command_args: tuple[str, ...] = ()
    artifact_root_path: str
    stdout_log_path: str
    stderr_log_path: str
    exit_code: int | None = None
    error_message: str | None = None
    reported_outputs: dict[str, str] = Field(default_factory=dict)
    result_summary: ExperimentRunResultSummaryPayload | None = None
