"""개발자용 실험 catalog API payload."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CatalogItemCompileSupport = Literal[
    "entrypoint",
    "preset_selector",
    "metadata_only",
]
CatalogSectionSelectionMode = Literal[
    "single_required",
    "single_optional",
    "multi_optional",
]


class CatalogItemPayload(BaseModel):
    """Catalog section 안의 개별 항목."""

    model_config = ConfigDict(extra="forbid")

    item_name: str
    display_name: str
    item_kind: str
    family_name: str | None = None
    core_method_name: str | None = None
    variant_profile_name: str | None = None
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
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_compile_surface(self) -> CatalogItemPayload:
        if self.compile_support == "entrypoint":
            if self.script_path is None:
                raise ValueError(
                    "Entrypoint catalog item must declare script_path."
                )
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
