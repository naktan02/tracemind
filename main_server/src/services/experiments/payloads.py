"""개발자용 실험 catalog API payload."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

CatalogMetadataScalar = str | int | float | bool | None


class CatalogItemPayload(BaseModel):
    """Catalog section 안의 개별 항목."""

    model_config = ConfigDict(extra="forbid")

    item_name: str
    display_name: str
    item_kind: str
    family_name: str | None = None
    method_name: str | None = None
    preset_group: str | None = None
    description: str | None = None
    source_of_truth: str
    source_kind: str
    supported_adapter_kinds: tuple[str, ...] = ()
    supported_runtime_paths: tuple[str, ...] = ()
    accepted_payload_formats: tuple[str, ...] = ()
    default_groups: tuple[str, ...] = ()
    declared_fields: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    metadata: dict[str, CatalogMetadataScalar] = Field(default_factory=dict)


class CatalogSectionPayload(BaseModel):
    """Track 내부의 한 전략 축 section."""

    model_config = ConfigDict(extra="forbid")

    section_name: str
    display_name: str
    item_kind: str
    description: str | None = None
    source_of_truth: str
    source_kind: str
    items: tuple[CatalogItemPayload, ...] = ()


class CatalogTrackPayload(BaseModel):
    """실험 workspace track 하나의 inventory."""

    model_config = ConfigDict(extra="forbid")

    track_name: str
    display_name: str
    description: str | None = None
    supported_runtime_paths: tuple[str, ...] = ()
    sections: tuple[CatalogSectionPayload, ...] = ()


class ExperimentCatalogPayload(BaseModel):
    """웹/CLI가 읽는 read-only 실험 catalog."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "experiment_catalog.v1"
    generated_at: datetime
    source_root: str
    tracks: tuple[CatalogTrackPayload, ...]
