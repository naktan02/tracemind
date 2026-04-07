"""Prototype rebuild용 입력/출력 모델."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from shared.src.contracts.prototype_build_state_contracts import (
    SinglePrototypeBuildStatePayload,
)
from shared.src.contracts.prototype_contracts import PrototypePackPayload
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.value_objects import EmbeddingAdapterSpec


@dataclass(slots=True)
class ReferencePrototypeSourceRow:
    """reference rebuild용 canonical row 표현."""

    text: str
    category: str


@dataclass(slots=True, frozen=True)
class PrototypeRebuildInputRecord:
    """server-owned canonical prototype rebuild 입력."""

    input_id: str
    embedding_spec: EmbeddingAdapterSpec
    rows: tuple[ReferencePrototypeSourceRow, ...]
    mapping_version: str
    normalize_embeddings: bool = True
    translation_model_id: str | None = None
    translation_model_revision: str | None = None
    translation_direction: str | None = None
    required_categories: tuple[str, ...] | None = None


@dataclass(slots=True)
class StoredReferencePrototypeRebuildRequest:
    """저장된 canonical input 기반 rebuild 요청."""

    adapter_state: SharedAdapterState
    prototype_version: str
    embedding_model_id: str
    embedding_model_revision: str
    input_id: str | None = None
    built_at: datetime | None = None


@dataclass(slots=True)
class ReferencePrototypeRebuildRequest:
    """reference row 기반 prototype rebuild 요청."""

    rows: tuple[ReferencePrototypeSourceRow, ...] | list[ReferencePrototypeSourceRow]
    adapter: Any
    adapter_state: SharedAdapterState
    prototype_version: str
    embedding_model_id: str
    embedding_model_revision: str
    embedding_backend: str
    mapping_version: str
    built_at: datetime | None = None
    normalize_embeddings: bool = True
    task_prefix: str = ""
    translation_model_id: str | None = None
    translation_model_revision: str | None = None
    translation_direction: str | None = None
    required_categories: tuple[str, ...] | list[str] | None = None


@dataclass(slots=True)
class PrototypeRebuildResult:
    """rebuild와 publication 이후의 결과 요약."""

    pack_payload: PrototypePackPayload
    build_state_payload: SinglePrototypeBuildStatePayload | None
    source_input_id: str | None = None
    published_pack_path: Path | None = None
    published_build_state_path: Path | None = None
    reference_pack_path: Path | None = None
    reference_build_state_path: Path | None = None
