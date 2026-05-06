"""Canonical prototype rebuild input 저장소."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from main_server.src.services.federation.assets.prototypes.models import (
    SERVER_REFERENCE_PROTOTYPE_SOURCE_KIND,
    PrototypeRebuildInputRecord,
    ServerReferencePrototypeSourceRow,
)
from shared.src.domain.value_objects.embedding_adapter_spec import EmbeddingAdapterSpec

MAIN_SERVER_ROOT = Path(__file__).resolve().parents[3]


class PrototypeRebuildInputRowPayload(BaseModel):
    """server-owned reference rebuild row payload."""

    model_config = ConfigDict(extra="forbid")

    text: str
    category: str
    source_kind: Literal["server_reference"] = SERVER_REFERENCE_PROTOTYPE_SOURCE_KIND


class PrototypeRebuildEmbeddingSpecPayload(BaseModel):
    """embedding adapter spec payload."""

    model_config = ConfigDict(extra="forbid")

    backend: str
    model_id: str = "mixedbread-ai/mxbai-embed-large-v1"
    revision: str = "main"
    device: str = "auto"
    batch_size: int = 16
    cache_dir: str | None = None
    task_prefix: str = ""
    normalize_embeddings: bool = True
    hash_dim: int = 256
    local_files_only: bool = False


class PrototypeRebuildInputPayload(BaseModel):
    """canonical rebuild input payload."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "prototype_rebuild_input.v1"
    input_id: str
    embedding_spec: PrototypeRebuildEmbeddingSpecPayload
    rows: list[PrototypeRebuildInputRowPayload]
    mapping_version: str
    normalize_embeddings: bool = True
    translation_model_id: str | None = None
    translation_model_revision: str | None = None
    translation_direction: str | None = None
    required_categories: list[str] | None = None


class PrototypeRebuildInputActivationPointer(BaseModel):
    """현재 활성 rebuild input 포인터."""

    model_config = ConfigDict(extra="forbid")

    input_id: str
    activated_at: datetime


def _dump_payload(path: Path, payload: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def dump_prototype_rebuild_input_payload(
    path: Path,
    payload: PrototypeRebuildInputPayload,
) -> None:
    """rebuild input payload를 JSON 파일로 기록한다."""
    _dump_payload(path, payload)


def load_prototype_rebuild_input_payload(path: Path) -> PrototypeRebuildInputPayload:
    """JSON 파일에서 rebuild input payload를 읽는다."""
    return PrototypeRebuildInputPayload.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def dump_activation_pointer_payload(
    path: Path,
    payload: PrototypeRebuildInputActivationPointer,
) -> None:
    """활성 input 포인터를 JSON 파일로 기록한다."""
    _dump_payload(path, payload)


def load_activation_pointer_payload(
    path: Path,
) -> PrototypeRebuildInputActivationPointer:
    """JSON 파일에서 활성 input 포인터를 읽는다."""
    return PrototypeRebuildInputActivationPointer.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def _record_to_payload(
    record: PrototypeRebuildInputRecord,
) -> PrototypeRebuildInputPayload:
    return PrototypeRebuildInputPayload(
        input_id=record.input_id,
        embedding_spec=PrototypeRebuildEmbeddingSpecPayload(
            backend=record.embedding_spec.backend,
            model_id=record.embedding_spec.model_id,
            revision=record.embedding_spec.revision,
            device=record.embedding_spec.device,
            batch_size=record.embedding_spec.batch_size,
            cache_dir=record.embedding_spec.cache_dir,
            task_prefix=record.embedding_spec.task_prefix,
            normalize_embeddings=record.embedding_spec.normalize_embeddings,
            hash_dim=record.embedding_spec.hash_dim,
            local_files_only=record.embedding_spec.local_files_only,
        ),
        rows=[
            PrototypeRebuildInputRowPayload(
                text=row.text,
                category=row.category,
                source_kind=row.source_kind,
            )
            for row in record.rows
        ],
        mapping_version=record.mapping_version,
        normalize_embeddings=record.normalize_embeddings,
        translation_model_id=record.translation_model_id,
        translation_model_revision=record.translation_model_revision,
        translation_direction=record.translation_direction,
        required_categories=(
            None
            if record.required_categories is None
            else list(record.required_categories)
        ),
    )


def _record_from_payload(
    payload: PrototypeRebuildInputPayload,
) -> PrototypeRebuildInputRecord:
    return PrototypeRebuildInputRecord(
        input_id=payload.input_id,
        embedding_spec=EmbeddingAdapterSpec(
            backend=payload.embedding_spec.backend,
            model_id=payload.embedding_spec.model_id,
            revision=payload.embedding_spec.revision,
            device=payload.embedding_spec.device,
            batch_size=payload.embedding_spec.batch_size,
            cache_dir=payload.embedding_spec.cache_dir,
            task_prefix=payload.embedding_spec.task_prefix,
            normalize_embeddings=payload.embedding_spec.normalize_embeddings,
            hash_dim=payload.embedding_spec.hash_dim,
            local_files_only=payload.embedding_spec.local_files_only,
        ),
        rows=tuple(
            ServerReferencePrototypeSourceRow(
                text=row.text,
                category=row.category,
                source_kind=row.source_kind,
            )
            for row in payload.rows
        ),
        mapping_version=payload.mapping_version,
        normalize_embeddings=payload.normalize_embeddings,
        translation_model_id=payload.translation_model_id,
        translation_model_revision=payload.translation_model_revision,
        translation_direction=payload.translation_direction,
        required_categories=(
            None
            if payload.required_categories is None
            else tuple(payload.required_categories)
        ),
    )


@dataclass(slots=True)
class PrototypeRebuildInputRepository:
    """중앙 서버의 canonical prototype rebuild input 파일을 관리한다."""

    state_root: Path = field(
        default_factory=lambda: MAIN_SERVER_ROOT / "state" / "prototype_rebuild_inputs"
    )

    @property
    def inputs_dir(self) -> Path:
        return self.state_root / "versions"

    @property
    def active_pointer_path(self) -> Path:
        return self.state_root / "active.json"

    def path_for_input(self, input_id: str) -> Path:
        return self.inputs_dir / f"{input_id}.json"

    def has_input(self, input_id: str) -> bool:
        return self.path_for_input(input_id).exists()

    def save_input(self, record: PrototypeRebuildInputRecord) -> Path:
        path = self.path_for_input(record.input_id)
        dump_prototype_rebuild_input_payload(path, _record_to_payload(record))
        return path

    def load_input(self, input_id: str) -> PrototypeRebuildInputRecord:
        path = self.path_for_input(input_id)
        if not path.exists():
            raise FileNotFoundError(f"Prototype rebuild input not found: {input_id}")
        return _record_from_payload(load_prototype_rebuild_input_payload(path))

    def load_active_pointer(self) -> PrototypeRebuildInputActivationPointer | None:
        if not self.active_pointer_path.exists():
            return None
        return load_activation_pointer_payload(self.active_pointer_path)

    def set_active(
        self,
        input_id: str,
        *,
        activated_at: datetime | None = None,
    ) -> PrototypeRebuildInputActivationPointer:
        if not self.has_input(input_id):
            raise FileNotFoundError(f"Prototype rebuild input not found: {input_id}")
        pointer = PrototypeRebuildInputActivationPointer(
            input_id=input_id,
            activated_at=activated_at or datetime.now(timezone.utc),
        )
        dump_activation_pointer_payload(self.active_pointer_path, pointer)
        return pointer

    def load_active_input(self) -> PrototypeRebuildInputRecord:
        active_pointer = self.load_active_pointer()
        if active_pointer is None:
            raise FileNotFoundError("No active prototype rebuild input is registered.")
        return self.load_input(active_pointer.input_id)
