"""ModelManifest contract와 직렬화 유틸리티."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict

from .common_types import TrainingScope, TrainingTaskType

MODEL_MANIFEST_V1 = "model_manifest.v1"
ModelManifestSchemaVersion: TypeAlias = Literal["model_manifest.v1"]


class ArtifactKind(StrEnum):
    """ModelManifest가 가리키는 artifact 종류."""

    EMBEDDING = "embedding"
    EMBEDDING_BACKBONE = "embedding_backbone"
    SHARED_ADAPTER_STATE = "shared_adapter_state"
    ADAPTER = "adapter"
    DECISION_HEAD = "decision_head"


class ModelManifest(BaseModel):
    """현재 활성 전역 모델 구성을 설명하는 canonical manifest contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: ModelManifestSchemaVersion = MODEL_MANIFEST_V1
    model_id: str
    model_revision: str
    published_at: datetime
    artifact_kind: ArtifactKind
    artifact_ref: str
    prototype_version: str
    training_scope: TrainingScope
    training_enabled: bool
    compatible_task_types: tuple[TrainingTaskType, ...] = ()
    base_model_id: str | None = None
    base_model_revision: str | None = None
    translation_model_id: str | None = None
    translation_model_revision: str | None = None
    notes: str | None = None


ModelManifestPayload = ModelManifest


def make_embedding_manifest(
    *,
    model_id: str,
    model_revision: str,
    prototype_version: str,
    artifact_ref: str,
    training_enabled: bool = True,
    compatible_task_types: (
        tuple[TrainingTaskType, ...] | list[TrainingTaskType] | None
    ) = None,
    published_at: datetime | None = None,
    training_scope: TrainingScope = TrainingScope.ADAPTER_ONLY,
    **kwargs,
) -> ModelManifest:
    """임베딩 모델용 manifest payload를 만드는 표준 factory.

    필수 필드(model_id, model_revision, prototype_version, artifact_ref)만
    지정하면 나머지는 임베딩 배포 기본값으로 채워진다.

    >>> p = make_embedding_manifest(
    ...     model_id="bg-m3",
    ...     model_revision="rev_001",
    ...     prototype_version="proto_v1",
    ...     artifact_ref="shared_adapter_state::rev_001",
    ... )
    """
    return ModelManifest(
        schema_version=MODEL_MANIFEST_V1,
        artifact_kind=ArtifactKind.EMBEDDING,
        model_id=model_id,
        model_revision=model_revision,
        prototype_version=prototype_version,
        artifact_ref=artifact_ref,
        training_scope=training_scope,
        training_enabled=training_enabled,
        compatible_task_types=tuple(compatible_task_types or ())
        or (TrainingTaskType.PSEUDO_LABEL_SELF_TRAINING,),
        published_at=published_at or datetime.now(tz=timezone.utc),
        **kwargs,
    )


def load_model_manifest_payload(path: Path) -> ModelManifest:
    """JSON 파일에서 manifest payload를 읽는다."""
    return ModelManifest.model_validate_json(path.read_text(encoding="utf-8"))


def dump_model_manifest_payload(path: Path, payload: ModelManifest) -> None:
    """manifest payload를 JSON 파일로 기록한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
