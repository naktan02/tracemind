"""ModelManifest payload와 직렬화 유틸리티."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ModelManifestPayload(BaseModel):
    """전역 모델 배포 구성을 설명하는 payload."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    model_id: str
    model_revision: str
    published_at: datetime
    artifact_kind: str
    artifact_ref: str
    prototype_version: str
    training_scope: str
    training_enabled: bool
    compatible_task_types: list[str]
    base_model_id: str | None = None
    base_model_revision: str | None = None
    translation_model_id: str | None = None
    translation_model_revision: str | None = None
    notes: str | None = None


def make_embedding_manifest(
    *,
    model_id: str,
    model_revision: str,
    prototype_version: str,
    artifact_ref: str,
    training_enabled: bool = True,
    compatible_task_types: list[str] | None = None,
    published_at: datetime | None = None,
    training_scope: str = "adapter_only",
    **kwargs,
) -> ModelManifestPayload:
    """임베딩 모델용 manifest payload를 만드는 표준 factory.

    필수 필드(model_id, model_revision, prototype_version, artifact_ref)만
    지정하면 나머지는 임베딩 배포 기본값으로 채워진다.

    >>> p = make_embedding_manifest(
    ...     model_id="bg-m3",
    ...     model_revision="rev_001",
    ...     prototype_version="proto_v1",
    ...     artifact_ref="/state/shared_adapter_states/versions/rev_001.json",
    ... )
    """
    return ModelManifestPayload(
        schema_version="model_manifest.v1",
        artifact_kind="embedding",
        model_id=model_id,
        model_revision=model_revision,
        prototype_version=prototype_version,
        artifact_ref=artifact_ref,
        training_scope=training_scope,
        training_enabled=training_enabled,
        compatible_task_types=compatible_task_types or ["pseudo_label_self_training"],
        published_at=published_at or datetime.now(tz=timezone.utc),
        **kwargs,
    )


def load_model_manifest_payload(path: Path) -> ModelManifestPayload:
    """JSON 파일에서 manifest payload를 읽는다."""
    return ModelManifestPayload.model_validate_json(path.read_text(encoding="utf-8"))


def dump_model_manifest_payload(path: Path, payload: ModelManifestPayload) -> None:
    """manifest payload를 JSON 파일로 기록한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
