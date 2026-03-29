"""전역 모델 배포 메타데이터."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class ModelManifest:
    """현재 활성 전역 모델 구성을 설명하는 배포용 manifest."""

    schema_version: str
    model_id: str
    model_revision: str
    published_at: datetime
    artifact_kind: str
    artifact_ref: str
    prototype_version: str
    training_scope: str
    training_enabled: bool
    compatible_task_types: tuple[str, ...] = field(default_factory=tuple)
    base_model_id: str | None = None
    base_model_revision: str | None = None
    translation_model_id: str | None = None
    translation_model_revision: str | None = None
    notes: str | None = None
