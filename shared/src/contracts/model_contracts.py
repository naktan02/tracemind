"""ModelManifest payload와 직렬화 유틸리티."""

from __future__ import annotations

import json
from datetime import datetime
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
