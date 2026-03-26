"""의미 계층용 prototype pack."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class CategoryPrototype:
    """의미 카테고리 하나에 대한 대표 벡터와 메타데이터."""

    centroid: list[float]
    sample_count: int


@dataclass(slots=True)
class PrototypePack:
    """하나의 임베딩 공간에 묶인 배포용 semantic layer 산출물."""

    schema_version: str
    prototype_version: str
    embedding_model_id: str
    translation_model_id: str | None
    built_at: datetime
    categories: dict[str, CategoryPrototype] = field(default_factory=dict)
