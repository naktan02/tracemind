"""Single-centroid prototype domain entities.

이 모듈의 타입은 canonical wire contract가 아니라,
single-centroid builder/runtime 보조용 domain 표현이다.

- `SinglePrototypePack`
  - category마다 정확히 하나의 centroid만 갖는 내부 표현
- `PrototypePack`
  - 하위 호환 alias
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class SingleCategoryPrototype:
    """single-centroid category 하나의 대표 벡터와 메타데이터."""

    centroid: list[float]
    sample_count: int


@dataclass(slots=True)
class SinglePrototypePack:
    """카테고리마다 centroid 하나만 갖는 single-centroid prototype pack."""

    schema_version: str
    prototype_version: str
    embedding_model_id: str
    embedding_model_revision: str
    translation_model_id: str | None
    translation_model_revision: str | None
    translation_direction: str | None
    mapping_version: str
    build_method: str
    distance_metric: str
    built_at: datetime
    categories: dict[str, SingleCategoryPrototype] = field(default_factory=dict)


# Backward-compatible aliases for legacy imports.
CategoryPrototype = SingleCategoryPrototype
PrototypePack = SinglePrototypePack

__all__ = [
    "CategoryPrototype",
    "PrototypePack",
    "SingleCategoryPrototype",
    "SinglePrototypePack",
]
