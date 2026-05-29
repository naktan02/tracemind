"""Prototype builder strategy 공통 인터페이스."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from typing import Any, Protocol

from shared.src.contracts.prototype_build_state_contracts import (
    SinglePrototypeBuildStatePayload,
)
from shared.src.contracts.prototype_contracts import PrototypePackPayload


@dataclass(slots=True)
class PrototypeBuildRequest:
    """Prototype 생성에 필요한 공통 입력."""

    embeddings_by_category: Mapping[str, Sequence[Sequence[float]]]
    prototype_version: str
    embedding_backend: str
    embedding_model_id: str
    embedding_model_revision: str
    normalize_embeddings: bool = True
    task_prefix: str = ""
    translation_model_id: str | None = None
    translation_model_revision: str | None = None
    translation_direction: str | None = None
    mapping_version: str = ""
    built_at: datetime | None = None
    required_categories: Sequence[str] | None = None


@dataclass(slots=True)
class PrototypeBuildArtifacts:
    """생성 전략 결과물."""

    pack_payload: PrototypePackPayload
    build_state_payload: SinglePrototypeBuildStatePayload | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PrototypeBuildStrategy(Protocol):
    """prototype 생성 전략 공통 인터페이스."""

    name: str
    supports_exact_build_state: bool

    def build(self, request: PrototypeBuildRequest) -> PrototypeBuildArtifacts:
        """입력 임베딩으로 pack/build-state 결과물을 생성한다."""


def describe_prototype_build_strategy(strategy: object) -> dict[str, Any]:
    """manifest에 남길 전략 설명을 만든다."""
    if is_dataclass(strategy):
        return asdict(strategy)
    return {
        "name": getattr(strategy, "name", type(strategy).__name__),
        "class_name": type(strategy).__name__,
    }
