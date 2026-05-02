"""Training input backend request models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from agent.src.infrastructure.repositories.scored_event_repository import (
    StoredScoredEvent,
)
from agent.src.services.inference.scoring_service import ScoringService
from shared.src.contracts.prototype_contracts import PrototypePackPayload
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.services.embedding_adapter import EmbeddingAdapter


@dataclass(slots=True)
class TrainingExampleSource:
    """학습 예시 생성에 필요한 최소 입력 단위."""

    query_id: str
    text: str
    occurred_at: datetime
    translated_text: str | None = None
    weak_text: str | None = None
    strong_text: str | None = None
    weak_translated_text: str | None = None
    strong_translated_text: str | None = None


@dataclass(slots=True)
class TrainingExampleBuildRequest:
    """학습 예시 생성 요청."""

    source_rows: tuple[TrainingExampleSource, ...] | list[TrainingExampleSource]
    adapter: EmbeddingAdapter
    adapter_state: SharedAdapterState
    prototype_pack: PrototypePackPayload
    model_id: str
    scoring_service: ScoringService


@dataclass(slots=True)
class StoredEventTrainingExampleBuildRequest:
    """저장된 scored event를 학습 예시로 재구성하는 요청."""

    stored_events: tuple[StoredScoredEvent, ...] | list[StoredScoredEvent]
    prototype_pack: PrototypePackPayload
    scoring_service: ScoringService
    adapter_state: SharedAdapterState | None = None


__all__ = [
    "StoredEventTrainingExampleBuildRequest",
    "TrainingExampleBuildRequest",
    "TrainingExampleSource",
]
