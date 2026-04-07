"""로컬 입력을 EmbeddedTrainingExample으로 변환한다."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

from agent.src.infrastructure.repositories.scored_event_repository import (
    StoredScoredEvent,
)
from agent.src.services.inference.scoring_service import ScoringService
from agent.src.services.training.local_training_service import EmbeddedTrainingExample
from shared.src.contracts.adapter_contracts import VectorAdapterState
from shared.src.contracts.prototype_contracts import (
    PrototypePackPayload,
    extract_category_prototypes,
)
from shared.src.domain.entities.inference.events import ScoredEvent


@dataclass(slots=True)
class TrainingExampleSource:
    """학습 예시 생성에 필요한 최소 입력 단위."""

    query_id: str
    text: str
    occurred_at: datetime
    translated_text: str | None = None


@dataclass(slots=True)
class TrainingExampleBuildRequest:
    """학습 예시 생성 요청."""

    source_rows: tuple[TrainingExampleSource, ...] | list[TrainingExampleSource]
    adapter: Any
    adapter_state: VectorAdapterState
    prototype_pack: PrototypePackPayload
    model_id: str
    scoring_service: ScoringService


@dataclass(slots=True)
class StoredEventTrainingExampleBuildRequest:
    """저장된 scored event를 학습 예시로 재구성하는 요청."""

    stored_events: tuple[StoredScoredEvent, ...] | list[StoredScoredEvent]
    prototype_pack: PrototypePackPayload
    scoring_service: ScoringService
    adapter_state: VectorAdapterState | None = None


@dataclass(slots=True)
class TrainingExampleService:
    """로컬 source row를 EmbeddedTrainingExample으로 변환한다."""

    def build_examples(
        self,
        request: TrainingExampleBuildRequest,
    ) -> tuple[EmbeddedTrainingExample, ...]:
        if not request.source_rows:
            return ()

        texts = [row.text for row in request.source_rows]
        base_embeddings = request.adapter.embed_texts(texts)
        prototypes = extract_category_prototypes(request.prototype_pack)
        examples: list[EmbeddedTrainingExample] = []
        for row, base_embedding in zip(
            request.source_rows,
            base_embeddings,
            strict=True,
        ):
            adapted_embedding = request.adapter_state.apply(base_embedding)
            scored_event = ScoredEvent(
                query_id=row.query_id,
                occurred_at=row.occurred_at,
                translated_text=row.translated_text,
                embedding_model_id=request.model_id,
                translation_model_id=request.prototype_pack.translation_model_id,
                category_scores=request.scoring_service.score(
                    adapted_embedding,
                    prototypes,
                ),
            )
            examples.append(
                EmbeddedTrainingExample(
                    scored_event=scored_event,
                    embedding=adapted_embedding,
                    base_embedding=list(base_embedding),
                )
            )
        return tuple(examples)

    def build_examples_from_stored_events(
        self,
        request: StoredEventTrainingExampleBuildRequest,
    ) -> tuple[EmbeddedTrainingExample, ...]:
        usable_events = [
            stored_event
            for stored_event in request.stored_events
            if stored_event.base_embedding is not None
            and len(stored_event.base_embedding) > 0
        ]
        if not usable_events:
            return ()

        adapter_state = request.adapter_state or VectorAdapterState.identity(
            model_id=usable_events[0].scored_event.embedding_model_id,
            model_revision="local_cached_identity",
            training_scope="adapter_only",
            embedding_dim=len(usable_events[0].base_embedding),
            updated_at=datetime.now(tz=timezone.utc),
        )
        prototypes = extract_category_prototypes(request.prototype_pack)
        examples: list[EmbeddedTrainingExample] = []
        for stored_event in usable_events:
            base_embedding = stored_event.base_embedding
            if base_embedding is None:
                continue
            adapted_embedding = adapter_state.apply(base_embedding)
            scored_event = replace(
                stored_event.scored_event,
                category_scores=request.scoring_service.score(
                    adapted_embedding,
                    prototypes,
                ),
            )
            examples.append(
                EmbeddedTrainingExample(
                    scored_event=scored_event,
                    embedding=adapted_embedding,
                    base_embedding=list(base_embedding),
                )
            )
        return tuple(examples)
