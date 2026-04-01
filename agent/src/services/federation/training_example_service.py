"""로컬 입력을 EmbeddedTrainingExample으로 변환한다."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from agent.src.services.inference.scoring_service import ScoringService
from agent.src.services.training.local_training_service import EmbeddedTrainingExample
from shared.src.contracts.prototype_contracts import (
    PrototypePackPayload,
    extract_category_prototypes,
)
from shared.src.domain.entities.inference.events import ScoredEvent
from shared.src.domain.entities.training.vector_adapter_state import VectorAdapterState


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
