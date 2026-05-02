"""Prototype rescoring training input backend."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone

from agent.src.services.training.examples.models import (
    EmbeddedTrainingExample,
)
from shared.src.contracts.common_types import TrainingScope
from shared.src.contracts.prototype_contracts import extract_category_prototypes
from shared.src.domain.entities.inference.events import ScoredEvent
from shared.src.domain.entities.training.shared_adapter_state import (
    IdentitySharedAdapterState,
)

from .base import ANY_ADAPTER_KIND, PROTOTYPE_RESCORE_BACKEND_NAME
from .models import (
    StoredEventTrainingExampleBuildRequest,
    TrainingExampleBuildRequest,
)


@dataclass(slots=True)
class PrototypeRescoringTrainingExampleBackend:
    """현재 prototype 재점수화 기반 학습 예시 재구성 backend."""

    backend_name: str = PROTOTYPE_RESCORE_BACKEND_NAME
    supported_adapter_kinds: tuple[str, ...] = (ANY_ADAPTER_KIND,)
    supports_stored_event_rebuild: bool = True

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

        adapter_state = request.adapter_state or IdentitySharedAdapterState(
            model_id=usable_events[0].scored_event.embedding_model_id,
            model_revision="local_cached_identity",
            training_scope=TrainingScope.ADAPTER_ONLY,
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


__all__ = ["PrototypeRescoringTrainingExampleBackend"]
