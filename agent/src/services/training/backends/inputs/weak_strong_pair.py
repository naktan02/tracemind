"""Weak/strong multiview training input backend."""

from __future__ import annotations

from dataclasses import dataclass

from agent.src.services.training.examples.models import (
    EmbeddedTrainingExample,
)
from shared.src.contracts.prototype_contracts import extract_category_prototypes
from shared.src.domain.entities.inference.events import ScoredEvent

from .base import ANY_ADAPTER_KIND, WEAK_STRONG_PAIR_BACKEND_NAME
from .models import (
    StoredEventTrainingExampleBuildRequest,
    TrainingExampleBuildRequest,
)


@dataclass(slots=True)
class WeakStrongPairTrainingExampleBackend:
    """weak/strong pair가 준비된 source row를 multiview example로 변환한다."""

    backend_name: str = WEAK_STRONG_PAIR_BACKEND_NAME
    supported_adapter_kinds: tuple[str, ...] = (ANY_ADAPTER_KIND,)
    supports_stored_event_rebuild: bool = False

    def build_examples(
        self,
        request: TrainingExampleBuildRequest,
    ) -> tuple[EmbeddedTrainingExample, ...]:
        if not request.source_rows:
            return ()

        weak_texts: list[str] = []
        strong_texts: list[str] = []
        for row in request.source_rows:
            if row.weak_text is None or row.strong_text is None:
                raise ValueError(
                    "weak_strong_pair backend requires both weak_text and "
                    "strong_text for every TrainingExampleSource."
                )
            weak_texts.append(row.weak_text)
            strong_texts.append(row.strong_text)

        weak_base_embeddings = request.adapter.embed_texts(weak_texts)
        strong_base_embeddings = request.adapter.embed_texts(strong_texts)
        prototypes = extract_category_prototypes(request.prototype_pack)
        examples: list[EmbeddedTrainingExample] = []
        for row, weak_base_embedding, strong_base_embedding in zip(
            request.source_rows,
            weak_base_embeddings,
            strong_base_embeddings,
            strict=True,
        ):
            weak_embedding = request.adapter_state.apply(weak_base_embedding)
            strong_embedding = request.adapter_state.apply(strong_base_embedding)
            weak_scored_event = ScoredEvent(
                query_id=row.query_id,
                occurred_at=row.occurred_at,
                translated_text=row.weak_translated_text or row.translated_text,
                embedding_model_id=request.model_id,
                translation_model_id=request.prototype_pack.translation_model_id,
                category_scores=request.scoring_service.score(
                    weak_embedding,
                    prototypes,
                ),
            )
            strong_scored_event = ScoredEvent(
                query_id=row.query_id,
                occurred_at=row.occurred_at,
                translated_text=row.strong_translated_text or row.translated_text,
                embedding_model_id=request.model_id,
                translation_model_id=request.prototype_pack.translation_model_id,
                category_scores=request.scoring_service.score(
                    strong_embedding,
                    prototypes,
                ),
            )
            examples.append(
                EmbeddedTrainingExample(
                    scored_event=weak_scored_event,
                    embedding=list(strong_embedding),
                    base_embedding=list(weak_base_embedding),
                    view_kind="weak_strong_pair",
                    weak_scored_event=weak_scored_event,
                    weak_embedding=list(weak_embedding),
                    strong_scored_event=strong_scored_event,
                    strong_embedding=list(strong_embedding),
                    strong_base_embedding=list(strong_base_embedding),
                    metadata={
                        "training_input_backend_name": self.backend_name,
                        "selection_view": "weak",
                        "update_view": "strong",
                    },
                )
            )
        return tuple(examples)

    def build_examples_from_stored_events(
        self,
        request: StoredEventTrainingExampleBuildRequest,
    ) -> tuple[EmbeddedTrainingExample, ...]:
        del request
        raise ValueError(
            "weak_strong_pair backend is not supported for stored scored events yet. "
            "Stored event reconstruction currently lacks weak/strong view data."
        )
