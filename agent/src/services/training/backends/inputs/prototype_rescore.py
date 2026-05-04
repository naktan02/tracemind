"""Prototype rescoring training input backend."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from agent.src.services.training.examples.models import (
    EmbeddedTrainingExample,
)
from methods.prototype.training_inputs.examples import (
    PrototypeSingleViewTrainingInput,
    build_prototype_rescore_inputs,
    build_prototype_rescore_inputs_from_stored_events,
)
from shared.src.contracts.common_types import TrainingScope
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
        inputs = build_prototype_rescore_inputs(
            source_rows=request.source_rows,
            base_embeddings=base_embeddings,
            adapter_state=request.adapter_state,
            prototype_pack=request.prototype_pack,
            model_id=request.model_id,
            scorer=request.scoring_service,
        )
        return tuple(_to_embedded_example(input_item) for input_item in inputs)

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
        inputs = build_prototype_rescore_inputs_from_stored_events(
            stored_events=usable_events,
            adapter_state=adapter_state,
            prototype_pack=request.prototype_pack,
            scorer=request.scoring_service,
        )
        return tuple(_to_embedded_example(input_item) for input_item in inputs)


def _to_embedded_example(
    input_item: PrototypeSingleViewTrainingInput,
) -> EmbeddedTrainingExample:
    return EmbeddedTrainingExample(
        scored_event=input_item.scored_event,
        embedding=list(input_item.embedding),
        base_embedding=list(input_item.base_embedding),
    )
