"""Weak/strong multiview training input backend."""

from __future__ import annotations

from dataclasses import dataclass

from agent.src.services.training.examples.models import (
    EmbeddedTrainingExample,
)
from methods.prototype.training_inputs.examples import (
    PrototypeWeakStrongTrainingInput,
    build_prototype_weak_strong_inputs,
    require_weak_strong_texts,
)

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

        weak_texts, strong_texts = require_weak_strong_texts(request.source_rows)
        weak_base_embeddings = request.adapter.embed_texts(weak_texts)
        strong_base_embeddings = request.adapter.embed_texts(strong_texts)
        inputs = build_prototype_weak_strong_inputs(
            source_rows=request.source_rows,
            weak_base_embeddings=weak_base_embeddings,
            strong_base_embeddings=strong_base_embeddings,
            adapter_state=request.adapter_state,
            prototype_pack=request.prototype_pack,
            model_id=request.model_id,
            scorer=request.scoring_service,
            backend_name=self.backend_name,
        )
        return tuple(_to_embedded_example(input_item) for input_item in inputs)

    def build_examples_from_stored_events(
        self,
        request: StoredEventTrainingExampleBuildRequest,
    ) -> tuple[EmbeddedTrainingExample, ...]:
        del request
        raise ValueError(
            "weak_strong_pair backend is not supported for stored scored events yet. "
            "Stored event reconstruction currently lacks weak/strong view data."
        )


def _to_embedded_example(
    input_item: PrototypeWeakStrongTrainingInput,
) -> EmbeddedTrainingExample:
    return EmbeddedTrainingExample(
        scored_event=input_item.weak_scored_event,
        embedding=list(input_item.strong_embedding),
        base_embedding=list(input_item.weak_base_embedding),
        view_kind="weak_strong_pair",
        weak_scored_event=input_item.weak_scored_event,
        weak_embedding=list(input_item.weak_embedding),
        strong_scored_event=input_item.strong_scored_event,
        strong_embedding=list(input_item.strong_embedding),
        strong_base_embedding=list(input_item.strong_base_embedding),
        metadata=dict(input_item.metadata),
    )
