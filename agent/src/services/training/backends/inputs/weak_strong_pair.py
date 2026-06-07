"""Weak/strong multiview training input backend."""

from __future__ import annotations

from dataclasses import dataclass

from agent.src.services.training.examples.models import (
    EmbeddedTrainingExample,
)
from shared.src.contracts.registry_catalog_metadata import RegistryCatalogEntry
from shared.src.contracts.training_contracts import TrainingObjectiveConfig
from shared.src.domain.entities.inference.events import AnalysisEvent

from .base import ANY_ADAPTER_KIND, WEAK_STRONG_PAIR_BACKEND_NAME
from .models import (
    StoredEventTrainingExampleBuildRequest,
    TrainingExampleBuildRequest,
    TrainingExampleSource,
)
from .registry import register_training_example_backend

WEAK_STRONG_PAIR_EXAMPLE_BACKEND_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name=WEAK_STRONG_PAIR_BACKEND_NAME,
    display_name=WEAK_STRONG_PAIR_BACKEND_NAME,
    implementation_module="agent.src.services.training.backends.inputs.weak_strong_pair",
    core_method_name=WEAK_STRONG_PAIR_BACKEND_NAME,
    family_name="example_generation",
    supported_adapter_kinds=(ANY_ADAPTER_KIND,),
    metadata={"supports_stored_event_rebuild": False},
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

        weak_texts, strong_texts = _require_weak_strong_texts(request.source_rows)
        weak_base_embeddings = request.adapter.embed_texts(weak_texts)
        strong_base_embeddings = request.adapter.embed_texts(strong_texts)
        return tuple(
            _to_embedded_example(
                source_row=source_row,
                weak_base_embedding=list(weak_base_embedding),
                strong_base_embedding=list(strong_base_embedding),
                request=request,
            )
            for source_row, weak_base_embedding, strong_base_embedding in zip(
                request.source_rows,
                weak_base_embeddings,
                strong_base_embeddings,
                strict=True,
            )
        )

    def build_examples_from_stored_events(
        self,
        request: StoredEventTrainingExampleBuildRequest,
    ) -> tuple[EmbeddedTrainingExample, ...]:
        del request
        raise ValueError(
            "weak_strong_pair backend is not supported for stored analysis events yet. "
            "Stored event reconstruction currently lacks weak/strong view data."
        )


def _require_weak_strong_texts(
    source_rows: tuple[TrainingExampleSource, ...] | list[TrainingExampleSource],
) -> tuple[list[str], list[str]]:
    weak_texts: list[str] = []
    strong_texts: list[str] = []
    for row in source_rows:
        weak_text = row.weak_translated_text or row.weak_text
        strong_text = row.strong_translated_text or row.strong_text
        if weak_text is None or strong_text is None:
            raise ValueError(
                "weak_strong_pair backend requires weak and strong text for each row."
            )
        weak_texts.append(weak_text)
        strong_texts.append(strong_text)
    return weak_texts, strong_texts


def _to_embedded_example(
    *,
    source_row: TrainingExampleSource,
    weak_base_embedding: list[float],
    strong_base_embedding: list[float],
    request: TrainingExampleBuildRequest,
) -> EmbeddedTrainingExample:
    weak_scores = request.scoring_service.score(
        weak_base_embedding,
        {},
        shared_state=request.adapter_state,
    )
    strong_scores = request.scoring_service.score(
        strong_base_embedding,
        {},
        shared_state=request.adapter_state,
    )
    weak_event = _analysis_event_for_view(
        source_row=source_row,
        category_scores=weak_scores,
        model_id=request.model_id,
        confidence_kind=request.scoring_service.confidence_kind,
        view_kind="weak",
    )
    strong_event = _analysis_event_for_view(
        source_row=source_row,
        category_scores=strong_scores,
        model_id=request.model_id,
        confidence_kind=request.scoring_service.confidence_kind,
        view_kind="strong",
    )
    metadata: dict[str, str | int | float | bool] = {
        "raw_text": source_row.text,
        "training_text": source_row.strong_text or source_row.text,
    }
    if source_row.weak_text is not None:
        metadata["weak_text"] = source_row.weak_text
    if source_row.strong_text is not None:
        metadata["strong_text"] = source_row.strong_text
    if source_row.translated_text is not None:
        metadata["translated_text"] = source_row.translated_text
    return EmbeddedTrainingExample(
        analysis_event=weak_event,
        embedding=strong_base_embedding,
        base_embedding=weak_base_embedding,
        view_kind=WEAK_STRONG_PAIR_BACKEND_NAME,
        weak_analysis_event=weak_event,
        weak_embedding=weak_base_embedding,
        strong_analysis_event=strong_event,
        strong_embedding=strong_base_embedding,
        strong_base_embedding=strong_base_embedding,
        metadata=metadata,
    )


def _analysis_event_for_view(
    *,
    source_row: TrainingExampleSource,
    category_scores: dict[str, float],
    model_id: str,
    confidence_kind: str,
    view_kind: str,
) -> AnalysisEvent:
    translated_text = (
        source_row.weak_translated_text
        if view_kind == "weak"
        else source_row.strong_translated_text
    )
    return AnalysisEvent(
        query_id=source_row.query_id,
        occurred_at=source_row.occurred_at,
        translated_text=translated_text or source_row.translated_text,
        embedding_model_id=model_id,
        translation_model_id=None,
        category_scores=category_scores,
        scorer_family="classifier",
        scorer_name="weak_strong_pair",
        confidence_kind=confidence_kind,
        metadata={"view_kind": view_kind},
    )


@register_training_example_backend(
    WEAK_STRONG_PAIR_BACKEND_NAME,
    catalog_entry=WEAK_STRONG_PAIR_EXAMPLE_BACKEND_CATALOG_ENTRY,
)
def build_weak_strong_pair_training_example_backend(
    objective_config: TrainingObjectiveConfig,
) -> WeakStrongPairTrainingExampleBackend:
    """registry용 weak/strong pair example backend factory."""

    del objective_config
    return WeakStrongPairTrainingExampleBackend()
