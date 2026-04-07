"""TrainingExampleService unit tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from agent.src.infrastructure.repositories.scored_event_repository import (
    StoredScoredEvent,
)
from agent.src.services.federation import (
    StoredEventTrainingExampleBuildRequest,
    TrainingExampleBuildRequest,
    TrainingExampleService,
    TrainingExampleSource,
)
from agent.src.services.inference.scoring_service import ScoringService
from shared.src.contracts.adapter_contracts import VectorAdapterState
from shared.src.contracts.prototype_contracts import PrototypePackPayload
from shared.src.domain.entities.inference.events import ScoredEvent


class _StaticEmbeddingAdapter:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vectors[text]) for text in texts]


@dataclass(slots=True)
class _CustomSharedAdapterState:
    schema_version: str = "custom_state.v1"
    adapter_kind: str = "custom_state"
    model_id: str = "hash_debug"
    model_revision: str = "main"
    training_scope: str = "adapter_only"
    updated_at: datetime = datetime(2026, 4, 2, tzinfo=timezone.utc)
    embedding_dim: int = 2

    def apply(self, embedding) -> list[float]:
        return [float(embedding[0]), 0.0]


def _pack_payload() -> PrototypePackPayload:
    return PrototypePackPayload.model_validate(
        {
            "schema_version": "prototype_pack.v1",
            "prototype_version": "proto_test_v1",
            "embedding_model_id": "hash_debug",
            "embedding_model_revision": "main",
            "mapping_version": "ourafla_to_4cat.v1",
            "build_method": "mean_centroid_l2_normalized",
            "distance_metric": "cosine",
            "built_at": "2026-04-02T00:00:00+00:00",
            "categories": {
                "anxiety": [
                    {
                        "prototype_id": "anxiety:single",
                        "centroid": [1.0, 0.0],
                        "sample_count": 2,
                    }
                ],
                "normal": [
                    {
                        "prototype_id": "normal:single",
                        "centroid": [0.0, 1.0],
                        "sample_count": 2,
                    }
                ],
            },
        }
    )


def test_training_example_service_builds_scored_examples_from_source_rows() -> None:
    service = TrainingExampleService()
    adapter = _StaticEmbeddingAdapter(
        {
            "panic panic": [1.0, 0.0],
            "calm calm": [0.0, 1.0],
        }
    )
    adapter_state = VectorAdapterState.identity(
        model_id="hash_debug",
        model_revision="main",
        training_scope="adapter_only",
        embedding_dim=2,
        updated_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
    )

    examples = service.build_examples(
        TrainingExampleBuildRequest(
            source_rows=(
                TrainingExampleSource(
                    query_id="q1",
                    text="panic panic",
                    occurred_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
                ),
                TrainingExampleSource(
                    query_id="q2",
                    text="calm calm",
                    occurred_at=datetime(2026, 4, 2, 0, 1, tzinfo=timezone.utc),
                ),
            ),
            adapter=adapter,
            adapter_state=adapter_state,
            prototype_pack=_pack_payload(),
            model_id="hash_debug",
            scoring_service=ScoringService(),
        )
    )

    assert len(examples) == 2
    assert examples[0].scored_event.query_id == "q1"
    assert examples[0].base_embedding == [1.0, 0.0]
    assert examples[0].embedding == [1.0, 0.0]
    assert examples[0].scored_event.category_scores["anxiety"] == 1.0
    assert examples[1].scored_event.query_id == "q2"
    assert examples[1].scored_event.category_scores["normal"] == 1.0


def test_training_example_service_returns_empty_tuple_for_empty_rows() -> None:
    service = TrainingExampleService()
    adapter_state = VectorAdapterState.identity(
        model_id="hash_debug",
        model_revision="main",
        training_scope="adapter_only",
        embedding_dim=2,
        updated_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
    )

    examples = service.build_examples(
        TrainingExampleBuildRequest(
            source_rows=(),
            adapter=_StaticEmbeddingAdapter({}),
            adapter_state=adapter_state,
            prototype_pack=_pack_payload(),
            model_id="hash_debug",
            scoring_service=ScoringService(),
        )
    )

    assert examples == ()


def test_training_example_service_rebuilds_examples_from_stored_events() -> None:
    service = TrainingExampleService()
    examples = service.build_examples_from_stored_events(
        StoredEventTrainingExampleBuildRequest(
            stored_events=(
                StoredScoredEvent(
                    scored_event=ScoredEvent(
                        query_id="q1",
                        occurred_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
                        translated_text="panic panic",
                        embedding_model_id="hash_debug",
                        translation_model_id=None,
                        category_scores={"anxiety": 0.4, "normal": 0.6},
                    ),
                    base_embedding=[1.0, 0.0],
                ),
            ),
            prototype_pack=_pack_payload(),
            scoring_service=ScoringService(),
        )
    )

    assert len(examples) == 1
    assert examples[0].base_embedding == [1.0, 0.0]
    assert examples[0].embedding == [1.0, 0.0]
    assert examples[0].scored_event.category_scores["anxiety"] == 1.0


def test_training_example_service_accepts_custom_shared_adapter_state() -> None:
    service = TrainingExampleService()
    adapter = _StaticEmbeddingAdapter({"panic panic": [0.2, 1.0]})

    examples = service.build_examples(
        TrainingExampleBuildRequest(
            source_rows=(
                TrainingExampleSource(
                    query_id="q1",
                    text="panic panic",
                    occurred_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
                ),
            ),
            adapter=adapter,
            adapter_state=_CustomSharedAdapterState(),
            prototype_pack=_pack_payload(),
            model_id="hash_debug",
            scoring_service=ScoringService(),
        )
    )

    assert len(examples) == 1
    assert examples[0].base_embedding == [0.2, 1.0]
    assert examples[0].embedding == [0.2, 0.0]
    assert examples[0].scored_event.category_scores["anxiety"] == 1.0
