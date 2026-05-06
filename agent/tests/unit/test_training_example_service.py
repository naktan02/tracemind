"""TrainingExampleService unit tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from agent.src.infrastructure.repositories.scored_event_repository import (
    StoredScoredEvent,
)
from agent.src.services.inference.scoring_service import ScoringService
from agent.src.services.training.backends.inputs import (
    registry as training_example_backend_registry,
)
from agent.src.services.training.backends.inputs.models import (
    StoredEventTrainingExampleBuildRequest,
    TrainingExampleBuildRequest,
    TrainingExampleSource,
)
from agent.src.services.training.backends.inputs.prototype_rescore import (
    PrototypeRescoringTrainingExampleBackend,
)
from agent.src.services.training.backends.inputs.weak_strong_pair import (
    WeakStrongPairTrainingExampleBackend,
)
from agent.src.services.training.backends.training.registry import (
    register_shared_adapter_training_backend,
)
from agent.src.services.training.examples.service import (
    TrainingExampleService,
)
from shared.src.config.registry_catalog_metadata import RegistryCatalogEntry
from shared.src.config.training_defaults import DEFAULT_TRAINING_PROFILE
from shared.src.contracts.adapter_contracts import VectorAdapterState
from shared.src.contracts.prototype_contracts import PrototypePackPayload
from shared.src.contracts.training_contracts import TrainingObjectiveConfig
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


def _registry_catalog_entry(
    *,
    item_name: str,
    family_name: str,
    supported_adapter_kinds: tuple[str, ...] = ("*",),
) -> RegistryCatalogEntry:
    return RegistryCatalogEntry(
        item_name=item_name,
        display_name=item_name,
        implementation_module=__name__,
        core_method_name=item_name,
        family_name=family_name,
        supported_adapter_kinds=supported_adapter_kinds,
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
    assert examples[0].metadata["raw_text"] == "panic panic"
    assert examples[0].metadata["training_text"] == "panic panic"
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


def test_weak_strong_pair_backend_builds_multiview_examples() -> None:
    service = TrainingExampleService(backend=WeakStrongPairTrainingExampleBackend())
    adapter = _StaticEmbeddingAdapter(
        {
            "panic weak": [1.0, 0.0],
            "panic strong": [0.8, 0.2],
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
                    query_id="q_fix",
                    text="panic panic",
                    weak_text="panic weak",
                    strong_text="panic strong",
                    occurred_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
                ),
            ),
            adapter=adapter,
            adapter_state=adapter_state,
            prototype_pack=_pack_payload(),
            model_id="hash_debug",
            scoring_service=ScoringService(),
        )
    )

    assert len(examples) == 1
    example = examples[0]
    assert example.view_kind == "weak_strong_pair"
    assert example.evidence_scored_event.query_id == "q_fix"
    assert example.update_scored_event.query_id == "q_fix"
    assert example.weak_embedding == [1.0, 0.0]
    assert example.strong_embedding == pytest.approx(
        [0.9701425001453318, 0.24253562503633294]
    )
    assert example.update_embedding == pytest.approx(
        [0.9701425001453318, 0.24253562503633294]
    )
    assert example.metadata["selection_view"] == "weak"
    assert example.metadata["update_view"] == "strong"
    assert example.metadata["raw_text"] == "panic panic"
    assert example.metadata["training_text"] == "panic strong"
    assert example.metadata["strong_text"] == "panic strong"


def test_weak_strong_pair_backend_rejects_stored_event_rebuild() -> None:
    service = TrainingExampleService(backend=WeakStrongPairTrainingExampleBackend())

    with pytest.raises(
        ValueError,
        match="not supported for stored scored events yet",
    ):
        service.build_examples_from_stored_events(
            StoredEventTrainingExampleBuildRequest(
                stored_events=(),
                prototype_pack=_pack_payload(),
                scoring_service=ScoringService(),
            )
        )


@dataclass(slots=True)
class _ConstantTrainingExampleBackend:
    backend_name: str = "constant_examples"
    supported_adapter_kinds: tuple[str, ...] = ("*",)

    def build_examples(
        self,
        request: TrainingExampleBuildRequest,
    ) -> tuple:
        return ()

    def build_examples_from_stored_events(
        self,
        request: StoredEventTrainingExampleBuildRequest,
    ) -> tuple:
        return ()


def test_training_example_service_selects_backend_from_objective_config() -> None:
    training_example_backend_registry.register_training_example_backend(
        "constant_examples",
        factory=lambda _objective_config: _ConstantTrainingExampleBackend(),
        catalog_entry=_registry_catalog_entry(
            item_name="constant_examples",
            family_name="example_generation",
        ),
    )

    service = TrainingExampleService.from_objective_config(
        TrainingObjectiveConfig(
            training_backend_name="diagonal_scale_heuristic",
            example_generation_backend_name="constant_examples",
        )
    )

    assert isinstance(service.backend, _ConstantTrainingExampleBackend)
    assert service.backend.backend_name == "constant_examples"
    assert not isinstance(service.backend, PrototypeRescoringTrainingExampleBackend)


def test_training_example_service_uses_profile_default_backend_when_omitted() -> None:
    service = TrainingExampleService.from_objective_config(
        TrainingObjectiveConfig(
            training_backend_name=DEFAULT_TRAINING_PROFILE.training_backend_name,
        )
    )

    assert isinstance(service.backend, PrototypeRescoringTrainingExampleBackend)
    assert service.backend.backend_name == (
        DEFAULT_TRAINING_PROFILE.example_generation_backend_name
    )


def test_training_example_service_rejects_incompatible_backend_family() -> None:
    @dataclass(slots=True)
    class _TestShiftTrainingBackend:
        backend_name: str = "test_shift_example_training_backend"
        payload_format: str = "test_shift_update"
        adapter_kind: str = "test_shift"

        def build_update(
            self,
            *,
            training_task,
            model_manifest,
            accepted_examples,
            created_at,
        ):
            raise AssertionError("이 테스트에서는 호출되면 안 됩니다.")

        def to_payload(self, update):
            return update

        def build_client_metrics(self, update) -> dict[str, float]:
            del update
            return {}

    @dataclass(slots=True)
    class _DiagonalOnlyTrainingExampleBackend:
        backend_name: str = "diagonal_only_training_examples"
        supported_adapter_kinds: tuple[str, ...] = ("diagonal_scale",)

        def build_examples(self, request: TrainingExampleBuildRequest) -> tuple:
            del request
            return ()

        def build_examples_from_stored_events(
            self,
            request: StoredEventTrainingExampleBuildRequest,
        ) -> tuple:
            del request
            return ()

    register_shared_adapter_training_backend(
        "test_shift_example_training_backend",
        factory=lambda _objective_config: _TestShiftTrainingBackend(),
        catalog_entry=_registry_catalog_entry(
            item_name="test_shift_example_training_backend",
            family_name="test_shift",
            supported_adapter_kinds=("test_shift",),
        ),
    )
    training_example_backend_registry.register_training_example_backend(
        "diagonal_only_training_examples",
        factory=lambda _objective_config: _DiagonalOnlyTrainingExampleBackend(),
        catalog_entry=_registry_catalog_entry(
            item_name="diagonal_only_training_examples",
            family_name="example_generation",
            supported_adapter_kinds=("diagonal_scale",),
        ),
    )

    with pytest.raises(
        ValueError,
        match="Incompatible training example backend",
    ):
        TrainingExampleService.from_objective_config(
            TrainingObjectiveConfig(
                training_backend_name="test_shift_example_training_backend",
                example_generation_backend_name="diagonal_only_training_examples",
            )
        )
