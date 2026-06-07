"""TrainingExampleService unit tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from agent.src.services.inference.scoring_backends.base import ScoringAssets
from agent.src.services.inference.scoring_service import ScoringService
from agent.src.services.training.backends.inputs import (
    registry as training_example_backend_registry,
)
from agent.src.services.training.backends.inputs.models import (
    StoredEventTrainingExampleBuildRequest,
    TrainingExampleBuildRequest,
    TrainingExampleSource,
)
from agent.src.services.training.backends.inputs.weak_strong_pair import (
    WeakStrongPairTrainingExampleBackend,
)
from agent.src.services.training.examples.service import (
    TrainingExampleService,
)
from methods.adaptation.local_update_registry import (
    register_shared_adapter_training_backend,
)
from methods.federated_ssl.runtime_fallbacks import RUNTIME_FALLBACK_TRAINING_PROFILE
from shared.src.contracts.registry_catalog_metadata import RegistryCatalogEntry
from shared.src.contracts.training_contracts import TrainingObjectiveConfig
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState


class _StaticEmbeddingAdapter:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vectors[text]) for text in texts]


@dataclass(slots=True)
class _IdentitySharedAdapterState:
    schema_version: str = "test_identity_state.v1"
    adapter_kind: str = "test_identity"
    model_id: str = "hash_debug"
    model_revision: str = "main"
    training_scope: str = "adapter_only"
    updated_at: datetime = datetime(2026, 4, 2, tzinfo=timezone.utc)
    embedding_dim: int = 2

    def apply(self, embedding) -> list[float]:
        return [float(value) for value in embedding]


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


@dataclass(slots=True)
class _VectorScoringBackend:
    backend_name: str = "test_vector_classifier"
    confidence_kind: str = "test_vector_top1"
    supported_adapter_kinds: tuple[str, ...] = ("*",)
    requires_shared_state: bool = True

    def score(
        self,
        embedding,
        scoring_assets: ScoringAssets,
        shared_state: SharedAdapterState | None = None,
    ) -> dict[str, float]:
        del scoring_assets
        transformed = (
            shared_state.apply(embedding) if shared_state is not None else embedding
        )
        return {"anxiety": float(transformed[0]), "normal": float(transformed[1])}


def _classifier_scoring_service() -> ScoringService:
    return ScoringService(backend=_VectorScoringBackend())


def test_weak_strong_pair_backend_builds_multiview_examples() -> None:
    service = TrainingExampleService(backend=WeakStrongPairTrainingExampleBackend())
    adapter = _StaticEmbeddingAdapter(
        {
            "panic weak": [1.0, 0.0],
            "panic strong": [0.8, 0.2],
        }
    )
    adapter_state = _IdentitySharedAdapterState()

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
            model_id="hash_debug",
            scoring_service=_classifier_scoring_service(),
        )
    )

    assert len(examples) == 1
    example = examples[0]
    assert example.view_kind == "weak_strong_pair"
    assert example.evidence_analysis_event.query_id == "q_fix"
    assert example.update_analysis_event.query_id == "q_fix"
    assert example.weak_embedding == [1.0, 0.0]
    assert example.strong_embedding == [0.8, 0.2]
    assert example.update_embedding == [0.8, 0.2]
    assert example.metadata["raw_text"] == "panic panic"
    assert example.metadata["training_text"] == "panic strong"
    assert example.metadata["strong_text"] == "panic strong"
    assert example.weak_analysis_event.category_scores["anxiety"] == 1.0
    assert example.strong_analysis_event.category_scores["anxiety"] == 0.8


def test_training_example_service_requires_explicit_backend() -> None:
    service = TrainingExampleService()

    with pytest.raises(ValueError, match="requires an explicit backend"):
        service.build_examples(
            TrainingExampleBuildRequest(
                source_rows=(),
                adapter=_StaticEmbeddingAdapter({}),
                adapter_state=_IdentitySharedAdapterState(),
                model_id="hash_debug",
                scoring_service=_classifier_scoring_service(),
            )
        )


def test_weak_strong_pair_backend_rejects_stored_event_rebuild() -> None:
    service = TrainingExampleService(backend=WeakStrongPairTrainingExampleBackend())

    with pytest.raises(
        ValueError,
        match="not supported for stored analysis events yet",
    ):
        service.build_examples_from_stored_events(
            StoredEventTrainingExampleBuildRequest(
                stored_events=(),
                scoring_service=_classifier_scoring_service(),
            )
        )


@dataclass(slots=True)
class _ConstantTrainingExampleBackend:
    backend_name: str = "constant_examples"
    supported_adapter_kinds: tuple[str, ...] = ("*",)
    supports_stored_event_rebuild: bool = True

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
            training_backend_name="peft_classifier_trainer",
            example_generation_backend_name="constant_examples",
        )
    )

    assert isinstance(service.backend, _ConstantTrainingExampleBackend)
    assert service.backend.backend_name == "constant_examples"


def test_training_example_service_uses_profile_default_backend_when_omitted() -> None:
    service = TrainingExampleService.from_objective_config(
        TrainingObjectiveConfig(
            training_backend_name=RUNTIME_FALLBACK_TRAINING_PROFILE.training_backend_name,
        )
    )

    assert service.backend.backend_name == (
        RUNTIME_FALLBACK_TRAINING_PROFILE.example_generation_backend_name
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
    class _PeftOnlyTrainingExampleBackend:
        backend_name: str = "peft_only_training_examples"
        supported_adapter_kinds: tuple[str, ...] = ("peft_classifier",)

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
        "peft_only_training_examples",
        factory=lambda _objective_config: _PeftOnlyTrainingExampleBackend(),
        catalog_entry=_registry_catalog_entry(
            item_name="peft_only_training_examples",
            family_name="example_generation",
            supported_adapter_kinds=("peft_classifier",),
        ),
    )

    with pytest.raises(
        ValueError,
        match="Incompatible training example backend",
    ):
        TrainingExampleService.from_objective_config(
            TrainingObjectiveConfig(
                training_backend_name="test_shift_example_training_backend",
                example_generation_backend_name="peft_only_training_examples",
            )
        )
