"""로컬 입력을 EmbeddedTrainingExample으로 변환한다."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Protocol

from agent.src.infrastructure.repositories.scored_event_repository import (
    StoredScoredEvent,
)
from agent.src.services.inference.scoring_service import ScoringService
from agent.src.services.training.local_training_service import EmbeddedTrainingExample
from shared.src.contracts.common_types import TrainingScope
from shared.src.contracts.prototype_contracts import (
    PrototypePackPayload,
    extract_category_prototypes,
)
from shared.src.contracts.training_contracts import TrainingObjectiveConfig
from shared.src.domain.entities.inference.events import ScoredEvent
from shared.src.domain.entities.training.shared_adapter_state import (
    IdentitySharedAdapterState,
    SharedAdapterState,
)


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
    adapter_state: SharedAdapterState
    prototype_pack: PrototypePackPayload
    model_id: str
    scoring_service: ScoringService


@dataclass(slots=True)
class StoredEventTrainingExampleBuildRequest:
    """저장된 scored event를 학습 예시로 재구성하는 요청."""

    stored_events: tuple[StoredScoredEvent, ...] | list[StoredScoredEvent]
    prototype_pack: PrototypePackPayload
    scoring_service: ScoringService
    adapter_state: SharedAdapterState | None = None


class TrainingExampleBackend(Protocol):
    """학습 예시 재구성 backend 인터페이스."""

    backend_name: str

    def build_examples(
        self,
        request: TrainingExampleBuildRequest,
    ) -> tuple[EmbeddedTrainingExample, ...]:
        """source row에서 학습 예시를 만든다."""

    def build_examples_from_stored_events(
        self,
        request: StoredEventTrainingExampleBuildRequest,
    ) -> tuple[EmbeddedTrainingExample, ...]:
        """stored event에서 학습 예시를 재구성한다."""


TrainingExampleBackendFactory = Callable[
    [TrainingObjectiveConfig],
    TrainingExampleBackend,
]


@dataclass(slots=True)
class PrototypeRescoringTrainingExampleBackend:
    """현재 prototype 재점수화 기반 학습 예시 재구성 backend."""

    backend_name: str = "prototype_rescore"

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


_TRAINING_EXAMPLE_BACKEND_REGISTRY: dict[str, TrainingExampleBackendFactory] = {}


def register_training_example_backend(
    *backend_names: str,
    factory: TrainingExampleBackendFactory,
) -> None:
    """얇은 wiring registry에 training example backend를 등록한다."""

    for backend_name in backend_names:
        _TRAINING_EXAMPLE_BACKEND_REGISTRY[backend_name.strip().lower()] = factory


def build_training_example_backend(
    backend_name: str,
    *,
    objective_config: TrainingObjectiveConfig,
) -> TrainingExampleBackend:
    """backend 이름과 objective config로 training example backend를 조립한다."""

    normalized_name = backend_name.strip().lower()
    factory = _TRAINING_EXAMPLE_BACKEND_REGISTRY.get(normalized_name)
    if factory is not None:
        return factory(objective_config)
    raise ValueError(f"Unsupported training example backend: {backend_name}.")


@dataclass(slots=True)
class TrainingExampleService:
    """로컬 source row를 EmbeddedTrainingExample으로 변환한다."""

    backend: TrainingExampleBackend = field(
        default_factory=PrototypeRescoringTrainingExampleBackend
    )

    @classmethod
    def from_objective_config(
        cls,
        objective_config: TrainingObjectiveConfig,
    ) -> "TrainingExampleService":
        backend_name = (
            objective_config.example_generation_backend_name
            or "prototype_rescore"
        )
        return cls(
            backend=build_training_example_backend(
                backend_name,
                objective_config=objective_config,
            )
        )

    def build_examples(
        self,
        request: TrainingExampleBuildRequest,
    ) -> tuple[EmbeddedTrainingExample, ...]:
        return self.backend.build_examples(request)

    def build_examples_from_stored_events(
        self,
        request: StoredEventTrainingExampleBuildRequest,
    ) -> tuple[EmbeddedTrainingExample, ...]:
        return self.backend.build_examples_from_stored_events(request)


register_training_example_backend(
    "prototype_rescore",
    factory=lambda _objective_config: PrototypeRescoringTrainingExampleBackend(),
)
