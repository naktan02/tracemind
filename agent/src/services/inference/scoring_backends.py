"""Scoring backend 구현과 resolver."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from agent.src.services.inference.scoring_policies import (
    MaxCosineScorePolicy,
    PrototypeScorePolicy,
    build_prototype_score_policy,
)
from shared.src.config.registry_catalog_metadata import (
    RegistryCatalogEntry,
    dedupe_registry_catalog_entries,
)
from shared.src.config.training_defaults import DEFAULT_TRAINING_PROFILE
from shared.src.contracts.adapter_contracts import ClassifierHeadState
from shared.src.contracts.training_contracts import TrainingObjectiveConfig
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState

PROTOTYPE_SIMILARITY_BACKEND_NAME = "prototype_similarity"
CLASSIFIER_HEAD_LOGITS_BACKEND_NAME = "classifier_head_logits"
PROTOTYPE_SIMILARITY_CONFIDENCE_KIND = "prototype_similarity_top1"
CLASSIFIER_HEAD_LOGITS_CONFIDENCE_KIND = "classifier_head_logit_top1"


class ScoringBackend(Protocol):
    """category score dict를 계산하는 backend 인터페이스."""

    backend_name: str
    confidence_kind: str
    supported_adapter_kinds: tuple[str, ...]
    requires_shared_state: bool

    def score(
        self,
        embedding: Sequence[float],
        prototypes: Mapping[str, Sequence[float] | Sequence[Sequence[float]]],
        shared_state: SharedAdapterState | None = None,
    ) -> dict[str, float]:
        """임베딩과 prototype들로 category score dict를 계산한다."""


ScoringBackendFactory = Callable[[TrainingObjectiveConfig, str], ScoringBackend]


@dataclass(slots=True)
class PrototypeSimilarityScoringBackend:
    """prototype similarity를 계산하고 policy로 category score를 접는다."""

    similarity_name: str = "cosine"
    policy: PrototypeScorePolicy = field(default_factory=MaxCosineScorePolicy)
    backend_name: str = PROTOTYPE_SIMILARITY_BACKEND_NAME
    confidence_kind: str = PROTOTYPE_SIMILARITY_CONFIDENCE_KIND
    supported_adapter_kinds: tuple[str, ...] = ("*",)
    requires_shared_state: bool = False

    def score(
        self,
        embedding: Sequence[float],
        prototypes: Mapping[str, Sequence[float] | Sequence[Sequence[float]]],
        shared_state: SharedAdapterState | None = None,
    ) -> dict[str, float]:
        del shared_state
        embedding_vector = _coerce_vector(embedding, vector_name="embedding")
        scores: dict[str, float] = {}
        for category, category_prototypes in prototypes.items():
            prototype_vectors = _coerce_prototype_vectors(
                category_prototypes,
                vector_name=f"prototype[{category}]",
            )
            scores[category] = self.policy.score_category(
                embedding_vector=embedding_vector,
                prototype_vectors=prototype_vectors,
                similarity_name=self.similarity_name,
                category=category,
            )
        return scores


@dataclass(slots=True)
class ClassifierHeadLogitsScoringBackend:
    """공통 classifier head state로 category logits를 계산한다."""

    backend_name: str = CLASSIFIER_HEAD_LOGITS_BACKEND_NAME
    confidence_kind: str = CLASSIFIER_HEAD_LOGITS_CONFIDENCE_KIND
    supported_adapter_kinds: tuple[str, ...] = ("classifier_head",)
    requires_shared_state: bool = True

    def score(
        self,
        embedding: Sequence[float],
        prototypes: Mapping[str, Sequence[float] | Sequence[Sequence[float]]],
        shared_state: SharedAdapterState | None = None,
    ) -> dict[str, float]:
        del prototypes
        if not isinstance(shared_state, ClassifierHeadState):
            raise ValueError(
                "classifier_head_logits backend requires "
                "ClassifierHeadState as shared_state."
            )
        embedding_vector = _coerce_vector(embedding, vector_name="embedding")
        return shared_state.compute_logits(embedding_vector)


_SCORING_BACKEND_REGISTRY: dict[
    str,
    tuple[ScoringBackendFactory, RegistryCatalogEntry],
] = {}


def register_scoring_backend(
    *backend_names: str,
    factory: ScoringBackendFactory,
    catalog_entry: RegistryCatalogEntry,
) -> None:
    """얇은 wiring registry에 scoring backend를 등록한다."""

    registered_backend = (factory, catalog_entry)
    for backend_name in backend_names:
        _SCORING_BACKEND_REGISTRY[backend_name.strip().lower()] = registered_backend


def build_scoring_backend(
    backend_name: str,
    *,
    objective_config: TrainingObjectiveConfig,
    similarity_name: str = "cosine",
) -> ScoringBackend:
    """backend 이름과 objective config로 scoring backend를 조립한다."""

    normalized_name = backend_name.strip().lower()
    registered_backend = _SCORING_BACKEND_REGISTRY.get(normalized_name)
    if registered_backend is not None:
        factory, _catalog_entry = registered_backend
        return factory(objective_config, similarity_name)
    raise ValueError(f"Unsupported scoring backend: {backend_name}.")


def list_registered_scoring_backend_names() -> tuple[str, ...]:
    """등록된 scoring backend 이름을 정렬된 tuple로 반환한다."""

    return tuple(sorted(_SCORING_BACKEND_REGISTRY))


def list_scoring_backend_catalog_entries() -> tuple[RegistryCatalogEntry, ...]:
    """등록된 scoring backend catalog entry를 canonical item 기준으로 반환한다."""

    return dedupe_registry_catalog_entries(
        catalog_entry
        for _factory, catalog_entry in _SCORING_BACKEND_REGISTRY.values()
    )


def resolve_scoring_backend_name(backend: ScoringBackend) -> str:
    """backend instance에서 canonical backend name을 읽는다."""

    backend_name = getattr(backend, "backend_name", None)
    return backend_name if isinstance(backend_name, str) else "unknown"


def resolve_scoring_confidence_kind(backend: ScoringBackend) -> str:
    """backend instance에서 query buffer confidence kind를 읽는다."""

    confidence_kind = getattr(backend, "confidence_kind", None)
    if isinstance(confidence_kind, str) and confidence_kind.strip():
        return confidence_kind
    backend_name = resolve_scoring_backend_name(backend)
    if backend_name == "unknown":
        return "unknown"
    return f"{backend_name}_top1"


def _build_prototype_similarity_backend(
    objective_config: TrainingObjectiveConfig,
    similarity_name: str,
) -> ScoringBackend:
    policy_name = (
        objective_config.score_policy_name or DEFAULT_TRAINING_PROFILE.score_policy_name
    )
    policy = build_prototype_score_policy(
        policy_name,
        top_k=objective_config.score_top_k,
    )
    return PrototypeSimilarityScoringBackend(
        similarity_name=similarity_name,
        policy=policy,
    )


def _build_classifier_head_logits_backend(
    objective_config: TrainingObjectiveConfig,
    similarity_name: str,
) -> ScoringBackend:
    del objective_config, similarity_name
    return ClassifierHeadLogitsScoringBackend()


def _coerce_vector(
    values: Sequence[float],
    *,
    vector_name: str,
) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if not vector:
        raise ValueError(f"{vector_name} must not be empty.")
    return vector


def _coerce_prototype_vectors(
    values: Sequence[float] | Sequence[Sequence[float]],
    *,
    vector_name: str,
) -> tuple[tuple[float, ...], ...]:
    raw_values = tuple(values)
    if not raw_values:
        raise ValueError(f"{vector_name} must not be empty.")

    if isinstance(raw_values[0], (int, float)):
        return (
            _coerce_vector(
                raw_values,  # type: ignore[arg-type]
                vector_name=vector_name,
            ),
        )

    return tuple(
        _coerce_vector(
            prototype,
            vector_name=f"{vector_name}[{index}]",
        )
        for index, prototype in enumerate(raw_values)
    )


register_scoring_backend(
    PROTOTYPE_SIMILARITY_BACKEND_NAME,
    factory=_build_prototype_similarity_backend,
    catalog_entry=RegistryCatalogEntry(
        item_name=PROTOTYPE_SIMILARITY_BACKEND_NAME,
        display_name=PROTOTYPE_SIMILARITY_BACKEND_NAME,
        implementation_module=PrototypeSimilarityScoringBackend.__module__,
        core_method_name=PROTOTYPE_SIMILARITY_BACKEND_NAME,
        family_name="scoring",
        supported_adapter_kinds=("*",),
        metadata={
            "requires_shared_state": False,
            "confidence_kind": PROTOTYPE_SIMILARITY_CONFIDENCE_KIND,
        },
    ),
)
register_scoring_backend(
    CLASSIFIER_HEAD_LOGITS_BACKEND_NAME,
    factory=_build_classifier_head_logits_backend,
    catalog_entry=RegistryCatalogEntry(
        item_name=CLASSIFIER_HEAD_LOGITS_BACKEND_NAME,
        display_name=CLASSIFIER_HEAD_LOGITS_BACKEND_NAME,
        implementation_module=ClassifierHeadLogitsScoringBackend.__module__,
        core_method_name=CLASSIFIER_HEAD_LOGITS_BACKEND_NAME,
        family_name="scoring",
        supported_adapter_kinds=("classifier_head",),
        tags=("requires_shared_state",),
        metadata={
            "requires_shared_state": True,
            "confidence_kind": CLASSIFIER_HEAD_LOGITS_CONFIDENCE_KIND,
        },
    ),
)


__all__ = [
    "CLASSIFIER_HEAD_LOGITS_BACKEND_NAME",
    "CLASSIFIER_HEAD_LOGITS_CONFIDENCE_KIND",
    "PROTOTYPE_SIMILARITY_BACKEND_NAME",
    "PROTOTYPE_SIMILARITY_CONFIDENCE_KIND",
    "ClassifierHeadLogitsScoringBackend",
    "PrototypeSimilarityScoringBackend",
    "ScoringBackend",
    "ScoringBackendFactory",
    "build_scoring_backend",
    "list_scoring_backend_catalog_entries",
    "list_registered_scoring_backend_names",
    "register_scoring_backend",
    "resolve_scoring_backend_name",
    "resolve_scoring_confidence_kind",
]
