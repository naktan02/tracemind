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
from shared.src.contracts.training_contracts import TrainingObjectiveConfig


class ScoringBackend(Protocol):
    """category score dict를 계산하는 backend 인터페이스."""

    backend_name: str
    supported_adapter_kinds: tuple[str, ...]

    def score(
        self,
        embedding: Sequence[float],
        prototypes: Mapping[str, Sequence[float] | Sequence[Sequence[float]]],
    ) -> dict[str, float]:
        """임베딩과 prototype들로 category score dict를 계산한다."""


ScoringBackendFactory = Callable[[TrainingObjectiveConfig, str], ScoringBackend]


@dataclass(slots=True)
class PrototypeSimilarityScoringBackend:
    """prototype similarity를 계산하고 policy로 category score를 접는다."""

    similarity_name: str = "cosine"
    policy: PrototypeScorePolicy = field(default_factory=MaxCosineScorePolicy)
    backend_name: str = "prototype_similarity"
    supported_adapter_kinds: tuple[str, ...] = ("*",)

    def score(
        self,
        embedding: Sequence[float],
        prototypes: Mapping[str, Sequence[float] | Sequence[Sequence[float]]],
    ) -> dict[str, float]:
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


_SCORING_BACKEND_REGISTRY: dict[str, ScoringBackendFactory] = {}


def register_scoring_backend(
    *backend_names: str,
    factory: ScoringBackendFactory,
) -> None:
    """얇은 wiring registry에 scoring backend를 등록한다."""

    for backend_name in backend_names:
        _SCORING_BACKEND_REGISTRY[backend_name.strip().lower()] = factory


def build_scoring_backend(
    backend_name: str,
    *,
    objective_config: TrainingObjectiveConfig,
    similarity_name: str = "cosine",
) -> ScoringBackend:
    """backend 이름과 objective config로 scoring backend를 조립한다."""

    normalized_name = backend_name.strip().lower()
    factory = _SCORING_BACKEND_REGISTRY.get(normalized_name)
    if factory is not None:
        return factory(objective_config, similarity_name)
    raise ValueError(f"Unsupported scoring backend: {backend_name}.")


def _build_prototype_similarity_backend(
    objective_config: TrainingObjectiveConfig,
    similarity_name: str,
) -> ScoringBackend:
    policy_name = objective_config.score_policy_name or "max_cosine"
    policy = build_prototype_score_policy(
        policy_name,
        top_k=objective_config.score_top_k,
    )
    return PrototypeSimilarityScoringBackend(
        similarity_name=similarity_name,
        policy=policy,
    )


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
    "prototype_similarity",
    factory=_build_prototype_similarity_backend,
)
