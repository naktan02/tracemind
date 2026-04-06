"""프로토타입 점수 계산 서비스."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from agent.src.services.inference.scoring_policies import (
    MaxCosineScorePolicy,
    PrototypeScorePolicy,
    build_prototype_score_policy,
)
from shared.src.contracts.training_contracts import TrainingObjectiveConfig


@dataclass(slots=True)
class ScoringService:
    """임베딩과 카테고리 프로토타입 간 cosine similarity를 계산한다."""

    similarity_name: str = "cosine"
    policy: PrototypeScorePolicy = field(default_factory=MaxCosineScorePolicy)

    @classmethod
    def from_objective_config(
        cls,
        objective_config: TrainingObjectiveConfig,
        *,
        similarity_name: str = "cosine",
    ) -> "ScoringService":
        """학습 objective config로부터 scoring service를 조립한다."""

        policy_name = objective_config.score_policy_name or "max_cosine"
        policy = build_prototype_score_policy(
            policy_name,
            top_k=objective_config.score_top_k,
        )
        return cls(similarity_name=similarity_name, policy=policy)

    def score(
        self,
        embedding: Sequence[float],
        prototypes: Mapping[str, Sequence[float] | Sequence[Sequence[float]]],
    ) -> dict[str, float]:
        embedding_vector = self._coerce_vector(embedding, vector_name="embedding")
        scores: dict[str, float] = {}
        for category, category_prototypes in prototypes.items():
            prototype_vectors = self._coerce_prototype_vectors(
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

    @staticmethod
    def _coerce_vector(
        values: Sequence[float],
        *,
        vector_name: str,
    ) -> tuple[float, ...]:
        vector = tuple(float(value) for value in values)
        if not vector:
            raise ValueError(f"{vector_name} must not be empty.")
        return vector

    @staticmethod
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
                ScoringService._coerce_vector(
                    raw_values,  # type: ignore[arg-type]
                    vector_name=vector_name,
                ),
            )

        return tuple(
            ScoringService._coerce_vector(
                prototype,
                vector_name=f"{vector_name}[{index}]",
            )
            for index, prototype in enumerate(raw_values)
        )
