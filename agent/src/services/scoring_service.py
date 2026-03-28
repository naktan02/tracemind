"""프로토타입 점수 계산 서비스."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(slots=True)
class ScoringService:
    """임베딩과 카테고리 프로토타입 간 cosine similarity를 계산한다."""

    similarity_name: str = "cosine"

    def score(
        self,
        embedding: Sequence[float],
        prototypes: Mapping[str, Sequence[float]],
    ) -> dict[str, float]:
        similarity_name = self.similarity_name.lower()
        if similarity_name != "cosine":
            raise ValueError(
                f"Unsupported similarity metric: {self.similarity_name}."
            )

        embedding_vector = self._coerce_vector(embedding, vector_name="embedding")
        embedding_norm = self._vector_norm(embedding_vector)
        if embedding_norm == 0.0:
            raise ValueError("Embedding vector norm must be non-zero.")

        scores: dict[str, float] = {}
        for category, prototype in prototypes.items():
            prototype_vector = self._coerce_vector(
                prototype,
                vector_name=f"prototype[{category}]",
            )
            if len(prototype_vector) != len(embedding_vector):
                raise ValueError(
                    "Embedding and prototype dimensions must match for scoring."
                )

            prototype_norm = self._vector_norm(prototype_vector)
            if prototype_norm == 0.0:
                raise ValueError(
                    f"Prototype vector norm must be non-zero for category '{category}'."
                )

            dot_product = math.fsum(
                left * right
                for left, right in zip(embedding_vector, prototype_vector, strict=True)
            )
            score = dot_product / (embedding_norm * prototype_norm)
            scores[category] = max(-1.0, min(1.0, score))

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
    def _vector_norm(vector: Sequence[float]) -> float:
        return math.sqrt(math.fsum(value * value for value in vector))
