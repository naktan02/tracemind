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
        prototypes: Mapping[str, Sequence[float] | Sequence[Sequence[float]]],
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
        for category, category_prototypes in prototypes.items():
            prototype_vectors = self._coerce_prototype_vectors(
                category_prototypes,
                vector_name=f"prototype[{category}]",
            )
            best_score = -1.0
            for prototype_vector in prototype_vectors:
                if len(prototype_vector) != len(embedding_vector):
                    raise ValueError(
                        "Embedding and prototype dimensions must match for scoring."
                    )

                prototype_norm = self._vector_norm(prototype_vector)
                if prototype_norm == 0.0:
                    raise ValueError(
                        "Prototype vector norm must be non-zero for category "
                        f"'{category}'."
                    )

                dot_product = math.fsum(
                    left * right
                    for left, right in zip(
                        embedding_vector,
                        prototype_vector,
                        strict=True,
                    )
                )
                score = dot_product / (embedding_norm * prototype_norm)
                best_score = max(best_score, max(-1.0, min(1.0, score)))
            scores[category] = best_score

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

    @staticmethod
    def _vector_norm(vector: Sequence[float]) -> float:
        return math.sqrt(math.fsum(value * value for value in vector))
