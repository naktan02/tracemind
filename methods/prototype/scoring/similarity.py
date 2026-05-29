"""Prototype similarity score 계산 helper."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from methods.prototype.scoring.base import PrototypeScorePolicy

PrototypeVectorInput = Sequence[float] | Sequence[Sequence[float]]


def score_prototype_categories(
    *,
    embedding: Sequence[float],
    prototypes: Mapping[str, PrototypeVectorInput],
    policy: PrototypeScorePolicy,
    similarity_name: str = "cosine",
) -> dict[str, float]:
    """embedding과 category별 prototype으로 score dict를 계산한다."""

    embedding_vector = coerce_vector(embedding, vector_name="embedding")
    scores: dict[str, float] = {}
    for category, category_prototypes in prototypes.items():
        prototype_vectors = coerce_prototype_vectors(
            category_prototypes,
            vector_name=f"prototype[{category}]",
        )
        scores[category] = policy.score_category(
            embedding_vector=embedding_vector,
            prototype_vectors=prototype_vectors,
            similarity_name=similarity_name,
            category=category,
        )
    return scores


def pairwise_scores(
    *,
    embedding_vector: Sequence[float],
    prototype_vectors: tuple[tuple[float, ...], ...],
    similarity_name: str,
    category: str,
) -> tuple[float, ...]:
    """단일 embedding과 category prototype들의 pairwise score를 계산한다."""

    normalized_similarity = similarity_name.strip().lower()
    if normalized_similarity != "cosine":
        raise ValueError(f"Unsupported similarity metric: {similarity_name}.")

    embedding_norm = _vector_norm(embedding_vector)
    if embedding_norm == 0.0:
        raise ValueError("Embedding vector norm must be non-zero.")

    scores: list[float] = []
    for prototype_vector in prototype_vectors:
        if len(prototype_vector) != len(embedding_vector):
            raise ValueError(
                "Embedding and prototype dimensions must match for scoring."
            )

        prototype_norm = _vector_norm(prototype_vector)
        if prototype_norm == 0.0:
            raise ValueError(
                f"Prototype vector norm must be non-zero for category '{category}'."
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
        scores.append(max(-1.0, min(1.0, score)))
    return tuple(scores)


def coerce_vector(
    values: Sequence[float],
    *,
    vector_name: str,
) -> tuple[float, ...]:
    """vector input을 non-empty float tuple로 정규화한다."""

    vector = tuple(float(value) for value in values)
    if not vector:
        raise ValueError(f"{vector_name} must not be empty.")
    return vector


def coerce_prototype_vectors(
    values: PrototypeVectorInput,
    *,
    vector_name: str,
) -> tuple[tuple[float, ...], ...]:
    """single 또는 multi-prototype input을 tuple-of-tuples로 정규화한다."""

    raw_values = tuple(values)
    if not raw_values:
        raise ValueError(f"{vector_name} must not be empty.")

    if isinstance(raw_values[0], (int, float)):
        return (
            coerce_vector(
                raw_values,  # type: ignore[arg-type]
                vector_name=vector_name,
            ),
        )

    return tuple(
        coerce_vector(
            prototype,
            vector_name=f"{vector_name}[{index}]",
        )
        for index, prototype in enumerate(raw_values)
    )


def _vector_norm(vector: Sequence[float]) -> float:
    return math.sqrt(math.fsum(value * value for value in vector))
