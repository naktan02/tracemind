"""프로토타입 점수 집계 정책."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class PrototypeScorePolicy(Protocol):
    """카테고리 내 다중 프로토타입 점수를 하나로 집계하는 정책."""

    def score_category(
        self,
        *,
        embedding_vector: Sequence[float],
        prototype_vectors: tuple[tuple[float, ...], ...],
        similarity_name: str,
        category: str,
    ) -> float:
        """단일 카테고리의 최종 score를 계산한다."""


@dataclass(slots=True)
class MaxCosineScorePolicy:
    """다중 프로토타입 중 가장 높은 cosine score를 사용한다."""

    def score_category(
        self,
        *,
        embedding_vector: Sequence[float],
        prototype_vectors: tuple[tuple[float, ...], ...],
        similarity_name: str,
        category: str,
    ) -> float:
        scores = _pairwise_scores(
            embedding_vector=embedding_vector,
            prototype_vectors=prototype_vectors,
            similarity_name=similarity_name,
            category=category,
        )
        return max(scores)


@dataclass(slots=True)
class TopKMeanCosineScorePolicy:
    """상위 k개 cosine score의 평균을 사용한다."""

    top_k: int = 2

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1.")

    def score_category(
        self,
        *,
        embedding_vector: Sequence[float],
        prototype_vectors: tuple[tuple[float, ...], ...],
        similarity_name: str,
        category: str,
    ) -> float:
        scores = sorted(
            _pairwise_scores(
                embedding_vector=embedding_vector,
                prototype_vectors=prototype_vectors,
                similarity_name=similarity_name,
                category=category,
            ),
            reverse=True,
        )
        selected = scores[: min(self.top_k, len(scores))]
        return math.fsum(selected) / len(selected)


def build_prototype_score_policy(
    policy_name: str,
    *,
    top_k: int | None = None,
) -> PrototypeScorePolicy:
    """정책 이름으로 prototype score policy를 생성한다."""

    normalized_name = policy_name.strip().lower()
    if normalized_name == "max_cosine":
        return MaxCosineScorePolicy()
    if normalized_name == "topk_mean_cosine":
        return TopKMeanCosineScorePolicy(top_k=top_k or 2)
    raise ValueError(f"Unsupported prototype score policy: {policy_name}.")


def _pairwise_scores(
    *,
    embedding_vector: Sequence[float],
    prototype_vectors: tuple[tuple[float, ...], ...],
    similarity_name: str,
    category: str,
) -> tuple[float, ...]:
    normalized_similarity = similarity_name.lower()
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
        scores.append(max(-1.0, min(1.0, score)))
    return tuple(scores)


def _vector_norm(vector: Sequence[float]) -> float:
    return math.sqrt(math.fsum(value * value for value in vector))
