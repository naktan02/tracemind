"""Max-cosine prototype score policy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from methods.prototype.scoring.base import PrototypeScorePolicy
from methods.prototype.scoring.policy_registry import register_prototype_score_policy
from methods.prototype.scoring.similarity import pairwise_scores


@dataclass(slots=True)
class MaxCosineScorePolicy:
    """다중 prototype 중 가장 높은 cosine score를 사용한다."""

    def score_category(
        self,
        *,
        embedding_vector: Sequence[float],
        prototype_vectors: tuple[tuple[float, ...], ...],
        similarity_name: str,
        category: str,
    ) -> float:
        scores = pairwise_scores(
            embedding_vector=embedding_vector,
            prototype_vectors=prototype_vectors,
            similarity_name=similarity_name,
            category=category,
        )
        return max(scores)


@register_prototype_score_policy("max_cosine")
def build_max_cosine_score_policy(
    top_k: int | None,
) -> PrototypeScorePolicy:
    """max-cosine score policy factory."""

    del top_k
    return MaxCosineScorePolicy()
