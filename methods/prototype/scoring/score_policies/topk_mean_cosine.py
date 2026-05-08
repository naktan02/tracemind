"""Top-k mean cosine prototype score policy."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from methods.prototype.scoring.base import PrototypeScorePolicy
from methods.prototype.scoring.policy_registry import register_prototype_score_policy
from methods.prototype.scoring.similarity import pairwise_scores


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
            pairwise_scores(
                embedding_vector=embedding_vector,
                prototype_vectors=prototype_vectors,
                similarity_name=similarity_name,
                category=category,
            ),
            reverse=True,
        )
        selected = scores[: min(self.top_k, len(scores))]
        return math.fsum(selected) / len(selected)


@register_prototype_score_policy("topk_mean_cosine")
def build_topk_mean_cosine_score_policy(
    top_k: int | None,
) -> PrototypeScorePolicy:
    """top-k mean cosine score policy factory."""

    return TopKMeanCosineScorePolicy(top_k=top_k or 2)
