"""프로토타입 점수 집계 정책."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from methods.prototype.scoring.base import (
    PrototypeScorePolicy,
    PrototypeScorePolicyFactory,
)
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


_PROTOTYPE_SCORE_POLICY_REGISTRY: dict[str, PrototypeScorePolicyFactory] = {}


def register_prototype_score_policy(
    *policy_names: str,
    factory: PrototypeScorePolicyFactory,
) -> None:
    """얇은 wiring registry에 prototype score policy를 등록한다."""

    for policy_name in policy_names:
        _PROTOTYPE_SCORE_POLICY_REGISTRY[policy_name.strip().lower()] = factory


def build_prototype_score_policy(
    policy_name: str,
    *,
    top_k: int | None = None,
) -> PrototypeScorePolicy:
    """정책 이름으로 prototype score policy를 생성한다."""

    normalized_name = policy_name.strip().lower()
    factory = _PROTOTYPE_SCORE_POLICY_REGISTRY.get(normalized_name)
    if factory is not None:
        return factory(top_k)
    raise ValueError(f"Unsupported prototype score policy: {policy_name}.")


register_prototype_score_policy(
    "max_cosine",
    factory=lambda _top_k: MaxCosineScorePolicy(),
)
register_prototype_score_policy(
    "topk_mean_cosine",
    factory=lambda top_k: TopKMeanCosineScorePolicy(top_k=top_k or 2),
)
