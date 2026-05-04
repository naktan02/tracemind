"""Prototype score policy 계약."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol


class PrototypeScorePolicy(Protocol):
    """카테고리 내 다중 prototype 점수를 하나로 집계하는 정책."""

    def score_category(
        self,
        *,
        embedding_vector: Sequence[float],
        prototype_vectors: tuple[tuple[float, ...], ...],
        similarity_name: str,
        category: str,
    ) -> float:
        """단일 카테고리의 최종 score를 계산한다."""


PrototypeScorePolicyFactory = Callable[[int | None], PrototypeScorePolicy]
