"""Prototype index scoring 유틸리티."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from scripts.experiments.prototype_strategy.models import PrototypeIndex


class MaxCosinePrototypeIndexScorer:
    """prototype index 전체에 대해 category별 max cosine score를 계산한다."""

    def score(
        self,
        embedding: Sequence[float],
        prototype_index: PrototypeIndex,
    ) -> dict[str, float]:
        vector = np.asarray(embedding, dtype=np.float64)
        vector_norm = np.linalg.norm(vector)
        if vector_norm == 0.0:
            raise ValueError("Embedding norm must not be zero.")
        normalized = vector / vector_norm

        scores: dict[str, float] = {}
        for category, prototypes in prototype_index.categories.items():
            best_score = max(
                float(np.dot(normalized, np.asarray(prototype.centroid)))
                for prototype in prototypes
            )
            scores[category] = best_score
        return scores
