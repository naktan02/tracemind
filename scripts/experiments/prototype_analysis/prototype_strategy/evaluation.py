"""임베딩 평가와 메트릭 계산."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from methods.prototype.index import PrototypeIndex
from methods.prototype.thresholding.evaluation import (
    evaluate_scored_predictions,
)
from methods.prototype.thresholding.models import (
    EvaluationMetrics,
    ScoredPrediction,
)
from scripts.experiments.prototype_analysis.prototype_strategy.scoring import (
    PrototypeIndexScorer,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.domain.services.embedding_adapter import EmbeddingAdapter


def embed_rows(
    rows: Sequence[LabeledQueryRow],
    adapter: EmbeddingAdapter,
) -> np.ndarray:
    """row의 text를 임베딩한다."""
    texts = [str(row["text"]) for row in rows]
    return np.asarray(adapter.embed_texts(texts), dtype=np.float64)


def group_embeddings_by_label(
    *,
    rows: Sequence[LabeledQueryRow],
    embeddings: np.ndarray,
) -> dict[str, np.ndarray]:
    """row와 임베딩을 라벨별로 묶는다."""
    buckets: dict[str, list[np.ndarray]] = defaultdict(list)
    for row, embedding in zip(rows, embeddings, strict=True):
        buckets[str(row["mapped_label_4"])].append(embedding)
    return {
        label: np.asarray(label_embeddings, dtype=np.float64)
        for label, label_embeddings in sorted(buckets.items())
    }


def evaluate_embeddings(
    *,
    rows: Sequence[LabeledQueryRow],
    embeddings: np.ndarray,
    prototype_index: PrototypeIndex,
    confidence_threshold: float,
    margin_threshold: float,
    scorer: PrototypeIndexScorer,
) -> EvaluationMetrics:
    """prototype 전략으로 평가셋 메트릭을 계산한다."""
    scored_predictions = score_embeddings(
        rows=rows,
        embeddings=embeddings,
        prototype_index=prototype_index,
        scorer=scorer,
    )
    return evaluate_scored_predictions(
        scored_predictions=scored_predictions,
        categories=sorted(prototype_index.categories.keys()),
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
    )


def score_embeddings(
    *,
    rows: Sequence[LabeledQueryRow],
    embeddings: np.ndarray,
    prototype_index: PrototypeIndex,
    scorer: PrototypeIndexScorer,
) -> tuple[ScoredPrediction, ...]:
    """row/embedding으로부터 threshold 재평가 가능한 score 목록을 만든다."""
    categories = sorted(prototype_index.categories.keys())
    if not categories:
        raise ValueError("Prototype index must contain at least one category.")

    scored_predictions: list[ScoredPrediction] = []

    for row, embedding in zip(rows, embeddings, strict=True):
        scores = scorer.score(embedding, prototype_index)
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top1_label, top1_score = ordered[0]
        top2_score = ordered[1][1] if len(ordered) > 1 else -1.0
        true_label = str(row["mapped_label_4"])
        margin = top1_score - top2_score

        scored_predictions.append(
            ScoredPrediction(
                actual_label=true_label,
                predicted_label=top1_label,
                true_label_score=float(scores[true_label]),
                top1_score=float(top1_score),
                top2_score=float(top2_score),
                margin_top1_top2=float(margin),
                is_correct=(true_label == top1_label),
            )
        )

    return tuple(scored_predictions)
