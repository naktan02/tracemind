"""Fixed classifier runner가 공유하는 작은 data model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn

from shared.src.domain.services.embedding_adapter import EmbeddingAdapter


@dataclass(slots=True)
class TrainedFixedClassifier:
    """학습된 fixed encoder + classifier bundle."""

    model: nn.Module
    adapter: EmbeddingAdapter
    embedding_spec: Any
    categories: list[str]
    label_to_index: dict[str, int]
    training_device: str
    history: list[dict[str, Any]]
    best_selection_report: dict[str, Any]
    eval_results: dict[str, Any]


@dataclass(slots=True)
class FixedClassifierPrediction:
    """teacher classifier의 unlabeled row 추론 결과."""

    query_id: str
    predicted_label: str
    confidence: float
    margin: float
    runner_up_label: str | None
    runner_up_score: float | None
    raw_scores: dict[str, float]
