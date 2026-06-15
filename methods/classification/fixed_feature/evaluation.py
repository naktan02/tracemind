"""Fixed-feature classifier 평가."""

from __future__ import annotations

from typing import Any

import numpy as np

from methods.evaluation.classification_report import (
    build_classification_evaluation_report,
)


def evaluate_fixed_feature_predictions(
    *,
    categories: list[str],
    actual_labels: list[str],
    predicted_labels: list[str],
    score_matrix: np.ndarray,
) -> dict[str, Any]:
    """기존 중앙 분류 report와 같은 metric shape를 만든다."""

    label_to_index = {label: index for index, label in enumerate(categories)}
    true_probs: list[float] = []
    top_1_values: list[float] = []
    margins: list[float] = []
    total_loss = 0.0
    for actual_label, scores in zip(actual_labels, score_matrix, strict=True):
        actual_index = label_to_index[actual_label]
        true_prob = float(scores[actual_index])
        sorted_scores = np.sort(scores)
        top_1 = float(sorted_scores[-1]) if sorted_scores.size else 0.0
        top_2 = float(sorted_scores[-2]) if sorted_scores.size > 1 else 0.0
        true_probs.append(true_prob)
        top_1_values.append(top_1)
        margins.append(top_1 - top_2)
        total_loss += -float(np.log(max(true_prob, 1e-12)))

    return build_classification_evaluation_report(
        categories=categories,
        actual_labels=actual_labels,
        predicted_labels=predicted_labels,
        true_probs=true_probs,
        top_1_values=top_1_values,
        margins=margins,
        total_loss=total_loss,
        total_rows=len(actual_labels),
    )
