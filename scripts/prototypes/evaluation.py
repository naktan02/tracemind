"""prototype pack 평가 재사용 함수."""

from __future__ import annotations

from agent.src.services.inference.scoring_service import ScoringService
from scripts.classification_report import (
    build_confusion_matrix,
    safe_divide,
    summarize_per_category,
)
from scripts.labeled_query_rows import LabeledQueryRow


def predict_label(scores: dict[str, float]) -> tuple[str, float, float]:
    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0]),
    )
    predicted_label, top_1_score = ranked[0]
    top_2_score = ranked[1][1] if len(ranked) > 1 else ranked[0][1]
    return predicted_label, top_1_score, top_1_score - top_2_score


def evaluate_rows(
    *,
    rows: list[LabeledQueryRow],
    prototypes: dict[str, tuple[list[float], ...]],
    embeddings: list[list[float]],
) -> dict[str, object]:
    scoring_service = ScoringService()
    categories = sorted(prototypes)
    actual_labels: list[str] = []
    predicted_labels: list[str] = []
    top_1_scores: list[float] = []
    true_scores: list[float] = []
    margins: list[float] = []

    for row, embedding in zip(rows, embeddings, strict=True):
        actual_label = row["mapped_label_4"]
        scores = scoring_service.score(embedding=embedding, prototypes=prototypes)
        predicted_label, top_1_score, margin = predict_label(scores)

        actual_labels.append(actual_label)
        predicted_labels.append(predicted_label)
        top_1_scores.append(top_1_score)
        true_scores.append(scores[actual_label])
        margins.append(margin)

    total = len(rows)
    correct = sum(
        1
        for actual, predicted in zip(actual_labels, predicted_labels, strict=True)
        if actual == predicted
    )
    accuracy = safe_divide(correct, total)
    confusion_matrix = build_confusion_matrix(
        categories=categories,
        actual_labels=actual_labels,
        predicted_labels=predicted_labels,
    )
    per_category = summarize_per_category(
        categories=categories,
        actual_labels=actual_labels,
        predicted_labels=predicted_labels,
        primary_values=true_scores,
        top_1_values=top_1_scores,
        margins=margins,
        primary_metric_key="mean_true_label_score",
        top_1_metric_key="mean_top_1_score",
    )

    return {
        "rows_total": total,
        "accuracy_top_1": round(accuracy, 6),
        "correct_top_1": correct,
        "mean_true_label_score": round(
            safe_divide(sum(true_scores), len(true_scores)),
            6,
        ),
        "mean_top_1_score": round(
            safe_divide(sum(top_1_scores), len(top_1_scores)),
            6,
        ),
        "mean_margin_top1_top2": round(
            safe_divide(sum(margins), len(margins)),
            6,
        ),
        "confusion_matrix": confusion_matrix,
        "per_category": per_category,
    }
