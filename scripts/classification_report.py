"""분류 리포트 공통 계산/렌더링 유틸리티."""

from __future__ import annotations

from collections import Counter, defaultdict


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def build_confusion_matrix(
    *,
    categories: list[str],
    actual_labels: list[str],
    predicted_labels: list[str],
) -> dict[str, dict[str, int]]:
    matrix = {
        actual: {predicted: 0 for predicted in categories} for actual in categories
    }
    for actual, predicted in zip(actual_labels, predicted_labels, strict=True):
        matrix[actual][predicted] += 1
    return matrix


def summarize_per_category(
    *,
    categories: list[str],
    actual_labels: list[str],
    predicted_labels: list[str],
    primary_values: list[float],
    top_1_values: list[float],
    margins: list[float],
    primary_metric_key: str,
    top_1_metric_key: str,
    round_digits: int | None = 6,
) -> dict[str, dict[str, float | int]]:
    support_counter = Counter(actual_labels)
    predicted_counter = Counter(predicted_labels)
    correct_counter = Counter(
        actual
        for actual, predicted in zip(actual_labels, predicted_labels, strict=True)
        if actual == predicted
    )
    primary_buckets: dict[str, list[float]] = defaultdict(list)
    top_1_buckets: dict[str, list[float]] = defaultdict(list)
    margin_buckets: dict[str, list[float]] = defaultdict(list)
    for actual, primary_value, top_1_value, margin in zip(
        actual_labels,
        primary_values,
        top_1_values,
        margins,
        strict=True,
    ):
        primary_buckets[actual].append(primary_value)
        top_1_buckets[actual].append(top_1_value)
        margin_buckets[actual].append(margin)

    per_category: dict[str, dict[str, float | int]] = {}
    for category in categories:
        support = support_counter[category]
        correct = correct_counter[category]
        predicted = predicted_counter[category]
        precision = safe_divide(correct, predicted)
        recall = safe_divide(correct, support)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        per_category[category] = {
            "support": support,
            "predicted": predicted,
            "correct": correct,
            "precision": _round_metric(precision, round_digits),
            "recall": _round_metric(recall, round_digits),
            "f1": _round_metric(f1, round_digits),
            primary_metric_key: _round_metric(
                safe_divide(
                    sum(primary_buckets[category]),
                    len(primary_buckets[category]),
                ),
                round_digits,
            ),
            top_1_metric_key: _round_metric(
                safe_divide(
                    sum(top_1_buckets[category]),
                    len(top_1_buckets[category]),
                ),
                round_digits,
            ),
            "mean_margin_top1_top2": _round_metric(
                safe_divide(
                    sum(margin_buckets[category]),
                    len(margin_buckets[category]),
                ),
                round_digits,
            ),
        }
    return per_category


def render_confusion_table(confusion_matrix: dict[str, dict[str, int]]) -> str:
    categories = list(confusion_matrix)
    header = ["actual \\ predicted"] + categories
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for actual in categories:
        row = [actual]
        for predicted in categories:
            row.append(str(confusion_matrix[actual][predicted]))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _round_metric(value: float, round_digits: int | None) -> float:
    if round_digits is None:
        return value
    return round(value, round_digits)


def render_per_category_table(
    per_category: dict[str, dict[str, float | int]],
    *,
    primary_metric_key: str,
    top_1_metric_key: str,
    primary_header: str,
    top_1_header: str,
) -> str:
    header = [
        "category",
        "support",
        "precision",
        "recall",
        "f1",
        primary_header,
        top_1_header,
        "mean_margin",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for category in sorted(per_category):
        metrics = per_category[category]
        row = [
            category,
            str(metrics["support"]),
            f"{float(metrics['precision']):.4f}",
            f"{float(metrics['recall']):.4f}",
            f"{float(metrics['f1']):.4f}",
            f"{float(metrics[primary_metric_key]):.4f}",
            f"{float(metrics[top_1_metric_key]):.4f}",
            f"{float(metrics['mean_margin_top1_top2']):.4f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
