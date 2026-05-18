"""Prototype scored-candidate 평가 payload 계산."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from methods.evaluation.classification_payload import (
    build_classification_evaluation_payload,
)
from methods.evaluation.classification_report import (
    build_classification_evaluation_report,
)
from methods.prototype.evidence.helpers import softmax_distribution


def build_prototype_candidate_evaluation_payload(
    *,
    row_count: int,
    true_labels: list[str],
    predicted_labels: list[str],
    candidates: list[Any],
    evidences: list[Any],
    accepted_ratio: float,
) -> dict[str, object]:
    """prototype/scored-candidate 평가 결과를 canonical evaluation payload로 만든다."""

    categories = sorted(set(true_labels) | set(predicted_labels))
    distributions: list[dict[str, float]] = []
    distribution_kinds: list[str] = []
    for candidate, evidence in zip(candidates, evidences, strict=True):
        distribution, distribution_kind = label_probability_distribution(
            candidate=candidate,
            evidence=evidence,
        )
        distributions.append(distribution)
        distribution_kinds.append(distribution_kind)

    true_probs = [
        distribution.get(true_label, 0.0)
        for true_label, distribution in zip(true_labels, distributions, strict=True)
    ]
    top_1_values = [
        distribution.get(predicted_label, 0.0)
        for predicted_label, distribution in zip(
            predicted_labels,
            distributions,
            strict=True,
        )
    ]
    margins = [
        probability_margin(distribution=distribution, predicted_label=predicted_label)
        for predicted_label, distribution in zip(
            predicted_labels,
            distributions,
            strict=True,
        )
    ]
    total_loss = sum(-math.log(max(probability, 1e-12)) for probability in true_probs)
    report = build_classification_evaluation_report(
        categories=categories,
        actual_labels=true_labels,
        predicted_labels=predicted_labels,
        true_probs=true_probs,
        top_1_values=top_1_values,
        margins=margins,
        total_loss=total_loss,
        total_rows=row_count,
    )
    return build_classification_evaluation_payload(
        report=report,
        row_count=row_count,
        accepted_ratio=accepted_ratio,
        loss_kind="negative_log_likelihood_from_score_distribution",
        score_distribution_kind=summarize_distribution_kind(distribution_kinds),
        selection_confidence_kind=summarize_selection_confidence_kind(candidates),
        mean_selection_confidence=mean(
            [float(candidate.confidence) for candidate in candidates]
        ),
        mean_selection_margin=mean(
            [float(candidate.margin) for candidate in candidates]
        ),
    )


def label_probability_distribution(
    *,
    candidate: Any,
    evidence: Any,
) -> tuple[dict[str, float], str]:
    """candidate/evidence에서 label probability distribution을 만든다."""

    label_distribution = getattr(evidence, "label_distribution", None)
    if label_distribution:
        return (
            float_distribution(label_distribution),
            "evidence_label_distribution",
        )

    raw_scores = getattr(evidence, "raw_scores", None)
    if raw_scores:
        return (
            softmax_distribution(raw_scores, temperature=1.0),
            "softmax_raw_scores_temperature_1.0",
        )

    sparse_scores = {str(candidate.label): float(candidate.confidence)}
    if candidate.runner_up_label is not None:
        sparse_scores[str(candidate.runner_up_label)] = float(
            candidate.runner_up_score or 0.0
        )
    if len(sparse_scores) > 1:
        distribution = softmax_distribution(sparse_scores, temperature=1.0)
    else:
        distribution = {str(candidate.label): 1.0}
    return (
        distribution,
        "softmax_candidate_top2_sparse_fallback",
    )


def float_distribution(distribution: Mapping[str, float]) -> dict[str, float]:
    """distribution 값을 float mapping으로 정규화한다."""

    return {
        str(label): float(probability) for label, probability in distribution.items()
    }


def probability_margin(
    *,
    distribution: Mapping[str, float],
    predicted_label: str,
) -> float:
    """top-1 label과 runner-up probability 차이를 계산한다."""

    top_value = float(distribution.get(predicted_label, 0.0))
    runner_up = max(
        (
            float(value)
            for label, value in distribution.items()
            if label != predicted_label
        ),
        default=0.0,
    )
    return top_value - runner_up


def summarize_distribution_kind(distribution_kinds: list[str]) -> str:
    """여러 distribution source kind를 하나의 report string으로 요약한다."""

    unique_kinds = sorted(set(distribution_kinds))
    if not unique_kinds:
        return "not_computed"
    if len(unique_kinds) == 1:
        return unique_kinds[0]
    return "mixed:" + ",".join(unique_kinds)


def summarize_selection_confidence_kind(candidates: list[Any]) -> str | None:
    """candidate confidence kind를 report string으로 요약한다."""

    unique_kinds = sorted(
        {
            str(candidate.confidence_kind)
            for candidate in candidates
            if candidate.confidence_kind is not None
        }
    )
    if not unique_kinds:
        return None
    if len(unique_kinds) == 1:
        return unique_kinds[0]
    return "mixed:" + ",".join(unique_kinds)


def mean(values: list[float]) -> float:
    """빈 입력을 0으로 다루는 평균 helper."""

    return sum(values) / len(values) if values else 0.0
