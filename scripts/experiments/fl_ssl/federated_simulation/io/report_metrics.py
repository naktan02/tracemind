"""FL simulation report metric payload helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence

from scripts.experiments.fl_ssl.federated_simulation.models import (
    SimulationEvaluation,
    SimulationResult,
)


def evaluation_to_payload(evaluation: SimulationEvaluation) -> dict[str, object]:
    """simulation validation 결과를 report 공통 metric shape로 변환한다."""

    payload = dict(evaluation.classification_report)
    payload.update(
        {
            "row_count": evaluation.row_count,
            "rows_total": evaluation.row_count,
            "top1_accuracy": evaluation.top1_accuracy,
            "accuracy_top_1": evaluation.accuracy_top_1,
            "correct_top_1": evaluation.correct_top_1,
            "accepted_ratio": evaluation.accepted_ratio,
            "loss": evaluation.loss,
            "loss_kind": evaluation.loss_kind,
            "macro_f1": evaluation.macro_f1,
            "macro_precision": evaluation.macro_precision,
            "macro_recall": evaluation.macro_recall,
            "weighted_f1": evaluation.weighted_f1,
            "balanced_accuracy": evaluation.balanced_accuracy,
            "worst_category_f1": evaluation.worst_category_f1,
            "worst_category_f1_value": evaluation.worst_category_f1_value,
            "worst_category_recall": evaluation.worst_category_recall,
            "worst_category_precision": evaluation.worst_category_precision,
            "expected_calibration_error": evaluation.expected_calibration_error,
            "max_calibration_error": evaluation.max_calibration_error,
            "overconfidence_gap": evaluation.overconfidence_gap,
            "mean_true_label_probability": evaluation.mean_true_label_probability,
            "mean_top_1_probability": evaluation.mean_top_1_probability,
            "mean_margin_top1_top2": evaluation.mean_margin_top1_top2,
            "mean_correct_top_1_probability": (
                evaluation.mean_correct_top_1_probability
            ),
            "mean_incorrect_top_1_probability": (
                evaluation.mean_incorrect_top_1_probability
            ),
            "score_distribution_kind": evaluation.score_distribution_kind,
            "selection_confidence_kind": evaluation.selection_confidence_kind,
            "mean_selection_confidence": evaluation.mean_selection_confidence,
            "mean_selection_margin": evaluation.mean_selection_margin,
            "per_label": evaluation.per_label,
            "per_category": evaluation.per_label,
            "confusion_matrix": evaluation.confusion_matrix,
        }
    )
    return payload


def build_communication_cost_summary(result: SimulationResult) -> dict[str, object]:
    """payload byte 계측 전까지 쓰는 FL communication proxy summary."""

    total_client_updates = sum(
        round_summary.update_count for round_summary in result.rounds
    )
    total_candidates = sum(
        client.candidate_count
        for round_summary in result.rounds
        for client in round_summary.clients
    )
    total_accepted = sum(
        client.accepted_count
        for round_summary in result.rounds
        for client in round_summary.clients
    )
    return {
        "unit": "client_update_envelopes",
        "value": total_client_updates,
        "total_client_updates": total_client_updates,
        "total_candidates": total_candidates,
        "total_accepted": total_accepted,
        "accepted_per_update": safe_ratio(total_accepted, total_client_updates),
        "acceptance_ratio": safe_ratio(total_accepted, total_candidates),
        "status": "proxy_until_payload_byte_accounting",
    }


def numeric_summary(values: Sequence[float | int]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": mean([float(value) for value in values]),
        "variance": population_variance([float(value) for value in values]),
    }


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def population_variance(values: Sequence[float]) -> float | None:
    if not values:
        return None
    average = sum(values) / len(values)
    return sum((value - average) ** 2 for value in values) / len(values)


def population_std(values: Sequence[float]) -> float | None:
    variance = population_variance(values)
    if variance is None:
        return None
    return math.sqrt(variance)


def safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator
