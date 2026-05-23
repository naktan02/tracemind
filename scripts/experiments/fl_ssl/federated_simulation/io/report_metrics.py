"""FL simulation report metric payload helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from scripts.experiments.fl_ssl.federated_simulation.io.report_math import (
    numeric_summary,
    safe_ratio,
    weighted_mean,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    ClientRoundSummary,
    SimulationEvaluation,
    SimulationResult,
    SimulationRoundSummary,
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
    """FL communication/system cost summary를 만든다."""

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
    payload_bytes = [
        client.client_payload_bytes
        for round_summary in result.rounds
        for client in round_summary.clients
        if client.client_payload_bytes is not None
    ]
    round_times = [
        round_summary.round_time_seconds
        for round_summary in result.rounds
        if round_summary.round_time_seconds is not None
    ]
    client_train_times = [
        client.client_train_time_seconds
        for round_summary in result.rounds
        for client in round_summary.clients
        if client.client_train_time_seconds is not None
    ]
    timing_breakdown_values: dict[str, list[float]] = {}
    for round_summary in result.rounds:
        for client in round_summary.clients:
            for key, value in client.timing_breakdown.items():
                timing_breakdown_values.setdefault(key, []).append(float(value))
    gpu_memory_peaks = [
        round_summary.gpu_memory_peak_mb
        for round_summary in result.rounds
        if round_summary.gpu_memory_peak_mb is not None
    ]
    return {
        "unit": "client_update_envelopes",
        "value": total_client_updates,
        "total_client_updates": total_client_updates,
        "total_candidates": total_candidates,
        "total_accepted": total_accepted,
        "total_payload_bytes": sum(payload_bytes) if payload_bytes else None,
        "payload_byte_accounting_status": (
            "measured" if payload_bytes else "not_instrumented"
        ),
        "accepted_per_update": safe_ratio(total_accepted, total_client_updates),
        "acceptance_ratio": safe_ratio(total_accepted, total_candidates),
        "round_time_seconds": numeric_summary(round_times),
        "client_train_time_seconds": numeric_summary(client_train_times),
        "timing_breakdown_summary": _timing_breakdown_summary(
            timing_breakdown_values
        ),
        "gpu_memory_peak_mb": numeric_summary(gpu_memory_peaks),
        "system_cost_status": (
            "partially_measured" if round_times or client_train_times else "proxy_only"
        ),
        "status": (
            "measured_with_update_payload_bytes"
            if payload_bytes
            else "proxy_until_payload_byte_accounting"
        ),
    }


def build_pseudo_label_quality_diagnostics(
    result: SimulationResult,
) -> dict[str, object]:
    """round/client selection result에서 pseudo-label 품질 요약을 만든다."""

    round_payloads = [
        _round_pseudo_label_quality(round_summary) for round_summary in result.rounds
    ]
    total_candidates = sum(
        int(payload["candidate_count"]) for payload in round_payloads
    )
    total_diagnostic_candidates = sum(
        int(payload["diagnostic_candidate_count"]) for payload in round_payloads
    )
    total_accepted = sum(int(payload["accepted_count"]) for payload in round_payloads)
    total_correct = sum(
        int(payload["pseudo_label_correct_count"]) for payload in round_payloads
    )
    total_evaluated = sum(
        int(payload["pseudo_label_evaluated_count"]) for payload in round_payloads
    )
    return {
        "summary": {
            "candidate_count": total_candidates,
            "diagnostic_candidate_count": total_diagnostic_candidates,
            "accepted_count": total_accepted,
            "accepted_ratio": safe_ratio(total_accepted, total_candidates),
            "pseudo_label_accuracy": safe_ratio(total_correct, total_evaluated),
            "pseudo_label_correct_count": total_correct,
            "pseudo_label_evaluated_count": total_evaluated,
            "pseudo_label_accuracy_basis": (
                "accepted_candidates_with_simulation_labels"
            ),
            "pseudo_label_confidence_mean": _weighted_round_mean(
                round_payloads,
                value_key="pseudo_label_confidence_mean",
                weight_key="diagnostic_candidate_count",
            ),
            "candidate_confidence_mean": _weighted_round_mean(
                round_payloads,
                value_key="pseudo_label_confidence_mean",
                weight_key="diagnostic_candidate_count",
            ),
            "pseudo_label_margin_mean": _weighted_round_mean(
                round_payloads,
                value_key="pseudo_label_margin_mean",
                weight_key="diagnostic_candidate_count",
            ),
            "candidate_margin_mean": _weighted_round_mean(
                round_payloads,
                value_key="pseudo_label_margin_mean",
                weight_key="diagnostic_candidate_count",
            ),
            "accepted_label_distribution": _sum_distributions(
                payload["accepted_label_distribution"] for payload in round_payloads
            ),
            "rejected_label_distribution": _sum_distributions(
                payload["rejected_label_distribution"] for payload in round_payloads
            ),
        },
        "rounds": round_payloads,
    }


def _round_pseudo_label_quality(
    round_summary: SimulationRoundSummary,
) -> dict[str, object]:
    candidate_count = sum(client.candidate_count for client in round_summary.clients)
    diagnostic_candidate_count = sum(
        client.diagnostic_candidate_count for client in round_summary.clients
    )
    accepted_count = sum(client.accepted_count for client in round_summary.clients)
    correct_count = sum(
        client.pseudo_label_correct_count for client in round_summary.clients
    )
    evaluated_count = sum(
        client.pseudo_label_evaluated_count for client in round_summary.clients
    )
    return {
        "round_id": round_summary.round_id,
        "candidate_count": candidate_count,
        "diagnostic_candidate_count": diagnostic_candidate_count,
        "accepted_count": accepted_count,
        "accepted_ratio": safe_ratio(accepted_count, candidate_count),
        "pseudo_label_accuracy": safe_ratio(correct_count, evaluated_count),
        "pseudo_label_correct_count": correct_count,
        "pseudo_label_evaluated_count": evaluated_count,
        "pseudo_label_accuracy_basis": "accepted_candidates_with_simulation_labels",
        "pseudo_label_confidence_mean": _weighted_client_mean(
            round_summary.clients,
            value_attr="pseudo_label_confidence_mean",
            weight_attr="diagnostic_candidate_count",
        ),
        "candidate_confidence_mean": _weighted_client_mean(
            round_summary.clients,
            value_attr="pseudo_label_confidence_mean",
            weight_attr="diagnostic_candidate_count",
        ),
        "pseudo_label_margin_mean": _weighted_client_mean(
            round_summary.clients,
            value_attr="pseudo_label_margin_mean",
            weight_attr="diagnostic_candidate_count",
        ),
        "candidate_margin_mean": _weighted_client_mean(
            round_summary.clients,
            value_attr="pseudo_label_margin_mean",
            weight_attr="diagnostic_candidate_count",
        ),
        "accepted_label_distribution": _sum_distributions(
            client.accepted_label_distribution for client in round_summary.clients
        ),
        "rejected_label_distribution": _sum_distributions(
            client.rejected_label_distribution for client in round_summary.clients
        ),
    }


def _weighted_client_mean(
    clients: tuple[ClientRoundSummary, ...],
    *,
    value_attr: str,
    weight_attr: str,
) -> float | None:
    return weighted_mean(
        (getattr(client, value_attr), int(getattr(client, weight_attr)))
        for client in clients
    )


def _timing_breakdown_summary(
    values_by_key: dict[str, list[float]],
) -> dict[str, dict[str, float | int | None]]:
    return {
        key: numeric_summary(values)
        for key, values in sorted(values_by_key.items())
        if values
    }


def _weighted_round_mean(
    round_payloads: list[dict[str, object]],
    *,
    value_key: str,
    weight_key: str,
) -> float | None:
    return weighted_mean(
        (
            None if payload[value_key] is None else float(payload[value_key]),
            int(payload[weight_key]),
        )
        for payload in round_payloads
    )


def _sum_distributions(
    distributions: object,
) -> dict[str, int]:
    total: Counter[str] = Counter()
    for distribution in distributions:
        if not isinstance(distribution, Mapping):
            continue
        total.update({str(label): int(count) for label, count in distribution.items()})
    return dict(sorted(total.items()))
