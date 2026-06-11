"""FL round progression report payload helpers."""

from __future__ import annotations

from methods.federated_ssl.diagnostics.client import (
    client_method_diagnostics_payload,
)
from scripts.experiments.fl_ssl.federated_simulation.io.report_math import safe_ratio
from scripts.experiments.fl_ssl.federated_simulation.io.report_metrics import (
    evaluation_to_payload,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    ClientRoundSummary,
    SimulationEvaluation,
    SimulationResult,
)


def build_round_payloads(result: SimulationResult) -> list[dict[str, object]]:
    previous_validation = result.initial_validation
    payloads: list[dict[str, object]] = []
    for round_index, round_summary in enumerate(result.rounds, start=1):
        payloads.append(
            {
                "round_id": round_summary.round_id,
                "round_index": round_index,
                "model_revision": round_summary.model_revision,
                "update_count": round_summary.update_count,
                "total_client_count": round_summary.total_client_count,
                "selected_client_count": round_summary.selected_client_count,
                "skipped_client_count": round_summary.skipped_client_count,
                "skipped_client_ids": list(round_summary.skipped_client_ids),
                "validation": evaluation_to_payload(round_summary.validation),
                "global_validation": evaluation_to_payload(round_summary.validation),
                "delta_from_previous_round": evaluation_delta(
                    previous=previous_validation,
                    current=round_summary.validation,
                ),
                "delta_from_initial": evaluation_delta(
                    previous=result.initial_validation,
                    current=round_summary.validation,
                ),
                "clients": [
                    _client_round_payload(client) for client in round_summary.clients
                ],
                "round_time_seconds": round_summary.round_time_seconds,
                "round_timing_breakdown": dict(round_summary.round_timing_breakdown),
                "aggregation_metrics": dict(round_summary.aggregation_metrics),
                "total_payload_bytes": round_summary.total_payload_bytes,
                "gpu_memory_peak_mb": round_summary.gpu_memory_peak_mb,
            }
        )
        previous_validation = round_summary.validation
    return payloads


def build_round_progression(result: SimulationResult) -> dict[str, object]:
    validation_points = [
        {
            "round_id": "initial",
            "round_index": 0,
            "validation": result.initial_validation,
        }
    ] + [
        {
            "round_id": round_summary.round_id,
            "round_index": index,
            "validation": round_summary.validation,
        }
        for index, round_summary in enumerate(result.rounds, start=1)
    ]
    best_macro_f1 = max(
        validation_points,
        key=lambda point: point["validation"].macro_f1,
    )
    best_loss = min(
        validation_points,
        key=lambda point: point["validation"].loss,
    )
    best_round = _round_point_to_payload(best_macro_f1)
    best_round.update(
        {
            "selection_metric": "macro_f1",
            "selection_mode": "max",
        }
    )
    return {
        "validation_curve": [
            _round_point_to_payload(point) for point in validation_points
        ],
        "best_round": best_round,
        "best_macro_f1_round": _round_point_to_payload(best_macro_f1),
        "best_loss_round": _round_point_to_payload(best_loss),
        "early_stop_candidate": _build_early_stop_candidate(result),
        "final_delta_from_initial": evaluation_delta(
            previous=result.initial_validation,
            current=result.final_validation,
        ),
        "round_count": len(result.rounds),
    }


def evaluation_delta(
    *,
    previous: SimulationEvaluation,
    current: SimulationEvaluation,
) -> dict[str, float]:
    return {
        "loss_delta": current.loss - previous.loss,
        "loss_reduction": previous.loss - current.loss,
        "macro_f1_delta": current.macro_f1 - previous.macro_f1,
        "accuracy_top_1_delta": current.accuracy_top_1 - previous.accuracy_top_1,
        "expected_calibration_error_delta": (
            current.expected_calibration_error - previous.expected_calibration_error
        ),
        "accepted_ratio_delta": current.accepted_ratio - previous.accepted_ratio,
    }


def _client_round_payload(client: ClientRoundSummary) -> dict[str, object]:
    payload = {
        "client_id": client.client_id,
        "candidate_count": client.candidate_count,
        "diagnostic_candidate_count": client.diagnostic_candidate_count,
        "accepted_count": client.accepted_count,
        "accepted_ratio": safe_ratio(
            client.accepted_count,
            client.candidate_count,
        ),
        "update_generated": client.update_generated,
        "aggregation_example_count": client.aggregation_example_count,
        "delta_l2_norm": client.delta_l2_norm,
        "client_train_time_seconds": client.client_train_time_seconds,
        "client_payload_bytes": client.client_payload_bytes,
        "client_artifact_bytes": client.client_artifact_bytes,
        "candidate_confidence_mean": client.pseudo_label_confidence_mean,
        "candidate_margin_mean": client.pseudo_label_margin_mean,
        "pseudo_label_confidence_mean": client.pseudo_label_confidence_mean,
        "pseudo_label_margin_mean": client.pseudo_label_margin_mean,
        "pseudo_label_accuracy": safe_ratio(
            client.pseudo_label_correct_count,
            client.pseudo_label_evaluated_count,
        ),
        "pseudo_label_correct_count": client.pseudo_label_correct_count,
        "pseudo_label_evaluated_count": client.pseudo_label_evaluated_count,
        "accepted_label_distribution": client.accepted_label_distribution,
        "rejected_label_distribution": client.rejected_label_distribution,
        "timing_breakdown": dict(client.timing_breakdown),
    }
    payload.update(client_method_diagnostics_payload(client.method_diagnostics))
    return payload


def _round_point_to_payload(point: dict[str, object]) -> dict[str, object]:
    validation = point["validation"]
    if not isinstance(validation, SimulationEvaluation):
        raise TypeError("round progression validation must be SimulationEvaluation.")
    return {
        "round_id": point["round_id"],
        "round_index": point["round_index"],
        "macro_f1": validation.macro_f1,
        "loss": validation.loss,
        "expected_calibration_error": validation.expected_calibration_error,
        "accepted_ratio": validation.accepted_ratio,
        "accuracy_top_1": validation.accuracy_top_1,
    }


def _build_early_stop_candidate(
    result: SimulationResult,
    *,
    patience_rounds: int = 3,
    min_delta: float = 1e-6,
) -> dict[str, object]:
    """macro-F1 정체 여부를 report에서 빠르게 확인하기 위한 진단값."""

    if not result.rounds:
        return {
            "status": "no_rounds",
            "is_candidate": False,
            "metric": "macro_f1",
            "mode": "maximize",
            "patience_rounds": patience_rounds,
            "min_delta": min_delta,
            "rounds_without_improvement": 0,
        }

    best_value = result.initial_validation.macro_f1
    rounds_without_improvement = 0
    for round_summary in result.rounds:
        current_value = round_summary.validation.macro_f1
        if current_value > best_value + min_delta:
            best_value = current_value
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1

    if len(result.rounds) < patience_rounds:
        status = "insufficient_rounds"
        is_candidate = False
    else:
        is_candidate = rounds_without_improvement >= patience_rounds
        status = "candidate" if is_candidate else "not_candidate"
    return {
        "status": status,
        "is_candidate": is_candidate,
        "metric": "macro_f1",
        "mode": "maximize",
        "patience_rounds": patience_rounds,
        "min_delta": min_delta,
        "rounds_without_improvement": rounds_without_improvement,
        "best_value": best_value,
    }
