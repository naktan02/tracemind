"""FL simulation sweep summary payload helpers."""

from __future__ import annotations

from pathlib import Path

from scripts.experiments.fl_ssl.federated_simulation.io.report_math import (
    population_variance,
)
from scripts.experiments.fl_ssl.federated_simulation.io.report_metrics import (
    build_communication_cost_summary,
    evaluation_to_payload,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    ClientEvaluationSummary,
    SimulationResult,
)


def build_sweep_run_payload(
    *,
    result: SimulationResult,
    output_dir: Path,
    protocol: dict[str, object] | None = None,
) -> dict[str, object]:
    """seed/client-count sweep가 공유하는 단일 run summary payload."""

    return {
        "output_dir": str(output_dir),
        "report_path": result.report_path,
        "protocol": {
            "completed_rounds": len(result.rounds),
            "initial_model_revision": result.initial_model_revision,
            "initial_prototype_version": result.initial_prototype_version,
            **(protocol or {}),
        },
        "metrics": {
            "primary": {
                "macro_f1": result.final_validation.macro_f1,
                "worst_client_macro_f1": worst_client_macro_f1(
                    result.client_evaluations
                ),
            },
            "secondary": {
                "loss": result.final_validation.loss,
                "weighted_f1": result.final_validation.weighted_f1,
                "balanced_accuracy": result.final_validation.balanced_accuracy,
                "worst_category_f1_value": (
                    result.final_validation.worst_category_f1_value
                ),
                "max_calibration_error": result.final_validation.max_calibration_error,
                "expected_calibration_error": (
                    result.final_validation.expected_calibration_error
                ),
                "communication_cost": build_communication_cost_summary(result),
                "per_client_macro_f1_variance": population_variance(
                    [
                        client.validation.macro_f1
                        for client in result.client_evaluations
                        if client.validation.row_count > 0
                    ]
                ),
            },
            "initial_validation": evaluation_to_payload(result.initial_validation),
            "final_validation": evaluation_to_payload(result.final_validation),
        },
    }


def worst_client_macro_f1(
    client_evaluations: tuple[ClientEvaluationSummary, ...],
) -> float | None:
    values = [
        client.validation.macro_f1
        for client in client_evaluations
        if client.validation.row_count > 0
    ]
    return min(values) if values else None
