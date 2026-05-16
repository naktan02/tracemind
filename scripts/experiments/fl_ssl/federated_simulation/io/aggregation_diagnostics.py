"""FL aggregation diagnostics payload helpers."""

from __future__ import annotations

from scripts.experiments.fl_ssl.federated_simulation.io.report_math import (
    numeric_summary,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    ClientRoundSummary,
    SimulationResult,
    SimulationRoundSummary,
)


def build_aggregation_diagnostics(result: SimulationResult) -> dict[str, object]:
    return {
        "weight_basis": "update_envelope.example_count",
        "client_weight_details_excluded": True,
        "rounds": [
            _round_aggregation_diagnostics(round_summary)
            for round_summary in result.rounds
        ],
    }


def aggregation_example_count(client: ClientRoundSummary) -> int:
    if client.aggregation_example_count is not None:
        return client.aggregation_example_count
    return client.accepted_count


def _round_aggregation_diagnostics(
    round_summary: SimulationRoundSummary,
) -> dict[str, object]:
    aggregation_example_counts = [
        aggregation_example_count(client)
        for client in round_summary.clients
        if client.update_generated
    ]
    accepted_counts = [
        client.accepted_count
        for client in round_summary.clients
        if client.update_generated
    ]
    total_aggregation_examples = sum(aggregation_example_counts)
    normalized_weights = [
        example_count / total_aggregation_examples
        for example_count in aggregation_example_counts
        if total_aggregation_examples > 0
    ]
    delta_l2_norms = [
        client.delta_l2_norm
        for client in round_summary.clients
        if client.update_generated and client.delta_l2_norm is not None
    ]
    zero_update_client_count = sum(
        1 for client in round_summary.clients if not client.update_generated
    )
    delta_l2_norm_summary = numeric_summary(delta_l2_norms)
    return {
        "round_id": round_summary.round_id,
        "participating_client_count": len(round_summary.clients),
        "update_count": round_summary.update_count,
        "zero_update_client_count": zero_update_client_count,
        "total_candidate_count": sum(
            client.candidate_count for client in round_summary.clients
        ),
        "total_accepted_count": sum(accepted_counts),
        "total_aggregation_examples": total_aggregation_examples,
        "aggregation_example_basis": "update_envelope.example_count",
        "accepted_count_summary": numeric_summary(accepted_counts),
        "aggregation_example_count_summary": numeric_summary(
            aggregation_example_counts
        ),
        "aggregation_weight_summary": numeric_summary(normalized_weights),
        "delta_l2_norm_summary": delta_l2_norm_summary,
        "mean_delta_l2_norm": delta_l2_norm_summary["mean"],
        "max_delta_l2_norm": delta_l2_norm_summary["max"],
        "update_norm_variance": delta_l2_norm_summary["variance"],
    }
