"""FL aggregation diagnostics payload helpers."""

from __future__ import annotations

from methods.federated.aggregation_weighting import (
    AggregationWeightPolicy,
    aggregation_example_count_for_diagnostics,
    aggregation_weight_basis_label,
    aggregation_weight_for_diagnostics,
)
from methods.federated_ssl.capabilities.plan import FederatedSslCapabilityPlan
from methods.federated_ssl.diagnostics.client import (
    client_method_diagnostics_summary_payload,
)
from scripts.experiments.fl_ssl.federated_simulation.io.report_math import (
    numeric_summary,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    ClientRoundSummary,
    SimulationResult,
    SimulationRoundSummary,
)


def build_aggregation_diagnostics(
    result: SimulationResult,
    *,
    capability_plan: FederatedSslCapabilityPlan | None = None,
) -> dict[str, object]:
    weight_policy = (
        AggregationWeightPolicy()
        if capability_plan is None
        else capability_plan.aggregation_weight_policy
    )
    return {
        "weight_basis": aggregation_weight_basis_label(weight_policy),
        "weight_policy_name": weight_policy.name,
        "client_weight_details_excluded": True,
        "rounds": [
            _round_aggregation_diagnostics(
                round_summary,
                weight_policy=weight_policy,
            )
            for round_summary in result.rounds
        ],
    }


def aggregation_example_count(client: ClientRoundSummary) -> int:
    return aggregation_example_count_for_diagnostics(client)


def _round_aggregation_diagnostics(
    round_summary: SimulationRoundSummary,
    *,
    weight_policy: AggregationWeightPolicy,
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
    aggregation_weights = [
        aggregation_weight_for_diagnostics(client, policy=weight_policy)
        for client in round_summary.clients
        if client.update_generated
    ]
    total_aggregation_weight = sum(aggregation_weights)
    normalized_weights = [
        weight / total_aggregation_weight
        for weight in aggregation_weights
        if total_aggregation_weight > 0
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
    payload = {
        "round_id": round_summary.round_id,
        "total_client_count": (
            round_summary.total_client_count or len(round_summary.clients)
        ),
        "selected_client_count": (
            round_summary.selected_client_count or len(round_summary.clients)
        ),
        "participating_client_count": len(round_summary.clients),
        "skipped_client_count": round_summary.skipped_client_count,
        "update_count": round_summary.update_count,
        "zero_update_client_count": zero_update_client_count,
        "total_candidate_count": sum(
            client.candidate_count for client in round_summary.clients
        ),
        "total_accepted_count": sum(accepted_counts),
        "total_aggregation_examples": sum(aggregation_example_counts),
        "total_aggregation_weight": total_aggregation_weight,
        "aggregation_example_basis": "update_envelope.example_count",
        "aggregation_weight_basis": weight_policy.name,
        "aggregation_metrics": dict(round_summary.aggregation_metrics),
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
    payload.update(
        client_method_diagnostics_summary_payload(
            (client.method_diagnostics for client in round_summary.clients),
            numeric_summary=numeric_summary,
        )
    )
    return payload
