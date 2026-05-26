"""FL aggregation diagnostics payload helpers."""

from __future__ import annotations

from methods.federated.aggregation_weighting import (
    AGGREGATION_WEIGHT_ACCEPTED_COUNT,
    AGGREGATION_WEIGHT_EXAMPLE_COUNT,
    AGGREGATION_WEIGHT_UNIFORM,
    AggregationWeightPolicy,
)
from methods.federated_ssl.capability_plan import FederatedSslCapabilityPlan
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
        "weight_basis": _aggregation_weight_basis_label(weight_policy),
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
    if client.aggregation_example_count is not None:
        return client.aggregation_example_count
    return client.accepted_count


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
        _client_aggregation_weight(client, policy=weight_policy)
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
    fedmatch_helper_counts = [
        client.fedmatch_helper_count
        for client in round_summary.clients
        if client.fedmatch_helper_count is not None
    ]
    fedmatch_peer_context_helper_counts = [
        client.fedmatch_peer_context_helper_count
        for client in round_summary.clients
        if client.fedmatch_peer_context_helper_count is not None
    ]
    fedmatch_helper_provider_counts = [
        client.fedmatch_helper_provider_count
        for client in round_summary.clients
        if client.fedmatch_helper_provider_count is not None
    ]
    fedmatch_missing_helper_snapshot_counts = [
        client.fedmatch_missing_helper_snapshot_count
        for client in round_summary.clients
        if client.fedmatch_missing_helper_snapshot_count is not None
    ]
    fedmatch_materialized_helper_model_counts = [
        client.fedmatch_materialized_helper_model_count
        for client in round_summary.clients
        if client.fedmatch_materialized_helper_model_count is not None
    ]
    fedmatch_c2s_sparse_upload_value_counts = [
        client.fedmatch_c2s_sparse_upload_value_count
        for client in round_summary.clients
        if client.fedmatch_c2s_sparse_upload_value_count is not None
    ]
    fedmatch_s2c_sparse_download_value_counts = [
        client.fedmatch_s2c_sparse_download_value_count
        for client in round_summary.clients
        if client.fedmatch_s2c_sparse_download_value_count is not None
    ]
    zero_update_client_count = sum(
        1 for client in round_summary.clients if not client.update_generated
    )
    delta_l2_norm_summary = numeric_summary(delta_l2_norms)
    return {
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
        "fedmatch_helper_count_summary": numeric_summary(fedmatch_helper_counts),
        "fedmatch_peer_context_helper_count_summary": numeric_summary(
            fedmatch_peer_context_helper_counts
        ),
        "fedmatch_helper_provider_count_summary": numeric_summary(
            fedmatch_helper_provider_counts
        ),
        "fedmatch_missing_helper_snapshot_count_summary": numeric_summary(
            fedmatch_missing_helper_snapshot_counts
        ),
        "fedmatch_materialized_helper_model_count_summary": numeric_summary(
            fedmatch_materialized_helper_model_counts
        ),
        "fedmatch_peer_context_refreshed_count": sum(
            1
            for client in round_summary.clients
            if client.fedmatch_peer_context_refreshed
        ),
        "fedmatch_c2s_sparse_upload_value_count_summary": numeric_summary(
            fedmatch_c2s_sparse_upload_value_counts
        ),
        "fedmatch_s2c_sparse_download_value_count_summary": numeric_summary(
            fedmatch_s2c_sparse_download_value_counts
        ),
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


def _client_aggregation_weight(
    client: ClientRoundSummary,
    *,
    policy: AggregationWeightPolicy,
) -> float:
    if policy.name == AGGREGATION_WEIGHT_UNIFORM:
        return 1.0
    if policy.name == AGGREGATION_WEIGHT_ACCEPTED_COUNT:
        return float(client.accepted_count)
    if policy.name == AGGREGATION_WEIGHT_EXAMPLE_COUNT:
        return float(aggregation_example_count(client))
    raise ValueError(f"Unsupported aggregation weight policy: {policy.name}")


def _aggregation_weight_basis_label(policy: AggregationWeightPolicy) -> str:
    if policy.name == AGGREGATION_WEIGHT_EXAMPLE_COUNT:
        return "update_envelope.example_count"
    return policy.name
