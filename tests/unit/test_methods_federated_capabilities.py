"""공통 FL capability policy 단위 테스트."""

from __future__ import annotations

import pytest

from methods.federated.aggregation_weighting import (
    AggregationWeightPolicy,
    normalized_aggregation_weights,
)
from methods.federated.participation import (
    ClientParticipationPolicy,
    select_participating_clients,
    select_participating_indices,
)
from methods.federated_ssl.capability_axes import (
    LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT,
    LOCAL_SSL_POLICY_FIXMATCH,
    LOCAL_SSL_POLICY_FLEXMATCH,
    LOCAL_SSL_POLICY_PROFILE_PSEUDO_LABEL,
    SERVER_UPDATE_FEDAVG_MERGED_DELTA,
    SERVER_UPDATE_FEDMATCH_PARTITIONED,
)
from methods.federated_ssl.capability_plan import FederatedSslCapabilityPlan
from methods.federated_ssl.compatibility import (
    validate_federated_ssl_capability_compatibility,
    validate_federated_ssl_local_ssl_policy_alignment,
)
from methods.federated_ssl.registry import resolve_federated_ssl_method_descriptor


class _Update:
    def __init__(
        self,
        *,
        example_count: int,
        accepted_count: int | None = None,
    ) -> None:
        self.example_count = example_count
        self.accepted_count = accepted_count


def test_participation_policy_defaults_to_all_clients() -> None:
    selected_clients, selection = select_participating_clients(
        clients=("agent_01", "agent_02", "agent_03"),
        policy=ClientParticipationPolicy(),
        seed=42,
        round_index=1,
    )

    assert selected_clients == ("agent_01", "agent_02", "agent_03")
    assert selection.selected_indices == (0, 1, 2)
    assert selection.skipped_indices == ()


def test_fraction_random_participation_is_round_deterministic() -> None:
    policy = ClientParticipationPolicy(
        name="fraction_random",
        fraction=0.4,
        min_clients=1,
    )

    first = select_participating_indices(
        total_clients=5,
        policy=policy,
        seed=7,
        round_index=2,
    )
    second = select_participating_indices(
        total_clients=5,
        policy=policy,
        seed=7,
        round_index=2,
    )

    assert first == second
    assert first.selected_count == 2
    assert first.skipped_count == 3


def test_aggregation_weight_policy_normalizes_example_and_uniform_weights() -> None:
    updates = [
        _Update(example_count=2),
        _Update(example_count=6),
    ]

    assert normalized_aggregation_weights(
        updates,
        policy=AggregationWeightPolicy(name="example_count"),
    ) == [pytest.approx(0.25), pytest.approx(0.75)]
    assert normalized_aggregation_weights(
        updates,
        policy=AggregationWeightPolicy(name="uniform"),
    ) == [pytest.approx(0.5), pytest.approx(0.5)]


def test_capability_plan_defaults_to_shared_client_seed() -> None:
    plan = FederatedSslCapabilityPlan.from_mappings(
        client_participation_policy=None,
        aggregation_weight_policy=None,
        labeled_exposure_policy=None,
        local_supervision_regime=None,
        server_step_policy=None,
        peer_context_policy=None,
        update_partition_policy=None,
        local_ssl_policy=None,
        server_update_policy=None,
        query_multiview_source=None,
    )

    assert plan.labeled_exposure_policy_name == "shared_client_seed"
    assert plan.client_participation_policy.name == "all_clients"
    assert plan.aggregation_weight_policy.name == "example_count"
    assert plan.server_step_policy_name == "none"
    assert plan.server_update_policy_name == SERVER_UPDATE_FEDAVG_MERGED_DELTA
    assert plan.peer_context_policy_name == "none"
    assert plan.local_ssl_policy_name == LOCAL_SSL_POLICY_PROFILE_PSEUDO_LABEL


def test_manual_capability_plan_rejects_server_only_seed() -> None:
    plan = FederatedSslCapabilityPlan.from_mappings(
        client_participation_policy=None,
        aggregation_weight_policy=None,
        labeled_exposure_policy={"name": "server_only_seed"},
        local_supervision_regime={"name": "client_unlabeled_only"},
        server_step_policy={"name": "supervised_seed_step"},
        peer_context_policy=None,
        update_partition_policy=None,
        local_ssl_policy=None,
        server_update_policy=None,
        query_multiview_source=None,
    )

    with pytest.raises(ValueError, match="method-owned"):
        validate_federated_ssl_capability_compatibility(
            method_descriptor=None,
            capability_plan=plan,
        )


def test_fedmatch_descriptor_requires_partition_and_weight_capabilities() -> None:
    descriptor = resolve_federated_ssl_method_descriptor("fedmatch")
    compatible_plan = FederatedSslCapabilityPlan.from_mappings(
        client_participation_policy={"name": "all_clients"},
        aggregation_weight_policy={"name": "uniform"},
        labeled_exposure_policy={"name": "shared_client_seed"},
        local_supervision_regime={"name": "client_labeled_and_unlabeled"},
        server_step_policy={"name": "none"},
        peer_context_policy={"name": "none"},
        update_partition_policy={"name": "partitioned"},
        local_ssl_policy={"name": LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT},
        server_update_policy={"name": SERVER_UPDATE_FEDAVG_MERGED_DELTA},
        query_multiview_source={"name": "materialized_rows"},
    )
    incompatible_plan = FederatedSslCapabilityPlan.from_mappings(
        client_participation_policy={"name": "all_clients"},
        aggregation_weight_policy={"name": "example_count"},
        labeled_exposure_policy={"name": "shared_client_seed"},
        local_supervision_regime={"name": "client_labeled_and_unlabeled"},
        server_step_policy={"name": "none"},
        peer_context_policy={"name": "none"},
        update_partition_policy={"name": "unified"},
        local_ssl_policy={"name": LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT},
        server_update_policy={"name": SERVER_UPDATE_FEDAVG_MERGED_DELTA},
        query_multiview_source={"name": "materialized_rows"},
    )

    validate_federated_ssl_capability_compatibility(
        method_descriptor=descriptor,
        capability_plan=compatible_plan,
    )
    with pytest.raises(ValueError, match="update_partition_policy"):
        validate_federated_ssl_capability_compatibility(
            method_descriptor=descriptor,
            capability_plan=incompatible_plan,
        )


def test_fedmatch_partitioned_server_update_requires_partitioned_update() -> None:
    descriptor = resolve_federated_ssl_method_descriptor("fedmatch")
    plan = FederatedSslCapabilityPlan.from_mappings(
        client_participation_policy={"name": "all_clients"},
        aggregation_weight_policy={"name": "uniform"},
        labeled_exposure_policy={"name": "shared_client_seed"},
        local_supervision_regime={"name": "client_labeled_and_unlabeled"},
        server_step_policy={"name": "none"},
        peer_context_policy={"name": "prediction_similarity_topk"},
        update_partition_policy={"name": "unified"},
        local_ssl_policy={"name": LOCAL_SSL_POLICY_FIXMATCH},
        server_update_policy={"name": SERVER_UPDATE_FEDMATCH_PARTITIONED},
        query_multiview_source={"name": "materialized_rows"},
    )

    with pytest.raises(ValueError, match="fedmatch_partitioned"):
        validate_federated_ssl_capability_compatibility(
            method_descriptor=descriptor,
            capability_plan=plan,
        )


def test_fedmatch_partitioned_server_update_can_pair_with_fixmatch_policy() -> None:
    descriptor = resolve_federated_ssl_method_descriptor("fedmatch")
    plan = FederatedSslCapabilityPlan.from_mappings(
        client_participation_policy={"name": "all_clients"},
        aggregation_weight_policy={"name": "uniform"},
        labeled_exposure_policy={"name": "shared_client_seed"},
        local_supervision_regime={"name": "client_labeled_and_unlabeled"},
        server_step_policy={"name": "none"},
        peer_context_policy={"name": "prediction_similarity_topk"},
        update_partition_policy={"name": "partitioned"},
        local_ssl_policy={"name": LOCAL_SSL_POLICY_FIXMATCH},
        server_update_policy={"name": SERVER_UPDATE_FEDMATCH_PARTITIONED},
        query_multiview_source={"name": "materialized_rows"},
    )

    validate_federated_ssl_capability_compatibility(
        method_descriptor=descriptor,
        capability_plan=plan,
    )


def test_fedmatch_partitioned_blocks_stateful_local_ssl_until_state_surface() -> None:
    descriptor = resolve_federated_ssl_method_descriptor("fedmatch")
    plan = FederatedSslCapabilityPlan.from_mappings(
        client_participation_policy={"name": "all_clients"},
        aggregation_weight_policy={"name": "uniform"},
        labeled_exposure_policy={"name": "shared_client_seed"},
        local_supervision_regime={"name": "client_labeled_and_unlabeled"},
        server_step_policy={"name": "none"},
        peer_context_policy={"name": "prediction_similarity_topk"},
        update_partition_policy={"name": "partitioned"},
        local_ssl_policy={"name": LOCAL_SSL_POLICY_FLEXMATCH},
        server_update_policy={"name": SERVER_UPDATE_FEDMATCH_PARTITIONED},
        query_multiview_source={"name": "materialized_rows"},
    )

    with pytest.raises(ValueError, match="state surface"):
        validate_federated_ssl_capability_compatibility(
            method_descriptor=descriptor,
            capability_plan=plan,
        )


def test_local_ssl_policy_must_match_query_ssl_algorithm_when_query_ssl_owned() -> None:
    plan = FederatedSslCapabilityPlan.from_mappings(
        client_participation_policy=None,
        aggregation_weight_policy=None,
        labeled_exposure_policy=None,
        local_supervision_regime=None,
        server_step_policy=None,
        peer_context_policy=None,
        update_partition_policy=None,
        local_ssl_policy={"name": LOCAL_SSL_POLICY_FIXMATCH},
        server_update_policy=None,
        query_multiview_source=None,
    )

    validate_federated_ssl_local_ssl_policy_alignment(
        capability_plan=plan,
        query_ssl_algorithm_name=LOCAL_SSL_POLICY_FIXMATCH,
    )
    with pytest.raises(ValueError, match="local_ssl_policy"):
        validate_federated_ssl_local_ssl_policy_alignment(
            capability_plan=plan,
            query_ssl_algorithm_name=LOCAL_SSL_POLICY_FLEXMATCH,
        )
