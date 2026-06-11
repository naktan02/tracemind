"""공통 FL capability policy 단위 테스트."""

from __future__ import annotations

import pytest

from methods.adaptation.federated_ssl_server_update import (
    resolve_federated_ssl_server_update_backend_name,
)
from methods.adaptation.implementation_modules import (
    adaptation_implementation_module_name,
)
from methods.adaptation.peft_text_encoder.aggregation import (
    peft_encoder_partitioned_projection as peft_part_projection,
)
from methods.federated.aggregation_weighting import (
    AggregationWeightPolicy,
    aggregation_example_count_for_diagnostics,
    aggregation_weight_basis_label,
    aggregation_weight_for_diagnostics,
    normalized_aggregation_weights,
)
from methods.federated.participation import (
    ClientParticipationPolicy,
    select_participating_clients,
    select_participating_indices,
)
from methods.federated_ssl.capabilities.axes import (
    LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT,
    LOCAL_SSL_POLICY_FIXMATCH,
    LOCAL_SSL_POLICY_FLEXMATCH,
    LOCAL_SSL_POLICY_PROFILE_PSEUDO_LABEL,
    SERVER_UPDATE_FEDAVG_MERGED_DELTA,
    SERVER_UPDATE_FEDMATCH_PARTITIONED,
    is_query_ssl_local_objective_policy,
)
from methods.federated_ssl.capabilities.plan import (
    LOCAL_SUPERVISION_CLIENT_LABELED_AND_UNLABELED,
    LOCAL_SUPERVISION_CLIENT_UNLABELED_ONLY,
    PEER_CONTEXT_FIXED_PROBE_OUTPUT_KNN,
    FederatedSslCapabilityPlan,
)
from methods.federated_ssl.compatibility import (
    validate_federated_ssl_capability_compatibility,
    validate_federated_ssl_local_ssl_policy_alignment,
    validate_federated_ssl_simulation_runtime_support,
)
from methods.federated_ssl.execution_plan import (
    COMPOSITION_MODE_MANUAL,
    COMPOSITION_MODE_METHOD_OWNED,
)
from methods.federated_ssl.hooks.local_objective import (
    requires_method_helper_probability_provider,
)
from methods.federated_ssl.hooks.server_step import (
    resolve_method_supervised_seed_step_parameters,
)
from methods.federated_ssl.local_supervision import (
    require_rows_match_local_supervision_regime,
    resolve_local_supervision_regime,
)
from methods.federated_ssl.registry import resolve_federated_ssl_method_descriptor
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PEFT_CLASSIFIER_ADAPTER_KIND,
)


class _Update:
    def __init__(
        self,
        *,
        example_count: int,
        accepted_count: int | None = None,
    ) -> None:
        self.example_count = example_count
        self.accepted_count = accepted_count


class _DiagnosticClient:
    def __init__(
        self,
        *,
        accepted_count: int,
        aggregation_example_count: int | None = None,
    ) -> None:
        self.accepted_count = accepted_count
        self.aggregation_example_count = aggregation_example_count


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


def test_unknown_peer_context_name_is_rejected() -> None:
    legacy_name = "prediction_similarity_topk"
    with pytest.raises(ValueError, match="peer_context_policy.name"):
        FederatedSslCapabilityPlan.from_mappings(
            client_participation_policy=None,
            aggregation_weight_policy=None,
            labeled_exposure_policy=None,
            local_supervision_regime=None,
            server_step_policy=None,
            peer_context_policy={"name": legacy_name},
            update_partition_policy=None,
            local_ssl_policy=None,
            server_update_policy=None,
            query_multiview_source=None,
        )


def test_local_supervision_regime_requires_client_labeled_rows_when_exposed() -> None:
    regime = resolve_local_supervision_regime(
        LOCAL_SUPERVISION_CLIENT_LABELED_AND_UNLABELED
    )

    with pytest.raises(ValueError, match="requires client labeled_rows"):
        require_rows_match_local_supervision_regime(
            regime=regime,
            labeled_rows=[],
            unlabeled_rows=[{"query_id": "u1", "text": "weak"}],
            context="test local runtime",
        )


def test_local_supervision_regime_rejects_labeled_rows_for_unlabeled_clients() -> None:
    regime = resolve_local_supervision_regime(LOCAL_SUPERVISION_CLIENT_UNLABELED_ONLY)

    with pytest.raises(ValueError, match="must not receive client labeled_rows"):
        require_rows_match_local_supervision_regime(
            regime=regime,
            labeled_rows=[{"query_id": "l1", "text": "labeled"}],
            unlabeled_rows=[{"query_id": "u1", "text": "weak"}],
            context="test local runtime",
        )


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


def test_aggregation_diagnostics_weight_helpers_follow_policy_meaning() -> None:
    client = _DiagnosticClient(accepted_count=3, aggregation_example_count=5)
    fallback_client = _DiagnosticClient(accepted_count=3)

    assert aggregation_example_count_for_diagnostics(client) == 5
    assert aggregation_example_count_for_diagnostics(fallback_client) == 3
    assert (
        aggregation_weight_for_diagnostics(
            client,
            policy=AggregationWeightPolicy(name="example_count"),
        )
        == 5.0
    )
    assert (
        aggregation_weight_for_diagnostics(
            client,
            policy=AggregationWeightPolicy(name="accepted_count"),
        )
        == 3.0
    )
    assert (
        aggregation_weight_for_diagnostics(
            client,
            policy=AggregationWeightPolicy(name="uniform"),
        )
        == 1.0
    )
    assert (
        aggregation_weight_basis_label(AggregationWeightPolicy(name="example_count"))
        == "update_envelope.example_count"
    )
    assert (
        aggregation_weight_basis_label(AggregationWeightPolicy(name="uniform"))
        == "uniform"
    )


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
    assert plan.aggregation_weight_policy.name == "uniform"
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
        server_update_policy={"name": SERVER_UPDATE_FEDMATCH_PARTITIONED},
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
        server_update_policy={"name": SERVER_UPDATE_FEDMATCH_PARTITIONED},
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
        peer_context_policy={"name": PEER_CONTEXT_FIXED_PROBE_OUTPUT_KNN},
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


def test_fedmatch_partitioned_server_update_can_express_fixmatch_policy() -> None:
    descriptor = resolve_federated_ssl_method_descriptor("fedmatch")
    plan = FederatedSslCapabilityPlan.from_mappings(
        client_participation_policy={"name": "all_clients"},
        aggregation_weight_policy={"name": "uniform"},
        labeled_exposure_policy={"name": "shared_client_seed"},
        local_supervision_regime={"name": "client_labeled_and_unlabeled"},
        server_step_policy={"name": "none"},
        peer_context_policy={"name": PEER_CONTEXT_FIXED_PROBE_OUTPUT_KNN},
        update_partition_policy={"name": "partitioned"},
        local_ssl_policy={"name": LOCAL_SSL_POLICY_FIXMATCH},
        server_update_policy={"name": SERVER_UPDATE_FEDMATCH_PARTITIONED},
        query_multiview_source={"name": "materialized_rows"},
    )

    validate_federated_ssl_capability_compatibility(
        method_descriptor=descriptor,
        capability_plan=plan,
    )


def test_fedmatch_agreement_requires_partitioned_server_update() -> None:
    descriptor = resolve_federated_ssl_method_descriptor("fedmatch")
    plan = FederatedSslCapabilityPlan.from_mappings(
        client_participation_policy={"name": "all_clients"},
        aggregation_weight_policy={"name": "uniform"},
        labeled_exposure_policy={"name": "shared_client_seed"},
        local_supervision_regime={"name": "client_labeled_and_unlabeled"},
        server_step_policy={"name": "none"},
        peer_context_policy={"name": PEER_CONTEXT_FIXED_PROBE_OUTPUT_KNN},
        update_partition_policy={"name": "partitioned"},
        local_ssl_policy={"name": LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT},
        server_update_policy={"name": SERVER_UPDATE_FEDAVG_MERGED_DELTA},
        query_multiview_source={"name": "materialized_rows"},
    )

    with pytest.raises(ValueError, match="fedmatch_partitioned"):
        validate_federated_ssl_capability_compatibility(
            method_descriptor=descriptor,
            capability_plan=plan,
        )


def test_fedmatch_partitioned_fixmatch_is_simulation_supported() -> None:
    descriptor = resolve_federated_ssl_method_descriptor("fedmatch")
    plan = FederatedSslCapabilityPlan.from_mappings(
        client_participation_policy={"name": "all_clients"},
        aggregation_weight_policy={"name": "uniform"},
        labeled_exposure_policy={"name": "shared_client_seed"},
        local_supervision_regime={"name": "client_labeled_and_unlabeled"},
        server_step_policy={"name": "none"},
        peer_context_policy={"name": PEER_CONTEXT_FIXED_PROBE_OUTPUT_KNN},
        update_partition_policy={"name": "partitioned"},
        local_ssl_policy={"name": LOCAL_SSL_POLICY_FIXMATCH},
        server_update_policy={"name": SERVER_UPDATE_FEDMATCH_PARTITIONED},
        query_multiview_source={"name": "materialized_rows"},
    )

    validate_federated_ssl_simulation_runtime_support(
        capability_plan=plan,
        composition_mode=COMPOSITION_MODE_METHOD_OWNED,
        method_descriptor=descriptor,
    )


def test_peft_classifier_payload_module_resolves_to_peft_text_encoder_owner() -> None:
    assert adaptation_implementation_module_name(
        payload_adapter_kind=PEFT_CLASSIFIER_ADAPTER_KIND,
        submodule="federated_ssl.server_update_policy",
    ) == ("methods.adaptation.peft_text_encoder.federated_ssl.server_update_policy")


def test_fedmatch_partitioned_server_update_resolves_for_peft_classifier_family() -> (
    None
):
    resolved_backend = resolve_federated_ssl_server_update_backend_name(
        payload_adapter_kind=PEFT_CLASSIFIER_ADAPTER_KIND,
        server_update_policy_name=SERVER_UPDATE_FEDMATCH_PARTITIONED,
        aggregation_backend_name="fedavg",
    )

    assert (
        resolved_backend == peft_part_projection.PARTITIONED_DELTA_AVERAGE_BACKEND_NAME
    )


def test_fedmatch_server_seed_parameters_resolve_by_method_convention() -> None:
    first_round = resolve_method_supervised_seed_step_parameters(
        method_name="fedmatch",
        effective_parameters={
            "server_pretrain_epochs": 2,
            "server_epochs": 1,
            "server_batch_size": 3,
        },
        default_epochs=9,
        default_batch_size=8,
        round_index=1,
    )
    later_round = resolve_method_supervised_seed_step_parameters(
        method_name="fedmatch",
        effective_parameters={
            "server_pretrain_epochs": 2,
            "server_epochs": 1,
            "server_batch_size": 3,
        },
        default_epochs=9,
        default_batch_size=8,
        round_index=2,
    )

    assert first_round.epochs == 2
    assert later_round.epochs == 1
    assert first_round.batch_size == 3


def test_fedmatch_server_seed_parameters_use_method_family() -> None:
    first_round = resolve_method_supervised_seed_step_parameters(
        method_name="fedmatch",
        effective_parameters={
            "server_pretrain_epochs": 2,
            "server_epochs": 1,
            "server_batch_size": 3,
        },
        default_epochs=9,
        default_batch_size=8,
        round_index=1,
    )

    assert first_round.epochs == 2
    assert first_round.batch_size == 3


def test_fedmatch_helper_provider_requirement_resolves_by_method_convention() -> None:
    assert requires_method_helper_probability_provider(
        method_name="fedmatch",
        local_ssl_policy_name=LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT,
    )
    assert not requires_method_helper_probability_provider(
        method_name="fedmatch",
        local_ssl_policy_name=LOCAL_SSL_POLICY_FIXMATCH,
    )


def test_fedmatch_server_seed_parameters_support_original_values() -> None:
    first_round = resolve_method_supervised_seed_step_parameters(
        method_name="fedmatch",
        effective_parameters={
            "server_pretrain_epochs": 2,
            "server_epochs": 1,
            "server_batch_size": 3,
        },
        default_epochs=9,
        default_batch_size=8,
        round_index=1,
    )
    later_round = resolve_method_supervised_seed_step_parameters(
        method_name="fedmatch",
        effective_parameters={
            "server_pretrain_epochs": 2,
            "server_epochs": 1,
            "server_batch_size": 3,
        },
        default_epochs=9,
        default_batch_size=8,
        round_index=2,
    )

    assert first_round.epochs == 2
    assert later_round.epochs == 1
    assert first_round.batch_size == 3


def test_manual_partitioned_server_update_waits_for_partition_producer() -> None:
    plan = FederatedSslCapabilityPlan.from_mappings(
        client_participation_policy={"name": "all_clients"},
        aggregation_weight_policy={"name": "uniform"},
        labeled_exposure_policy={"name": "shared_client_seed"},
        local_supervision_regime={"name": "client_labeled_and_unlabeled"},
        server_step_policy={"name": "none"},
        peer_context_policy={"name": "none"},
        update_partition_policy={"name": "partitioned"},
        local_ssl_policy={"name": LOCAL_SSL_POLICY_FIXMATCH},
        server_update_policy={"name": SERVER_UPDATE_FEDMATCH_PARTITIONED},
        query_multiview_source={"name": "materialized_rows"},
    )

    with pytest.raises(ValueError, match="partitioned_deltas"):
        validate_federated_ssl_simulation_runtime_support(
            capability_plan=plan,
            composition_mode=COMPOSITION_MODE_MANUAL,
        )


def test_fedmatch_partitioned_blocks_stateful_local_ssl_until_state_surface() -> None:
    descriptor = resolve_federated_ssl_method_descriptor("fedmatch")
    plan = FederatedSslCapabilityPlan.from_mappings(
        client_participation_policy={"name": "all_clients"},
        aggregation_weight_policy={"name": "uniform"},
        labeled_exposure_policy={"name": "shared_client_seed"},
        local_supervision_regime={"name": "client_labeled_and_unlabeled"},
        server_step_policy={"name": "none"},
        peer_context_policy={"name": PEER_CONTEXT_FIXED_PROBE_OUTPUT_KNN},
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


def test_query_ssl_local_objective_policy_predicate_normalizes_names() -> None:
    assert is_query_ssl_local_objective_policy("fixmatch")
    assert is_query_ssl_local_objective_policy("FLEXMATCH")
    assert not is_query_ssl_local_objective_policy(
        LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT
    )
