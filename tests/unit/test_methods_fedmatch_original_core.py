"""FedMatch 원본 core mapping 단위 테스트."""

from __future__ import annotations

import pytest

from methods.federated_ssl.fedmatch.helper_selection import (
    select_helper_client_ids,
    should_refresh_helper_context,
)
from methods.federated_ssl.fedmatch.local_objective import (
    FEDMATCH_LOSS_COMPONENTS,
    agreement_pseudo_label_indices,
    fedmatch_loss_weights,
    select_confident_prediction_indices,
)
from methods.federated_ssl.fedmatch.original_spec import (
    FEDMATCH_ORIGINAL_COMMIT,
    FEDMATCH_ORIGINAL_REPOSITORY,
    FEDMATCH_SCENARIO_LABELS_AT_CLIENT,
    FEDMATCH_SCENARIO_LABELS_AT_SERVER,
    fedmatch_original_parameter_mapping,
    resolve_original_scenario_spec,
)
from methods.federated_ssl.fedmatch.parameter_routing import (
    FEDMATCH_PSI_PARTITION,
    FEDMATCH_SIGMA_PARTITION,
    trace_parameter_mapping,
    upload_partitions_for_scenario,
)
from methods.federated_ssl.method_parameters import (
    build_federated_ssl_method_parameter_snapshot,
)


def test_fedmatch_original_source_snapshot_is_pinned() -> None:
    params = fedmatch_original_parameter_mapping()

    assert FEDMATCH_ORIGINAL_REPOSITORY == "https://github.com/wyjeong/FedMatch.git"
    assert FEDMATCH_ORIGINAL_COMMIT == ("4947aa255d59bd37915e25a719763aaaf5d7e067")
    assert params["scenario"] == FEDMATCH_SCENARIO_LABELS_AT_CLIENT
    assert params["num_clients"] == 100
    assert params["num_rounds"] == 200
    assert params["client_fraction"] == pytest.approx(0.05)
    assert params["confidence_threshold"] == pytest.approx(0.75)
    assert params["psi_factor"] == pytest.approx(0.2)
    assert params["num_helpers"] == 2
    assert params["helper_refresh_interval"] == 10
    assert params["lambda_s"] == pytest.approx(10.0)
    assert params["lambda_i"] == pytest.approx(1e-2)
    assert params["lambda_a"] == pytest.approx(1e-2)
    assert params["lambda_l2"] == pytest.approx(10.0)
    assert params["lambda_l1"] == pytest.approx(1e-4)
    assert params["l1_threshold"] == pytest.approx(5e-6)
    assert params["delta_threshold"] == pytest.approx(5e-5)


def test_fedmatch_labels_at_server_keeps_original_scenario_differences() -> None:
    scenario = resolve_original_scenario_spec(FEDMATCH_SCENARIO_LABELS_AT_SERVER)
    params = fedmatch_original_parameter_mapping(
        scenario_name=FEDMATCH_SCENARIO_LABELS_AT_SERVER,
    )

    assert scenario.client_batch_size == 100
    assert scenario.server_epochs == 1
    assert scenario.server_pretrain_epochs == 1
    assert params["lambda_l1"] == pytest.approx(1e-5)
    assert params["l1_threshold"] == pytest.approx(1e-5)
    assert params["delta_threshold"] == pytest.approx(1e-5)


def test_fedmatch_loss_components_preserve_original_partition_routing() -> None:
    components = {component.name: component for component in FEDMATCH_LOSS_COMPONENTS}
    weights = fedmatch_loss_weights()

    assert components["supervised_cross_entropy"].updated_partition == (
        FEDMATCH_SIGMA_PARTITION
    )
    assert components["inter_client_consistency_kl"].updated_partition == (
        FEDMATCH_PSI_PARTITION
    )
    assert components["agreement_pseudo_label_cross_entropy"].updated_partition == (
        FEDMATCH_PSI_PARTITION
    )
    assert components["psi_l1_regularization"].updated_partition == (
        FEDMATCH_PSI_PARTITION
    )
    assert components["sigma_psi_l2_regularization"].updated_partition == (
        FEDMATCH_PSI_PARTITION
    )
    assert weights == {
        "lambda_s": pytest.approx(10.0),
        "lambda_i": pytest.approx(1e-2),
        "lambda_a": pytest.approx(1e-2),
        "lambda_l2": pytest.approx(10.0),
        "lambda_l1": pytest.approx(1e-4),
    }


def test_fedmatch_confidence_filter_is_inclusive_like_original_numpy_where() -> None:
    selected = select_confident_prediction_indices(
        (
            (0.2, 0.5, 0.3),
            (0.1, 0.75, 0.15),
            (0.8, 0.1, 0.1),
        ),
        confidence_threshold=0.75,
    )

    assert selected == (1, 2)


def test_fedmatch_agreement_pseudo_label_uses_client_and_helper_argmax_votes() -> None:
    labels = agreement_pseudo_label_indices(
        client_probabilities=(
            (0.1, 0.8, 0.1),
            (0.6, 0.3, 0.1),
            (0.4, 0.4, 0.2),
        ),
        helper_probabilities_by_helper=(
            (
                (0.2, 0.3, 0.5),
                (0.1, 0.8, 0.1),
                (0.1, 0.7, 0.2),
            ),
            (
                (0.1, 0.2, 0.7),
                (0.7, 0.2, 0.1),
                (0.3, 0.5, 0.2),
            ),
        ),
    )

    assert labels == (2, 0, 1)


def test_fedmatch_helper_refresh_and_topk_selection_preserve_original_policy() -> None:
    assert should_refresh_helper_context(round_index_zero_based=8) is False
    assert should_refresh_helper_context(round_index_zero_based=9) is True

    helpers = select_helper_client_ids(
        client_id="client_a",
        client_vectors={
            "client_a": (0.0, 0.0),
            "client_b": (0.2, 0.0),
            "client_c": (3.0, 0.0),
            "client_d": (0.1, 0.0),
        },
        num_helpers=2,
    )

    assert helpers == ("client_d", "client_b")


def test_fedmatch_lora_classifier_partition_mapping_is_explicit() -> None:
    assert trace_parameter_mapping.original_trainable_scope == (
        "ResNet9 Conv/Dense full weights"
    )
    assert trace_parameter_mapping.trace_trainable_scope == (
        "LoRA adapter tensors plus classifier head tensors"
    )
    assert trace_parameter_mapping.frozen_scope == "Transformer backbone base weights"
    assert upload_partitions_for_scenario(
        scenario_name=FEDMATCH_SCENARIO_LABELS_AT_CLIENT,
    ) == (FEDMATCH_SIGMA_PARTITION, FEDMATCH_PSI_PARTITION)
    assert upload_partitions_for_scenario(
        scenario_name=FEDMATCH_SCENARIO_LABELS_AT_SERVER,
    ) == (FEDMATCH_PSI_PARTITION,)


def test_fedmatch_generic_parameter_snapshot_applies_overrides() -> None:
    snapshot = build_federated_ssl_method_parameter_snapshot(
        method_name="fedmatch",
        method_config={
            "scenario": FEDMATCH_SCENARIO_LABELS_AT_CLIENT,
            "use_original_parameters": True,
            "parameter_overrides": {
                "confidence_threshold": 0.85,
                "num_helpers": 4,
            },
        },
    )

    assert snapshot.original_parameters["confidence_threshold"] == pytest.approx(0.75)
    assert snapshot.effective_parameters["confidence_threshold"] == pytest.approx(0.85)
    assert snapshot.effective_parameters["num_helpers"] == 4
    assert snapshot.parameter_override_status == "ablation"


def test_fedmatch_generic_parameter_snapshot_rejects_unknown_overrides() -> None:
    with pytest.raises(ValueError, match="Unknown parameter_overrides"):
        build_federated_ssl_method_parameter_snapshot(
            method_name="fedmatch",
            method_config={
                "scenario": FEDMATCH_SCENARIO_LABELS_AT_CLIENT,
                "use_original_parameters": True,
                "parameter_overrides": {"unknown": 1},
            },
        )
