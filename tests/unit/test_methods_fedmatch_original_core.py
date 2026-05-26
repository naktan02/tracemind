"""FedMatch 원본 core mapping 단위 테스트."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from methods.federated_ssl.fedmatch.helper_selection import (
    build_helper_index,
    select_helper_client_ids,
    should_refresh_helper_context,
)
from methods.federated_ssl.fedmatch.local_objective import (
    FEDMATCH_AGREEMENT_PSEUDO_LABEL_CE,
    FEDMATCH_INTER_CLIENT_KL,
    FEDMATCH_LOSS_COMPONENTS,
    FEDMATCH_PSI_L1_REGULARIZATION,
    FEDMATCH_SIGMA_PSI_L2_REGULARIZATION,
    FEDMATCH_SUPERVISED_CE,
    FedMatchLocalObjectiveParameters,
    FedMatchParameterPartitions,
    agreement_pseudo_label_indices,
    compute_fedmatch_supervised_loss,
    compute_fedmatch_unsupervised_loss,
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
from methods.federated_ssl.fedmatch.partitioned_runtime_plan import (
    build_fedmatch_partitioned_runtime_plan,
    normalize_fedmatch_scenario_name,
    resolve_fedmatch_psi_factor,
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


def test_fedmatch_supervised_tensor_loss_routes_to_sigma_partition() -> None:
    parameters = FedMatchLocalObjectiveParameters.from_original_scenario()
    logits = torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)

    result = compute_fedmatch_supervised_loss(
        labeled_logits=logits,
        labels=labels,
        parameters=parameters,
    )

    expected = F.cross_entropy(logits, labels) * parameters.lambda_s
    torch.testing.assert_close(result.total_loss, expected)
    torch.testing.assert_close(result.partition_losses["sigma"], expected)
    torch.testing.assert_close(result.loss_components[FEDMATCH_SUPERVISED_CE], expected)
    torch.testing.assert_close(result.metrics["labeled_count"], torch.tensor(2.0))


def test_fedmatch_unsupervised_tensor_loss_preserves_original_components() -> None:
    parameters = FedMatchLocalObjectiveParameters(
        confidence_threshold=0.75,
        lambda_s=10.0,
        lambda_i=1.0,
        lambda_a=2.0,
        lambda_l2=0.5,
        lambda_l1=0.1,
    )
    weak_probabilities = torch.tensor(
        [[0.8, 0.2], [0.6, 0.4], [0.1, 0.9]],
        dtype=torch.float32,
    )
    weak_logits = torch.log(weak_probabilities)
    strong_logits = torch.tensor(
        [[0.0, 2.0], [2.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    helper_probabilities = torch.tensor(
        [
            [[0.2, 0.8], [0.6, 0.4], [0.3, 0.7]],
            [[0.1, 0.9], [0.4, 0.6], [0.8, 0.2]],
        ],
        dtype=torch.float32,
    )
    partitions = FedMatchParameterPartitions(
        sigma={"layer": torch.tensor([1.0, 3.0])},
        psi={"layer": torch.tensor([0.5, -1.0])},
    )

    result = compute_fedmatch_unsupervised_loss(
        weak_logits=weak_logits,
        selected_strong_logits=strong_logits[[0, 2]],
        selected_helper_weak_probabilities=helper_probabilities[:, [0, 2], :],
        parameter_partitions=partitions,
        parameters=parameters,
    )

    selected_weak = weak_probabilities[[0, 2]]
    selected_helpers = helper_probabilities[:, [0, 2], :]
    expected_kl = F.kl_div(
        torch.log(selected_weak).unsqueeze(0).expand(2, -1, -1).reshape(-1, 2),
        selected_helpers.reshape(-1, 2),
        reduction="batchmean",
    )
    expected_agreement = (
        F.cross_entropy(
            strong_logits[[0, 2]],
            torch.tensor([1, 1], dtype=torch.long),
        )
        * 2.0
    )
    expected_l1 = torch.tensor((abs(0.5) + abs(-1.0)) * 0.1)
    expected_l2 = torch.tensor(((1.0 - 0.5) ** 2 + (3.0 - -1.0) ** 2) * 0.5)
    expected_total = expected_kl + expected_agreement + expected_l1 + expected_l2

    torch.testing.assert_close(
        result.loss_components[FEDMATCH_INTER_CLIENT_KL],
        expected_kl,
    )
    torch.testing.assert_close(
        result.loss_components[FEDMATCH_AGREEMENT_PSEUDO_LABEL_CE],
        expected_agreement,
    )
    torch.testing.assert_close(
        result.loss_components[FEDMATCH_PSI_L1_REGULARIZATION],
        expected_l1,
    )
    torch.testing.assert_close(
        result.loss_components[FEDMATCH_SIGMA_PSI_L2_REGULARIZATION],
        expected_l2,
    )
    torch.testing.assert_close(result.total_loss, expected_total)
    torch.testing.assert_close(result.partition_losses["psi"], expected_total)
    torch.testing.assert_close(result.metrics["confident_count"], torch.tensor(2.0))
    torch.testing.assert_close(result.metrics["util_ratio"], torch.tensor(2.0 / 3.0))
    torch.testing.assert_close(result.metrics["helper_count"], torch.tensor(2.0))
    torch.testing.assert_close(
        result.debug_tensors["confidence_mask"],
        torch.tensor([True, False, True]),
    )
    torch.testing.assert_close(
        result.debug_tensors["agreement_pseudo_labels"],
        torch.tensor([1, 1]),
    )


def test_fedmatch_unsupervised_tensor_loss_can_disable_round_zero_kl() -> None:
    parameters = FedMatchLocalObjectiveParameters(
        confidence_threshold=0.5,
        lambda_s=10.0,
        lambda_i=1.0,
        lambda_a=1.0,
        lambda_l2=0.0,
        lambda_l1=0.0,
    )

    result = compute_fedmatch_unsupervised_loss(
        weak_logits=torch.log(torch.tensor([[0.8, 0.2]], dtype=torch.float32)),
        selected_strong_logits=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        selected_helper_weak_probabilities=torch.tensor(
            [[[0.1, 0.9]]],
            dtype=torch.float32,
        ),
        parameter_partitions=FedMatchParameterPartitions(sigma={}, psi={}),
        parameters=parameters,
        enable_inter_client_consistency=False,
    )

    torch.testing.assert_close(
        result.loss_components[FEDMATCH_INTER_CLIENT_KL],
        torch.tensor(0.0),
    )


def test_fedmatch_inter_client_kl_keeps_weak_prediction_gradient() -> None:
    parameters = FedMatchLocalObjectiveParameters(
        confidence_threshold=0.5,
        lambda_s=10.0,
        lambda_i=1.0,
        lambda_a=0.0,
        lambda_l2=0.0,
        lambda_l1=0.0,
    )
    weak_logits = torch.log(torch.tensor([[0.8, 0.2]], dtype=torch.float32))
    weak_logits.requires_grad_(True)

    result = compute_fedmatch_unsupervised_loss(
        weak_logits=weak_logits,
        selected_strong_logits=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        selected_helper_weak_probabilities=torch.tensor(
            [[[0.1, 0.9]]],
            dtype=torch.float32,
        ),
        parameter_partitions=FedMatchParameterPartitions(sigma={}, psi={}),
        parameters=parameters,
    )
    result.total_loss.backward()

    assert weak_logits.grad is not None
    assert torch.count_nonzero(weak_logits.grad).item() > 0


def test_fedmatch_unsupervised_loss_keeps_regularization_without_confidence() -> None:
    parameters = FedMatchLocalObjectiveParameters(
        confidence_threshold=0.99,
        lambda_s=10.0,
        lambda_i=1.0,
        lambda_a=1.0,
        lambda_l2=1.0,
        lambda_l1=1.0,
    )

    result = compute_fedmatch_unsupervised_loss(
        weak_logits=torch.log(torch.tensor([[0.6, 0.4]], dtype=torch.float32)),
        selected_strong_logits=torch.empty((0, 2), dtype=torch.float32),
        parameter_partitions=FedMatchParameterPartitions(
            sigma={"layer": torch.tensor([2.0])},
            psi={"layer": torch.tensor([-1.0])},
        ),
        parameters=parameters,
    )

    torch.testing.assert_close(
        result.loss_components[FEDMATCH_AGREEMENT_PSEUDO_LABEL_CE],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        result.loss_components[FEDMATCH_PSI_L1_REGULARIZATION],
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        result.loss_components[FEDMATCH_SIGMA_PSI_L2_REGULARIZATION],
        torch.tensor(9.0),
    )
    torch.testing.assert_close(result.metrics["confident_count"], torch.tensor(0.0))


def test_fedmatch_unsupervised_loss_requires_selected_rows_to_match_confidence() -> (
    None
):
    parameters = FedMatchLocalObjectiveParameters(
        confidence_threshold=0.75,
        lambda_s=10.0,
        lambda_i=1.0,
        lambda_a=1.0,
        lambda_l2=0.0,
        lambda_l1=0.0,
    )

    with pytest.raises(ValueError, match="confident_count"):
        compute_fedmatch_unsupervised_loss(
            weak_logits=torch.log(
                torch.tensor([[0.8, 0.2], [0.4, 0.6]], dtype=torch.float32)
            ),
            selected_strong_logits=torch.tensor(
                [[0.0, 1.0], [1.0, 0.0]],
                dtype=torch.float32,
            ),
            parameter_partitions=FedMatchParameterPartitions(sigma={}, psi={}),
            parameters=parameters,
        )


def test_fedmatch_parameter_partitions_reject_shape_drift() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        FedMatchParameterPartitions(
            sigma={"layer": torch.tensor([1.0, 2.0])},
            psi={"layer": torch.tensor([1.0])},
        )


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


def test_fedmatch_helper_refresh_and_nearest_selection_match_original() -> None:
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


def test_fedmatch_helper_index_prefers_kdtree_when_scipy_is_available() -> None:
    index = build_helper_index(
        client_vectors={
            "client_a": (0.0, 0.0),
            "client_b": (0.2, 0.0),
            "client_c": (3.0, 0.0),
            "client_d": (0.1, 0.0),
        },
    )

    try:
        import scipy.spatial  # noqa: F401
    except ImportError:
        expected_backend = "full_scan"
    else:
        expected_backend = "scipy_kdtree"
    assert index.backend_name == expected_backend
    assert index.query_size_including_self(peer_count=2) == 3
    assert index.query(client_id="client_a", peer_count=2) == (
        "client_d",
        "client_b",
    )


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


def test_fedmatch_partitioned_runtime_plan_owns_sigma_psi_routing() -> None:
    parameters = fedmatch_original_parameter_mapping()

    plan = build_fedmatch_partitioned_runtime_plan(
        scenario_name=FEDMATCH_SCENARIO_LABELS_AT_SERVER,
        effective_parameters=parameters,
    )

    assert plan.scenario_name == FEDMATCH_SCENARIO_LABELS_AT_SERVER
    assert plan.partition_names == (FEDMATCH_SIGMA_PARTITION, FEDMATCH_PSI_PARTITION)
    assert plan.supervised_partition == FEDMATCH_SIGMA_PARTITION
    assert plan.unsupervised_partition == FEDMATCH_PSI_PARTITION
    assert plan.upload_partitions == (FEDMATCH_PSI_PARTITION,)
    assert not plan.emit_supervised_partition
    assert plan.l1_sparse_partitions == (FEDMATCH_PSI_PARTITION,)
    assert plan.psi_factor == pytest.approx(0.2)
    assert plan.parameters.confidence_threshold == pytest.approx(0.75)
    assert plan.physical_objective.parameters == plan.parameters
    assert plan.sequential_objective.omit_regularization_for_single_trainable_model
    assert not plan.local_supervision_regime.uses_client_labeled_rows


def test_fedmatch_partitioned_runtime_plan_normalizes_scenario_and_psi_factor() -> None:
    parameters = {
        **fedmatch_original_parameter_mapping(),
        "psi_factor": 0.35,
    }

    plan = build_fedmatch_partitioned_runtime_plan(
        scenario_name="labels_at_client",
        effective_parameters=parameters,
    )

    assert normalize_fedmatch_scenario_name(None) == FEDMATCH_SCENARIO_LABELS_AT_CLIENT
    assert plan.scenario_name == FEDMATCH_SCENARIO_LABELS_AT_CLIENT
    assert plan.emit_supervised_partition
    assert plan.local_supervision_regime.uses_client_labeled_rows
    assert resolve_fedmatch_psi_factor(parameters) == pytest.approx(0.35)


def test_fedmatch_partitioned_runtime_plan_rejects_invalid_psi_factor() -> None:
    parameters = {
        **fedmatch_original_parameter_mapping(),
        "psi_factor": -0.1,
    }

    with pytest.raises(ValueError, match="psi_factor"):
        build_fedmatch_partitioned_runtime_plan(
            scenario_name=FEDMATCH_SCENARIO_LABELS_AT_CLIENT,
            effective_parameters=parameters,
        )


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
