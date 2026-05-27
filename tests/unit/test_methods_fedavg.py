"""Reusable FedAvg method core tests."""

from __future__ import annotations

import pytest

from methods.adaptation.peft_text_classifier.aggregation import (
    peft_encoder_fedavg_projection as peft_fedavg_projection,
)
from methods.classification.linear_head.aggregation import (
    linear_head_fedavg_projection as feature_head_projection,
)
from methods.federated.aggregation import registry as aggregation_registry
from methods.federated.aggregation.fedavg.update_metrics import (
    FedAvgObservationMetricUpdate,
    aggregate_update_observation_metrics,
)
from methods.federated.aggregation.fedavg.weighted_average import (
    WeightedScalarUpdate,
    WeightedVectorMappingUpdate,
    WeightedVectorUpdate,
    weighted_average_scalars,
    weighted_average_vector_mappings,
    weighted_average_vectors,
)
from methods.federated.aggregation.registry import (
    get_federated_aggregation_method_spec,
    register_federated_aggregation_strategy,
)


def test_weighted_average_scalars_uses_fedavg_weights() -> None:
    result = weighted_average_scalars(
        [
            WeightedScalarUpdate(value=2.0, weight=1.0),
            WeightedScalarUpdate(value=8.0, weight=3.0),
        ]
    )

    assert result == pytest.approx(6.5)


def test_weighted_average_vectors_uses_fedavg_weights_without_mutating_input() -> None:
    first_values = [1.0, 3.0]
    second_values = [5.0, 7.0]

    result = weighted_average_vectors(
        [
            WeightedVectorUpdate(values=first_values, weight=1.0),
            WeightedVectorUpdate(values=second_values, weight=3.0),
        ]
    )

    assert result == pytest.approx([4.0, 6.0])
    assert first_values == [1.0, 3.0]
    assert second_values == [5.0, 7.0]


def test_weighted_average_rejects_zero_total_weight() -> None:
    with pytest.raises(ValueError, match="total update weight"):
        weighted_average_scalars(
            [
                WeightedScalarUpdate(value=1.0, weight=0.0),
                WeightedScalarUpdate(value=2.0, weight=0.0),
            ]
        )


def test_weighted_average_vector_mappings_rejects_key_mismatch() -> None:
    with pytest.raises(ValueError, match="same keys"):
        weighted_average_vector_mappings(
            [
                WeightedVectorMappingUpdate(values={"a": [1.0]}, weight=1.0),
                WeightedVectorMappingUpdate(values={"b": [2.0]}, weight=1.0),
            ]
        )


def test_aggregate_update_observation_metrics_uses_example_weights() -> None:
    result = aggregate_update_observation_metrics(
        [
            FedAvgObservationMetricUpdate(
                example_count=2,
                mean_confidence=0.9,
                mean_margin=0.3,
                delta_l2_norm=0.5,
            ),
            FedAvgObservationMetricUpdate(
                example_count=1,
                mean_confidence=None,
                mean_margin=None,
                delta_l2_norm=0.2,
            ),
        ]
    )

    assert result["client_count"] == 2.0
    assert result["example_count"] == 3.0
    assert result["mean_confidence"] == pytest.approx(0.9)
    assert result["mean_confidence_observed_count"] == 1.0
    assert result["mean_confidence_missing_count"] == 1.0
    assert result["mean_margin"] == pytest.approx(0.3)
    assert result["mean_margin_observed_count"] == 1.0
    assert result["mean_margin_missing_count"] == 1.0
    assert result["mean_delta_l2_norm"] == pytest.approx(0.4)


def test_federated_aggregation_registry_resolves_alias_without_package_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_package_scan() -> None:
        raise AssertionError("package-wide adaptation aggregation scan must not run")

    monkeypatch.setattr(
        aggregation_registry,
        "_import_adaptation_aggregation_modules",
        _fail_package_scan,
    )

    spec = get_federated_aggregation_method_spec(
        adapter_kind="classifier_head",
        method_name="classifier_head_fedavg",
    )

    assert spec.method_name == "fedavg"
    assert spec.core_function_name == "compute_classifier_head_fedavg"


def test_federated_aggregation_registry_rejects_duplicate_strategy() -> None:
    get_federated_aggregation_method_spec(
        adapter_kind="classifier_head",
        method_name="fedavg",
    )

    with pytest.raises(ValueError, match="Duplicate federated aggregation strategy"):
        register_federated_aggregation_strategy(
            adapter_kind="classifier_head",
            method_name="fedavg",
            implementation_module="tests.duplicate",
            core_function_name="duplicate",
            factory=lambda _overrides: None,
        )


def test_classifier_head_fedavg_updates_values_without_mutation() -> None:
    base_label_weights = {
        "anxiety": [1.0, 0.0],
        "normal": [0.0, 1.0],
    }
    base_label_biases = {"anxiety": 0.1}
    first_weight_deltas = {
        "anxiety": [0.2, -0.1],
        "normal": [-0.2, 0.1],
    }
    second_weight_deltas = {
        "anxiety": [0.1, 0.2],
        "normal": [-0.1, -0.2],
    }

    result = feature_head_projection.compute_classifier_head_fedavg(
        base_label_weights=base_label_weights,
        base_label_biases=base_label_biases,
        updates=[
            feature_head_projection.ClassifierHeadFedAvgUpdate(
                label_weight_deltas=first_weight_deltas,
                label_bias_deltas={"anxiety": 0.05, "normal": -0.05},
                example_count=2,
                mean_confidence=0.9,
                mean_margin=0.3,
                delta_l2_norm=0.5,
            ),
            feature_head_projection.ClassifierHeadFedAvgUpdate(
                label_weight_deltas=second_weight_deltas,
                label_bias_deltas={"anxiety": 0.03},
                example_count=1,
                mean_confidence=0.8,
                mean_margin=0.2,
                delta_l2_norm=0.2,
            ),
        ],
    )

    assert result.label_weights["anxiety"] == pytest.approx([1.1666666666666667, 0.0])
    assert result.label_weights["normal"] == pytest.approx([-0.16666666666666669, 1.0])
    assert result.label_biases["anxiety"] == pytest.approx(0.14333333333333334)
    assert result.label_biases["normal"] == pytest.approx(-0.03333333333333333)
    assert result.aggregated_metrics["mean_confidence"] == pytest.approx(
        0.8666666666666667
    )
    assert result.update_count == 2
    assert base_label_weights == {
        "anxiety": [1.0, 0.0],
        "normal": [0.0, 1.0],
    }
    assert base_label_biases == {"anxiety": 0.1}
    assert first_weight_deltas["anxiety"] == [0.2, -0.1]
    assert second_weight_deltas["normal"] == [-0.1, -0.2]


def test_peft_encoder_fedavg_averages_adapter_and_head_deltas() -> None:
    result = peft_fedavg_projection.compute_peft_encoder_fedavg(
        label_schema=("anxiety", "normal"),
        updates=[
            peft_fedavg_projection.PeftEncoderFedAvgUpdate(
                lora_parameter_deltas={
                    "encoder.q_proj.lora_A": [0.2, 0.4],
                    "encoder.q_proj.lora_B": [0.1, -0.1],
                },
                classifier_head_weight_deltas={
                    "anxiety": [0.2, -0.1],
                    "normal": [-0.2, 0.1],
                },
                classifier_head_bias_deltas={"anxiety": 0.05, "normal": -0.05},
                example_count=2,
                mean_confidence=0.9,
                mean_margin=0.3,
                delta_l2_norm=0.5,
            ),
            peft_fedavg_projection.PeftEncoderFedAvgUpdate(
                lora_parameter_deltas={
                    "encoder.q_proj.lora_A": [0.0, 0.1],
                    "encoder.q_proj.lora_B": [0.2, 0.2],
                },
                classifier_head_weight_deltas={
                    "anxiety": [0.1, 0.2],
                    "normal": [-0.1, -0.2],
                },
                classifier_head_bias_deltas={"anxiety": 0.02},
                example_count=1,
                mean_confidence=0.6,
                mean_margin=None,
                delta_l2_norm=0.2,
            ),
        ],
    )

    assert result.lora_parameter_deltas["encoder.q_proj.lora_A"] == pytest.approx(
        [0.13333333333333333, 0.3]
    )
    assert result.lora_parameter_deltas["encoder.q_proj.lora_B"] == pytest.approx(
        [0.13333333333333333, 0.0]
    )
    assert result.classifier_head_weight_deltas["anxiety"] == pytest.approx(
        [0.16666666666666666, 0.0]
    )
    assert result.classifier_head_weight_deltas["normal"] == pytest.approx(
        [-0.16666666666666666, 0.0]
    )
    assert result.classifier_head_bias_deltas["anxiety"] == pytest.approx(0.04)
    assert result.classifier_head_bias_deltas["normal"] == pytest.approx(
        -0.03333333333333333
    )
    assert result.aggregated_metrics["example_count"] == 3.0
    assert result.aggregated_metrics["mean_confidence"] == pytest.approx(0.8)
    assert result.aggregated_metrics["mean_delta_l2_norm"] == pytest.approx(0.4)
    assert result.update_count == 2


def test_federated_aggregation_method_registry_points_to_lora_core() -> None:
    spec = get_federated_aggregation_method_spec(
        adapter_kind="lora_classifier",
        method_name="fedavg",
    )

    assert spec.method_name == "fedavg"
    assert (
        spec.implementation_module
        == "methods.adaptation.peft_text_classifier.aggregation."
        "peft_encoder_fedavg_projection"
    )
    assert spec.core_function_name == "compute_peft_encoder_fedavg"


def test_federated_aggregation_method_registry_points_to_peft_classifier_core() -> None:
    spec = get_federated_aggregation_method_spec(
        adapter_kind="peft_classifier",
        method_name="fedavg",
    )

    assert spec.method_name == "fedavg"
    assert (
        spec.implementation_module
        == "methods.adaptation.peft_text_classifier.aggregation."
        "peft_encoder_fedavg_projection"
    )
    assert spec.core_function_name == "compute_peft_encoder_fedavg"


def test_federated_aggregation_method_registry_points_to_partitioned_lora_core() -> (
    None
):
    spec = get_federated_aggregation_method_spec(
        adapter_kind="lora_classifier",
        method_name="partitioned_delta_average",
    )

    assert spec.method_name == "partitioned_delta_average"
    assert (
        spec.implementation_module
        == "methods.adaptation.peft_text_classifier.aggregation."
        "peft_encoder_partitioned_projection"
    )
    assert spec.core_function_name == "compute_peft_encoder_partitioned_delta_average"


def test_federated_aggregation_method_registry_points_to_partitioned_peft_core() -> (
    None
):
    spec = get_federated_aggregation_method_spec(
        adapter_kind="peft_classifier",
        method_name="partitioned_delta_average",
    )

    assert spec.method_name == "partitioned_delta_average"
    assert (
        spec.implementation_module
        == "methods.adaptation.peft_text_classifier.aggregation."
        "peft_encoder_partitioned_projection"
    )
    assert spec.core_function_name == "compute_peft_encoder_partitioned_delta_average"
