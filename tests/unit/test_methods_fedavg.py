"""Reusable FedAvg method core tests."""

from __future__ import annotations

import pytest

from methods.federated.aggregation.fedavg.classifier_head_fedavg import (
    ClassifierHeadFedAvgUpdate,
    compute_classifier_head_fedavg,
)
from methods.federated.aggregation.fedavg.diagonal_scale_fedavg import (
    DiagonalScaleFedAvgUpdate,
    compute_diagonal_scale_fedavg,
)
from methods.federated.aggregation.fedavg.fedavg import (
    WeightedScalarUpdate,
    WeightedVectorMappingUpdate,
    WeightedVectorUpdate,
    weighted_average_scalars,
    weighted_average_vector_mappings,
    weighted_average_vectors,
)
from methods.federated.aggregation.registry import (
    get_federated_aggregation_method_spec,
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


def test_diagonal_scale_fedavg_clamps_next_scales_and_reports_metrics() -> None:
    result = compute_diagonal_scale_fedavg(
        base_dimension_scales=[1.0, 1.0],
        updates=[
            DiagonalScaleFedAvgUpdate(
                dimension_deltas=[0.3, -0.6],
                example_count=2,
                mean_confidence=0.9,
                mean_margin=0.3,
                delta_l2_norm=0.5,
            ),
            DiagonalScaleFedAvgUpdate(
                dimension_deltas=[0.0, 0.3],
                example_count=1,
                mean_confidence=0.6,
                mean_margin=None,
                delta_l2_norm=0.2,
            ),
        ],
        min_scale=0.8,
        max_scale=1.1,
    )

    assert result.next_dimension_scales == pytest.approx([1.1, 0.8])
    assert result.update_count == 2
    assert result.aggregated_metrics["client_count"] == 2.0
    assert result.aggregated_metrics["example_count"] == 3.0
    assert result.aggregated_metrics["mean_confidence"] == pytest.approx(0.8)
    assert result.aggregated_metrics["mean_margin"] == pytest.approx(0.2)
    assert result.aggregated_metrics["mean_delta_l2_norm"] == pytest.approx(0.4)


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

    result = compute_classifier_head_fedavg(
        base_label_weights=base_label_weights,
        base_label_biases=base_label_biases,
        updates=[
            ClassifierHeadFedAvgUpdate(
                label_weight_deltas=first_weight_deltas,
                label_bias_deltas={"anxiety": 0.05, "normal": -0.05},
                example_count=2,
                mean_confidence=0.9,
                mean_margin=0.3,
                delta_l2_norm=0.5,
            ),
            ClassifierHeadFedAvgUpdate(
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


def test_federated_aggregation_method_registry_points_to_methods_core() -> None:
    spec = get_federated_aggregation_method_spec(
        adapter_kind="diagonal_scale",
        method_name="fedavg",
    )

    assert spec.method_name == "fedavg"
    assert spec.implementation_module.endswith("diagonal_scale_fedavg")
    assert spec.core_function_name == "compute_diagonal_scale_fedavg"
