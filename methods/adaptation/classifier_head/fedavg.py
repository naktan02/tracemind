"""Classifier-head family용 FedAvg 계산 core."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from methods.federated.aggregation.fedavg.update_metrics import (
    FedAvgObservationMetricUpdate,
    aggregate_update_observation_metrics,
)
from methods.federated.aggregation.fedavg.weighted_average import (
    WeightedScalarMappingUpdate,
    WeightedVectorMappingUpdate,
    weighted_average_scalar_mappings,
    weighted_average_vector_mappings,
)


@dataclass(frozen=True, slots=True)
class ClassifierHeadFedAvgUpdate:
    """main_server boundary와 분리된 classifier-head FedAvg 입력."""

    label_weight_deltas: Mapping[str, Sequence[float]]
    label_bias_deltas: Mapping[str, float]
    example_count: int
    mean_confidence: float
    mean_margin: float | None
    delta_l2_norm: float


@dataclass(frozen=True, slots=True)
class ClassifierHeadFedAvgResult:
    """classifier-head FedAvg 계산 결과."""

    label_weights: dict[str, list[float]]
    label_biases: dict[str, float]
    aggregated_metrics: dict[str, float]
    update_count: int


def compute_classifier_head_fedavg(
    *,
    base_label_weights: Mapping[str, Sequence[float]],
    base_label_biases: Mapping[str, float],
    updates: Sequence[ClassifierHeadFedAvgUpdate],
) -> ClassifierHeadFedAvgResult:
    """label별 weight/bias delta를 example_count로 평균해 다음 head를 계산한다."""

    normalized_base_weights = _normalize_base_label_weights(base_label_weights)
    labels = tuple(sorted(normalized_base_weights))
    normalized_base_biases = _normalize_base_label_biases(
        base_label_biases=base_label_biases,
        labels=labels,
    )
    valid_updates = tuple(update for update in updates if update.example_count > 0)

    for update in valid_updates:
        if set(update.label_weight_deltas) != set(labels):
            raise ValueError(
                "Classifier-head FedAvg updates must share the base label keys."
            )
    weighted_weight_deltas = weighted_average_vector_mappings(
        [
            WeightedVectorMappingUpdate(
                values=update.label_weight_deltas,
                weight=float(update.example_count),
            )
            for update in valid_updates
        ]
    )
    for label in labels:
        if len(weighted_weight_deltas[label]) != len(normalized_base_weights[label]):
            raise ValueError(
                "Classifier-head FedAvg delta dimensions must match base weights."
            )

    weighted_bias_deltas = weighted_average_scalar_mappings(
        [
            WeightedScalarMappingUpdate(
                values=_normalize_bias_deltas(update, labels=labels),
                weight=float(update.example_count),
            )
            for update in valid_updates
        ]
    )

    return ClassifierHeadFedAvgResult(
        label_weights={
            label: [
                base_value + delta
                for base_value, delta in zip(
                    normalized_base_weights[label],
                    weighted_weight_deltas[label],
                    strict=True,
                )
            ]
            for label in labels
        },
        label_biases={
            label: normalized_base_biases[label] + weighted_bias_deltas[label]
            for label in labels
        },
        aggregated_metrics=_aggregate_common_metrics(valid_updates),
        update_count=len(valid_updates),
    )


def _normalize_base_label_weights(
    base_label_weights: Mapping[str, Sequence[float]],
) -> dict[str, list[float]]:
    if not base_label_weights:
        raise ValueError("base_label_weights must not be empty.")
    normalized = {
        str(label): [float(value) for value in weights]
        for label, weights in base_label_weights.items()
    }
    dims = {len(weights) for weights in normalized.values()}
    if dims == {0}:
        raise ValueError("base_label_weights vectors must not be empty.")
    if len(dims) != 1:
        raise ValueError("base_label_weights vectors must share one dimension.")
    return {label: normalized[label] for label in sorted(normalized)}


def _normalize_base_label_biases(
    *,
    base_label_biases: Mapping[str, float],
    labels: Sequence[str],
) -> dict[str, float]:
    extra_labels = set(base_label_biases) - set(labels)
    if extra_labels:
        raise ValueError(
            "base_label_biases contains labels missing from base_label_weights: "
            f"{sorted(extra_labels)}"
        )
    return {label: float(base_label_biases.get(label, 0.0)) for label in labels}


def _normalize_bias_deltas(
    update: ClassifierHeadFedAvgUpdate,
    *,
    labels: Sequence[str],
) -> dict[str, float]:
    extra_labels = set(update.label_bias_deltas) - set(labels)
    if extra_labels:
        raise ValueError(
            "Classifier-head FedAvg bias deltas contain unknown labels: "
            f"{sorted(extra_labels)}"
        )
    return {label: float(update.label_bias_deltas.get(label, 0.0)) for label in labels}


def _aggregate_common_metrics(
    updates: Sequence[ClassifierHeadFedAvgUpdate],
) -> dict[str, float]:
    return aggregate_update_observation_metrics(
        [
            FedAvgObservationMetricUpdate(
                example_count=update.example_count,
                mean_confidence=update.mean_confidence,
                mean_margin=update.mean_margin,
                delta_l2_norm=update.delta_l2_norm,
            )
            for update in updates
        ]
    )
