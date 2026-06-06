"""Classifier-head family용 FedAvg 계산 core."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from methods.federated.aggregation.base import (
    AggregationConfigScalar,
    FederatedAggregationContext,
    FederatedAggregationResult,
)
from methods.federated.aggregation.fedavg.strategy import (
    FedAvgAdapterStrategySpec,
    register_fedavg_adapter_strategy,
)
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
from methods.federated.aggregation_weighting import (
    AGGREGATION_WEIGHT_UNIFORM,
    AggregationWeightPolicy,
    aggregation_weight_for_update,
)
from shared.src.contracts.adapter_contract_families.classifier_head import (
    CLASSIFIER_HEAD_ADAPTER_KIND,
    ClassifierHeadDelta,
    ClassifierHeadState,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)


@dataclass(frozen=True, slots=True)
class ClassifierHeadFedAvgUpdate:
    """main_server boundary와 분리된 classifier-head FedAvg 입력."""

    label_weight_deltas: Mapping[str, Sequence[float]]
    label_bias_deltas: Mapping[str, float]
    example_count: int
    mean_confidence: float | None
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
    weight_policy_name: str = AGGREGATION_WEIGHT_UNIFORM,
) -> ClassifierHeadFedAvgResult:
    """label별 weight/bias delta를 policy weight로 평균해 다음 head를 계산한다."""

    normalized_base_weights = _normalize_base_label_weights(base_label_weights)
    labels = tuple(sorted(normalized_base_weights))
    weight_policy = AggregationWeightPolicy(name=weight_policy_name)
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
                weight=aggregation_weight_for_update(update, policy=weight_policy),
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
                weight=aggregation_weight_for_update(update, policy=weight_policy),
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


def aggregate_classifier_head_fedavg(
    base_state: SharedAdapterState,
    update_payloads: Sequence[SharedAdapterUpdate],
    context: FederatedAggregationContext,
    overrides: Mapping[str, AggregationConfigScalar] | None,
) -> FederatedAggregationResult:
    """Classifier-head update payload를 FedAvg core 입력으로 변환한다."""

    base_state = cast(ClassifierHeadState, base_state)
    updates = [cast(ClassifierHeadDelta, payload) for payload in update_payloads]
    labels = base_state.labels
    embedding_dim = base_state.embedding_dim
    method_updates: list[ClassifierHeadFedAvgUpdate] = []

    for payload in updates:
        if payload.labels != labels:
            raise ValueError(
                "Classifier head updates must share the same ordered labels."
            )
        if payload.embedding_dim != embedding_dim:
            raise ValueError("All update payloads must share the same embedding_dim.")
        method_updates.append(
            ClassifierHeadFedAvgUpdate(
                label_weight_deltas=payload.label_weight_deltas,
                label_bias_deltas=payload.label_bias_deltas,
                example_count=payload.example_count,
                mean_confidence=payload.mean_confidence,
                mean_margin=payload.mean_margin,
                delta_l2_norm=payload.l2_norm(),
            )
        )

    method_result = compute_classifier_head_fedavg(
        base_label_weights=base_state.label_weights,
        base_label_biases=base_state.label_biases,
        updates=method_updates,
        weight_policy_name=str(
            (overrides or {}).get("weight_policy", AGGREGATION_WEIGHT_UNIFORM)
        ),
    )

    next_state = ClassifierHeadState(
        schema_version=base_state.schema_version,
        adapter_kind=base_state.adapter_kind,
        model_id=base_state.model_id,
        model_revision=context.next_model_revision,
        training_scope=base_state.training_scope,
        updated_at=context.aggregated_at,
        label_weights=method_result.label_weights,
        label_biases=method_result.label_biases,
    )
    return FederatedAggregationResult(
        next_state=next_state,
        aggregated_metrics=method_result.aggregated_metrics,
        update_count=method_result.update_count,
    )


register_fedavg_adapter_strategy(
    FedAvgAdapterStrategySpec(
        adapter_kind=CLASSIFIER_HEAD_ADAPTER_KIND,
        state_type=ClassifierHeadState,
        update_type=ClassifierHeadDelta,
        context="classifier-head",
        aliases=("classifier_head_fedavg",),
        implementation_module=compute_classifier_head_fedavg.__module__,
        core_function_name=compute_classifier_head_fedavg.__name__,
        metadata={"adapter_kind": CLASSIFIER_HEAD_ADAPTER_KIND},
        aggregate=aggregate_classifier_head_fedavg,
    )
)
