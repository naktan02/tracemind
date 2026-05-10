"""Classifier-head payload projection for FedAvg aggregation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from methods.adaptation.classifier_head.fedavg import (
    ClassifierHeadFedAvgUpdate,
    compute_classifier_head_fedavg,
)
from methods.federated.aggregation.base import (
    AggregationConfigScalar,
    FederatedAggregationContext,
    FederatedAggregationResult,
)
from methods.federated.aggregation.fedavg.strategy import (
    FedAvgAdapterStrategySpec,
    register_fedavg_adapter_strategy,
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


def aggregate_classifier_head_fedavg(
    base_state: SharedAdapterState,
    update_payloads: Sequence[SharedAdapterUpdate],
    context: FederatedAggregationContext,
    overrides: Mapping[str, AggregationConfigScalar] | None,
) -> FederatedAggregationResult:
    """Classifier-head update payload를 FedAvg core 입력으로 변환한다."""

    del overrides

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
