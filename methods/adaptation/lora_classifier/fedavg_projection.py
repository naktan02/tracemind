"""LoRA-classifier payload projection for FedAvg aggregation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from methods.adaptation.lora_classifier.fedavg import (
    LoraClassifierFedAvgUpdate,
    compute_lora_classifier_fedavg,
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
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LORA_CLASSIFIER_ADAPTER_KIND,
)
from shared.src.contracts.adapter_contracts import (
    LoraClassifierDelta,
    LoraClassifierState,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)

LORA_ADAPTER_ARTIFACT_SLOT = "lora_adapter"
CLASSIFIER_HEAD_ARTIFACT_SLOT = "classifier_head"


def aggregate_lora_classifier_fedavg(
    base_state: SharedAdapterState,
    update_payloads: Sequence[SharedAdapterUpdate],
    context: FederatedAggregationContext,
    overrides: Mapping[str, AggregationConfigScalar] | None,
) -> FederatedAggregationResult:
    """LoRA-classifier update payload를 FedAvg core 입력으로 변환한다."""

    _validate_lora_classifier_fedavg_overrides(overrides)

    base_state = cast(LoraClassifierState, base_state)
    updates = [cast(LoraClassifierDelta, payload) for payload in update_payloads]
    method_updates = [
        _to_lora_classifier_method_update(base_state=base_state, payload=payload)
        for payload in updates
    ]
    method_result = compute_lora_classifier_fedavg(
        label_schema=base_state.label_schema,
        updates=method_updates,
    )
    artifact_ref_resolver = context.require_artifact_ref_resolver(
        context="LoRA-classifier FedAvg"
    )
    next_state = LoraClassifierState(
        schema_version=base_state.schema_version,
        adapter_kind=base_state.adapter_kind,
        model_id=base_state.model_id,
        model_revision=context.next_model_revision,
        training_scope=base_state.training_scope,
        updated_at=context.aggregated_at,
        backbone=base_state.backbone,
        lora_config=base_state.lora_config,
        label_schema=base_state.label_schema,
        lora_adapter_artifact_ref=artifact_ref_resolver.build_ref(
            next_model_revision=context.next_model_revision,
            artifact_name=LORA_ADAPTER_ARTIFACT_SLOT,
        ),
        classifier_head_artifact_ref=artifact_ref_resolver.build_ref(
            next_model_revision=context.next_model_revision,
            artifact_name=CLASSIFIER_HEAD_ARTIFACT_SLOT,
        ),
        artifact_format=artifact_ref_resolver.artifact_format,
    )
    return FederatedAggregationResult(
        next_state=next_state,
        aggregated_metrics=method_result.aggregated_metrics,
        update_count=method_result.update_count,
    )


def _to_lora_classifier_method_update(
    *,
    base_state: LoraClassifierState,
    payload: LoraClassifierDelta,
) -> LoraClassifierFedAvgUpdate:
    if _payload_snapshot(payload.backbone) != _payload_snapshot(base_state.backbone):
        raise ValueError("All LoRA-classifier updates must match the backbone.")
    if _payload_snapshot(payload.lora_config) != _payload_snapshot(
        base_state.lora_config
    ):
        raise ValueError("All LoRA-classifier updates must match the LoRA config.")
    if payload.labels != base_state.labels:
        raise ValueError(
            "LoRA-classifier updates must share the base ordered label_schema."
        )
    if (
        payload.lora_parameter_deltas is None
        or payload.classifier_head_weight_deltas is None
    ):
        raise ValueError(
            "LoRA-classifier FedAvg currently requires inline "
            "lora_parameter_deltas and classifier_head_weight_deltas. "
            "Artifact-ref-only updates require a server artifact materializer."
        )
    return LoraClassifierFedAvgUpdate(
        lora_parameter_deltas=payload.lora_parameter_deltas,
        classifier_head_weight_deltas=payload.classifier_head_weight_deltas,
        classifier_head_bias_deltas=payload.classifier_head_bias_deltas,
        example_count=payload.example_count,
        mean_confidence=payload.mean_confidence,
        mean_margin=payload.mean_margin,
        delta_l2_norm=payload.l2_norm(),
    )


def _payload_snapshot(payload) -> dict[str, object]:
    return payload.model_dump(mode="json")


def _validate_lora_classifier_fedavg_overrides(
    overrides: Mapping[str, AggregationConfigScalar] | None,
) -> None:
    if overrides is None:
        return
    unknown_keys = sorted(set(overrides) - {"artifact_ref_prefix", "artifact_format"})
    if unknown_keys:
        raise ValueError(
            "Unsupported LoRA-classifier aggregate artifact config key(s): "
            f"{unknown_keys}."
        )


register_fedavg_adapter_strategy(
    FedAvgAdapterStrategySpec(
        adapter_kind=LORA_CLASSIFIER_ADAPTER_KIND,
        state_type=LoraClassifierState,
        update_type=LoraClassifierDelta,
        context="LoRA-classifier",
        aliases=("lora_classifier_fedavg",),
        implementation_module=compute_lora_classifier_fedavg.__module__,
        core_function_name=compute_lora_classifier_fedavg.__name__,
        metadata={
            "adapter_kind": LORA_CLASSIFIER_ADAPTER_KIND,
            "requires_inline_or_materialized_artifacts": True,
        },
        aggregate=aggregate_lora_classifier_fedavg,
    )
)
