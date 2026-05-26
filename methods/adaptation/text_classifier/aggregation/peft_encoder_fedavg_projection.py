"""LoRA-classifier family용 FedAvg 계산 core."""

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
    AGGREGATION_WEIGHT_EXAMPLE_COUNT,
    AggregationWeightPolicy,
    aggregation_weight_for_update,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LORA_CLASSIFIER_ADAPTER_KIND,
    LoraClassifierDelta,
    LoraClassifierState,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)

from ..peft_encoder.update.materialization import (
    materialize_base_lora_classifier_state,
    materialize_lora_classifier_update,
)
from .peft_encoder_state_projection import build_lora_classifier_state_projection

LORA_ADAPTER_ARTIFACT_SLOT = "lora_adapter"
CLASSIFIER_HEAD_ARTIFACT_SLOT = "classifier_head"


@dataclass(frozen=True, slots=True)
class LoraClassifierFedAvgUpdate:
    """main_server boundary와 분리된 LoRA-classifier FedAvg 입력."""

    lora_parameter_deltas: Mapping[str, Sequence[float]]
    classifier_head_weight_deltas: Mapping[str, Sequence[float]]
    classifier_head_bias_deltas: Mapping[str, float]
    example_count: int
    mean_confidence: float | None
    mean_margin: float | None
    delta_l2_norm: float


@dataclass(frozen=True, slots=True)
class LoraClassifierFedAvgResult:
    """LoRA-classifier FedAvg 계산 결과."""

    lora_parameter_deltas: dict[str, list[float]]
    classifier_head_weight_deltas: dict[str, list[float]]
    classifier_head_bias_deltas: dict[str, float]
    aggregated_metrics: dict[str, float]
    update_count: int


def compute_lora_classifier_fedavg(
    *,
    label_schema: Sequence[str],
    updates: Sequence[LoraClassifierFedAvgUpdate],
    weight_policy_name: str = AGGREGATION_WEIGHT_EXAMPLE_COUNT,
) -> LoraClassifierFedAvgResult:
    """LoRA parameter delta와 classifier-head delta를 policy weight로 평균한다."""

    labels = _normalize_label_schema(label_schema)
    weight_policy = AggregationWeightPolicy(name=weight_policy_name)
    valid_updates = tuple(update for update in updates if update.example_count > 0)
    if not valid_updates:
        raise ValueError("At least one non-empty LoRA-classifier update is required.")

    lora_parameter_deltas = weighted_average_vector_mappings(
        [
            WeightedVectorMappingUpdate(
                values=update.lora_parameter_deltas,
                weight=aggregation_weight_for_update(update, policy=weight_policy),
            )
            for update in valid_updates
        ]
    )
    classifier_head_weight_deltas = weighted_average_vector_mappings(
        [
            WeightedVectorMappingUpdate(
                values=_normalize_classifier_head_weight_deltas(
                    update,
                    labels=labels,
                ),
                weight=aggregation_weight_for_update(update, policy=weight_policy),
            )
            for update in valid_updates
        ]
    )
    classifier_head_bias_deltas = weighted_average_scalar_mappings(
        [
            WeightedScalarMappingUpdate(
                values=_normalize_classifier_head_bias_deltas(
                    update,
                    labels=labels,
                ),
                weight=aggregation_weight_for_update(update, policy=weight_policy),
            )
            for update in valid_updates
        ]
    )

    return LoraClassifierFedAvgResult(
        lora_parameter_deltas=lora_parameter_deltas,
        classifier_head_weight_deltas=classifier_head_weight_deltas,
        classifier_head_bias_deltas=classifier_head_bias_deltas,
        aggregated_metrics={
            **_aggregate_common_metrics(valid_updates),
            "aggregation_weight_policy_example_count": float(
                weight_policy.name == AGGREGATION_WEIGHT_EXAMPLE_COUNT
            ),
            "lora_parameter_count": float(len(lora_parameter_deltas)),
            "classifier_head_label_count": float(len(classifier_head_weight_deltas)),
        },
        update_count=len(valid_updates),
    )


def _normalize_label_schema(label_schema: Sequence[str]) -> tuple[str, ...]:
    labels = tuple(str(label).strip() for label in label_schema if str(label).strip())
    if not labels:
        raise ValueError("label_schema must not be empty.")
    if len(set(labels)) != len(labels):
        raise ValueError("label_schema must not contain duplicates.")
    return labels


def _normalize_classifier_head_weight_deltas(
    update: LoraClassifierFedAvgUpdate,
    *,
    labels: Sequence[str],
) -> dict[str, Sequence[float]]:
    if set(update.classifier_head_weight_deltas) != set(labels):
        raise ValueError(
            "LoRA-classifier FedAvg classifier head weight delta keys must match "
            "label_schema."
        )
    return {
        label: update.classifier_head_weight_deltas[label]
        for label in sorted(update.classifier_head_weight_deltas)
    }


def _normalize_classifier_head_bias_deltas(
    update: LoraClassifierFedAvgUpdate,
    *,
    labels: Sequence[str],
) -> dict[str, float]:
    extra_labels = set(update.classifier_head_bias_deltas) - set(labels)
    if extra_labels:
        raise ValueError(
            "LoRA-classifier FedAvg bias deltas contain unknown labels: "
            f"{sorted(extra_labels)}"
        )
    return {
        label: float(update.classifier_head_bias_deltas.get(label, 0.0))
        for label in labels
    }


def _aggregate_common_metrics(
    updates: Sequence[LoraClassifierFedAvgUpdate],
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
        _to_lora_classifier_method_update(
            base_state=base_state,
            payload=payload,
            context=context,
        )
        for payload in updates
    ]
    method_result = compute_lora_classifier_fedavg(
        label_schema=base_state.label_schema,
        updates=method_updates,
        weight_policy_name=str((overrides or {}).get("weight_policy", "example_count")),
    )
    artifact_ref_resolver = context.require_artifact_ref_resolver(
        context="LoRA-classifier FedAvg"
    )
    lora_adapter_artifact_ref = artifact_ref_resolver.build_ref(
        next_model_revision=context.next_model_revision,
        artifact_name=LORA_ADAPTER_ARTIFACT_SLOT,
    )
    classifier_head_artifact_ref = artifact_ref_resolver.build_ref(
        next_model_revision=context.next_model_revision,
        artifact_name=CLASSIFIER_HEAD_ARTIFACT_SLOT,
    )
    state_projection = build_lora_classifier_state_projection(
        base_state=base_state,
        base_parameters=materialize_base_lora_classifier_state(
            base_state=base_state,
            context=context,
        ),
        next_model_revision=context.next_model_revision,
        updated_at=context.aggregated_at,
        lora_adapter_artifact_ref=lora_adapter_artifact_ref,
        classifier_head_artifact_ref=classifier_head_artifact_ref,
        artifact_format=artifact_ref_resolver.artifact_format,
        lora_parameter_deltas=method_result.lora_parameter_deltas,
        classifier_head_weight_deltas=method_result.classifier_head_weight_deltas,
        classifier_head_bias_deltas=method_result.classifier_head_bias_deltas,
    )
    return FederatedAggregationResult(
        next_state=state_projection.next_state,
        aggregated_metrics=method_result.aggregated_metrics,
        update_count=method_result.update_count,
        aggregated_artifacts=state_projection.artifacts,
    )


def _to_lora_classifier_method_update(
    *,
    base_state: LoraClassifierState,
    payload: LoraClassifierDelta,
    context: FederatedAggregationContext,
) -> LoraClassifierFedAvgUpdate:
    validate_lora_classifier_update_matches_base(
        base_state=base_state,
        payload=payload,
    )
    materialized = materialize_lora_classifier_update(
        payload=payload,
        context=context,
    )
    return LoraClassifierFedAvgUpdate(
        lora_parameter_deltas=materialized.lora_parameter_deltas,
        classifier_head_weight_deltas=materialized.classifier_head_weight_deltas,
        classifier_head_bias_deltas=materialized.classifier_head_bias_deltas,
        example_count=payload.example_count,
        mean_confidence=payload.mean_confidence,
        mean_margin=payload.mean_margin,
        delta_l2_norm=materialized.delta_l2_norm,
    )


def validate_lora_classifier_update_matches_base(
    *,
    base_state: LoraClassifierState,
    payload: LoraClassifierDelta,
) -> None:
    """LoRA-classifier update가 base global state와 같은 lineage인지 검증한다."""

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


def _payload_snapshot(payload) -> dict[str, object]:
    return payload.model_dump(mode="json")


def _validate_lora_classifier_fedavg_overrides(
    overrides: Mapping[str, AggregationConfigScalar] | None,
) -> None:
    if overrides is None:
        return
    unknown_keys = sorted(
        set(overrides) - {"artifact_ref_prefix", "artifact_format", "weight_policy"}
    )
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
