"""LoRA-classifier partitioned delta average backend."""

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
from methods.federated.aggregation_weighting import (
    AGGREGATION_WEIGHT_EXAMPLE_COUNT,
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
    materialize_base_lora_classifier_partitioned_state,
    materialize_base_lora_classifier_state,
    materialize_lora_classifier_partitioned_update,
)
from ..peft_encoder.update.partitioned_delta import LoraClassifierPartitionDelta
from .peft_encoder_fedavg_projection import (
    CLASSIFIER_HEAD_ARTIFACT_SLOT,
    LORA_ADAPTER_ARTIFACT_SLOT,
    LoraClassifierFedAvgResult,
    LoraClassifierFedAvgUpdate,
    compute_lora_classifier_fedavg,
    validate_lora_classifier_update_matches_base,
)
from .peft_encoder_partitioned_state import (
    apply_lora_classifier_partition_deltas_to_partitioned_state,
    merge_partitioned_lora_classifier_deltas,
)
from .peft_encoder_state_projection import build_lora_classifier_state_projection

PARTITIONED_DELTA_AVERAGE_BACKEND_NAME = "partitioned_delta_average"


@dataclass(frozen=True, slots=True)
class LoraClassifierPartitionedDeltaAverageUpdate:
    """partitioned server update policy가 소비하는 client delta."""

    partitions: Mapping[str, LoraClassifierPartitionDelta]
    example_count: int
    mean_confidence: float | None
    mean_margin: float | None
    delta_l2_norm: float


def compute_lora_classifier_partitioned_delta_average(
    *,
    label_schema: Sequence[str],
    updates: Sequence[LoraClassifierPartitionedDeltaAverageUpdate],
    weight_policy_name: str = AGGREGATION_WEIGHT_EXAMPLE_COUNT,
) -> LoraClassifierFedAvgResult:
    """client별 partition을 병합한 뒤 LoRA/head delta를 policy weight로 평균한다."""

    valid_updates = tuple(update for update in updates if update.example_count > 0)
    if not valid_updates:
        raise ValueError(
            "At least one non-empty partitioned LoRA-classifier update is required."
        )

    partition_counts: list[int] = []
    merged_updates: list[LoraClassifierFedAvgUpdate] = []
    for update in valid_updates:
        if not update.partitions:
            raise ValueError("partitioned LoRA-classifier update must have partitions.")
        merged = merge_partitioned_lora_classifier_deltas(update.partitions)
        partition_counts.append(len(update.partitions))
        merged_updates.append(
            LoraClassifierFedAvgUpdate(
                lora_parameter_deltas=merged.lora_parameter_deltas,
                classifier_head_weight_deltas=merged.classifier_head_weight_deltas,
                classifier_head_bias_deltas=merged.classifier_head_bias_deltas,
                example_count=update.example_count,
                mean_confidence=update.mean_confidence,
                mean_margin=update.mean_margin,
                delta_l2_norm=update.delta_l2_norm,
            )
        )

    result = compute_lora_classifier_fedavg(
        label_schema=label_schema,
        updates=merged_updates,
        weight_policy_name=weight_policy_name,
    )
    return LoraClassifierFedAvgResult(
        lora_parameter_deltas=result.lora_parameter_deltas,
        classifier_head_weight_deltas=result.classifier_head_weight_deltas,
        classifier_head_bias_deltas=result.classifier_head_bias_deltas,
        aggregated_metrics={
            **result.aggregated_metrics,
            "server_update_partitioned": 1.0,
            "partitioned_update_count": float(len(valid_updates)),
            "partition_count_total": float(sum(partition_counts)),
            "partition_count_mean": float(
                sum(partition_counts) / len(partition_counts)
            ),
        },
        update_count=result.update_count,
    )


def aggregate_lora_classifier_partitioned_delta_average(
    base_state: SharedAdapterState,
    update_payloads: Sequence[SharedAdapterUpdate],
    context: FederatedAggregationContext,
    overrides: Mapping[str, AggregationConfigScalar] | None,
) -> FederatedAggregationResult:
    """partitioned payload를 materialize한 뒤 다음 LoRA-classifier state를 만든다."""

    _validate_lora_classifier_partitioned_delta_average_overrides(overrides)

    base_state = cast(LoraClassifierState, base_state)
    updates = [cast(LoraClassifierDelta, payload) for payload in update_payloads]
    method_updates = [
        _to_lora_classifier_partitioned_method_update(
            base_state=base_state,
            payload=payload,
            context=context,
        )
        for payload in updates
    ]
    method_result = compute_lora_classifier_partitioned_delta_average(
        label_schema=base_state.label_schema,
        updates=method_updates,
        weight_policy_name=str((overrides or {}).get("weight_policy", "example_count")),
    )
    partition_average_deltas = _compute_average_partition_deltas(
        label_schema=base_state.label_schema,
        updates=method_updates,
        weight_policy_name=str((overrides or {}).get("weight_policy", "example_count")),
    )
    base_parameters = materialize_base_lora_classifier_state(
        base_state=base_state,
        context=context,
    )
    base_partition_parameters = materialize_base_lora_classifier_partitioned_state(
        base_state=base_state,
        context=context,
    )
    next_partition_parameters = (
        apply_lora_classifier_partition_deltas_to_partitioned_state(
            base_parameters=base_parameters,
            base_partition_parameters=base_partition_parameters,
            partition_deltas=partition_average_deltas,
        )
    )
    artifact_ref_resolver = context.require_artifact_ref_resolver(
        context="LoRA-classifier partitioned delta average"
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
        base_parameters=base_parameters,
        next_model_revision=context.next_model_revision,
        updated_at=context.aggregated_at,
        lora_adapter_artifact_ref=lora_adapter_artifact_ref,
        classifier_head_artifact_ref=classifier_head_artifact_ref,
        artifact_format=artifact_ref_resolver.artifact_format,
        lora_parameter_deltas=method_result.lora_parameter_deltas,
        classifier_head_weight_deltas=method_result.classifier_head_weight_deltas,
        classifier_head_bias_deltas=method_result.classifier_head_bias_deltas,
        partitioned_parameters=next_partition_parameters,
    )
    return FederatedAggregationResult(
        next_state=state_projection.next_state,
        aggregated_metrics={
            **method_result.aggregated_metrics,
            "partitioned_global_state_count": float(len(next_partition_parameters)),
        },
        update_count=method_result.update_count,
        aggregated_artifacts=state_projection.artifacts,
    )


def _to_lora_classifier_partitioned_method_update(
    *,
    base_state: LoraClassifierState,
    payload: LoraClassifierDelta,
    context: FederatedAggregationContext,
) -> LoraClassifierPartitionedDeltaAverageUpdate:
    validate_lora_classifier_update_matches_base(
        base_state=base_state,
        payload=payload,
    )
    return LoraClassifierPartitionedDeltaAverageUpdate(
        partitions=materialize_lora_classifier_partitioned_update(
            payload=payload,
            context=context,
        ),
        example_count=payload.example_count,
        mean_confidence=payload.mean_confidence,
        mean_margin=payload.mean_margin,
        delta_l2_norm=payload.l2_norm(),
    )


def _compute_average_partition_deltas(
    *,
    label_schema: Sequence[str],
    updates: Sequence[LoraClassifierPartitionedDeltaAverageUpdate],
    weight_policy_name: str,
) -> dict[str, LoraClassifierPartitionDelta]:
    partition_names = sorted(
        {
            partition_name
            for update in updates
            if update.example_count > 0
            for partition_name in update.partitions
        }
    )
    averaged: dict[str, LoraClassifierPartitionDelta] = {}
    for partition_name in partition_names:
        partition_updates = [
            LoraClassifierFedAvgUpdate(
                lora_parameter_deltas=partition.lora_parameter_deltas,
                classifier_head_weight_deltas=partition.classifier_head_weight_deltas,
                classifier_head_bias_deltas=partition.classifier_head_bias_deltas,
                example_count=update.example_count,
                mean_confidence=update.mean_confidence,
                mean_margin=update.mean_margin,
                delta_l2_norm=update.delta_l2_norm,
            )
            for update in updates
            if update.example_count > 0
            for name, partition in update.partitions.items()
            if name == partition_name
        ]
        if not partition_updates:
            continue
        partition_result = compute_lora_classifier_fedavg(
            label_schema=label_schema,
            updates=partition_updates,
            weight_policy_name=weight_policy_name,
        )
        averaged[partition_name] = LoraClassifierPartitionDelta(
            partition_name=partition_name,
            lora_parameter_deltas=partition_result.lora_parameter_deltas,
            classifier_head_weight_deltas=(
                partition_result.classifier_head_weight_deltas
            ),
            classifier_head_bias_deltas=partition_result.classifier_head_bias_deltas,
        )
    return averaged


def _validate_lora_classifier_partitioned_delta_average_overrides(
    overrides: Mapping[str, AggregationConfigScalar] | None,
) -> None:
    if overrides is None:
        return
    unknown_keys = sorted(
        set(overrides) - {"artifact_ref_prefix", "artifact_format", "weight_policy"}
    )
    if unknown_keys:
        raise ValueError(
            "Unsupported LoRA-classifier partitioned delta average config key(s): "
            f"{unknown_keys}."
        )


register_fedavg_adapter_strategy(
    FedAvgAdapterStrategySpec(
        adapter_kind=LORA_CLASSIFIER_ADAPTER_KIND,
        state_type=LoraClassifierState,
        update_type=LoraClassifierDelta,
        context="LoRA-classifier partitioned",
        aliases=("lora_classifier_partitioned_delta_average",),
        implementation_module=(
            compute_lora_classifier_partitioned_delta_average.__module__
        ),
        core_function_name=compute_lora_classifier_partitioned_delta_average.__name__,
        metadata={
            "adapter_kind": LORA_CLASSIFIER_ADAPTER_KIND,
            "requires_partitioned_deltas": True,
        },
        aggregate=aggregate_lora_classifier_partitioned_delta_average,
        method_name=PARTITIONED_DELTA_AVERAGE_BACKEND_NAME,
    )
)
