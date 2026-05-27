"""PEFT-encoder classifier partitioned delta average backend."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PEFT_CLASSIFIER_ADAPTER_KIND,
    PeftClassifierDelta,
    PeftClassifierState,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)

from ..update.materialization import (
    materialize_base_peft_encoder_partitioned_state,
    materialize_base_peft_encoder_state,
    materialize_peft_encoder_partitioned_update,
)
from ..update.partitioned_delta import LoraClassifierPartitionDelta
from .peft_encoder_fedavg_projection import (
    CLASSIFIER_HEAD_ARTIFACT_SLOT,
    LORA_ADAPTER_ARTIFACT_SLOT,
    PEFT_ADAPTER_ARTIFACT_SLOT,
    LoraClassifierFedAvgResult,
    PeftEncoderDeltaPayload,
    PeftEncoderFedAvgUpdate,
    PeftEncoderStatePayload,
    compute_peft_encoder_fedavg,
    validate_peft_encoder_update_matches_base,
)
from .peft_encoder_partitioned_state import (
    apply_peft_encoder_partition_deltas_to_partitioned_state,
    merge_partitioned_peft_encoder_deltas,
)
from .peft_encoder_state_projection import build_peft_encoder_state_projection

PARTITIONED_DELTA_AVERAGE_BACKEND_NAME = "partitioned_delta_average"


@dataclass(frozen=True, slots=True)
class PeftEncoderPartitionedDeltaAverageUpdate:
    """partitioned server update policy가 소비하는 client delta."""

    partitions: Mapping[str, LoraClassifierPartitionDelta]
    example_count: int
    mean_confidence: float | None
    mean_margin: float | None
    delta_l2_norm: float


LoraClassifierPartitionedDeltaAverageUpdate = PeftEncoderPartitionedDeltaAverageUpdate


def compute_peft_encoder_partitioned_delta_average(
    *,
    label_schema: Sequence[str],
    updates: Sequence[PeftEncoderPartitionedDeltaAverageUpdate],
    weight_policy_name: str = AGGREGATION_WEIGHT_EXAMPLE_COUNT,
) -> LoraClassifierFedAvgResult:
    """client별 partition을 병합한 뒤 PEFT/head delta를 policy weight로 평균한다."""

    valid_updates = tuple(update for update in updates if update.example_count > 0)
    if not valid_updates:
        raise ValueError(
            "At least one non-empty partitioned PEFT-classifier update is required."
        )

    partition_counts: list[int] = []
    merged_updates: list[PeftEncoderFedAvgUpdate] = []
    for update in valid_updates:
        if not update.partitions:
            raise ValueError("partitioned PEFT-classifier update must have partitions.")
        merged = merge_partitioned_peft_encoder_deltas(update.partitions)
        partition_counts.append(len(update.partitions))
        merged_updates.append(
            PeftEncoderFedAvgUpdate(
                lora_parameter_deltas=merged.lora_parameter_deltas,
                classifier_head_weight_deltas=merged.classifier_head_weight_deltas,
                classifier_head_bias_deltas=merged.classifier_head_bias_deltas,
                example_count=update.example_count,
                mean_confidence=update.mean_confidence,
                mean_margin=update.mean_margin,
                delta_l2_norm=update.delta_l2_norm,
            )
        )

    result = compute_peft_encoder_fedavg(
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


compute_lora_classifier_partitioned_delta_average = (
    compute_peft_encoder_partitioned_delta_average
)


def aggregate_peft_encoder_partitioned_delta_average(
    base_state: SharedAdapterState,
    update_payloads: Sequence[SharedAdapterUpdate],
    context: FederatedAggregationContext,
    overrides: Mapping[str, AggregationConfigScalar] | None,
) -> FederatedAggregationResult:
    """partitioned payload를 materialize한 뒤 다음 PEFT classifier state를 만든다."""

    _validate_peft_encoder_partitioned_delta_average_overrides(overrides)

    base_state = cast(PeftEncoderStatePayload, base_state)
    updates = [cast(PeftEncoderDeltaPayload, payload) for payload in update_payloads]
    method_updates = [
        _to_lora_classifier_partitioned_method_update(
            base_state=base_state,
            payload=payload,
            context=context,
        )
        for payload in updates
    ]
    method_result = compute_peft_encoder_partitioned_delta_average(
        label_schema=base_state.label_schema,
        updates=method_updates,
        weight_policy_name=str((overrides or {}).get("weight_policy", "example_count")),
    )
    partition_average_deltas = _compute_average_partition_deltas(
        label_schema=base_state.label_schema,
        updates=method_updates,
        weight_policy_name=str((overrides or {}).get("weight_policy", "example_count")),
    )
    base_parameters = materialize_base_peft_encoder_state(
        base_state=base_state,
        context=context,
    )
    base_partition_parameters = materialize_base_peft_encoder_partitioned_state(
        base_state=base_state,
        context=context,
    )
    next_partition_parameters = (
        apply_peft_encoder_partition_deltas_to_partitioned_state(
            base_parameters=base_parameters,
            base_partition_parameters=base_partition_parameters,
            partition_deltas=partition_average_deltas,
        )
    )
    artifact_ref_resolver = context.require_artifact_ref_resolver(
        context="PEFT-classifier partitioned delta average"
    )
    lora_adapter_artifact_ref = artifact_ref_resolver.build_ref(
        next_model_revision=context.next_model_revision,
        artifact_name=_adapter_artifact_slot(base_state),
    )
    classifier_head_artifact_ref = artifact_ref_resolver.build_ref(
        next_model_revision=context.next_model_revision,
        artifact_name=CLASSIFIER_HEAD_ARTIFACT_SLOT,
    )
    state_projection = build_peft_encoder_state_projection(
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
    base_state: PeftEncoderStatePayload,
    payload: PeftEncoderDeltaPayload,
    context: FederatedAggregationContext,
) -> PeftEncoderPartitionedDeltaAverageUpdate:
    validate_peft_encoder_update_matches_base(
        base_state=base_state,
        payload=payload,
    )
    return PeftEncoderPartitionedDeltaAverageUpdate(
        partitions=materialize_peft_encoder_partitioned_update(
            payload=payload,
            context=context,
        ),
        example_count=payload.example_count,
        mean_confidence=payload.mean_confidence,
        mean_margin=payload.mean_margin,
        delta_l2_norm=payload.l2_norm(),
    )


def _adapter_artifact_slot(base_state: PeftEncoderStatePayload) -> str:
    if isinstance(base_state, PeftClassifierState):
        return PEFT_ADAPTER_ARTIFACT_SLOT
    return LORA_ADAPTER_ARTIFACT_SLOT


def _compute_average_partition_deltas(
    *,
    label_schema: Sequence[str],
    updates: Sequence[PeftEncoderPartitionedDeltaAverageUpdate],
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
            PeftEncoderFedAvgUpdate(
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
        partition_result = compute_peft_encoder_fedavg(
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


def _validate_peft_encoder_partitioned_delta_average_overrides(
    overrides: Mapping[str, AggregationConfigScalar] | None,
) -> None:
    if overrides is None:
        return
    unknown_keys = sorted(
        set(overrides) - {"artifact_ref_prefix", "artifact_format", "weight_policy"}
    )
    if unknown_keys:
        raise ValueError(
            "Unsupported PEFT-classifier partitioned delta average config key(s): "
            f"{unknown_keys}."
        )


aggregate_lora_classifier_partitioned_delta_average = (
    aggregate_peft_encoder_partitioned_delta_average
)


def _register_peft_encoder_partitioned_strategy(
    *,
    adapter_kind: str,
    state_type: type[object],
    update_type: type[object],
    context: str,
    aliases: tuple[str, ...],
    core_function_name: str,
    aggregate: Callable[
        [
            SharedAdapterState,
            Sequence[SharedAdapterUpdate],
            FederatedAggregationContext,
            Mapping[str, AggregationConfigScalar] | None,
        ],
        FederatedAggregationResult,
    ],
) -> None:
    register_fedavg_adapter_strategy(
        FedAvgAdapterStrategySpec(
            adapter_kind=adapter_kind,
            state_type=state_type,
            update_type=update_type,
            context=context,
            aliases=aliases,
            implementation_module=(
                compute_peft_encoder_partitioned_delta_average.__module__
            ),
            core_function_name=core_function_name,
            metadata={
                "adapter_kind": adapter_kind,
                "requires_partitioned_deltas": True,
            },
            aggregate=aggregate,
            method_name=PARTITIONED_DELTA_AVERAGE_BACKEND_NAME,
        )
    )


_register_peft_encoder_partitioned_strategy(
    adapter_kind=LORA_CLASSIFIER_ADAPTER_KIND,
    state_type=LoraClassifierState,
    update_type=LoraClassifierDelta,
    context="LoRA-classifier partitioned",
    aliases=("lora_classifier_partitioned_delta_average",),
    core_function_name=compute_peft_encoder_partitioned_delta_average.__name__,
    aggregate=aggregate_peft_encoder_partitioned_delta_average,
)
_register_peft_encoder_partitioned_strategy(
    adapter_kind=PEFT_CLASSIFIER_ADAPTER_KIND,
    state_type=PeftClassifierState,
    update_type=PeftClassifierDelta,
    context="PEFT-classifier partitioned",
    aliases=("peft_classifier_partitioned_delta_average",),
    core_function_name=compute_peft_encoder_partitioned_delta_average.__name__,
    aggregate=aggregate_peft_encoder_partitioned_delta_average,
)
