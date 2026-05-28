"""PEFT-encoder classifier family용 FedAvg projection."""

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
    materialize_base_peft_encoder_state,
    materialize_peft_encoder_update,
)
from .peft_encoder_state_projection import build_peft_encoder_state_projection

PEFT_ADAPTER_ARTIFACT_SLOT = "peft_adapter"
CLASSIFIER_HEAD_ARTIFACT_SLOT = "classifier_head"
PeftEncoderStatePayload = PeftClassifierState
PeftEncoderDeltaPayload = PeftClassifierDelta


@dataclass(frozen=True, slots=True)
class PeftEncoderFedAvgUpdate:
    """main_server boundary와 분리된 PEFT encoder/head FedAvg 입력."""

    peft_parameter_deltas: Mapping[str, Sequence[float]]
    classifier_head_weight_deltas: Mapping[str, Sequence[float]]
    classifier_head_bias_deltas: Mapping[str, float]
    example_count: int
    mean_confidence: float | None
    mean_margin: float | None
    delta_l2_norm: float


@dataclass(frozen=True, slots=True)
class PeftEncoderFedAvgResult:
    """PEFT encoder/head FedAvg 계산 결과."""

    peft_parameter_deltas: dict[str, list[float]]
    classifier_head_weight_deltas: dict[str, list[float]]
    classifier_head_bias_deltas: dict[str, float]
    aggregated_metrics: dict[str, float]
    update_count: int


def compute_peft_encoder_fedavg(
    *,
    label_schema: Sequence[str],
    updates: Sequence[PeftEncoderFedAvgUpdate],
    weight_policy_name: str = AGGREGATION_WEIGHT_EXAMPLE_COUNT,
) -> PeftEncoderFedAvgResult:
    """PEFT encoder/head delta를 policy weight로 평균한다."""

    labels = _normalize_label_schema(label_schema)
    weight_policy = AggregationWeightPolicy(name=weight_policy_name)
    valid_updates = tuple(update for update in updates if update.example_count > 0)
    if not valid_updates:
        raise ValueError(
            "At least one non-empty PEFT text encoder/head update is required."
        )

    peft_parameter_deltas = weighted_average_vector_mappings(
        [
            WeightedVectorMappingUpdate(
                values=update.peft_parameter_deltas,
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

    return PeftEncoderFedAvgResult(
        peft_parameter_deltas=peft_parameter_deltas,
        classifier_head_weight_deltas=classifier_head_weight_deltas,
        classifier_head_bias_deltas=classifier_head_bias_deltas,
        aggregated_metrics={
            **_aggregate_common_metrics(valid_updates),
            "aggregation_weight_policy_example_count": float(
                weight_policy.name == AGGREGATION_WEIGHT_EXAMPLE_COUNT
            ),
            "peft_parameter_count": float(len(peft_parameter_deltas)),
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
    update: PeftEncoderFedAvgUpdate,
    *,
    labels: Sequence[str],
) -> dict[str, Sequence[float]]:
    if set(update.classifier_head_weight_deltas) != set(labels):
        raise ValueError(
            "PEFT encoder/head FedAvg classifier head weight delta keys must match "
            "label_schema."
        )
    return {
        label: update.classifier_head_weight_deltas[label]
        for label in sorted(update.classifier_head_weight_deltas)
    }


def _normalize_classifier_head_bias_deltas(
    update: PeftEncoderFedAvgUpdate,
    *,
    labels: Sequence[str],
) -> dict[str, float]:
    extra_labels = set(update.classifier_head_bias_deltas) - set(labels)
    if extra_labels:
        raise ValueError(
            "PEFT encoder/head FedAvg bias deltas contain unknown labels: "
            f"{sorted(extra_labels)}"
        )
    return {
        label: float(update.classifier_head_bias_deltas.get(label, 0.0))
        for label in labels
    }


def _aggregate_common_metrics(
    updates: Sequence[PeftEncoderFedAvgUpdate],
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


def aggregate_peft_encoder_fedavg(
    base_state: SharedAdapterState,
    update_payloads: Sequence[SharedAdapterUpdate],
    context: FederatedAggregationContext,
    overrides: Mapping[str, AggregationConfigScalar] | None,
) -> FederatedAggregationResult:
    """PEFT text encoder/head update payload를 FedAvg core 입력으로 변환한다."""

    _validate_peft_encoder_fedavg_overrides(overrides)

    base_state = cast(PeftEncoderStatePayload, base_state)
    updates = [cast(PeftEncoderDeltaPayload, payload) for payload in update_payloads]
    method_updates = [
        _to_peft_encoder_method_update(
            base_state=base_state,
            payload=payload,
            context=context,
        )
        for payload in updates
    ]
    method_result = compute_peft_encoder_fedavg(
        label_schema=base_state.label_schema,
        updates=method_updates,
        weight_policy_name=str((overrides or {}).get("weight_policy", "example_count")),
    )
    artifact_ref_resolver = context.require_artifact_ref_resolver(
        context="PEFT encoder/head FedAvg"
    )
    peft_adapter_artifact_ref = artifact_ref_resolver.build_ref(
        next_model_revision=context.next_model_revision,
        artifact_name=_adapter_artifact_slot(base_state),
    )
    classifier_head_artifact_ref = artifact_ref_resolver.build_ref(
        next_model_revision=context.next_model_revision,
        artifact_name=CLASSIFIER_HEAD_ARTIFACT_SLOT,
    )
    state_projection = build_peft_encoder_state_projection(
        base_state=base_state,
        base_parameters=materialize_base_peft_encoder_state(
            base_state=base_state,
            context=context,
        ),
        next_model_revision=context.next_model_revision,
        updated_at=context.aggregated_at,
        peft_adapter_artifact_ref=peft_adapter_artifact_ref,
        classifier_head_artifact_ref=classifier_head_artifact_ref,
        artifact_format=artifact_ref_resolver.artifact_format,
        peft_parameter_deltas=method_result.peft_parameter_deltas,
        classifier_head_weight_deltas=method_result.classifier_head_weight_deltas,
        classifier_head_bias_deltas=method_result.classifier_head_bias_deltas,
    )
    return FederatedAggregationResult(
        next_state=state_projection.next_state,
        aggregated_metrics=method_result.aggregated_metrics,
        update_count=method_result.update_count,
        aggregated_artifacts=state_projection.artifacts,
    )


def _to_peft_encoder_method_update(
    *,
    base_state: PeftEncoderStatePayload,
    payload: PeftEncoderDeltaPayload,
    context: FederatedAggregationContext,
) -> PeftEncoderFedAvgUpdate:
    validate_peft_encoder_update_matches_base(
        base_state=base_state,
        payload=payload,
    )
    materialized = materialize_peft_encoder_update(
        payload=payload,
        context=context,
    )
    return PeftEncoderFedAvgUpdate(
        peft_parameter_deltas=materialized.peft_parameter_deltas,
        classifier_head_weight_deltas=materialized.classifier_head_weight_deltas,
        classifier_head_bias_deltas=materialized.classifier_head_bias_deltas,
        example_count=payload.example_count,
        mean_confidence=payload.mean_confidence,
        mean_margin=payload.mean_margin,
        delta_l2_norm=materialized.delta_l2_norm,
    )


def validate_peft_encoder_update_matches_base(
    *,
    base_state: PeftEncoderStatePayload,
    payload: PeftEncoderDeltaPayload,
) -> None:
    """PEFT text encoder/head update가 base state와 같은 lineage인지 검증한다."""

    if payload.adapter_kind != base_state.adapter_kind:
        raise ValueError(
            "PEFT text encoder/head updates must match the base adapter_kind."
        )
    if _payload_snapshot(payload.backbone) != _payload_snapshot(base_state.backbone):
        raise ValueError("All PEFT text encoder/head updates must match the backbone.")
    if _adapter_config_snapshot(payload) != _adapter_config_snapshot(base_state):
        raise ValueError(
            "All PEFT text encoder/head updates must match the adapter config."
        )
    if payload.labels != base_state.labels:
        raise ValueError(
            "PEFT text encoder/head updates must share the base ordered label_schema."
        )


def _payload_snapshot(payload) -> dict[str, object]:
    return payload.model_dump(mode="json")


def _adapter_config_snapshot(
    payload: PeftEncoderStatePayload | PeftEncoderDeltaPayload,
) -> dict[str, object]:
    return payload.peft_adapter_config.model_dump(mode="json")


def _adapter_artifact_slot(base_state: PeftEncoderStatePayload) -> str:
    return PEFT_ADAPTER_ARTIFACT_SLOT


def _validate_peft_encoder_fedavg_overrides(
    overrides: Mapping[str, AggregationConfigScalar] | None,
) -> None:
    if overrides is None:
        return
    unknown_keys = sorted(
        set(overrides) - {"artifact_ref_prefix", "artifact_format", "weight_policy"}
    )
    if unknown_keys:
        raise ValueError(
            "Unsupported PEFT text encoder/head aggregate artifact config key(s): "
            f"{unknown_keys}."
        )


def _register_peft_encoder_fedavg_strategy(
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
            implementation_module=compute_peft_encoder_fedavg.__module__,
            core_function_name=core_function_name,
            metadata={
                "adapter_kind": adapter_kind,
                "requires_inline_or_materialized_artifacts": True,
            },
            aggregate=aggregate,
        )
    )


_register_peft_encoder_fedavg_strategy(
    adapter_kind=PEFT_CLASSIFIER_ADAPTER_KIND,
    state_type=PeftClassifierState,
    update_type=PeftClassifierDelta,
    context="PEFT text encoder/head",
    aliases=("peft_classifier_fedavg",),
    core_function_name=compute_peft_encoder_fedavg.__name__,
    aggregate=aggregate_peft_encoder_fedavg,
)
