"""LoRA-classifier family용 FedAvg 계산 core."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from methods.federated.aggregation.base import (
    AggregationConfigScalar,
    AggregationJsonArtifactLoader,
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
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LORA_CLASSIFIER_ADAPTER_KIND,
    LoraClassifierDelta,
    LoraClassifierState,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)

LORA_ADAPTER_ARTIFACT_SLOT = "lora_adapter"
CLASSIFIER_HEAD_ARTIFACT_SLOT = "classifier_head"
LORA_STATE_PARAMETERS_KEY = "lora_parameters"
CLASSIFIER_HEAD_STATE_WEIGHTS_KEY = "classifier_head_weights"
CLASSIFIER_HEAD_STATE_BIASES_KEY = "classifier_head_biases"


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
) -> LoraClassifierFedAvgResult:
    """LoRA parameter delta와 classifier-head delta를 example_count로 평균한다."""

    labels = _normalize_label_schema(label_schema)
    valid_updates = tuple(update for update in updates if update.example_count > 0)
    if not valid_updates:
        raise ValueError("At least one non-empty LoRA-classifier update is required.")

    lora_parameter_deltas = weighted_average_vector_mappings(
        [
            WeightedVectorMappingUpdate(
                values=update.lora_parameter_deltas,
                weight=float(update.example_count),
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
                weight=float(update.example_count),
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
                weight=float(update.example_count),
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
    base_parameters = _materialize_base_lora_classifier_state(
        base_state=base_state,
        context=context,
    )
    next_lora_parameters = _apply_vector_deltas(
        base_parameters.lora_parameters,
        method_result.lora_parameter_deltas,
        field_name=LORA_STATE_PARAMETERS_KEY,
    )
    next_classifier_head_weights = _apply_vector_deltas(
        base_parameters.classifier_head_weights,
        method_result.classifier_head_weight_deltas,
        field_name=CLASSIFIER_HEAD_STATE_WEIGHTS_KEY,
    )
    next_classifier_head_biases = _apply_scalar_deltas(
        base_parameters.classifier_head_biases,
        method_result.classifier_head_bias_deltas,
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
        lora_adapter_artifact_ref=lora_adapter_artifact_ref,
        classifier_head_artifact_ref=classifier_head_artifact_ref,
        artifact_format=artifact_ref_resolver.artifact_format,
    )
    return FederatedAggregationResult(
        next_state=next_state,
        aggregated_metrics=method_result.aggregated_metrics,
        update_count=method_result.update_count,
        aggregated_artifacts={
            lora_adapter_artifact_ref: {
                LORA_STATE_PARAMETERS_KEY: next_lora_parameters,
                "applied_lora_parameter_deltas": (method_result.lora_parameter_deltas),
            },
            classifier_head_artifact_ref: {
                CLASSIFIER_HEAD_STATE_WEIGHTS_KEY: next_classifier_head_weights,
                CLASSIFIER_HEAD_STATE_BIASES_KEY: next_classifier_head_biases,
                "applied_classifier_head_weight_deltas": (
                    method_result.classifier_head_weight_deltas
                ),
                "applied_classifier_head_bias_deltas": (
                    method_result.classifier_head_bias_deltas
                ),
            },
        },
    )


def _to_lora_classifier_method_update(
    *,
    base_state: LoraClassifierState,
    payload: LoraClassifierDelta,
    context: FederatedAggregationContext,
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
    materialized = _materialize_lora_classifier_update(
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


def _payload_snapshot(payload) -> dict[str, object]:
    return payload.model_dump(mode="json")


@dataclass(frozen=True, slots=True)
class _MaterializedLoraClassifierUpdate:
    lora_parameter_deltas: dict[str, list[float]]
    classifier_head_weight_deltas: dict[str, list[float]]
    classifier_head_bias_deltas: dict[str, float]
    delta_l2_norm: float


@dataclass(frozen=True, slots=True)
class _MaterializedLoraClassifierState:
    lora_parameters: dict[str, list[float]]
    classifier_head_weights: dict[str, list[float]]
    classifier_head_biases: dict[str, float]


def _materialize_base_lora_classifier_state(
    *,
    base_state: LoraClassifierState,
    context: FederatedAggregationContext,
) -> _MaterializedLoraClassifierState:
    """base global state artifact를 읽고 없으면 zero-initialized state로 본다."""

    loader = None
    if (
        base_state.lora_adapter_artifact_ref is not None
        or base_state.classifier_head_artifact_ref is not None
    ):
        loader = context.require_artifact_loader(
            context="LoRA-classifier base state materialization"
        )

    lora_parameters: dict[str, list[float]] = {}
    if base_state.lora_adapter_artifact_ref is not None:
        if loader is None:
            raise AssertionError("artifact loader must be resolved before load.")
        lora_parameters = _load_base_lora_parameters(
            artifact_ref=base_state.lora_adapter_artifact_ref,
            loader=loader,
        )

    classifier_head_weights: dict[str, list[float]] = {}
    classifier_head_biases: dict[str, float] = {}
    if base_state.classifier_head_artifact_ref is not None:
        if loader is None:
            raise AssertionError("artifact loader must be resolved before load.")
        classifier_head_weights, classifier_head_biases = (
            _load_base_classifier_head_parameters(
                artifact_ref=base_state.classifier_head_artifact_ref,
                loader=loader,
            )
        )

    return _MaterializedLoraClassifierState(
        lora_parameters=lora_parameters,
        classifier_head_weights=classifier_head_weights,
        classifier_head_biases=classifier_head_biases,
    )


def _materialize_lora_classifier_update(
    *,
    payload: LoraClassifierDelta,
    context: FederatedAggregationContext,
) -> _MaterializedLoraClassifierUpdate:
    loader = None
    if (
        payload.lora_parameter_deltas is None
        or payload.classifier_head_weight_deltas is None
    ):
        loader = context.require_artifact_loader(context="LoRA-classifier FedAvg")

    if payload.lora_parameter_deltas is not None:
        lora_parameter_deltas = _normalize_vector_mapping(
            payload.lora_parameter_deltas,
            field_name="lora_parameter_deltas",
        )
    else:
        if loader is None:
            raise AssertionError("artifact loader must be resolved before load.")
        lora_parameter_deltas = _load_lora_parameter_deltas(
            payload=payload,
            loader=loader,
        )

    if payload.classifier_head_weight_deltas is not None:
        head_artifact = None
        classifier_head_weight_deltas = _normalize_vector_mapping(
            payload.classifier_head_weight_deltas,
            field_name="classifier_head_weight_deltas",
        )
    else:
        if loader is None:
            raise AssertionError("artifact loader must be resolved before load.")
        head_artifact = _load_classifier_head_artifact(
            payload=payload,
            loader=loader,
        )
        classifier_head_weight_deltas = head_artifact[0]
    classifier_head_bias_deltas = dict(payload.classifier_head_bias_deltas)
    if not classifier_head_bias_deltas and head_artifact is not None:
        classifier_head_bias_deltas = head_artifact[1]

    return _MaterializedLoraClassifierUpdate(
        lora_parameter_deltas=lora_parameter_deltas,
        classifier_head_weight_deltas=classifier_head_weight_deltas,
        classifier_head_bias_deltas=classifier_head_bias_deltas,
        delta_l2_norm=(
            payload.delta_l2_norm
            if payload.delta_l2_norm is not None
            else _l2_norm(
                lora_parameter_deltas=lora_parameter_deltas,
                classifier_head_weight_deltas=classifier_head_weight_deltas,
                classifier_head_bias_deltas=classifier_head_bias_deltas,
            )
        ),
    )


def _load_lora_parameter_deltas(
    *,
    payload: LoraClassifierDelta,
    loader: AggregationJsonArtifactLoader,
) -> dict[str, list[float]]:
    if payload.lora_delta_artifact_ref is None:
        raise ValueError(
            "LoRA-classifier artifact materialization requires lora_delta_artifact_ref."
        )
    artifact = loader.load_json_artifact(artifact_ref=payload.lora_delta_artifact_ref)
    source = artifact.get("lora_parameter_deltas", artifact)
    return _normalize_vector_mapping(source, field_name="lora_parameter_deltas")


def _load_base_lora_parameters(
    *,
    artifact_ref: str,
    loader: AggregationJsonArtifactLoader,
) -> dict[str, list[float]]:
    artifact = loader.load_json_artifact(artifact_ref=artifact_ref)
    source = artifact.get(
        LORA_STATE_PARAMETERS_KEY,
        artifact.get("lora_parameter_deltas", artifact),
    )
    return _normalize_vector_mapping(source, field_name=LORA_STATE_PARAMETERS_KEY)


def _load_base_classifier_head_parameters(
    *,
    artifact_ref: str,
    loader: AggregationJsonArtifactLoader,
) -> tuple[dict[str, list[float]], dict[str, float]]:
    artifact = loader.load_json_artifact(artifact_ref=artifact_ref)
    weight_source = artifact.get(
        CLASSIFIER_HEAD_STATE_WEIGHTS_KEY,
        artifact.get("classifier_head_weight_deltas", artifact),
    )
    bias_source = artifact.get(
        CLASSIFIER_HEAD_STATE_BIASES_KEY,
        artifact.get("classifier_head_bias_deltas", {}),
    )
    return (
        _normalize_vector_mapping(
            weight_source,
            field_name=CLASSIFIER_HEAD_STATE_WEIGHTS_KEY,
        ),
        _normalize_scalar_mapping(
            bias_source,
            field_name=CLASSIFIER_HEAD_STATE_BIASES_KEY,
        ),
    )


def _load_classifier_head_artifact(
    *,
    payload: LoraClassifierDelta,
    loader: AggregationJsonArtifactLoader,
) -> tuple[dict[str, list[float]], dict[str, float]]:
    if payload.classifier_head_delta_artifact_ref is None:
        raise ValueError(
            "LoRA-classifier artifact materialization requires "
            "classifier_head_delta_artifact_ref."
        )
    artifact = loader.load_json_artifact(
        artifact_ref=payload.classifier_head_delta_artifact_ref
    )
    weight_source = artifact.get("classifier_head_weight_deltas", artifact)
    bias_source = artifact.get("classifier_head_bias_deltas", {})
    return (
        _normalize_vector_mapping(
            weight_source,
            field_name="classifier_head_weight_deltas",
        ),
        _normalize_scalar_mapping(
            bias_source,
            field_name="classifier_head_bias_deltas",
        ),
    )


def _normalize_vector_mapping(
    source: object,
    *,
    field_name: str,
) -> dict[str, list[float]]:
    if not isinstance(source, Mapping) or not source:
        raise ValueError(f"{field_name} artifact must be a non-empty mapping.")
    result: dict[str, list[float]] = {}
    for key, values in source.items():
        normalized_key = str(key).strip()
        if not normalized_key:
            raise ValueError(f"{field_name} artifact keys must not be empty.")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"{field_name} artifact values must be numeric vectors.")
        vector = [float(value) for value in values]
        if not vector:
            raise ValueError(f"{field_name} artifact vectors must not be empty.")
        result[normalized_key] = vector
    return result


def _normalize_scalar_mapping(
    source: object,
    *,
    field_name: str,
) -> dict[str, float]:
    if not isinstance(source, Mapping):
        raise ValueError(f"{field_name} artifact must be a mapping.")
    result: dict[str, float] = {}
    for key, value in source.items():
        normalized_key = str(key).strip()
        if not normalized_key:
            raise ValueError(f"{field_name} artifact keys must not be empty.")
        result[normalized_key] = float(value)
    return result


def _apply_vector_deltas(
    base_values: Mapping[str, Sequence[float]],
    deltas: Mapping[str, Sequence[float]],
    *,
    field_name: str,
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for key in sorted(set(base_values) | set(deltas)):
        base_vector = [float(value) for value in base_values.get(key, [])]
        delta_vector = [float(value) for value in deltas.get(key, [])]
        if base_vector and delta_vector and len(base_vector) != len(delta_vector):
            raise ValueError(f"{field_name} delta dimension mismatch for {key!r}.")
        if not base_vector:
            result[key] = delta_vector
            continue
        if not delta_vector:
            result[key] = base_vector
            continue
        result[key] = [
            base_value + delta_value
            for base_value, delta_value in zip(
                base_vector,
                delta_vector,
                strict=True,
            )
        ]
    return result


def _apply_scalar_deltas(
    base_values: Mapping[str, float],
    deltas: Mapping[str, float],
) -> dict[str, float]:
    return {
        key: float(base_values.get(key, 0.0)) + float(deltas.get(key, 0.0))
        for key in sorted(set(base_values) | set(deltas))
    }


def _l2_norm(
    *,
    lora_parameter_deltas: Mapping[str, Sequence[float]],
    classifier_head_weight_deltas: Mapping[str, Sequence[float]],
    classifier_head_bias_deltas: Mapping[str, float],
) -> float:
    squared_norm = 0.0
    for vectors in (lora_parameter_deltas, classifier_head_weight_deltas):
        squared_norm += sum(
            float(value) * float(value)
            for vector in vectors.values()
            for value in vector
        )
    squared_norm += sum(
        float(value) * float(value) for value in classifier_head_bias_deltas.values()
    )
    return math.sqrt(squared_norm)


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
