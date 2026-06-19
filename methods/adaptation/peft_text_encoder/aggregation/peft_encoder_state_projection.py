"""PEFT text encoder/head aggregation 결과를 다음 global state artifact로 투영한다."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from methods.adaptation.text_encoder_classifier.classifier_head_tensor_artifact import (
    build_classifier_head_state_tensor_artifact,
)
from methods.federated.aggregation.base import build_safetensors_aggregated_artifact
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PeftClassifierState,
)

from ..update.materialization import (
    CLASSIFIER_HEAD_STATE_WEIGHTS_KEY,
    PEFT_STATE_PARAMETERS_KEY,
    PeftEncoderMaterializedState,
)
from ..update.merged_tensor_artifact import build_peft_adapter_state_tensor_artifact

PeftEncoderStatePayload = PeftClassifierState


@dataclass(frozen=True, slots=True)
class PeftEncoderStateProjection:
    """server publication runtime이 저장할 next state와 artifact payload."""

    next_state: PeftEncoderStatePayload
    artifacts: dict[str, dict[str, object]]


def build_peft_encoder_state_projection(
    *,
    base_state: PeftEncoderStatePayload,
    base_parameters: PeftEncoderMaterializedState,
    next_model_revision: str,
    updated_at: datetime,
    peft_adapter_artifact_ref: str,
    classifier_head_artifact_ref: str,
    artifact_format: str,
    peft_parameter_deltas: Mapping[str, Sequence[float]],
    classifier_head_weight_deltas: Mapping[str, Sequence[float]],
    classifier_head_bias_deltas: Mapping[str, float],
    partitioned_parameters: (Mapping[str, PeftEncoderMaterializedState] | None) = None,
) -> PeftEncoderStateProjection:
    """base global snapshot에 aggregated delta를 적용해 next state를 만든다."""

    next_peft_parameters = _apply_vector_deltas(
        base_parameters.peft_parameters,
        peft_parameter_deltas,
        field_name=PEFT_STATE_PARAMETERS_KEY,
    )
    next_classifier_head_weights = _apply_vector_deltas(
        base_parameters.classifier_head_weights,
        classifier_head_weight_deltas,
        field_name=CLASSIFIER_HEAD_STATE_WEIGHTS_KEY,
    )
    next_classifier_head_biases = _apply_scalar_deltas(
        base_parameters.classifier_head_biases,
        classifier_head_bias_deltas,
    )
    applied_adapter_deltas = {
        key: [float(value) for value in values]
        for key, values in peft_parameter_deltas.items()
    }
    partitioned_peft_parameters: dict[str, dict[str, list[float]]] = {}
    partitioned_head_weights: dict[str, dict[str, list[float]]] = {}
    partitioned_head_biases: dict[str, dict[str, float]] = {}
    if partitioned_parameters:
        partitioned_peft_parameters = {
            partition_name: _json_vector_mapping(partition.peft_parameters)
            for partition_name, partition in sorted(partitioned_parameters.items())
        }
        partitioned_head_weights = {
            partition_name: _json_vector_mapping(partition.classifier_head_weights)
            for partition_name, partition in sorted(partitioned_parameters.items())
        }
        partitioned_head_biases = {
            partition_name: partition.classifier_head_biases
            for partition_name, partition in sorted(partitioned_parameters.items())
        }
    peft_state_tensors, peft_state_metadata = build_peft_adapter_state_tensor_artifact(
        peft_parameters=next_peft_parameters,
        applied_peft_parameter_deltas=applied_adapter_deltas,
        partitioned_peft_parameters=partitioned_peft_parameters,
    )
    head_state_tensors, head_state_metadata = (
        build_classifier_head_state_tensor_artifact(
            classifier_head_weights=next_classifier_head_weights,
            classifier_head_biases=next_classifier_head_biases,
            label_schema=base_state.label_schema,
            applied_classifier_head_weight_deltas={
                key: [float(value) for value in values]
                for key, values in classifier_head_weight_deltas.items()
            },
            applied_classifier_head_bias_deltas={
                key: float(value) for key, value in classifier_head_bias_deltas.items()
            },
            partitioned_classifier_head_weights=partitioned_head_weights,
            partitioned_classifier_head_biases=partitioned_head_biases,
        )
    )

    return PeftEncoderStateProjection(
        next_state=_build_next_state(
            base_state=base_state,
            next_model_revision=next_model_revision,
            updated_at=updated_at,
            peft_adapter_artifact_ref=peft_adapter_artifact_ref,
            classifier_head_artifact_ref=classifier_head_artifact_ref,
            artifact_format=artifact_format,
        ),
        artifacts={
            peft_adapter_artifact_ref: build_safetensors_aggregated_artifact(
                tensors=peft_state_tensors,
                metadata=peft_state_metadata,
            ),
            classifier_head_artifact_ref: build_safetensors_aggregated_artifact(
                tensors=head_state_tensors,
                metadata=head_state_metadata,
            ),
        },
    )


def _build_next_state(
    *,
    base_state: PeftEncoderStatePayload,
    next_model_revision: str,
    updated_at: datetime,
    peft_adapter_artifact_ref: str,
    classifier_head_artifact_ref: str,
    artifact_format: str,
) -> PeftEncoderStatePayload:
    return PeftClassifierState(
        schema_version=base_state.schema_version,
        adapter_kind=base_state.adapter_kind,
        model_id=base_state.model_id,
        model_revision=next_model_revision,
        training_scope=base_state.training_scope,
        updated_at=updated_at,
        backbone=base_state.backbone,
        peft_adapter_config=base_state.peft_adapter_config,
        label_schema=base_state.label_schema,
        peft_adapter_artifact_ref=peft_adapter_artifact_ref,
        classifier_head_artifact_ref=classifier_head_artifact_ref,
        artifact_format=artifact_format,
    )


def _json_vector_mapping(
    values: Mapping[str, Sequence[float]],
) -> dict[str, list[float]]:
    return {
        str(key): [float(value) for value in vector]
        for key, vector in sorted(values.items())
    }


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
