"""LoRA-classifier FedAvg 결과를 다음 global state artifact로 투영한다."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LoraClassifierState,
)
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PeftClassifierState,
)

from ..peft_encoder.update.materialization import (
    CLASSIFIER_HEAD_STATE_BIASES_KEY,
    CLASSIFIER_HEAD_STATE_WEIGHTS_KEY,
    LORA_STATE_PARAMETERS_KEY,
    PARTITIONED_CLASSIFIER_HEAD_STATE_BIASES_KEY,
    PARTITIONED_CLASSIFIER_HEAD_STATE_WEIGHTS_KEY,
    PARTITIONED_LORA_STATE_PARAMETERS_KEY,
    PARTITIONED_PEFT_STATE_PARAMETERS_KEY,
    PEFT_STATE_PARAMETERS_KEY,
    LoraClassifierMaterializedState,
)

PeftEncoderStatePayload = LoraClassifierState | PeftClassifierState


@dataclass(frozen=True, slots=True)
class LoraClassifierStateProjection:
    """server publication runtime이 저장할 next state와 artifact payload."""

    next_state: PeftEncoderStatePayload
    artifacts: dict[str, dict[str, object]]


def build_lora_classifier_state_projection(
    *,
    base_state: PeftEncoderStatePayload,
    base_parameters: LoraClassifierMaterializedState,
    next_model_revision: str,
    updated_at: datetime,
    lora_adapter_artifact_ref: str,
    classifier_head_artifact_ref: str,
    artifact_format: str,
    lora_parameter_deltas: Mapping[str, Sequence[float]],
    classifier_head_weight_deltas: Mapping[str, Sequence[float]],
    classifier_head_bias_deltas: Mapping[str, float],
    partitioned_parameters: (
        Mapping[str, LoraClassifierMaterializedState] | None
    ) = None,
) -> LoraClassifierStateProjection:
    """base global snapshot에 aggregated delta를 적용해 next state를 만든다."""

    next_lora_parameters = _apply_vector_deltas(
        base_parameters.lora_parameters,
        lora_parameter_deltas,
        field_name=LORA_STATE_PARAMETERS_KEY,
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
    adapter_parameters_key = _adapter_parameters_key(base_state)
    applied_adapter_deltas_key = _applied_adapter_deltas_key(base_state)
    lora_artifact: dict[str, object] = {
        adapter_parameters_key: next_lora_parameters,
        applied_adapter_deltas_key: {
            key: [float(value) for value in values]
            for key, values in lora_parameter_deltas.items()
        },
    }
    classifier_head_artifact: dict[str, object] = {
        CLASSIFIER_HEAD_STATE_WEIGHTS_KEY: next_classifier_head_weights,
        CLASSIFIER_HEAD_STATE_BIASES_KEY: next_classifier_head_biases,
        "applied_classifier_head_weight_deltas": {
            key: [float(value) for value in values]
            for key, values in classifier_head_weight_deltas.items()
        },
        "applied_classifier_head_bias_deltas": {
            key: float(value) for key, value in classifier_head_bias_deltas.items()
        },
    }
    if partitioned_parameters:
        lora_artifact[_partitioned_adapter_parameters_key(base_state)] = {
            partition_name: _json_vector_mapping(partition.lora_parameters)
            for partition_name, partition in sorted(partitioned_parameters.items())
        }
        classifier_head_artifact[PARTITIONED_CLASSIFIER_HEAD_STATE_WEIGHTS_KEY] = {
            partition_name: _json_vector_mapping(partition.classifier_head_weights)
            for partition_name, partition in sorted(partitioned_parameters.items())
        }
        classifier_head_artifact[PARTITIONED_CLASSIFIER_HEAD_STATE_BIASES_KEY] = {
            partition_name: partition.classifier_head_biases
            for partition_name, partition in sorted(partitioned_parameters.items())
        }

    return LoraClassifierStateProjection(
        next_state=_build_next_state(
            base_state=base_state,
            next_model_revision=next_model_revision,
            updated_at=updated_at,
            lora_adapter_artifact_ref=lora_adapter_artifact_ref,
            classifier_head_artifact_ref=classifier_head_artifact_ref,
            artifact_format=artifact_format,
        ),
        artifacts={
            lora_adapter_artifact_ref: {
                **lora_artifact,
            },
            classifier_head_artifact_ref: {
                **classifier_head_artifact,
            },
        },
    )


def _build_next_state(
    *,
    base_state: PeftEncoderStatePayload,
    next_model_revision: str,
    updated_at: datetime,
    lora_adapter_artifact_ref: str,
    classifier_head_artifact_ref: str,
    artifact_format: str,
) -> PeftEncoderStatePayload:
    if isinstance(base_state, PeftClassifierState):
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
            peft_adapter_artifact_ref=lora_adapter_artifact_ref,
            classifier_head_artifact_ref=classifier_head_artifact_ref,
            artifact_format=artifact_format,
        )
    return LoraClassifierState(
        schema_version=base_state.schema_version,
        adapter_kind=base_state.adapter_kind,
        model_id=base_state.model_id,
        model_revision=next_model_revision,
        training_scope=base_state.training_scope,
        updated_at=updated_at,
        backbone=base_state.backbone,
        lora_config=base_state.lora_config,
        label_schema=base_state.label_schema,
        lora_adapter_artifact_ref=lora_adapter_artifact_ref,
        classifier_head_artifact_ref=classifier_head_artifact_ref,
        artifact_format=artifact_format,
    )


def _adapter_parameters_key(base_state: PeftEncoderStatePayload) -> str:
    if isinstance(base_state, PeftClassifierState):
        return PEFT_STATE_PARAMETERS_KEY
    return LORA_STATE_PARAMETERS_KEY


def _partitioned_adapter_parameters_key(base_state: PeftEncoderStatePayload) -> str:
    if isinstance(base_state, PeftClassifierState):
        return PARTITIONED_PEFT_STATE_PARAMETERS_KEY
    return PARTITIONED_LORA_STATE_PARAMETERS_KEY


def _applied_adapter_deltas_key(base_state: PeftEncoderStatePayload) -> str:
    if isinstance(base_state, PeftClassifierState):
        return "applied_peft_parameter_deltas"
    return "applied_lora_parameter_deltas"


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
