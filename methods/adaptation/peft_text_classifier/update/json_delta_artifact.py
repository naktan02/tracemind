"""PEFT encoder JSON delta artifact payload builders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from methods.adaptation.peft_text_classifier.update.partitioned_delta import (
    PeftEncoderPartitionDelta,
)
from shared.src.contracts.training_contracts import TrainingTask

PEFT_ADAPTER_DELTA_ARTIFACT_SCHEMA_VERSION = "peft_encoder_client_delta_artifact.v1"
HEAD_DELTA_ARTIFACT_SCHEMA_VERSION = "peft_encoder_client_head_delta_artifact.v1"
PARTITIONED_DELTA_ARTIFACT_SCHEMA_VERSION = (
    "peft_encoder_client_partitioned_delta_artifact.v1"
)


def build_peft_adapter_delta_json_artifact_payload(
    *,
    update_id: str,
    training_task: TrainingTask,
    client_id: str,
    peft_parameter_deltas: Mapping[str, Sequence[float]],
) -> dict[str, object]:
    return {
        "schema_version": PEFT_ADAPTER_DELTA_ARTIFACT_SCHEMA_VERSION,
        "update_id": update_id,
        "round_id": training_task.round_id,
        "client_id": client_id,
        "peft_parameter_deltas": {
            str(key): [float(value) for value in values]
            for key, values in peft_parameter_deltas.items()
        },
    }


def build_classifier_head_delta_json_artifact_payload(
    *,
    update_id: str,
    training_task: TrainingTask,
    client_id: str,
    classifier_head_weight_deltas: Mapping[str, Sequence[float]],
    classifier_head_bias_deltas: Mapping[str, float],
) -> dict[str, object]:
    return {
        "schema_version": HEAD_DELTA_ARTIFACT_SCHEMA_VERSION,
        "update_id": update_id,
        "round_id": training_task.round_id,
        "client_id": client_id,
        "classifier_head_weight_deltas": {
            str(key): [float(value) for value in values]
            for key, values in classifier_head_weight_deltas.items()
        },
        "classifier_head_bias_deltas": {
            str(key): float(value) for key, value in classifier_head_bias_deltas.items()
        },
    }


def build_partitioned_delta_json_artifact_payload(
    *,
    update_id: str,
    training_task: TrainingTask,
    client_id: str,
    partitioned_deltas: Mapping[str, PeftEncoderPartitionDelta],
) -> dict[str, object]:
    return {
        "schema_version": PARTITIONED_DELTA_ARTIFACT_SCHEMA_VERSION,
        "update_id": update_id,
        "round_id": training_task.round_id,
        "client_id": client_id,
        "partitions": {
            str(name): {
                "peft_parameter_deltas": {
                    str(key): [float(value) for value in values]
                    for key, values in delta.peft_parameter_deltas.items()
                },
                "classifier_head_weight_deltas": {
                    str(key): [float(value) for value in values]
                    for key, values in delta.classifier_head_weight_deltas.items()
                },
                "classifier_head_bias_deltas": {
                    str(key): float(value)
                    for key, value in delta.classifier_head_bias_deltas.items()
                },
            }
            for name, delta in sorted(partitioned_deltas.items())
        },
    }
