"""LoRA-classifier partitioned delta tensor artifact helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping

import torch
from torch import Tensor

from methods.adaptation.peft_text_classifier.update.partitioned_delta import (
    LoraClassifierPartitionDelta,
)

PARTITIONED_DELTA_TENSOR_ARTIFACT_SCHEMA_VERSION = (
    "lora_classifier_client_partitioned_delta_tensor_artifact.v1"
)
PARTITIONED_DELTA_TENSOR_ARTIFACT_FORMAT = "safetensors"
PARTITIONED_DELTA_TENSOR_ARTIFACT_INDEX_METADATA_KEY = "partition_index_json"


def build_partitioned_delta_tensor_artifact(
    partitioned_deltas: Mapping[str, LoraClassifierPartitionDelta],
) -> tuple[dict[str, Tensor], dict[str, str]]:
    """partitioned delta를 safetensors payload와 metadata index로 변환한다."""

    if not partitioned_deltas:
        raise ValueError("Partitioned delta tensor artifact requires partitions.")
    tensors: dict[str, Tensor] = {}
    index: dict[str, object] = {
        "schema_version": PARTITIONED_DELTA_TENSOR_ARTIFACT_SCHEMA_VERSION,
        "artifact_format": PARTITIONED_DELTA_TENSOR_ARTIFACT_FORMAT,
        "partitions": {},
    }
    partitions_index: dict[str, object] = {}
    for partition_index, (partition_name, delta) in enumerate(
        sorted(partitioned_deltas.items())
    ):
        partition_payload: dict[str, dict[str, str]] = {
            "lora_parameter_deltas": {},
            "classifier_head_weight_deltas": {},
            "classifier_head_bias_deltas": {},
        }
        _add_vector_mapping_tensors(
            tensors=tensors,
            target=partition_payload["lora_parameter_deltas"],
            prefix=f"p{partition_index:03d}.lora",
            values=delta.lora_parameter_deltas,
        )
        _add_vector_mapping_tensors(
            tensors=tensors,
            target=partition_payload["classifier_head_weight_deltas"],
            prefix=f"p{partition_index:03d}.head_weight",
            values=delta.classifier_head_weight_deltas,
        )
        for label, value in sorted(delta.classifier_head_bias_deltas.items()):
            bias_offset = len(partition_payload["classifier_head_bias_deltas"])
            tensor_key = f"p{partition_index:03d}.head_bias.{bias_offset:04d}"
            tensors[tensor_key] = torch.tensor([float(value)], dtype=torch.float32)
            partition_payload["classifier_head_bias_deltas"][str(label)] = tensor_key
        partitions_index[str(partition_name)] = partition_payload
    index["partitions"] = partitions_index
    return tensors, {
        PARTITIONED_DELTA_TENSOR_ARTIFACT_INDEX_METADATA_KEY: json.dumps(
            index,
            ensure_ascii=True,
            sort_keys=True,
        )
    }


def parse_partitioned_delta_tensor_artifact(
    *,
    tensors: Mapping[str, Tensor],
    metadata: Mapping[str, str],
) -> dict[str, LoraClassifierPartitionDelta]:
    """safetensors payload와 metadata index를 partitioned delta로 복원한다."""

    raw_index = metadata.get(PARTITIONED_DELTA_TENSOR_ARTIFACT_INDEX_METADATA_KEY)
    if raw_index is None:
        raise ValueError("Partitioned delta tensor artifact metadata is missing.")
    index = json.loads(raw_index)
    if not isinstance(index, dict):
        raise ValueError("Partitioned delta tensor artifact index must be an object.")
    if index.get("schema_version") != PARTITIONED_DELTA_TENSOR_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported partitioned delta tensor artifact schema_version: "
            f"{index.get('schema_version')!r}."
        )
    partitions = index.get("partitions")
    if not isinstance(partitions, dict) or not partitions:
        raise ValueError("Partitioned delta tensor artifact partitions must exist.")

    result: dict[str, LoraClassifierPartitionDelta] = {}
    for partition_name, partition_payload in partitions.items():
        if not isinstance(partition_payload, dict):
            raise ValueError("Partitioned delta partition index must be an object.")
        result[str(partition_name)] = LoraClassifierPartitionDelta(
            partition_name=str(partition_name),
            lora_parameter_deltas=_read_vector_mapping(
                tensors=tensors,
                source=partition_payload.get("lora_parameter_deltas", {}),
            ),
            classifier_head_weight_deltas=_read_vector_mapping(
                tensors=tensors,
                source=partition_payload.get("classifier_head_weight_deltas", {}),
            ),
            classifier_head_bias_deltas=_read_scalar_mapping(
                tensors=tensors,
                source=partition_payload.get("classifier_head_bias_deltas", {}),
            ),
        )
    return result


def _add_vector_mapping_tensors(
    *,
    tensors: dict[str, Tensor],
    target: dict[str, str],
    prefix: str,
    values: Mapping[str, list[float]],
) -> None:
    for offset, (name, vector) in enumerate(sorted(values.items())):
        tensor_key = f"{prefix}.{offset:04d}"
        tensors[tensor_key] = torch.tensor(
            [float(value) for value in vector],
            dtype=torch.float32,
        )
        target[str(name)] = tensor_key


def _read_vector_mapping(
    *,
    tensors: Mapping[str, Tensor],
    source: object,
) -> dict[str, list[float]]:
    if not isinstance(source, dict):
        raise ValueError("Partitioned delta vector index must be an object.")
    return {
        str(name): _tensor_to_float_list(_require_tensor(tensors, str(tensor_key)))
        for name, tensor_key in sorted(source.items())
    }


def _read_scalar_mapping(
    *,
    tensors: Mapping[str, Tensor],
    source: object,
) -> dict[str, float]:
    if not isinstance(source, dict):
        raise ValueError("Partitioned delta scalar index must be an object.")
    result: dict[str, float] = {}
    for name, tensor_key in sorted(source.items()):
        values = _tensor_to_float_list(_require_tensor(tensors, str(tensor_key)))
        if len(values) != 1:
            raise ValueError("Partitioned delta scalar tensor must have one value.")
        result[str(name)] = values[0]
    return result


def _require_tensor(tensors: Mapping[str, Tensor], tensor_key: str) -> Tensor:
    tensor = tensors.get(tensor_key)
    if tensor is None:
        raise ValueError(f"Partitioned delta tensor is missing: {tensor_key}")
    return tensor


def _tensor_to_float_list(tensor: Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().reshape(-1).tolist()]
