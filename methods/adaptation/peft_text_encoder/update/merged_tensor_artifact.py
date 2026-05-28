"""PEFT encoder merged delta tensor artifact helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

MERGED_PEFT_ADAPTER_DELTA_TENSOR_ARTIFACT_SCHEMA_VERSION = (
    "peft_encoder_client_merged_adapter_delta_tensor_artifact.v1"
)
MERGED_HEAD_DELTA_TENSOR_ARTIFACT_SCHEMA_VERSION = (
    "peft_encoder_client_merged_head_delta_tensor_artifact.v1"
)
MERGED_DELTA_TENSOR_ARTIFACT_FORMAT = "safetensors"
PEFT_ADAPTER_DELTA_TENSOR_ARTIFACT_INDEX_METADATA_KEY = "peft_adapter_delta_index_json"
HEAD_DELTA_TENSOR_ARTIFACT_INDEX_METADATA_KEY = "head_delta_index_json"


def build_peft_adapter_delta_tensor_artifact(
    peft_parameter_deltas: Mapping[str, Sequence[float]],
) -> tuple[dict[str, Tensor], dict[str, str]]:
    """merged PEFT adapter delta를 safetensors payload와 metadata index로 변환한다."""

    if not peft_parameter_deltas:
        raise ValueError("PEFT adapter delta tensor artifact requires deltas.")
    tensors: dict[str, Tensor] = {}
    index: dict[str, object] = {
        "schema_version": MERGED_PEFT_ADAPTER_DELTA_TENSOR_ARTIFACT_SCHEMA_VERSION,
        "artifact_format": MERGED_DELTA_TENSOR_ARTIFACT_FORMAT,
        "peft_parameter_deltas": {},
    }
    _add_vector_mapping_tensors(
        tensors=tensors,
        target=index["peft_parameter_deltas"],
        prefix="peft_adapter",
        values=peft_parameter_deltas,
    )
    return tensors, {
        PEFT_ADAPTER_DELTA_TENSOR_ARTIFACT_INDEX_METADATA_KEY: json.dumps(
            index,
            ensure_ascii=True,
            sort_keys=True,
        )
    }


def parse_peft_adapter_delta_tensor_artifact(
    *,
    tensors: Mapping[str, Tensor],
    metadata: Mapping[str, str],
) -> dict[str, list[float]]:
    """safetensors payload와 metadata index를 merged PEFT adapter delta로 복원한다."""

    index = _load_index(
        metadata=metadata,
        metadata_key=PEFT_ADAPTER_DELTA_TENSOR_ARTIFACT_INDEX_METADATA_KEY,
        schema_version=MERGED_PEFT_ADAPTER_DELTA_TENSOR_ARTIFACT_SCHEMA_VERSION,
        artifact_name="PEFT adapter delta",
    )
    return _read_vector_mapping(
        tensors=tensors,
        source=index.get("peft_parameter_deltas", {}),
        artifact_name="PEFT adapter delta",
    )


def build_classifier_head_delta_tensor_artifact(
    *,
    classifier_head_weight_deltas: Mapping[str, Sequence[float]],
    classifier_head_bias_deltas: Mapping[str, float],
) -> tuple[dict[str, Tensor], dict[str, str]]:
    """merged classifier-head delta를 safetensors payload와 index로 변환한다."""

    if not classifier_head_weight_deltas:
        raise ValueError("Classifier-head tensor artifact requires weight deltas.")
    tensors: dict[str, Tensor] = {}
    index: dict[str, object] = {
        "schema_version": MERGED_HEAD_DELTA_TENSOR_ARTIFACT_SCHEMA_VERSION,
        "artifact_format": MERGED_DELTA_TENSOR_ARTIFACT_FORMAT,
        "classifier_head_weight_deltas": {},
        "classifier_head_bias_deltas": {},
    }
    _add_vector_mapping_tensors(
        tensors=tensors,
        target=index["classifier_head_weight_deltas"],
        prefix="head_weight",
        values=classifier_head_weight_deltas,
    )
    bias_index = index["classifier_head_bias_deltas"]
    if not isinstance(bias_index, dict):
        raise AssertionError("classifier_head_bias_deltas index must be a dict.")
    for offset, (label, value) in enumerate(
        sorted(classifier_head_bias_deltas.items())
    ):
        tensor_key = f"head_bias.{offset:04d}"
        tensors[tensor_key] = torch.tensor([float(value)], dtype=torch.float32)
        bias_index[str(label)] = tensor_key
    return tensors, {
        HEAD_DELTA_TENSOR_ARTIFACT_INDEX_METADATA_KEY: json.dumps(
            index,
            ensure_ascii=True,
            sort_keys=True,
        )
    }


def parse_classifier_head_delta_tensor_artifact(
    *,
    tensors: Mapping[str, Tensor],
    metadata: Mapping[str, str],
) -> tuple[dict[str, list[float]], dict[str, float]]:
    """safetensors payload와 index를 merged classifier-head delta로 복원한다."""

    index = _load_index(
        metadata=metadata,
        metadata_key=HEAD_DELTA_TENSOR_ARTIFACT_INDEX_METADATA_KEY,
        schema_version=MERGED_HEAD_DELTA_TENSOR_ARTIFACT_SCHEMA_VERSION,
        artifact_name="Classifier-head delta",
    )
    return (
        _read_vector_mapping(
            tensors=tensors,
            source=index.get("classifier_head_weight_deltas", {}),
            artifact_name="Classifier-head weight delta",
        ),
        _read_scalar_mapping(
            tensors=tensors,
            source=index.get("classifier_head_bias_deltas", {}),
            artifact_name="Classifier-head bias delta",
        ),
    )


def _add_vector_mapping_tensors(
    *,
    tensors: dict[str, Tensor],
    target: object,
    prefix: str,
    values: Mapping[str, Sequence[float]],
) -> None:
    if not isinstance(target, dict):
        raise AssertionError("tensor index target must be a dict.")
    for offset, (name, vector) in enumerate(sorted(values.items())):
        tensor_key = f"{prefix}.{offset:04d}"
        tensors[tensor_key] = torch.tensor(
            [float(value) for value in vector],
            dtype=torch.float32,
        )
        target[str(name)] = tensor_key


def _load_index(
    *,
    metadata: Mapping[str, str],
    metadata_key: str,
    schema_version: str,
    artifact_name: str,
) -> dict[str, object]:
    raw_index = metadata.get(metadata_key)
    if raw_index is None:
        raise ValueError(f"{artifact_name} tensor artifact metadata is missing.")
    index = json.loads(raw_index)
    if not isinstance(index, dict):
        raise ValueError(f"{artifact_name} tensor artifact index must be an object.")
    if index.get("schema_version") != schema_version:
        raise ValueError(
            f"Unsupported {artifact_name} tensor artifact schema_version: "
            f"{index.get('schema_version')!r}."
        )
    return index


def _read_vector_mapping(
    *,
    tensors: Mapping[str, Tensor],
    source: object,
    artifact_name: str,
) -> dict[str, list[float]]:
    if not isinstance(source, dict) or not source:
        raise ValueError(f"{artifact_name} vector index must be a non-empty object.")
    return {
        str(name): _tensor_to_float_list(_require_tensor(tensors, str(tensor_key)))
        for name, tensor_key in sorted(source.items())
    }


def _read_scalar_mapping(
    *,
    tensors: Mapping[str, Tensor],
    source: object,
    artifact_name: str,
) -> dict[str, float]:
    if not isinstance(source, dict):
        raise ValueError(f"{artifact_name} scalar index must be an object.")
    result: dict[str, float] = {}
    for name, tensor_key in sorted(source.items()):
        values = _tensor_to_float_list(_require_tensor(tensors, str(tensor_key)))
        if len(values) != 1:
            raise ValueError(f"{artifact_name} scalar tensor must have one value.")
        result[str(name)] = values[0]
    return result


def _require_tensor(tensors: Mapping[str, Tensor], tensor_key: str) -> Tensor:
    tensor = tensors.get(tensor_key)
    if tensor is None:
        raise ValueError(f"Merged delta tensor is missing: {tensor_key}")
    return tensor


def _tensor_to_float_list(tensor: Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().reshape(-1).tolist()]
