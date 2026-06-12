"""Linear classifier head tensor artifact helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor

CLASSIFIER_HEAD_STATE_TENSOR_ARTIFACT_SCHEMA_VERSION = (
    "linear_classifier_head_state_tensor_artifact.v1"
)
CLASSIFIER_HEAD_STATE_TENSOR_ARTIFACT_FORMAT = "safetensors"
CLASSIFIER_HEAD_STATE_TENSOR_ARTIFACT_INDEX_METADATA_KEY = (
    "classifier_head_state_index_json"
)


@dataclass(frozen=True, slots=True)
class LinearClassifierHeadState:
    """Label-keyed linear classifier head parameters."""

    label_schema: tuple[str, ...]
    classifier_head_weights: dict[str, list[float]]
    classifier_head_biases: dict[str, float]

    @property
    def hidden_size(self) -> int:
        first_label = self.label_schema[0]
        return len(self.classifier_head_weights[first_label])

    def to_state_dict(self) -> dict[str, Tensor]:
        """PyTorch ``nn.Linear`` state_dict로 변환한다."""

        return {
            "weight": torch.tensor(
                [self.classifier_head_weights[label] for label in self.label_schema],
                dtype=torch.float32,
            ),
            "bias": torch.tensor(
                [self.classifier_head_biases[label] for label in self.label_schema],
                dtype=torch.float32,
            ),
        }


def classifier_state_dict_to_label_mappings(
    *,
    classifier_state_dict: Mapping[str, object],
    label_schema: Sequence[str],
) -> LinearClassifierHeadState:
    """PyTorch linear head state_dict를 label-keyed mapping으로 변환한다."""

    labels = _normalize_label_schema(label_schema)
    weight = _required_tensor(classifier_state_dict.get("weight"), field_name="weight")
    bias = _required_tensor(classifier_state_dict.get("bias"), field_name="bias")
    if weight.ndim != 2:
        raise ValueError("classifier head weight tensor must be 2D.")
    if bias.ndim != 1:
        raise ValueError("classifier head bias tensor must be 1D.")
    if weight.shape[0] != len(labels) or bias.shape[0] != len(labels):
        raise ValueError(
            "classifier head tensor label dimension does not match label schema."
        )
    return LinearClassifierHeadState(
        label_schema=labels,
        classifier_head_weights={
            label: [float(value) for value in weight[index].detach().cpu().tolist()]
            for index, label in enumerate(labels)
        },
        classifier_head_biases={
            label: float(bias[index].detach().cpu().item())
            for index, label in enumerate(labels)
        },
    )


def build_classifier_head_state_tensor_artifact(
    *,
    classifier_head_weights: Mapping[str, Sequence[float]],
    classifier_head_biases: Mapping[str, float],
    label_schema: Sequence[str] | None = None,
    applied_classifier_head_weight_deltas: Mapping[str, Sequence[float]] | None = None,
    applied_classifier_head_bias_deltas: Mapping[str, float] | None = None,
    partitioned_classifier_head_weights: (
        Mapping[str, Mapping[str, Sequence[float]]] | None
    ) = None,
    partitioned_classifier_head_biases: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[dict[str, Tensor], dict[str, str]]:
    """classifier head state를 safetensors payload와 metadata index로 변환한다."""

    labels = _resolve_label_schema(
        label_schema=label_schema,
        classifier_head_weights=classifier_head_weights,
        classifier_head_biases=classifier_head_biases,
    )
    tensors: dict[str, Tensor] = {}
    index: dict[str, object] = {
        "schema_version": CLASSIFIER_HEAD_STATE_TENSOR_ARTIFACT_SCHEMA_VERSION,
        "artifact_format": CLASSIFIER_HEAD_STATE_TENSOR_ARTIFACT_FORMAT,
        "label_schema": list(labels),
        "classifier_head_weights": {},
        "classifier_head_biases": {},
        "applied_classifier_head_weight_deltas": {},
        "applied_classifier_head_bias_deltas": {},
        "partitioned_classifier_head_weights": {},
        "partitioned_classifier_head_biases": {},
    }
    _add_vector_mapping_tensors(
        tensors=tensors,
        target=_required_dict(index["classifier_head_weights"]),
        prefix="head_state_weight",
        values={label: classifier_head_weights[label] for label in labels},
    )
    _add_scalar_mapping_tensors(
        tensors=tensors,
        target=_required_dict(index["classifier_head_biases"]),
        prefix="head_state_bias",
        values={label: classifier_head_biases[label] for label in labels},
    )
    if applied_classifier_head_weight_deltas:
        _add_vector_mapping_tensors(
            tensors=tensors,
            target=_required_dict(index["applied_classifier_head_weight_deltas"]),
            prefix="head_applied_weight_delta",
            values=applied_classifier_head_weight_deltas,
        )
    if applied_classifier_head_bias_deltas:
        _add_scalar_mapping_tensors(
            tensors=tensors,
            target=_required_dict(index["applied_classifier_head_bias_deltas"]),
            prefix="head_applied_bias_delta",
            values=applied_classifier_head_bias_deltas,
        )
    _add_partitioned_vector_mapping_tensors(
        tensors=tensors,
        target=_required_dict(index["partitioned_classifier_head_weights"]),
        prefix="head_partition_weight",
        values=partitioned_classifier_head_weights or {},
    )
    _add_partitioned_scalar_mapping_tensors(
        tensors=tensors,
        target=_required_dict(index["partitioned_classifier_head_biases"]),
        prefix="head_partition_bias",
        values=partitioned_classifier_head_biases or {},
    )
    return tensors, {
        CLASSIFIER_HEAD_STATE_TENSOR_ARTIFACT_INDEX_METADATA_KEY: json.dumps(
            index,
            ensure_ascii=True,
            sort_keys=True,
        )
    }


def save_classifier_head_state_tensor_artifact(
    *,
    path: Path,
    classifier_state_dict: Mapping[str, object],
    label_schema: Sequence[str],
) -> LinearClassifierHeadState:
    """PyTorch linear head state_dict를 safetensors artifact로 저장한다."""

    state = classifier_state_dict_to_label_mappings(
        classifier_state_dict=classifier_state_dict,
        label_schema=label_schema,
    )
    tensors, metadata = build_classifier_head_state_tensor_artifact(
        classifier_head_weights=state.classifier_head_weights,
        classifier_head_biases=state.classifier_head_biases,
        label_schema=state.label_schema,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path, metadata=metadata)
    return state


def load_classifier_head_state_tensor_artifact(
    path: str | Path,
) -> LinearClassifierHeadState:
    """safetensors 파일에서 classifier head state를 읽는다."""

    artifact_path = Path(path)
    with safe_open(artifact_path, framework="pt", device="cpu") as artifact:
        metadata = dict(artifact.metadata() or {})
    tensors = load_file(artifact_path, device="cpu")
    return parse_classifier_head_state_tensor_artifact(
        tensors=tensors,
        metadata=metadata,
    )


def parse_classifier_head_state_tensor_artifact(
    *,
    tensors: Mapping[str, Tensor],
    metadata: Mapping[str, str],
) -> LinearClassifierHeadState:
    """safetensors payload와 metadata index를 classifier head state로 복원한다."""

    index = _load_index(metadata)
    labels = _normalize_label_schema(index.get("label_schema", ()))
    weights = _read_vector_mapping(
        tensors=tensors,
        source=index.get("classifier_head_weights", {}),
        field_name="classifier_head_weights",
    )
    biases = _read_scalar_mapping(
        tensors=tensors,
        source=index.get("classifier_head_biases", {}),
        field_name="classifier_head_biases",
    )
    _require_labels_present(labels=labels, weights=weights, biases=biases)
    return LinearClassifierHeadState(
        label_schema=labels,
        classifier_head_weights={label: weights[label] for label in labels},
        classifier_head_biases={label: biases[label] for label in labels},
    )


def parse_applied_classifier_head_deltas_tensor_artifact(
    *,
    tensors: Mapping[str, Tensor],
    metadata: Mapping[str, str],
) -> tuple[dict[str, list[float]], dict[str, float]]:
    """safetensors state artifact에서 server-applied head delta를 복원한다."""

    index = _load_index(metadata)
    return (
        _read_vector_mapping(
            tensors=tensors,
            source=index.get("applied_classifier_head_weight_deltas", {}),
            field_name="applied_classifier_head_weight_deltas",
            allow_empty=True,
        ),
        _read_scalar_mapping(
            tensors=tensors,
            source=index.get("applied_classifier_head_bias_deltas", {}),
            field_name="applied_classifier_head_bias_deltas",
            allow_empty=True,
        ),
    )


def parse_partitioned_classifier_head_state_tensor_artifact(
    *,
    tensors: Mapping[str, Tensor],
    metadata: Mapping[str, str],
) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, float]]]:
    """safetensors state artifact에서 partition별 head state를 복원한다."""

    index = _load_index(metadata)
    return (
        _read_partitioned_vector_mapping(
            tensors=tensors,
            source=index.get("partitioned_classifier_head_weights", {}),
            field_name="partitioned_classifier_head_weights",
        ),
        _read_partitioned_scalar_mapping(
            tensors=tensors,
            source=index.get("partitioned_classifier_head_biases", {}),
            field_name="partitioned_classifier_head_biases",
        ),
    )


def _load_index(metadata: Mapping[str, str]) -> dict[str, object]:
    raw_index = metadata.get(CLASSIFIER_HEAD_STATE_TENSOR_ARTIFACT_INDEX_METADATA_KEY)
    if raw_index is None:
        metadata_key = CLASSIFIER_HEAD_STATE_TENSOR_ARTIFACT_INDEX_METADATA_KEY
        raise KeyError(f"Missing metadata key: {metadata_key}")
    index = json.loads(raw_index)
    if not isinstance(index, dict):
        raise ValueError("classifier head state index must be a JSON object.")
    if (
        index.get("schema_version")
        != CLASSIFIER_HEAD_STATE_TENSOR_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError(
            "Unsupported classifier head state tensor artifact schema: "
            f"{index.get('schema_version')!r}"
        )
    return index


def _resolve_label_schema(
    *,
    label_schema: Sequence[str] | None,
    classifier_head_weights: Mapping[str, Sequence[float]],
    classifier_head_biases: Mapping[str, float],
) -> tuple[str, ...]:
    labels = (
        _normalize_label_schema(label_schema)
        if label_schema is not None
        else tuple(sorted(str(label) for label in classifier_head_weights))
    )
    _require_labels_present(
        labels=labels,
        weights=classifier_head_weights,
        biases=classifier_head_biases,
    )
    return labels


def _require_labels_present(
    *,
    labels: Sequence[str],
    weights: Mapping[str, Sequence[float]],
    biases: Mapping[str, float],
) -> None:
    if not labels:
        raise ValueError("classifier head artifact requires label_schema.")
    missing_weights = [label for label in labels if label not in weights]
    missing_biases = [label for label in labels if label not in biases]
    if missing_weights or missing_biases:
        raise ValueError(
            "classifier head artifact missing labels: "
            f"weights={missing_weights}, biases={missing_biases}"
        )


def _normalize_label_schema(label_schema: Sequence[object]) -> tuple[str, ...]:
    labels = tuple(str(label).strip() for label in label_schema)
    if not labels or any(not label for label in labels):
        raise ValueError("classifier head label_schema must contain non-empty labels.")
    if len(set(labels)) != len(labels):
        raise ValueError("classifier head label_schema labels must be unique.")
    return labels


def _required_tensor(value: object, *, field_name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(
            f"classifier head state_dict must contain tensor {field_name}."
        )
    return value


def _required_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AssertionError("classifier head tensor artifact index must use dicts.")
    return value


def _add_vector_mapping_tensors(
    *,
    tensors: dict[str, Tensor],
    target: dict[str, object],
    prefix: str,
    values: Mapping[str, Sequence[float]],
) -> None:
    for offset, (label, vector) in enumerate(sorted(values.items())):
        tensor_key = f"{prefix}.{offset:04d}"
        tensors[tensor_key] = torch.tensor(
            [float(value) for value in vector],
            dtype=torch.float32,
        )
        target[str(label)] = tensor_key


def _add_scalar_mapping_tensors(
    *,
    tensors: dict[str, Tensor],
    target: dict[str, object],
    prefix: str,
    values: Mapping[str, float],
) -> None:
    for offset, (label, value) in enumerate(sorted(values.items())):
        tensor_key = f"{prefix}.{offset:04d}"
        tensors[tensor_key] = torch.tensor([float(value)], dtype=torch.float32)
        target[str(label)] = tensor_key


def _add_partitioned_vector_mapping_tensors(
    *,
    tensors: dict[str, Tensor],
    target: dict[str, object],
    prefix: str,
    values: Mapping[str, Mapping[str, Sequence[float]]],
) -> None:
    for partition_index, (partition_name, partition_values) in enumerate(
        sorted(values.items())
    ):
        partition_target: dict[str, object] = {}
        target[str(partition_name)] = partition_target
        _add_vector_mapping_tensors(
            tensors=tensors,
            target=partition_target,
            prefix=f"{prefix}.{partition_index:04d}",
            values=partition_values,
        )


def _add_partitioned_scalar_mapping_tensors(
    *,
    tensors: dict[str, Tensor],
    target: dict[str, object],
    prefix: str,
    values: Mapping[str, Mapping[str, float]],
) -> None:
    for partition_index, (partition_name, partition_values) in enumerate(
        sorted(values.items())
    ):
        partition_target: dict[str, object] = {}
        target[str(partition_name)] = partition_target
        _add_scalar_mapping_tensors(
            tensors=tensors,
            target=partition_target,
            prefix=f"{prefix}.{partition_index:04d}",
            values=partition_values,
        )


def _read_vector_mapping(
    *,
    tensors: Mapping[str, Tensor],
    source: object,
    field_name: str,
    allow_empty: bool = False,
) -> dict[str, list[float]]:
    if not isinstance(source, Mapping):
        raise ValueError(f"{field_name} index must be a mapping.")
    if not source:
        if allow_empty:
            return {}
        raise ValueError(f"{field_name} index must not be empty.")
    return {
        str(label): [
            float(value)
            for value in _required_artifact_tensor(
                tensors=tensors,
                tensor_key=str(tensor_key),
                field_name=field_name,
            )
            .detach()
            .cpu()
            .flatten()
            .tolist()
        ]
        for label, tensor_key in source.items()
    }


def _read_scalar_mapping(
    *,
    tensors: Mapping[str, Tensor],
    source: object,
    field_name: str,
    allow_empty: bool = False,
) -> dict[str, float]:
    if not isinstance(source, Mapping):
        raise ValueError(f"{field_name} index must be a mapping.")
    if not source:
        if allow_empty:
            return {}
        raise ValueError(f"{field_name} index must not be empty.")
    return {
        str(label): float(
            _required_artifact_tensor(
                tensors=tensors,
                tensor_key=str(tensor_key),
                field_name=field_name,
            )
            .detach()
            .cpu()
            .flatten()[0]
            .item()
        )
        for label, tensor_key in source.items()
    }


def _read_partitioned_vector_mapping(
    *,
    tensors: Mapping[str, Tensor],
    source: object,
    field_name: str,
) -> dict[str, dict[str, list[float]]]:
    if not isinstance(source, Mapping):
        raise ValueError(f"{field_name} index must be a mapping.")
    return {
        str(partition_name): _read_vector_mapping(
            tensors=tensors,
            source=partition_source,
            field_name=f"{field_name}.{partition_name}",
            allow_empty=True,
        )
        for partition_name, partition_source in source.items()
    }


def _read_partitioned_scalar_mapping(
    *,
    tensors: Mapping[str, Tensor],
    source: object,
    field_name: str,
) -> dict[str, dict[str, float]]:
    if not isinstance(source, Mapping):
        raise ValueError(f"{field_name} index must be a mapping.")
    return {
        str(partition_name): _read_scalar_mapping(
            tensors=tensors,
            source=partition_source,
            field_name=f"{field_name}.{partition_name}",
            allow_empty=True,
        )
        for partition_name, partition_source in source.items()
    }


def _required_artifact_tensor(
    *,
    tensors: Mapping[str, Tensor],
    tensor_key: str,
    field_name: str,
) -> Tensor:
    tensor = tensors.get(tensor_key)
    if tensor is None:
        raise KeyError(f"{field_name} tensor key not found: {tensor_key}")
    return tensor
