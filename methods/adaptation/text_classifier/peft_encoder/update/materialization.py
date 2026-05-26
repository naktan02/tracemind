"""LoRA-classifier aggregation artifact materialization helpers."""

from __future__ import annotations

import math
from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from torch import Tensor

from methods.federated.aggregation.base import (
    AggregationJsonArtifactLoader,
    FederatedAggregationContext,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LoraClassifierDelta,
    LoraClassifierState,
)
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PeftClassifierDelta,
    PeftClassifierState,
)

from .merged_tensor_artifact import (
    parse_classifier_head_delta_tensor_artifact,
    parse_lora_delta_tensor_artifact,
)
from .partitioned_delta import LoraClassifierPartitionDelta
from .partitioned_tensor_artifact import parse_partitioned_delta_tensor_artifact

LORA_STATE_PARAMETERS_KEY = "lora_parameters"
PEFT_STATE_PARAMETERS_KEY = "peft_parameters"
CLASSIFIER_HEAD_STATE_WEIGHTS_KEY = "classifier_head_weights"
CLASSIFIER_HEAD_STATE_BIASES_KEY = "classifier_head_biases"
PARTITIONED_LORA_STATE_PARAMETERS_KEY = "partitioned_lora_parameters"
PARTITIONED_PEFT_STATE_PARAMETERS_KEY = "partitioned_peft_parameters"
PARTITIONED_CLASSIFIER_HEAD_STATE_WEIGHTS_KEY = "partitioned_classifier_head_weights"
PARTITIONED_CLASSIFIER_HEAD_STATE_BIASES_KEY = "partitioned_classifier_head_biases"
PeftEncoderStatePayload = LoraClassifierState | PeftClassifierState
PeftEncoderDeltaPayload = LoraClassifierDelta | PeftClassifierDelta


@dataclass(frozen=True, slots=True)
class LoraClassifierMaterializedUpdate:
    """LoRA-classifier aggregation core가 소비하는 materialized client delta."""

    lora_parameter_deltas: dict[str, list[float]]
    classifier_head_weight_deltas: dict[str, list[float]]
    classifier_head_bias_deltas: dict[str, float]
    delta_l2_norm: float


@dataclass(frozen=True, slots=True)
class LoraClassifierMaterializedState:
    """다음 global state projection이 소비하는 base global snapshot."""

    lora_parameters: dict[str, list[float]]
    classifier_head_weights: dict[str, list[float]]
    classifier_head_biases: dict[str, float]


def compact_lora_classifier_materialized_state(
    state: LoraClassifierMaterializedState,
) -> LoraClassifierMaterializedState:
    """simulation memory 보존용으로 vector payload를 float32 array로 압축한다."""

    return LoraClassifierMaterializedState(
        lora_parameters=cast(
            dict[str, list[float]],
            {
                str(key): array("f", (float(value) for value in values))
                for key, values in state.lora_parameters.items()
            },
        ),
        classifier_head_weights=cast(
            dict[str, list[float]],
            {
                str(key): array("f", (float(value) for value in values))
                for key, values in state.classifier_head_weights.items()
            },
        ),
        classifier_head_biases={
            str(key): float(value)
            for key, value in state.classifier_head_biases.items()
        },
    )


class _AggregationTensorArtifactLoader(Protocol):
    def load_safetensors_artifact(
        self,
        *,
        artifact_ref: str,
    ) -> tuple[Mapping[str, Tensor], Mapping[str, str]]:
        """opaque artifact ref를 safetensors tensor payload로 materialize한다."""


def materialize_base_lora_classifier_state(
    *,
    base_state: PeftEncoderStatePayload,
    context: FederatedAggregationContext,
) -> LoraClassifierMaterializedState:
    """base global state artifact를 읽고 없으면 zero-initialized state로 본다."""

    loader = None
    peft_adapter_artifact_ref = _peft_adapter_artifact_ref(base_state)
    if (
        peft_adapter_artifact_ref is not None
        or base_state.classifier_head_artifact_ref is not None
    ):
        loader = context.require_artifact_loader(
            context="LoRA-classifier base state materialization"
        )

    lora_parameters: dict[str, list[float]] = {}
    if peft_adapter_artifact_ref is not None:
        if loader is None:
            raise AssertionError("artifact loader must be resolved before load.")
        lora_parameters = _load_base_lora_parameters(
            artifact_ref=peft_adapter_artifact_ref,
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

    return LoraClassifierMaterializedState(
        lora_parameters=lora_parameters,
        classifier_head_weights=classifier_head_weights,
        classifier_head_biases=classifier_head_biases,
    )


def materialize_base_lora_classifier_partitioned_state(
    *,
    base_state: PeftEncoderStatePayload,
    context: FederatedAggregationContext,
) -> dict[str, LoraClassifierMaterializedState]:
    """server-published artifact에 저장된 partition별 global state를 읽는다.

    shared `LoraClassifierState` 계약은 merged LoRA/head artifact ref만 가진다.
    partitioned state는 그 artifact payload 안의 optional methods-owned metadata로
    보존하고, 없으면 아직 partitioned global state가 없는 것으로 본다.
    """

    peft_adapter_artifact_ref = _peft_adapter_artifact_ref(base_state)
    if (
        peft_adapter_artifact_ref is None
        and base_state.classifier_head_artifact_ref is None
    ):
        return {}
    loader = context.require_artifact_loader(
        context="LoRA-classifier partitioned base state materialization"
    )
    partitioned_lora_parameters: dict[str, dict[str, list[float]]] = {}
    if peft_adapter_artifact_ref is not None:
        lora_artifact = loader.load_json_artifact(
            artifact_ref=peft_adapter_artifact_ref
        )
        partitioned_lora_parameters = _normalize_partitioned_vector_mapping(
            lora_artifact.get(
                PARTITIONED_LORA_STATE_PARAMETERS_KEY,
                lora_artifact.get(PARTITIONED_PEFT_STATE_PARAMETERS_KEY, {}),
            ),
            field_name=PARTITIONED_LORA_STATE_PARAMETERS_KEY,
        )

    partitioned_head_weights: dict[str, dict[str, list[float]]] = {}
    partitioned_head_biases: dict[str, dict[str, float]] = {}
    if base_state.classifier_head_artifact_ref is not None:
        head_artifact = loader.load_json_artifact(
            artifact_ref=base_state.classifier_head_artifact_ref
        )
        partitioned_head_weights = _normalize_partitioned_vector_mapping(
            head_artifact.get(PARTITIONED_CLASSIFIER_HEAD_STATE_WEIGHTS_KEY, {}),
            field_name=PARTITIONED_CLASSIFIER_HEAD_STATE_WEIGHTS_KEY,
        )
        partitioned_head_biases = _normalize_partitioned_scalar_mapping(
            head_artifact.get(PARTITIONED_CLASSIFIER_HEAD_STATE_BIASES_KEY, {}),
            field_name=PARTITIONED_CLASSIFIER_HEAD_STATE_BIASES_KEY,
        )

    partition_names = sorted(
        set(partitioned_lora_parameters)
        | set(partitioned_head_weights)
        | set(partitioned_head_biases)
    )
    return {
        partition_name: LoraClassifierMaterializedState(
            lora_parameters=partitioned_lora_parameters.get(partition_name, {}),
            classifier_head_weights=partitioned_head_weights.get(partition_name, {}),
            classifier_head_biases=partitioned_head_biases.get(partition_name, {}),
        )
        for partition_name in partition_names
    }


def materialize_lora_classifier_update(
    *,
    payload: PeftEncoderDeltaPayload,
    context: FederatedAggregationContext,
) -> LoraClassifierMaterializedUpdate:
    """inline delta 또는 server-owned artifact ref update를 delta mapping으로 읽는다."""

    loader = None
    peft_parameter_deltas = _peft_parameter_deltas(payload)
    if peft_parameter_deltas is None or payload.classifier_head_weight_deltas is None:
        loader = context.require_artifact_loader(
            context="LoRA-classifier aggregation materialization"
        )

    if peft_parameter_deltas is not None:
        lora_parameter_deltas = _normalize_vector_mapping(
            peft_parameter_deltas,
            field_name=_peft_parameter_deltas_field_name(payload),
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

    return LoraClassifierMaterializedUpdate(
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


def materialize_lora_classifier_partitioned_update(
    *,
    payload: PeftEncoderDeltaPayload,
    context: FederatedAggregationContext | None = None,
) -> dict[str, LoraClassifierPartitionDelta]:
    """shared payload의 partitioned delta를 methods-owned delta object로 읽는다."""

    partitioned_deltas = payload.partitioned_deltas
    if partitioned_deltas is None and payload.partitioned_deltas_artifact_ref:
        if context is None:
            raise ValueError(
                "LoRA-classifier partitioned artifact materialization requires an "
                "aggregation context."
            )
        loader = context.require_artifact_loader(
            context="LoRA-classifier partitioned aggregation materialization"
        )
        tensor_partitions = _try_load_partitioned_tensor_artifact(
            loader=loader,
            artifact_ref=payload.partitioned_deltas_artifact_ref,
        )
        if tensor_partitions is not None:
            return tensor_partitions
        artifact = loader.load_json_artifact(
            artifact_ref=payload.partitioned_deltas_artifact_ref
        )
        source = artifact.get("partitions", artifact)
        if not isinstance(source, Mapping):
            raise ValueError(
                "LoRA-classifier partitioned delta artifact must contain a "
                "mapping payload."
            )
        partitioned_deltas = source

    if partitioned_deltas is None:
        raise ValueError(
            "LoRA-classifier partitioned aggregation requires partitioned_deltas "
            "or partitioned_deltas_artifact_ref."
        )
    partitions: dict[str, LoraClassifierPartitionDelta] = {}
    for partition_name, partition in partitioned_deltas.items():
        if isinstance(partition, Mapping):
            lora_parameter_deltas = partition.get(
                "lora_parameter_deltas",
                partition.get("peft_parameter_deltas", {}),
            )
            classifier_head_weight_deltas = partition.get(
                "classifier_head_weight_deltas",
                {},
            )
            classifier_head_bias_deltas = partition.get(
                "classifier_head_bias_deltas",
                {},
            )
        else:
            lora_parameter_deltas = _partition_peft_parameter_deltas(partition)
            classifier_head_weight_deltas = partition.classifier_head_weight_deltas
            classifier_head_bias_deltas = partition.classifier_head_bias_deltas
        partitions[partition_name] = LoraClassifierPartitionDelta(
            partition_name=partition_name,
            lora_parameter_deltas=_normalize_optional_vector_mapping(
                lora_parameter_deltas,
                field_name=(
                    f"partitioned_deltas.{partition_name}.lora_parameter_deltas"
                ),
            ),
            classifier_head_weight_deltas=_normalize_optional_vector_mapping(
                classifier_head_weight_deltas,
                field_name=(
                    f"partitioned_deltas.{partition_name}.classifier_head_weight_deltas"
                ),
            ),
            classifier_head_bias_deltas=_normalize_scalar_mapping(
                classifier_head_bias_deltas,
                field_name=(
                    f"partitioned_deltas.{partition_name}.classifier_head_bias_deltas"
                ),
            ),
        )
    return partitions


def _try_load_partitioned_tensor_artifact(
    *,
    loader: AggregationJsonArtifactLoader,
    artifact_ref: str,
) -> dict[str, LoraClassifierPartitionDelta] | None:
    tensor_loader = getattr(loader, "load_safetensors_artifact", None)
    if tensor_loader is None:
        return None
    tensor_artifact_loader = cast(_AggregationTensorArtifactLoader, loader)
    try:
        tensors, metadata = tensor_artifact_loader.load_safetensors_artifact(
            artifact_ref=artifact_ref
        )
    except FileNotFoundError:
        return None
    return parse_partitioned_delta_tensor_artifact(
        tensors=tensors,
        metadata=metadata,
    )


def _load_lora_parameter_deltas(
    *,
    payload: PeftEncoderDeltaPayload,
    loader: AggregationJsonArtifactLoader,
) -> dict[str, list[float]]:
    artifact_ref = _peft_adapter_delta_artifact_ref(payload)
    if artifact_ref is None:
        raise ValueError(
            "PEFT-classifier artifact materialization requires adapter delta "
            "artifact ref."
        )
    tensor_deltas = _try_load_lora_delta_tensor_artifact(
        loader=loader,
        artifact_ref=artifact_ref,
    )
    if tensor_deltas is not None:
        return tensor_deltas
    artifact = loader.load_json_artifact(artifact_ref=artifact_ref)
    source = artifact.get(
        "lora_parameter_deltas",
        artifact.get("peft_parameter_deltas", artifact),
    )
    return _normalize_vector_mapping(
        source,
        field_name=_peft_parameter_deltas_field_name(payload),
    )


def _load_classifier_head_artifact(
    *,
    payload: PeftEncoderDeltaPayload,
    loader: AggregationJsonArtifactLoader,
) -> tuple[dict[str, list[float]], dict[str, float]]:
    if payload.classifier_head_delta_artifact_ref is None:
        raise ValueError(
            "LoRA-classifier artifact materialization requires "
            "classifier_head_delta_artifact_ref."
        )
    tensor_artifact = _try_load_classifier_head_tensor_artifact(
        loader=loader,
        artifact_ref=payload.classifier_head_delta_artifact_ref,
    )
    if tensor_artifact is not None:
        return tensor_artifact
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


def _try_load_lora_delta_tensor_artifact(
    *,
    loader: AggregationJsonArtifactLoader,
    artifact_ref: str,
) -> dict[str, list[float]] | None:
    tensor_loader = getattr(loader, "load_safetensors_artifact", None)
    if tensor_loader is None:
        return None
    tensor_artifact_loader = cast(_AggregationTensorArtifactLoader, loader)
    try:
        tensors, metadata = tensor_artifact_loader.load_safetensors_artifact(
            artifact_ref=artifact_ref
        )
    except FileNotFoundError:
        return None
    return parse_lora_delta_tensor_artifact(
        tensors=tensors,
        metadata=metadata,
    )


def _try_load_classifier_head_tensor_artifact(
    *,
    loader: AggregationJsonArtifactLoader,
    artifact_ref: str,
) -> tuple[dict[str, list[float]], dict[str, float]] | None:
    tensor_loader = getattr(loader, "load_safetensors_artifact", None)
    if tensor_loader is None:
        return None
    tensor_artifact_loader = cast(_AggregationTensorArtifactLoader, loader)
    try:
        tensors, metadata = tensor_artifact_loader.load_safetensors_artifact(
            artifact_ref=artifact_ref
        )
    except FileNotFoundError:
        return None
    return parse_classifier_head_delta_tensor_artifact(
        tensors=tensors,
        metadata=metadata,
    )


def _load_base_lora_parameters(
    *,
    artifact_ref: str,
    loader: AggregationJsonArtifactLoader,
) -> dict[str, list[float]]:
    artifact = loader.load_json_artifact(artifact_ref=artifact_ref)
    source = artifact.get(
        LORA_STATE_PARAMETERS_KEY,
        artifact.get(
            PEFT_STATE_PARAMETERS_KEY,
            artifact.get("lora_parameter_deltas", artifact),
        ),
    )
    return _normalize_vector_mapping(source, field_name=LORA_STATE_PARAMETERS_KEY)


def _peft_adapter_artifact_ref(
    state: PeftEncoderStatePayload,
) -> str | None:
    if isinstance(state, PeftClassifierState):
        return state.peft_adapter_artifact_ref
    return state.lora_adapter_artifact_ref


def _peft_adapter_delta_artifact_ref(
    payload: PeftEncoderDeltaPayload,
) -> str | None:
    if isinstance(payload, PeftClassifierDelta):
        return payload.peft_adapter_delta_artifact_ref
    return payload.lora_delta_artifact_ref


def _peft_parameter_deltas(
    payload: PeftEncoderDeltaPayload,
) -> dict[str, list[float]] | None:
    if isinstance(payload, PeftClassifierDelta):
        return payload.peft_parameter_deltas
    return payload.lora_parameter_deltas


def _partition_peft_parameter_deltas(partition: object) -> object:
    if hasattr(partition, "lora_parameter_deltas"):
        return partition.lora_parameter_deltas
    return getattr(partition, "peft_parameter_deltas")


def _peft_parameter_deltas_field_name(payload: PeftEncoderDeltaPayload) -> str:
    if isinstance(payload, PeftClassifierDelta):
        return "peft_parameter_deltas"
    return "lora_parameter_deltas"


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


def _normalize_optional_vector_mapping(
    source: object,
    *,
    field_name: str,
) -> dict[str, list[float]]:
    if source == {}:
        return {}
    return _normalize_vector_mapping(source, field_name=field_name)


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


def _normalize_partitioned_vector_mapping(
    source: object,
    *,
    field_name: str,
) -> dict[str, dict[str, list[float]]]:
    if source == {}:
        return {}
    if not isinstance(source, Mapping):
        raise ValueError(f"{field_name} artifact must be a mapping.")
    return {
        str(partition_name): _normalize_vector_mapping(
            values,
            field_name=f"{field_name}.{partition_name}",
        )
        for partition_name, values in source.items()
    }


def _normalize_partitioned_scalar_mapping(
    source: object,
    *,
    field_name: str,
) -> dict[str, dict[str, float]]:
    if source == {}:
        return {}
    if not isinstance(source, Mapping):
        raise ValueError(f"{field_name} artifact must be a mapping.")
    return {
        str(partition_name): _normalize_scalar_mapping(
            values,
            field_name=f"{field_name}.{partition_name}",
        )
        for partition_name, values in source.items()
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
