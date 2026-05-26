"""LoRA-classifier partitioned aggregation state helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from methods.adaptation.text_classifier.peft_encoder.update.partitioned_delta import (
    LoraClassifierPartitionDelta,
)

from ..peft_encoder.update.materialization import LoraClassifierMaterializedState


def merge_partitioned_lora_classifier_deltas(
    partitions: Mapping[str, LoraClassifierPartitionDelta],
) -> LoraClassifierPartitionDelta:
    """partition별 delta를 하나의 LoRA-classifier delta로 합친다.

    같은 parameter key가 여러 partition에 있으면 값을 element-wise로 더한다.
    """

    merged_lora: dict[str, list[float]] = {}
    merged_weights: dict[str, list[float]] = {}
    merged_biases: dict[str, float] = {}
    for partition in partitions.values():
        _merge_vector_mapping(merged_lora, partition.lora_parameter_deltas)
        _merge_vector_mapping(
            merged_weights,
            partition.classifier_head_weight_deltas,
        )
        for key, value in partition.classifier_head_bias_deltas.items():
            merged_biases[str(key)] = merged_biases.get(str(key), 0.0) + float(value)
    return LoraClassifierPartitionDelta(
        partition_name="merged",
        lora_parameter_deltas=merged_lora,
        classifier_head_weight_deltas=merged_weights,
        classifier_head_bias_deltas=merged_biases,
    )


def apply_lora_classifier_partition_delta_to_state(
    *,
    base_parameters: LoraClassifierMaterializedState,
    delta: LoraClassifierPartitionDelta,
) -> LoraClassifierMaterializedState:
    """base state에 merged partition delta를 적용한 materialized state를 만든다."""

    return LoraClassifierMaterializedState(
        lora_parameters=_apply_vector_mapping(
            base_parameters.lora_parameters,
            delta.lora_parameter_deltas,
        ),
        classifier_head_weights=_apply_vector_mapping(
            base_parameters.classifier_head_weights,
            delta.classifier_head_weight_deltas,
        ),
        classifier_head_biases={
            key: float(base_parameters.classifier_head_biases.get(key, 0.0))
            + float(delta.classifier_head_bias_deltas.get(key, 0.0))
            for key in sorted(
                set(base_parameters.classifier_head_biases)
                | set(delta.classifier_head_bias_deltas)
            )
        },
    )


def apply_lora_classifier_partition_deltas_to_partitioned_state(
    *,
    base_parameters: LoraClassifierMaterializedState,
    base_partition_parameters: Mapping[str, LoraClassifierMaterializedState],
    partition_deltas: Mapping[str, LoraClassifierPartitionDelta],
) -> dict[str, LoraClassifierMaterializedState]:
    """partition별 base state에 partition delta를 적용해 다음 partition state를 만든다.

    첫 partitioned round처럼 partition별 base artifact가 아직 없으면 merged global
    base를 해당 partition의 시작점으로 사용한다.
    """

    return {
        partition_name: apply_lora_classifier_partition_delta_to_state(
            base_parameters=base_partition_parameters.get(
                partition_name,
                base_parameters,
            ),
            delta=delta,
        )
        for partition_name, delta in sorted(partition_deltas.items())
    }


def split_lora_classifier_state_by_residual_factor(
    *,
    published_parameters: LoraClassifierMaterializedState,
    base_partition_name: str,
    residual_partition_name: str,
    residual_factor: float,
) -> dict[str, LoraClassifierMaterializedState]:
    """`published = base + residual`, `residual = base * factor`로 초기 분해한다."""

    if residual_factor < 0.0:
        raise ValueError("residual_factor must not be negative.")
    base_partition_name = str(base_partition_name).strip()
    residual_partition_name = str(residual_partition_name).strip()
    if not base_partition_name or not residual_partition_name:
        raise ValueError("partition names must not be empty.")
    if base_partition_name == residual_partition_name:
        raise ValueError("partition names must be different.")

    base_scale = 1.0 / (1.0 + residual_factor)
    residual_scale = residual_factor / (1.0 + residual_factor)
    return {
        base_partition_name: _scale_lora_classifier_state(
            published_parameters,
            scale=base_scale,
        ),
        residual_partition_name: _scale_lora_classifier_state(
            published_parameters,
            scale=residual_scale,
        ),
    }


def _apply_vector_mapping(
    base_values: Mapping[str, Sequence[float]],
    deltas: Mapping[str, Sequence[float]],
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for key in sorted(set(base_values) | set(deltas)):
        base_vector = [float(value) for value in base_values.get(key, [])]
        delta_vector = [float(value) for value in deltas.get(key, [])]
        if base_vector and delta_vector and len(base_vector) != len(delta_vector):
            raise ValueError(
                f"Partitioned LoRA-classifier delta dimension mismatch for {key!r}."
            )
        if not base_vector:
            result[key] = delta_vector
            continue
        if not delta_vector:
            result[key] = base_vector
            continue
        result[key] = [
            left + right for left, right in zip(base_vector, delta_vector, strict=True)
        ]
    return result


def _scale_lora_classifier_state(
    state: LoraClassifierMaterializedState,
    *,
    scale: float,
) -> LoraClassifierMaterializedState:
    return LoraClassifierMaterializedState(
        lora_parameters={
            key: [float(value) * scale for value in values]
            for key, values in state.lora_parameters.items()
        },
        classifier_head_weights={
            key: [float(value) * scale for value in values]
            for key, values in state.classifier_head_weights.items()
        },
        classifier_head_biases={
            key: float(value) * scale
            for key, value in state.classifier_head_biases.items()
        },
    )


def _merge_vector_mapping(
    target: dict[str, list[float]],
    source: Mapping[str, Sequence[float]],
) -> None:
    for key, values in source.items():
        vector = [float(value) for value in values]
        normalized_key = str(key)
        if normalized_key not in target:
            target[normalized_key] = vector
            continue
        if len(target[normalized_key]) != len(vector):
            raise ValueError(
                "Partitioned LoRA-classifier deltas must share dimensions per key."
            )
        target[normalized_key] = [
            left + right for left, right in zip(target[normalized_key], vector)
        ]
