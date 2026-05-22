"""LoRA-classifier partitioned aggregation state helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from methods.adaptation.lora_classifier.update.partitioned_delta import (
    LoraClassifierPartitionDelta,
)


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
