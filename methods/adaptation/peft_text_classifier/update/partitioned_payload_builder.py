"""LoRA-classifier partitioned payload builder helpers."""

from __future__ import annotations

from collections.abc import Mapping

from methods.adaptation.peft_text_classifier.update.partitioned_delta import (
    LoraClassifierPartitionDelta,
    normalize_partition_deltas,
)


def build_partitioned_delta_payload(
    deltas: tuple[LoraClassifierPartitionDelta, ...],
) -> dict[str, object]:
    """partitioned delta를 artifact/diagnostics가 쓰기 쉬운 JSON shape로 바꾼다."""

    normalized = normalize_partition_deltas(deltas)
    return {
        "partitions": {
            name: _partition_to_payload(delta)
            for name, delta in sorted(normalized.items())
        }
    }


def _partition_to_payload(delta: LoraClassifierPartitionDelta) -> Mapping[str, object]:
    return {
        "lora_parameter_deltas": {
            key: list(values)
            for key, values in sorted(delta.lora_parameter_deltas.items())
        },
        "classifier_head_weight_deltas": {
            key: list(values)
            for key, values in sorted(delta.classifier_head_weight_deltas.items())
        },
        "classifier_head_bias_deltas": dict(
            sorted(delta.classifier_head_bias_deltas.items())
        ),
    }
