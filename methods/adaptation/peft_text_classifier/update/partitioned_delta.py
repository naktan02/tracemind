"""PEFT encoder partitioned delta mechanism helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class PeftEncoderPartitionDelta:
    """하나의 partition에 속한 PEFT adapter/head delta 묶음."""

    partition_name: str
    lora_parameter_deltas: Mapping[str, Sequence[float]] = field(default_factory=dict)
    classifier_head_weight_deltas: Mapping[str, Sequence[float]] = field(
        default_factory=dict
    )
    classifier_head_bias_deltas: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.partition_name.strip():
            raise ValueError("partition_name must not be empty.")


def normalize_partition_deltas(
    deltas: Sequence[PeftEncoderPartitionDelta],
) -> dict[str, PeftEncoderPartitionDelta]:
    """partition 이름 중복을 막고 name -> delta mapping으로 정규화한다."""

    normalized: dict[str, PeftEncoderPartitionDelta] = {}
    for delta in deltas:
        name = delta.partition_name.strip()
        if name in normalized:
            raise ValueError(f"Duplicate PEFT encoder partition delta: {name}")
        normalized[name] = delta
    return normalized
