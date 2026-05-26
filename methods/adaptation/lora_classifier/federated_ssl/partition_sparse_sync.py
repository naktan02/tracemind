"""Partitioned adapter/head sparse sync helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from methods.adaptation.lora_classifier.aggregation.materialization import (
    LoraClassifierMaterializedState,
)
from methods.adaptation.lora_classifier.update.partitioned_delta import (
    LoraClassifierPartitionDelta,
)


@dataclass(frozen=True, slots=True)
class PartitionSparseSyncParameters:
    """partitioned C2S/S2C sparse sync threshold surface."""

    l1_threshold: float
    delta_threshold: float
    l1_sparse_partitions: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.l1_threshold < 0.0:
            raise ValueError("l1_threshold must not be negative.")
        if self.delta_threshold < 0.0:
            raise ValueError("delta_threshold must not be negative.")

    @classmethod
    def from_mapping(
        cls,
        parameters: Mapping[str, object],
        *,
        l1_sparse_partitions: Sequence[str],
    ) -> PartitionSparseSyncParameters:
        """method effective parameter mapping에서 sparse sync 설정을 읽는다."""

        return cls(
            l1_threshold=float(parameters["l1_threshold"]),
            delta_threshold=float(parameters["delta_threshold"]),
            l1_sparse_partitions=tuple(str(name) for name in l1_sparse_partitions),
        )


def apply_partitioned_c2s_sparse_upload(
    *,
    base_parameters: LoraClassifierMaterializedState,
    base_partition_parameters: Mapping[str, LoraClassifierMaterializedState],
    partition_deltas: Mapping[str, LoraClassifierPartitionDelta],
    parameters: PartitionSparseSyncParameters,
) -> dict[str, LoraClassifierPartitionDelta]:
    """원본 FedMatch `cal_c2s`처럼 의미 있게 변한 partition update만 남긴다.

    `psi`처럼 L1 sparse partition으로 지정된 partition은 먼저
    `sparsify(base + delta)`를 만든 뒤 server base와의 차이를 `delta_threshold`로
    자른다. 그 외 partition은 `delta` 자체를 `delta_threshold`로 자른다.
    """

    sparse_partitions = set(parameters.l1_sparse_partitions)
    return {
        partition_name: _apply_sparse_upload_to_partition(
            base_parameters=base_partition_parameters.get(
                partition_name,
                base_parameters,
            ),
            delta=delta,
            l1_sparse=partition_name in sparse_partitions,
            parameters=parameters,
        )
        for partition_name, delta in sorted(partition_deltas.items())
    }


def _apply_sparse_upload_to_partition(
    *,
    base_parameters: LoraClassifierMaterializedState,
    delta: LoraClassifierPartitionDelta,
    l1_sparse: bool,
    parameters: PartitionSparseSyncParameters,
) -> LoraClassifierPartitionDelta:
    return LoraClassifierPartitionDelta(
        partition_name=delta.partition_name,
        lora_parameter_deltas=_sparse_vector_mapping(
            base_values=base_parameters.lora_parameters,
            deltas=delta.lora_parameter_deltas,
            l1_sparse=l1_sparse,
            parameters=parameters,
        ),
        classifier_head_weight_deltas=_sparse_vector_mapping(
            base_values=base_parameters.classifier_head_weights,
            deltas=delta.classifier_head_weight_deltas,
            l1_sparse=l1_sparse,
            parameters=parameters,
        ),
        classifier_head_bias_deltas=_sparse_scalar_mapping(
            base_values=base_parameters.classifier_head_biases,
            deltas=delta.classifier_head_bias_deltas,
            l1_sparse=l1_sparse,
            parameters=parameters,
        ),
    )


def _sparse_vector_mapping(
    *,
    base_values: Mapping[str, Sequence[float]],
    deltas: Mapping[str, Sequence[float]],
    l1_sparse: bool,
    parameters: PartitionSparseSyncParameters,
) -> dict[str, list[float]]:
    return {
        key: _sparse_vector_delta(
            base_vector=[float(value) for value in base_values.get(key, [])],
            delta_vector=[float(value) for value in values],
            l1_sparse=l1_sparse,
            parameters=parameters,
            key=key,
        )
        for key, values in sorted(deltas.items())
    }


def _sparse_scalar_mapping(
    *,
    base_values: Mapping[str, float],
    deltas: Mapping[str, float],
    l1_sparse: bool,
    parameters: PartitionSparseSyncParameters,
) -> dict[str, float]:
    return {
        key: _sparse_scalar_delta(
            base_value=float(base_values.get(key, 0.0)),
            delta_value=float(value),
            l1_sparse=l1_sparse,
            parameters=parameters,
        )
        for key, value in sorted(deltas.items())
    }


def _sparse_vector_delta(
    *,
    base_vector: Sequence[float],
    delta_vector: Sequence[float],
    l1_sparse: bool,
    parameters: PartitionSparseSyncParameters,
    key: str,
) -> list[float]:
    if base_vector and len(base_vector) != len(delta_vector):
        raise ValueError(f"partition sparse sync dimension mismatch for {key!r}.")
    if not base_vector:
        base_vector = [0.0 for _value in delta_vector]
    return [
        _sparse_scalar_delta(
            base_value=base,
            delta_value=delta,
            l1_sparse=l1_sparse,
            parameters=parameters,
        )
        for base, delta in zip(base_vector, delta_vector, strict=True)
    ]


def _sparse_scalar_delta(
    *,
    base_value: float,
    delta_value: float,
    l1_sparse: bool,
    parameters: PartitionSparseSyncParameters,
) -> float:
    candidate = base_value + delta_value
    if l1_sparse and abs(candidate) <= parameters.l1_threshold:
        candidate = 0.0
    sparse_delta = candidate - base_value if l1_sparse else delta_value
    if abs(sparse_delta) <= parameters.delta_threshold:
        return 0.0
    return sparse_delta
