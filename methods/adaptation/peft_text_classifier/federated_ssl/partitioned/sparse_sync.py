"""Partitioned adapter/head sparse sync helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ...update.materialization import PeftEncoderMaterializedState
from ...update.partitioned_delta import PeftEncoderPartitionDelta


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


@dataclass(frozen=True, slots=True)
class PartitionSparseUploadProjection:
    """C2S sparse upload payload와 upload 후 client-visible partition state."""

    upload_partition_deltas: dict[str, PeftEncoderPartitionDelta]
    client_partition_parameters: dict[str, PeftEncoderMaterializedState]


def count_partition_delta_nonzero_values(
    partition_deltas: Mapping[str, PeftEncoderPartitionDelta],
) -> int:
    """partition delta payload 안에서 실제 transport되는 non-zero scalar 수를 센다."""

    return sum(
        _partition_delta_nonzero_count(delta) for delta in partition_deltas.values()
    )


def apply_partitioned_c2s_sparse_upload(
    *,
    base_parameters: PeftEncoderMaterializedState,
    base_partition_parameters: Mapping[str, PeftEncoderMaterializedState],
    partition_deltas: Mapping[str, PeftEncoderPartitionDelta],
    parameters: PartitionSparseSyncParameters,
) -> dict[str, PeftEncoderPartitionDelta]:
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


def project_partitioned_c2s_sparse_upload(
    *,
    base_parameters: PeftEncoderMaterializedState,
    server_partition_parameters: Mapping[str, PeftEncoderMaterializedState],
    client_partition_parameters: Mapping[str, PeftEncoderMaterializedState],
    parameters: PartitionSparseSyncParameters,
) -> PartitionSparseUploadProjection:
    """client final state를 sparse C2S payload와 post-upload state로 투영한다."""

    sparse_partitions = set(parameters.l1_sparse_partitions)
    upload_deltas: dict[str, PeftEncoderPartitionDelta] = {}
    projected_client_parameters: dict[str, PeftEncoderMaterializedState] = {}
    for partition_name, client_partition in sorted(client_partition_parameters.items()):
        server_partition = server_partition_parameters.get(
            partition_name,
            base_parameters,
        )
        upload_delta = _build_sparse_upload_partition_delta_from_states(
            partition_name=partition_name,
            server_parameters=server_partition,
            client_parameters=client_partition,
            l1_sparse=partition_name in sparse_partitions,
            parameters=parameters,
        )
        upload_deltas[partition_name] = upload_delta
        projected_client_parameters[partition_name] = _apply_upload_delta_to_state(
            server_parameters=server_partition,
            upload_delta=upload_delta,
        )
    return PartitionSparseUploadProjection(
        upload_partition_deltas=upload_deltas,
        client_partition_parameters=projected_client_parameters,
    )


def apply_partitioned_s2c_sparse_download(
    *,
    server_partition_parameters: Mapping[str, PeftEncoderMaterializedState],
    client_partition_parameters: Mapping[str, PeftEncoderMaterializedState],
    parameters: PartitionSparseSyncParameters,
) -> dict[str, PeftEncoderPartitionDelta]:
    """원본 FedMatch `cal_s2c`처럼 server-client sparse diff를 만든다.

    반환 delta는 client partition state에 적용할 S2C transport payload다. `psi`
    같은 L1 sparse partition은 server/client 양쪽 값을 먼저 sparsify한 뒤
    `delta_threshold`로 의미 있게 달라진 원소만 남긴다.
    """

    sparse_partitions = set(parameters.l1_sparse_partitions)
    return {
        partition_name: _build_sparse_download_partition_delta(
            partition_name=partition_name,
            server_parameters=server_partition,
            client_parameters=client_partition_parameters.get(
                partition_name,
                PeftEncoderMaterializedState(
                    lora_parameters={},
                    classifier_head_weights={},
                    classifier_head_biases={},
                ),
            ),
            l1_sparse=partition_name in sparse_partitions,
            parameters=parameters,
        )
        for partition_name, server_partition in sorted(
            server_partition_parameters.items()
        )
    }


def project_partitioned_s2c_sparse_download(
    *,
    server_partition_parameters: Mapping[str, PeftEncoderMaterializedState],
    client_partition_parameters: Mapping[str, PeftEncoderMaterializedState],
    parameters: PartitionSparseSyncParameters,
) -> dict[str, PeftEncoderMaterializedState]:
    """S2C sparse mask로 client partition state를 raw server 값으로 부분 갱신한다."""

    sparse_partitions = set(parameters.l1_sparse_partitions)
    return {
        partition_name: _project_sparse_download_partition_state(
            partition_name=partition_name,
            server_parameters=server_partition,
            client_parameters=client_partition_parameters.get(
                partition_name,
                PeftEncoderMaterializedState(
                    lora_parameters={},
                    classifier_head_weights={},
                    classifier_head_biases={},
                ),
            ),
            l1_sparse=partition_name in sparse_partitions,
            parameters=parameters,
        )
        for partition_name, server_partition in sorted(
            server_partition_parameters.items()
        )
    }


def _apply_sparse_upload_to_partition(
    *,
    base_parameters: PeftEncoderMaterializedState,
    delta: PeftEncoderPartitionDelta,
    l1_sparse: bool,
    parameters: PartitionSparseSyncParameters,
) -> PeftEncoderPartitionDelta:
    return PeftEncoderPartitionDelta(
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


def _partition_delta_nonzero_count(delta: PeftEncoderPartitionDelta) -> int:
    return (
        _nested_nonzero_float_count(delta.lora_parameter_deltas)
        + _nested_nonzero_float_count(delta.classifier_head_weight_deltas)
        + _nested_nonzero_float_count(delta.classifier_head_bias_deltas)
    )


def _nested_nonzero_float_count(value: object) -> int:
    if isinstance(value, Mapping):
        return sum(_nested_nonzero_float_count(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return sum(_nested_nonzero_float_count(child) for child in value)
    try:
        return int(float(value) != 0.0)
    except (TypeError, ValueError):
        return 0


def _build_sparse_upload_partition_delta_from_states(
    *,
    partition_name: str,
    server_parameters: PeftEncoderMaterializedState,
    client_parameters: PeftEncoderMaterializedState,
    l1_sparse: bool,
    parameters: PartitionSparseSyncParameters,
) -> PeftEncoderPartitionDelta:
    return PeftEncoderPartitionDelta(
        partition_name=partition_name,
        lora_parameter_deltas=_sparse_upload_vector_mapping_from_states(
            server_values=server_parameters.lora_parameters,
            client_values=client_parameters.lora_parameters,
            l1_sparse=l1_sparse,
            parameters=parameters,
        ),
        classifier_head_weight_deltas=_sparse_upload_vector_mapping_from_states(
            server_values=server_parameters.classifier_head_weights,
            client_values=client_parameters.classifier_head_weights,
            l1_sparse=l1_sparse,
            parameters=parameters,
        ),
        classifier_head_bias_deltas=_sparse_upload_scalar_mapping_from_states(
            server_values=server_parameters.classifier_head_biases,
            client_values=client_parameters.classifier_head_biases,
            l1_sparse=l1_sparse,
            parameters=parameters,
        ),
    )


def _apply_upload_delta_to_state(
    *,
    server_parameters: PeftEncoderMaterializedState,
    upload_delta: PeftEncoderPartitionDelta,
) -> PeftEncoderMaterializedState:
    return PeftEncoderMaterializedState(
        lora_parameters=_apply_vector_delta_mapping(
            base_values=server_parameters.lora_parameters,
            deltas=upload_delta.lora_parameter_deltas,
        ),
        classifier_head_weights=_apply_vector_delta_mapping(
            base_values=server_parameters.classifier_head_weights,
            deltas=upload_delta.classifier_head_weight_deltas,
        ),
        classifier_head_biases={
            key: float(server_parameters.classifier_head_biases.get(key, 0.0))
            + float(upload_delta.classifier_head_bias_deltas.get(key, 0.0))
            for key in sorted(
                set(server_parameters.classifier_head_biases)
                | set(upload_delta.classifier_head_bias_deltas)
            )
        },
    )


def _build_sparse_download_partition_delta(
    *,
    partition_name: str,
    server_parameters: PeftEncoderMaterializedState,
    client_parameters: PeftEncoderMaterializedState,
    l1_sparse: bool,
    parameters: PartitionSparseSyncParameters,
) -> PeftEncoderPartitionDelta:
    return PeftEncoderPartitionDelta(
        partition_name=partition_name,
        lora_parameter_deltas=_sparse_download_vector_mapping(
            server_values=server_parameters.lora_parameters,
            client_values=client_parameters.lora_parameters,
            l1_sparse=l1_sparse,
            parameters=parameters,
        ),
        classifier_head_weight_deltas=_sparse_download_vector_mapping(
            server_values=server_parameters.classifier_head_weights,
            client_values=client_parameters.classifier_head_weights,
            l1_sparse=l1_sparse,
            parameters=parameters,
        ),
        classifier_head_bias_deltas=_sparse_download_scalar_mapping(
            server_values=server_parameters.classifier_head_biases,
            client_values=client_parameters.classifier_head_biases,
            l1_sparse=l1_sparse,
            parameters=parameters,
        ),
    )


def _project_sparse_download_partition_state(
    *,
    partition_name: str,
    server_parameters: PeftEncoderMaterializedState,
    client_parameters: PeftEncoderMaterializedState,
    l1_sparse: bool,
    parameters: PartitionSparseSyncParameters,
) -> PeftEncoderMaterializedState:
    return PeftEncoderMaterializedState(
        lora_parameters=_sparse_download_projected_vector_mapping(
            server_values=server_parameters.lora_parameters,
            client_values=client_parameters.lora_parameters,
            l1_sparse=l1_sparse,
            parameters=parameters,
            context=partition_name,
        ),
        classifier_head_weights=_sparse_download_projected_vector_mapping(
            server_values=server_parameters.classifier_head_weights,
            client_values=client_parameters.classifier_head_weights,
            l1_sparse=l1_sparse,
            parameters=parameters,
            context=partition_name,
        ),
        classifier_head_biases=_sparse_download_projected_scalar_mapping(
            server_values=server_parameters.classifier_head_biases,
            client_values=client_parameters.classifier_head_biases,
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


def _sparse_upload_vector_mapping_from_states(
    *,
    server_values: Mapping[str, Sequence[float]],
    client_values: Mapping[str, Sequence[float]],
    l1_sparse: bool,
    parameters: PartitionSparseSyncParameters,
) -> dict[str, list[float]]:
    return {
        key: _sparse_upload_vector_delta_from_state(
            server_vector=[float(value) for value in server_values.get(key, [])],
            client_vector=[float(value) for value in client_values.get(key, [])],
            l1_sparse=l1_sparse,
            parameters=parameters,
            key=key,
        )
        for key in sorted(set(server_values) | set(client_values))
    }


def _sparse_download_vector_mapping(
    *,
    server_values: Mapping[str, Sequence[float]],
    client_values: Mapping[str, Sequence[float]],
    l1_sparse: bool,
    parameters: PartitionSparseSyncParameters,
) -> dict[str, list[float]]:
    return {
        key: _sparse_download_vector_delta(
            server_vector=[float(value) for value in server_values.get(key, [])],
            client_vector=[float(value) for value in client_values.get(key, [])],
            l1_sparse=l1_sparse,
            parameters=parameters,
            key=key,
        )
        for key in sorted(set(server_values) | set(client_values))
    }


def _sparse_download_projected_vector_mapping(
    *,
    server_values: Mapping[str, Sequence[float]],
    client_values: Mapping[str, Sequence[float]],
    l1_sparse: bool,
    parameters: PartitionSparseSyncParameters,
    context: str,
) -> dict[str, list[float]]:
    return {
        key: _sparse_download_projected_vector(
            server_vector=[float(value) for value in server_values.get(key, [])],
            client_vector=[float(value) for value in client_values.get(key, [])],
            l1_sparse=l1_sparse,
            parameters=parameters,
            key=f"{context}.{key}",
        )
        for key in sorted(set(server_values) | set(client_values))
    }


def _sparse_upload_scalar_mapping_from_states(
    *,
    server_values: Mapping[str, float],
    client_values: Mapping[str, float],
    l1_sparse: bool,
    parameters: PartitionSparseSyncParameters,
) -> dict[str, float]:
    return {
        key: _sparse_upload_scalar_delta_from_state(
            server_value=float(server_values.get(key, 0.0)),
            client_value=float(client_values.get(key, 0.0)),
            l1_sparse=l1_sparse,
            parameters=parameters,
        )
        for key in sorted(set(server_values) | set(client_values))
    }


def _sparse_download_scalar_mapping(
    *,
    server_values: Mapping[str, float],
    client_values: Mapping[str, float],
    l1_sparse: bool,
    parameters: PartitionSparseSyncParameters,
) -> dict[str, float]:
    return {
        key: _sparse_download_scalar_delta(
            server_value=float(server_values.get(key, 0.0)),
            client_value=float(client_values.get(key, 0.0)),
            l1_sparse=l1_sparse,
            parameters=parameters,
        )
        for key in sorted(set(server_values) | set(client_values))
    }


def _sparse_download_projected_scalar_mapping(
    *,
    server_values: Mapping[str, float],
    client_values: Mapping[str, float],
    l1_sparse: bool,
    parameters: PartitionSparseSyncParameters,
) -> dict[str, float]:
    return {
        key: _sparse_download_projected_scalar(
            server_value=float(server_values.get(key, 0.0)),
            client_value=float(client_values.get(key, 0.0)),
            l1_sparse=l1_sparse,
            parameters=parameters,
        )
        for key in sorted(set(server_values) | set(client_values))
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


def _apply_vector_delta_mapping(
    *,
    base_values: Mapping[str, Sequence[float]],
    deltas: Mapping[str, Sequence[float]],
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for key in sorted(set(base_values) | set(deltas)):
        base_vector = [float(value) for value in base_values.get(key, [])]
        delta_vector = [float(value) for value in deltas.get(key, [])]
        if not base_vector:
            base_vector = [0.0 for _value in delta_vector]
        if not delta_vector:
            delta_vector = [0.0 for _value in base_vector]
        if len(base_vector) != len(delta_vector):
            raise ValueError(f"partition sparse sync dimension mismatch for {key!r}.")
        result[key] = [
            base + delta for base, delta in zip(base_vector, delta_vector, strict=True)
        ]
    return result


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


def _sparse_upload_vector_delta_from_state(
    *,
    server_vector: Sequence[float],
    client_vector: Sequence[float],
    l1_sparse: bool,
    parameters: PartitionSparseSyncParameters,
    key: str,
) -> list[float]:
    server_vector, client_vector = _align_sparse_vectors(
        server_vector=server_vector,
        client_vector=client_vector,
        key=key,
    )
    return [
        _sparse_upload_scalar_delta_from_state(
            server_value=server,
            client_value=client,
            l1_sparse=l1_sparse,
            parameters=parameters,
        )
        for server, client in zip(server_vector, client_vector, strict=True)
    ]


def _sparse_download_vector_delta(
    *,
    server_vector: Sequence[float],
    client_vector: Sequence[float],
    l1_sparse: bool,
    parameters: PartitionSparseSyncParameters,
    key: str,
) -> list[float]:
    if not server_vector:
        server_vector = [0.0 for _value in client_vector]
    if not client_vector:
        client_vector = [0.0 for _value in server_vector]
    if len(server_vector) != len(client_vector):
        raise ValueError(f"partition sparse sync dimension mismatch for {key!r}.")
    return [
        _sparse_download_scalar_delta(
            server_value=server,
            client_value=client,
            l1_sparse=l1_sparse,
            parameters=parameters,
        )
        for server, client in zip(server_vector, client_vector, strict=True)
    ]


def _sparse_download_projected_vector(
    *,
    server_vector: Sequence[float],
    client_vector: Sequence[float],
    l1_sparse: bool,
    parameters: PartitionSparseSyncParameters,
    key: str,
) -> list[float]:
    server_vector, client_vector = _align_sparse_vectors(
        server_vector=server_vector,
        client_vector=client_vector,
        key=key,
    )
    return [
        _sparse_download_projected_scalar(
            server_value=server,
            client_value=client,
            l1_sparse=l1_sparse,
            parameters=parameters,
        )
        for server, client in zip(server_vector, client_vector, strict=True)
    ]


def _align_sparse_vectors(
    *,
    server_vector: Sequence[float],
    client_vector: Sequence[float],
    key: str,
) -> tuple[Sequence[float], Sequence[float]]:
    if not server_vector:
        server_vector = [0.0 for _value in client_vector]
    if not client_vector:
        client_vector = [0.0 for _value in server_vector]
    if len(server_vector) != len(client_vector):
        raise ValueError(f"partition sparse sync dimension mismatch for {key!r}.")
    return server_vector, client_vector


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


def _sparse_upload_scalar_delta_from_state(
    *,
    server_value: float,
    client_value: float,
    l1_sparse: bool,
    parameters: PartitionSparseSyncParameters,
) -> float:
    projected_client_value = client_value
    if l1_sparse and abs(projected_client_value) <= parameters.l1_threshold:
        projected_client_value = 0.0
    sparse_delta = projected_client_value - server_value
    if abs(sparse_delta) <= parameters.delta_threshold:
        return 0.0
    return sparse_delta


def _sparse_download_scalar_delta(
    *,
    server_value: float,
    client_value: float,
    l1_sparse: bool,
    parameters: PartitionSparseSyncParameters,
) -> float:
    if l1_sparse:
        masked_server_value = (
            0.0 if abs(server_value) <= parameters.l1_threshold else server_value
        )
        masked_client_value = (
            0.0 if abs(client_value) <= parameters.l1_threshold else client_value
        )
    else:
        masked_server_value = server_value
        masked_client_value = client_value
    sparse_delta = masked_server_value - masked_client_value
    if abs(sparse_delta) <= parameters.delta_threshold:
        return 0.0
    return server_value - client_value


def _sparse_download_projected_scalar(
    *,
    server_value: float,
    client_value: float,
    l1_sparse: bool,
    parameters: PartitionSparseSyncParameters,
) -> float:
    if (
        _sparse_download_scalar_delta(
            server_value=server_value,
            client_value=client_value,
            l1_sparse=l1_sparse,
            parameters=parameters,
        )
        == 0.0
    ):
        return client_value
    return server_value
