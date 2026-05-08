"""FedAvg 공통 가중 평균 산술."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WeightedScalarUpdate:
    """하나의 scalar 값과 FedAvg weight."""

    value: float
    weight: float


@dataclass(frozen=True, slots=True)
class WeightedVectorUpdate:
    """하나의 vector 값과 FedAvg weight."""

    values: Sequence[float]
    weight: float


@dataclass(frozen=True, slots=True)
class WeightedScalarMappingUpdate:
    """동일 key 집합을 갖는 scalar mapping 값과 FedAvg weight."""

    values: Mapping[str, float]
    weight: float


@dataclass(frozen=True, slots=True)
class WeightedVectorMappingUpdate:
    """동일 key 집합을 갖는 vector mapping 값과 FedAvg weight."""

    values: Mapping[str, Sequence[float]]
    weight: float


def _validate_weight(weight: float) -> float:
    if isinstance(weight, bool):
        raise ValueError("FedAvg update weight must not be bool.")
    normalized_weight = float(weight)
    if normalized_weight < 0.0:
        raise ValueError("FedAvg update weight must be non-negative.")
    return normalized_weight


def _normalized_weights(weights: Sequence[float]) -> list[float]:
    if not weights:
        raise ValueError("At least one FedAvg update is required.")
    normalized_weights = [_validate_weight(weight) for weight in weights]
    total_weight = sum(normalized_weights)
    if total_weight <= 0.0:
        raise ValueError("FedAvg total update weight must be positive.")
    return [weight / total_weight for weight in normalized_weights]


def weighted_average_scalars(updates: Sequence[WeightedScalarUpdate]) -> float:
    """scalar update를 FedAvg weight로 평균한다."""

    weights = _normalized_weights([update.weight for update in updates])
    return sum(float(update.value) * weight for update, weight in zip(updates, weights))


def weighted_average_vectors(updates: Sequence[WeightedVectorUpdate]) -> list[float]:
    """vector update를 FedAvg weight로 평균한다."""

    weights = _normalized_weights([update.weight for update in updates])
    vectors = [tuple(float(value) for value in update.values) for update in updates]
    if not vectors[0]:
        raise ValueError("FedAvg vector update values must not be empty.")
    vector_dim = len(vectors[0])
    if any(len(vector) != vector_dim for vector in vectors):
        raise ValueError("FedAvg vector updates must share the same dimension.")
    return [
        sum(vector[index] * weight for vector, weight in zip(vectors, weights))
        for index in range(vector_dim)
    ]


def weighted_average_scalar_mappings(
    updates: Sequence[WeightedScalarMappingUpdate],
) -> dict[str, float]:
    """동일 key 집합의 scalar mapping update를 FedAvg weight로 평균한다."""

    weights = _normalized_weights([update.weight for update in updates])
    mappings = [
        {str(key): float(value) for key, value in update.values.items()}
        for update in updates
    ]
    expected_keys = set(mappings[0])
    if not expected_keys:
        raise ValueError("FedAvg scalar mapping update keys must not be empty.")
    for mapping in mappings:
        if set(mapping) != expected_keys:
            raise ValueError("FedAvg scalar mapping updates must share the same keys.")
    return {
        key: sum(mapping[key] * weight for mapping, weight in zip(mappings, weights))
        for key in sorted(expected_keys)
    }


def weighted_average_vector_mappings(
    updates: Sequence[WeightedVectorMappingUpdate],
) -> dict[str, list[float]]:
    """동일 key 집합의 vector mapping update를 FedAvg weight로 평균한다."""

    weights = _normalized_weights([update.weight for update in updates])
    mappings = [
        {
            str(key): tuple(float(value) for value in values)
            for key, values in update.values.items()
        }
        for update in updates
    ]
    expected_keys = set(mappings[0])
    if not expected_keys:
        raise ValueError("FedAvg vector mapping update keys must not be empty.")
    expected_dims = {key: len(mappings[0][key]) for key in expected_keys}
    if any(dim == 0 for dim in expected_dims.values()):
        raise ValueError("FedAvg vector mapping values must not be empty.")
    for mapping in mappings:
        if set(mapping) != expected_keys:
            raise ValueError("FedAvg vector mapping updates must share the same keys.")
        for key in expected_keys:
            if len(mapping[key]) != expected_dims[key]:
                raise ValueError(
                    "FedAvg vector mapping updates must share dimensions per key."
                )
    return {
        key: [
            sum(
                mapping[key][index] * weight
                for mapping, weight in zip(mappings, weights)
            )
            for index in range(expected_dims[key])
        ]
        for key in sorted(expected_keys)
    }
