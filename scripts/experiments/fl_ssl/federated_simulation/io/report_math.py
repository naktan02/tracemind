"""FL report numeric helper."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def numeric_summary(values: Sequence[float | int]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": mean([float(value) for value in values]),
        "variance": population_variance([float(value) for value in values]),
    }


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def population_variance(values: Sequence[float]) -> float | None:
    if not values:
        return None
    average = sum(values) / len(values)
    return sum((value - average) ** 2 for value in values) / len(values)


def population_std(values: Sequence[float]) -> float | None:
    variance = population_variance(values)
    if variance is None:
        return None
    return math.sqrt(variance)


def weighted_mean(values: Iterable[tuple[float | None, int]]) -> float | None:
    weighted_sum = 0.0
    total_weight = 0
    for value, weight in values:
        if value is None or weight <= 0:
            continue
        weighted_sum += value * weight
        total_weight += weight
    if total_weight <= 0:
        return None
    return weighted_sum / total_weight


def safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator
