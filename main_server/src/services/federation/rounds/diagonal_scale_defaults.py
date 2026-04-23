"""Diagonal-scale family의 server-owned aggregation 설정값."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

AggregationConfigScalar = str | int | float | bool


def _read_float(
    source: Mapping[str, AggregationConfigScalar] | None,
    key: str,
    default: float,
) -> float:
    if source is None:
        return default
    value = source.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must not be bool.")
    return float(value)


@dataclass(frozen=True, slots=True)
class DiagonalScaleFedAvgAggregationConfig:
    """Diagonal-scale FedAvg 서버 집계 설정."""

    min_scale: float = 0.75
    max_scale: float = 1.25

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, AggregationConfigScalar] | None,
    ) -> "DiagonalScaleFedAvgAggregationConfig":
        return cls(
            min_scale=_read_float(
                source,
                "min_scale",
                DEFAULT_DIAGONAL_SCALE_FEDAVG_AGGREGATION_CONFIG.min_scale,
            ),
            max_scale=_read_float(
                source,
                "max_scale",
                DEFAULT_DIAGONAL_SCALE_FEDAVG_AGGREGATION_CONFIG.max_scale,
            ),
        )


DEFAULT_DIAGONAL_SCALE_FEDAVG_AGGREGATION_CONFIG = (
    DiagonalScaleFedAvgAggregationConfig()
)


__all__ = [
    "AggregationConfigScalar",
    "DEFAULT_DIAGONAL_SCALE_FEDAVG_AGGREGATION_CONFIG",
    "DiagonalScaleFedAvgAggregationConfig",
]
