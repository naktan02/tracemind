"""Diagonal-scale family용 FedAvg 계산 core."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from methods.federated.aggregation.fedavg.fedavg import (
    WeightedScalarUpdate,
    WeightedVectorUpdate,
    weighted_average_scalars,
    weighted_average_vectors,
)


@dataclass(frozen=True, slots=True)
class DiagonalScaleFedAvgUpdate:
    """main_server boundary와 분리된 diagonal-scale FedAvg 입력."""

    dimension_deltas: Sequence[float]
    example_count: int
    mean_confidence: float
    mean_margin: float | None
    delta_l2_norm: float


@dataclass(frozen=True, slots=True)
class DiagonalScaleFedAvgResult:
    """diagonal-scale FedAvg 계산 결과."""

    next_dimension_scales: list[float]
    aggregated_metrics: dict[str, float]
    update_count: int


def compute_diagonal_scale_fedavg(
    *,
    base_dimension_scales: Sequence[float],
    updates: Sequence[DiagonalScaleFedAvgUpdate],
    min_scale: float,
    max_scale: float,
) -> DiagonalScaleFedAvgResult:
    """차원별 scale delta를 example_count로 평균하고 clamp한다."""

    if min_scale > max_scale:
        raise ValueError("min_scale must be less than or equal to max_scale.")

    base_scales = [float(scale) for scale in base_dimension_scales]
    if not base_scales:
        raise ValueError("base_dimension_scales must not be empty.")

    valid_updates = tuple(update for update in updates if update.example_count > 0)
    weighted_delta = weighted_average_vectors(
        [
            WeightedVectorUpdate(
                values=update.dimension_deltas,
                weight=float(update.example_count),
            )
            for update in valid_updates
        ]
    )
    if len(weighted_delta) != len(base_scales):
        raise ValueError("FedAvg delta dimension must match base_dimension_scales.")

    next_dimension_scales = [
        max(min_scale, min(max_scale, scale + delta))
        for scale, delta in zip(base_scales, weighted_delta, strict=True)
    ]
    return DiagonalScaleFedAvgResult(
        next_dimension_scales=next_dimension_scales,
        aggregated_metrics=_aggregate_common_metrics(valid_updates),
        update_count=len(valid_updates),
    )


def _aggregate_common_metrics(
    updates: Sequence[DiagonalScaleFedAvgUpdate],
) -> dict[str, float]:
    return {
        "client_count": float(len(updates)),
        "example_count": float(sum(update.example_count for update in updates)),
        "mean_confidence": weighted_average_scalars(
            [
                WeightedScalarUpdate(
                    value=update.mean_confidence,
                    weight=float(update.example_count),
                )
                for update in updates
            ]
        ),
        "mean_margin": weighted_average_scalars(
            [
                WeightedScalarUpdate(
                    value=update.mean_margin or 0.0,
                    weight=float(update.example_count),
                )
                for update in updates
            ]
        ),
        "mean_delta_l2_norm": weighted_average_scalars(
            [
                WeightedScalarUpdate(
                    value=update.delta_l2_norm,
                    weight=float(update.example_count),
                )
                for update in updates
            ]
        ),
    }
