"""FedAvg update observation metric aggregation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from methods.federated.aggregation.fedavg.weighted_average import (
    WeightedScalarUpdate,
    weighted_average_scalars,
)


@dataclass(frozen=True, slots=True)
class FedAvgObservationMetricUpdate:
    """adapter family와 무관하게 FedAvg가 관측하는 client update metric."""

    example_count: int
    mean_confidence: float | None
    mean_margin: float | None
    delta_l2_norm: float


def aggregate_update_observation_metrics(
    updates: Sequence[FedAvgObservationMetricUpdate],
) -> dict[str, float]:
    """adapter family와 무관한 FedAvg update 관측 metric을 example weight로 집계한다."""

    weights = [float(update.example_count) for update in updates]
    mean_confidences = [update.mean_confidence for update in updates]
    mean_margins = [update.mean_margin for update in updates]
    return {
        "client_count": float(len(updates)),
        "example_count": float(sum(update.example_count for update in updates)),
        "mean_confidence": _weighted_observed_metric_mean(
            values=mean_confidences,
            weights=weights,
        ),
        "mean_confidence_observed_count": float(
            sum(1 for value in mean_confidences if value is not None)
        ),
        "mean_confidence_missing_count": float(
            sum(1 for value in mean_confidences if value is None)
        ),
        "mean_margin": _weighted_observed_metric_mean(
            values=mean_margins,
            weights=weights,
        ),
        "mean_margin_observed_count": float(
            sum(1 for value in mean_margins if value is not None)
        ),
        "mean_margin_missing_count": float(
            sum(1 for value in mean_margins if value is None)
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


def _weighted_observed_metric_mean(
    *,
    values: Sequence[float | None],
    weights: Sequence[float],
) -> float:
    observed_updates = [
        WeightedScalarUpdate(value=float(value), weight=weight)
        for value, weight in zip(values, weights, strict=True)
        if value is not None
    ]
    if not observed_updates:
        return 0.0
    return weighted_average_scalars(observed_updates)
