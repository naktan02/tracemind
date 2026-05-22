"""FL aggregation weight policy helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

AGGREGATION_WEIGHT_EXAMPLE_COUNT = "example_count"
AGGREGATION_WEIGHT_UNIFORM = "uniform"
AGGREGATION_WEIGHT_ACCEPTED_COUNT = "accepted_count"
AGGREGATION_WEIGHT_POLICY_NAMES = frozenset(
    {
        AGGREGATION_WEIGHT_EXAMPLE_COUNT,
        AGGREGATION_WEIGHT_UNIFORM,
        AGGREGATION_WEIGHT_ACCEPTED_COUNT,
    }
)


class AggregationWeightUpdate(Protocol):
    """Aggregation weight 계산에 필요한 update 최소 surface."""

    example_count: int


@dataclass(frozen=True, slots=True)
class AggregationWeightPolicy:
    """client update를 aggregate할 때 사용할 weight 기준."""

    name: str = AGGREGATION_WEIGHT_EXAMPLE_COUNT

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, object] | None,
    ) -> "AggregationWeightPolicy":
        if source is None:
            return cls()
        return cls(name=str(source.get("name", AGGREGATION_WEIGHT_EXAMPLE_COUNT)))

    def __post_init__(self) -> None:
        if self.name not in AGGREGATION_WEIGHT_POLICY_NAMES:
            raise ValueError(
                "aggregation_weight_policy.name must be one of "
                f"{sorted(AGGREGATION_WEIGHT_POLICY_NAMES)}."
            )

    def to_payload(self) -> dict[str, object]:
        return {"name": self.name}


def aggregation_weight_for_update(
    update: AggregationWeightUpdate,
    *,
    policy: AggregationWeightPolicy,
) -> float:
    """단일 update의 raw aggregation weight를 계산한다."""

    if policy.name == AGGREGATION_WEIGHT_UNIFORM:
        return 1.0
    if policy.name == AGGREGATION_WEIGHT_ACCEPTED_COUNT:
        accepted_count = _accepted_count(update)
        if accepted_count is None:
            raise ValueError(
                "accepted_count aggregation weight requires update payloads to "
                "expose accepted_count or accepted label_counts."
            )
        return float(accepted_count)
    return float(update.example_count)


def normalized_aggregation_weights(
    updates: Sequence[AggregationWeightUpdate],
    *,
    policy: AggregationWeightPolicy,
) -> list[float]:
    """update sequence의 normalized aggregation weights를 반환한다."""

    weights = [
        aggregation_weight_for_update(update, policy=policy) for update in updates
    ]
    if not weights:
        return []
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("aggregation weights must sum to a positive value.")
    return [weight / total for weight in weights]


def _accepted_count(update: AggregationWeightUpdate) -> int | None:
    value = getattr(update, "accepted_count", None)
    if value is not None:
        return int(value)
    label_counts = getattr(update, "label_counts", None)
    if isinstance(label_counts, Mapping) and label_counts:
        return sum(int(count) for count in label_counts.values())
    return None
