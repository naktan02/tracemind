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


class AggregationWeightDiagnosticClient(Protocol):
    """round diagnostics에서 aggregation weight 계산에 필요한 client 최소 surface."""

    accepted_count: int
    aggregation_example_count: int | None


@dataclass(frozen=True, slots=True)
class AggregationWeightPolicy:
    """client update를 aggregate할 때 사용할 weight 기준."""

    name: str = AGGREGATION_WEIGHT_UNIFORM

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, object] | None,
    ) -> "AggregationWeightPolicy":
        if source is None:
            return cls()
        return cls(name=str(source.get("name", AGGREGATION_WEIGHT_UNIFORM)))

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
                "expose accepted_count."
            )
        return float(accepted_count)
    return float(update.example_count)


def aggregation_example_count_for_diagnostics(
    client: AggregationWeightDiagnosticClient,
) -> int:
    """diagnostics용 aggregation example count를 canonical fallback으로 계산한다."""

    if client.aggregation_example_count is not None:
        return client.aggregation_example_count
    return client.accepted_count


def aggregation_weight_for_diagnostics(
    client: AggregationWeightDiagnosticClient,
    *,
    policy: AggregationWeightPolicy,
) -> float:
    """round diagnostics에서 단일 client의 raw aggregation weight를 계산한다."""

    if policy.name == AGGREGATION_WEIGHT_UNIFORM:
        return 1.0
    if policy.name == AGGREGATION_WEIGHT_ACCEPTED_COUNT:
        return float(client.accepted_count)
    return float(aggregation_example_count_for_diagnostics(client))


def aggregation_weight_basis_label(policy: AggregationWeightPolicy) -> str:
    """report에서 사용할 aggregation weight basis label을 반환한다."""

    if policy.name == AGGREGATION_WEIGHT_EXAMPLE_COUNT:
        return "update_envelope.example_count"
    return policy.name


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
    return None
