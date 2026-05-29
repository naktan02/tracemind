"""FL round client participation policy helpers."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

PARTICIPATION_ALL_CLIENTS = "all_clients"
PARTICIPATION_FRACTION_RANDOM = "fraction_random"
PARTICIPATION_FIXED_COUNT_RANDOM = "fixed_count_random"
PARTICIPATION_POLICY_NAMES = frozenset(
    {
        PARTICIPATION_ALL_CLIENTS,
        PARTICIPATION_FRACTION_RANDOM,
        PARTICIPATION_FIXED_COUNT_RANDOM,
    }
)

ClientT = TypeVar("ClientT")


@dataclass(frozen=True, slots=True)
class ClientParticipationPolicy:
    """한 FL round에서 학습에 참여할 client를 고르는 공통 정책."""

    name: str = PARTICIPATION_ALL_CLIENTS
    fraction: float | None = None
    count: int | None = None
    min_clients: int = 1

    @classmethod
    def from_mapping(
        cls,
        source: dict[str, object] | None,
    ) -> "ClientParticipationPolicy":
        if source is None:
            return cls()
        raw_fraction = source.get("fraction")
        raw_count = source.get("count")
        return cls(
            name=str(source.get("name", PARTICIPATION_ALL_CLIENTS)),
            fraction=None if raw_fraction is None else float(raw_fraction),
            count=None if raw_count is None else int(raw_count),
            min_clients=int(source.get("min_clients", 1)),
        )

    def __post_init__(self) -> None:
        if self.name not in PARTICIPATION_POLICY_NAMES:
            raise ValueError(
                "client_participation_policy.name must be one of "
                f"{sorted(PARTICIPATION_POLICY_NAMES)}."
            )
        if self.min_clients <= 0:
            raise ValueError(
                "client_participation_policy.min_clients must be positive."
            )
        if self.name == PARTICIPATION_FRACTION_RANDOM:
            if self.fraction is None or not 0.0 < self.fraction <= 1.0:
                raise ValueError(
                    "client_participation_policy.fraction must be in (0, 1] "
                    "when name is fraction_random."
                )
            if self.count is not None:
                raise ValueError(
                    "client_participation_policy.count must be null when "
                    "name is fraction_random."
                )
        elif self.name == PARTICIPATION_FIXED_COUNT_RANDOM:
            if self.count is None or self.count <= 0:
                raise ValueError(
                    "client_participation_policy.count must be positive when "
                    "name is fixed_count_random."
                )
            if self.fraction is not None:
                raise ValueError(
                    "client_participation_policy.fraction must be null when "
                    "name is fixed_count_random."
                )
        elif self.fraction is not None or self.count is not None:
            raise ValueError(
                "client_participation_policy.fraction/count must be null when "
                "name is all_clients."
            )

    def selected_count(self, total_clients: int) -> int:
        """전체 client 수에서 이 round 참여 client 수를 계산한다."""

        if total_clients <= 0:
            raise ValueError("total_clients must be positive.")
        if self.name == PARTICIPATION_ALL_CLIENTS:
            return total_clients
        if self.name == PARTICIPATION_FIXED_COUNT_RANDOM:
            assert self.count is not None
            return min(total_clients, max(self.min_clients, self.count))
        assert self.fraction is not None
        rounded = int(round(total_clients * self.fraction))
        return min(total_clients, max(self.min_clients, rounded))

    def to_payload(self) -> dict[str, object]:
        """report/diagnostics용 policy payload를 만든다."""

        return {
            "name": self.name,
            "fraction": self.fraction,
            "count": self.count,
            "min_clients": self.min_clients,
        }


@dataclass(frozen=True, slots=True)
class ParticipationSelection:
    """round별 client 선택 결과."""

    selected_indices: tuple[int, ...]
    skipped_indices: tuple[int, ...]

    @property
    def selected_count(self) -> int:
        return len(self.selected_indices)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped_indices)

    def to_payload(self) -> dict[str, object]:
        return {
            "selected_indices": list(self.selected_indices),
            "skipped_indices": list(self.skipped_indices),
            "selected_count": self.selected_count,
            "skipped_count": self.skipped_count,
        }


def select_participating_indices(
    *,
    total_clients: int,
    policy: ClientParticipationPolicy,
    seed: int,
    round_index: int,
) -> ParticipationSelection:
    """round별 deterministic client index 선택을 수행한다."""

    if total_clients <= 0:
        raise ValueError("total_clients must be positive.")
    if round_index <= 0:
        raise ValueError("round_index must be positive.")
    count = policy.selected_count(total_clients)
    all_indices = tuple(range(total_clients))
    if count == total_clients:
        return ParticipationSelection(
            selected_indices=all_indices,
            skipped_indices=(),
        )
    rng = random.Random(seed + round_index * 1_000_003)
    selected = tuple(sorted(rng.sample(all_indices, count)))
    selected_set = set(selected)
    skipped = tuple(index for index in all_indices if index not in selected_set)
    return ParticipationSelection(selected_indices=selected, skipped_indices=skipped)


def select_participating_clients(
    *,
    clients: Sequence[ClientT],
    policy: ClientParticipationPolicy,
    seed: int,
    round_index: int,
) -> tuple[tuple[ClientT, ...], ParticipationSelection]:
    """client sequence에서 round 참여 subset을 반환한다."""

    selection = select_participating_indices(
        total_clients=len(clients),
        policy=policy,
        seed=seed,
        round_index=round_index,
    )
    return tuple(clients[index] for index in selection.selected_indices), selection
