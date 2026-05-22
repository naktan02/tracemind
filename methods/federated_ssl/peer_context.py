"""Method-owned peer/helper context policy specs and common selection helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class FederatedSslPeerContextPolicy:
    """round 전 client/helper context를 준비하는 policy metadata."""

    policy_name: str
    parameters: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.policy_name.strip():
            raise ValueError("policy_name must not be empty.")


@dataclass(frozen=True, slots=True)
class FederatedSslPeerContext:
    """round-local peer/helper context selected before client training."""

    client_id: str
    policy_name: str
    round_index_zero_based: int
    helper_client_ids: tuple[str, ...] = ()
    refreshed: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.client_id.strip():
            raise ValueError("client_id must not be empty.")
        if not self.policy_name.strip():
            raise ValueError("policy_name must not be empty.")
        if self.round_index_zero_based < 0:
            raise ValueError("round_index_zero_based must be non-negative.")
        object.__setattr__(
            self,
            "helper_client_ids",
            tuple(str(client_id) for client_id in self.helper_client_ids),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )

    @property
    def helper_count(self) -> int:
        return len(self.helper_client_ids)


@dataclass(frozen=True, slots=True)
class FederatedSslPeerClientSnapshot:
    """다음 round peer selection/prediction에 쓰는 client-local snapshot."""

    client_id: str
    selection_vector: tuple[float, ...]
    payload_kind: str
    payload: object
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.client_id.strip():
            raise ValueError("client_id must not be empty.")
        if not self.selection_vector:
            raise ValueError("selection_vector must not be empty.")
        if not self.payload_kind.strip():
            raise ValueError("payload_kind must not be empty.")
        object.__setattr__(
            self,
            "selection_vector",
            tuple(float(value) for value in self.selection_vector),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


FederatedSslPeerContextByClient = Mapping[str, FederatedSslPeerContext]


def should_refresh_peer_context(
    *,
    round_index_zero_based: int,
    refresh_interval: int,
) -> bool:
    """원본 FedMatch의 `(curr_round + 1) % interval == 0` 계열 조건."""

    if round_index_zero_based < 0:
        raise ValueError("round_index_zero_based must be non-negative.")
    if refresh_interval <= 0:
        raise ValueError("refresh_interval must be positive.")
    return (round_index_zero_based + 1) % refresh_interval == 0


def select_nearest_peer_client_ids(
    *,
    client_id: str,
    client_vectors: Mapping[str, Sequence[float]],
    peer_count: int,
) -> tuple[str, ...]:
    """client vector 기준 최근접 peer client를 deterministic하게 고른다."""

    if peer_count < 0:
        raise ValueError("peer_count must be non-negative.")
    if peer_count == 0 or client_id not in client_vectors:
        return ()

    target = _validate_vector(client_vectors[client_id], name=client_id)
    distances: list[tuple[float, str]] = []
    for candidate_id, candidate_vector in client_vectors.items():
        if candidate_id == client_id:
            continue
        vector = _validate_vector(candidate_vector, name=candidate_id)
        if len(vector) != len(target):
            raise ValueError("all peer selection vectors must share one dimension.")
        distances.append((_squared_euclidean_distance(target, vector), candidate_id))

    distances.sort(key=lambda item: (item[0], item[1]))
    return tuple(candidate_id for _, candidate_id in distances[:peer_count])


def _squared_euclidean_distance(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    return sum((float(a) - float(b)) ** 2 for a, b in zip(left, right, strict=True))


def _validate_vector(vector: Sequence[float], *, name: str) -> tuple[float, ...]:
    if not vector:
        raise ValueError(f"peer selection vector for {name!r} must not be empty.")
    return tuple(float(value) for value in vector)
