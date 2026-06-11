"""Method-owned peer/helper context policy specs and common selection helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

_SCIPY_KDTREE_BACKEND_NAME = "scipy_kdtree"
_FULL_SCAN_BACKEND_NAME = "full_scan"


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


@dataclass(frozen=True, slots=True)
class FixedProbePeerContextParameters:
    """fixed-probe nearest-neighbor helper context에 필요한 method parameter."""

    num_helpers: int
    refresh_interval: int

    def __post_init__(self) -> None:
        if self.num_helpers < 0:
            raise ValueError("peer context num_helpers must be non-negative.")
        if self.refresh_interval <= 0:
            raise ValueError("peer context refresh_interval must be positive.")


def should_refresh_peer_context(
    *,
    round_index_zero_based: int,
    refresh_interval: int,
) -> bool:
    """one-based round index가 refresh interval에 도달했는지 반환한다."""

    if round_index_zero_based < 0:
        raise ValueError("round_index_zero_based must be non-negative.")
    if refresh_interval <= 0:
        raise ValueError("refresh_interval must be positive.")
    return (round_index_zero_based + 1) % refresh_interval == 0


def resolve_fixed_probe_peer_context_parameters(
    *,
    round_state_exchange: Mapping[str, object],
    effective_parameters: Mapping[str, object] | None = None,
) -> FixedProbePeerContextParameters:
    """method descriptor/effective parameter에서 fixed-probe helper 설정을 해석한다."""

    parameters = dict(round_state_exchange)
    overrides = {} if effective_parameters is None else dict(effective_parameters)
    if "num_helpers" in overrides:
        parameters["num_helpers"] = overrides["num_helpers"]
    if "helper_refresh_interval" in overrides:
        parameters["refresh_interval"] = overrides["helper_refresh_interval"]
    elif "refresh_interval" in overrides:
        parameters["refresh_interval"] = overrides["refresh_interval"]
    return FixedProbePeerContextParameters(
        num_helpers=_required_positive_or_zero_int(parameters, "num_helpers"),
        refresh_interval=_required_positive_int(parameters, "refresh_interval"),
    )


def build_fixed_probe_peer_context_by_client(
    *,
    policy_name: str,
    parameters: FixedProbePeerContextParameters,
    selected_client_ids: Sequence[str],
    round_index: int,
    client_vectors: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, FederatedSslPeerContext]:
    """fixed-probe output vector 기준 client별 nearest helper context를 만든다."""

    if round_index <= 0:
        raise ValueError("round_index must be one-based and positive.")
    round_index_zero_based = round_index - 1
    refresh_due = should_refresh_peer_context(
        round_index_zero_based=round_index_zero_based,
        refresh_interval=parameters.refresh_interval,
    )
    vectors = {} if client_vectors is None else dict(client_vectors)
    helper_index = (
        NearestPeerClientIndex(client_vectors=vectors, prefer_kdtree=True)
        if refresh_due and vectors
        else None
    )
    contexts: dict[str, FederatedSslPeerContext] = {}
    for client_id in selected_client_ids:
        has_selection_vector = client_id in vectors
        helper_client_ids: tuple[str, ...] = ()
        if refresh_due and has_selection_vector and helper_index is not None:
            helper_client_ids = helper_index.query(
                client_id=client_id,
                peer_count=parameters.num_helpers,
            )
        contexts[client_id] = FederatedSslPeerContext(
            client_id=client_id,
            policy_name=policy_name,
            round_index_zero_based=round_index_zero_based,
            helper_client_ids=helper_client_ids,
            refreshed=refresh_due and has_selection_vector,
            metadata={
                "num_helpers": parameters.num_helpers,
                "refresh_interval": parameters.refresh_interval,
                "refresh_due": refresh_due,
                "has_selection_vector": has_selection_vector,
                "selection_vector_source": (
                    "provided" if has_selection_vector else "unavailable"
                ),
                "selection_index_backend": (
                    helper_index.backend_name if helper_index is not None else "none"
                ),
                "selection_query_size": (
                    helper_index.query_size_including_self(
                        peer_count=parameters.num_helpers,
                    )
                    if helper_index is not None
                    else 0
                ),
                "parameter_source": "effective_parameters",
            },
        )
    return contexts


def select_nearest_peer_client_ids(
    *,
    client_id: str,
    client_vectors: Mapping[str, Sequence[float]],
    peer_count: int,
    prefer_kdtree: bool = False,
) -> tuple[str, ...]:
    """client vector 기준 최근접 peer client를 deterministic하게 고른다."""

    index = NearestPeerClientIndex(
        client_vectors=client_vectors,
        prefer_kdtree=prefer_kdtree,
    )
    return index.query(client_id=client_id, peer_count=peer_count)


class NearestPeerClientIndex:
    """client output vector의 최근접 peer 조회 index.

    `scipy.spatial.KDTree.query(k=peer_count + 1)`를 우선 쓰되, experiments
    dependency가 없는 실행도 깨지지 않도록 scipy를 사용할 수 없으면 같은
    Euclidean 기준의 full-scan으로 내린다.
    """

    def __init__(
        self,
        *,
        client_vectors: Mapping[str, Sequence[float]],
        prefer_kdtree: bool = False,
    ) -> None:
        self._client_ids: tuple[str, ...] = tuple(client_vectors)
        self._vectors: dict[str, tuple[float, ...]] = {
            client_id: _validate_vector(vector, name=client_id)
            for client_id, vector in client_vectors.items()
        }
        self._dimension = self._validate_shared_dimension()
        self._rows: tuple[tuple[float, ...], ...] = tuple(
            self._vectors[client_id] for client_id in self._client_ids
        )
        self._tree: Any | None = None
        self._backend_name = _FULL_SCAN_BACKEND_NAME
        if prefer_kdtree and self._rows:
            self._tree = _build_scipy_kdtree(self._rows)
            if self._tree is not None:
                self._backend_name = _SCIPY_KDTREE_BACKEND_NAME

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def client_count(self) -> int:
        return len(self._client_ids)

    @property
    def dimension(self) -> int:
        return self._dimension

    def query_size_including_self(self, *, peer_count: int) -> int:
        """nearest query에서 self 제거를 고려해 `peer_count + 1`개를 요청한다."""

        if peer_count < 0:
            raise ValueError("peer_count must be non-negative.")
        return min(self.client_count, peer_count + 1)

    def query(self, *, client_id: str, peer_count: int) -> tuple[str, ...]:
        if peer_count < 0:
            raise ValueError("peer_count must be non-negative.")
        if peer_count == 0 or client_id not in self._vectors:
            return ()
        if self._tree is not None:
            return self._query_kdtree(client_id=client_id, peer_count=peer_count)
        return self._query_full_scan(client_id=client_id, peer_count=peer_count)

    def _query_kdtree(self, *, client_id: str, peer_count: int) -> tuple[str, ...]:
        query_size = self.query_size_including_self(peer_count=peer_count)
        if query_size <= 0:
            return ()
        distances, indices = self._tree.query(self._vectors[client_id], k=query_size)
        candidates: list[tuple[float, str]] = []
        for distance, index in zip(
            _as_tuple(distances),
            _as_tuple(indices),
            strict=True,
        ):
            candidate_index = int(index)
            if candidate_index < 0 or candidate_index >= len(self._client_ids):
                continue
            candidate_id = self._client_ids[candidate_index]
            if candidate_id == client_id:
                continue
            candidates.append((float(distance), candidate_id))
        candidates.sort(key=lambda item: (item[0], item[1]))
        return tuple(candidate_id for _, candidate_id in candidates[:peer_count])

    def _query_full_scan(self, *, client_id: str, peer_count: int) -> tuple[str, ...]:
        target = self._vectors[client_id]
        distances: list[tuple[float, str]] = []
        for candidate_id, vector in self._vectors.items():
            if candidate_id == client_id:
                continue
            distances.append(
                (_squared_euclidean_distance(target, vector), candidate_id)
            )
        distances.sort(key=lambda item: (item[0], item[1]))
        return tuple(candidate_id for _, candidate_id in distances[:peer_count])

    def _validate_shared_dimension(self) -> int:
        dimension = 0
        for client_id, vector in self._vectors.items():
            if not client_id.strip():
                raise ValueError("client_id must not be empty.")
            if dimension == 0:
                dimension = len(vector)
                continue
            if len(vector) != dimension:
                raise ValueError("all peer selection vectors must share one dimension.")
        return dimension


def _squared_euclidean_distance(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    return sum((float(a) - float(b)) ** 2 for a, b in zip(left, right, strict=True))


def _validate_vector(vector: Sequence[float], *, name: str) -> tuple[float, ...]:
    if not vector:
        raise ValueError(f"peer selection vector for {name!r} must not be empty.")
    return tuple(float(value) for value in vector)


def _required_positive_or_zero_int(
    parameters: Mapping[str, object],
    key: str,
) -> int:
    if key not in parameters:
        raise ValueError(f"peer context method parameter is missing: {key}.")
    value = int(parameters[key])
    if value < 0:
        raise ValueError(f"peer context method parameter must be non-negative: {key}.")
    return value


def _required_positive_int(
    parameters: Mapping[str, object],
    key: str,
) -> int:
    value = _required_positive_or_zero_int(parameters, key)
    if value <= 0:
        raise ValueError(f"peer context method parameter must be positive: {key}.")
    return value


def _build_scipy_kdtree(rows: Sequence[Sequence[float]]) -> Any | None:
    try:
        from scipy import spatial
    except ImportError:
        return None
    return spatial.KDTree(rows)


def _as_tuple(value: object) -> tuple[object, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return tuple(converted)
    return (value,)
