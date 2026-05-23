"""FL simulation run-scoped runtime resource cache."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from threading import RLock


@dataclass(frozen=True, slots=True)
class RoundBaseSnapshotCacheKey:
    """round-local base snapshot cache key.

    cache는 family/method 의미를 해석하지 않고, server-published global state의
    식별자만 비교한다.
    """

    adapter_kind: str
    model_revision: str
    schema_version: str
    artifact_refs: tuple[tuple[str, str], ...]
    materializer_name: str


@dataclass(slots=True)
class RoundBaseSnapshotCache:
    """한 round 안에서 동일 active global base snapshot materialization을 공유한다."""

    _snapshots: dict[RoundBaseSnapshotCacheKey, object] = field(default_factory=dict)
    _lock: RLock = field(default_factory=RLock)
    hit_count: int = 0
    miss_count: int = 0

    def get_or_materialize(
        self,
        *,
        key: RoundBaseSnapshotCacheKey,
        materialize: Callable[[], object],
    ) -> object:
        """key에 해당하는 snapshot을 반환하고 없으면 materialize한다."""

        with self._lock:
            cached = self._snapshots.get(key)
            if cached is not None:
                self.hit_count += 1
                return cached
            snapshot = materialize()
            self._snapshots[key] = snapshot
            self.miss_count += 1
            return snapshot

    def clear(self) -> None:
        """round boundary에서 cached snapshot을 폐기한다."""

        with self._lock:
            self._snapshots.clear()


@dataclass(slots=True)
class InMemoryRuntimeResourceCache:
    """한 simulation run 동안만 유지되는 opaque resource cache."""

    _resources: dict[str, object] = field(default_factory=dict)

    def get_resource(self, key: str) -> object | None:
        return self._resources.get(key)

    def set_resource(self, key: str, value: object) -> None:
        self._resources[key] = value
