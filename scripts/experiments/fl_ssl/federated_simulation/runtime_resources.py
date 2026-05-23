"""FL simulation run-scoped runtime resource cache."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class InMemoryRuntimeResourceCache:
    """한 simulation run 동안만 유지되는 opaque resource cache."""

    _resources: dict[str, object] = field(default_factory=dict)

    def get_resource(self, key: str) -> object | None:
        return self._resources.get(key)

    def set_resource(self, key: str, value: object) -> None:
        self._resources[key] = value
