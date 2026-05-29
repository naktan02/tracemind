"""Runtime-scoped reusable resource cache protocol."""

from __future__ import annotations

from typing import Protocol


class RuntimeResourceCache(Protocol):
    """runtime별 lifecycle이 소유하는 opaque resource cache."""

    def get_resource(self, key: str) -> object | None:
        """key에 해당하는 resource를 반환한다."""

    def set_resource(self, key: str, value: object) -> None:
        """key에 해당하는 resource를 저장한다."""
