"""Method 계층에서 공유하는 작은 이름 registry helper."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Generic, TypeVar

RegistryItem = TypeVar("RegistryItem")


class MethodRegistry(Generic[RegistryItem]):
    """method family별 registry instance를 위한 공통 helper."""

    def __init__(self, *, item_label: str) -> None:
        self._item_label = item_label
        self._items: dict[str, RegistryItem] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def register(self, *names: str, item: RegistryItem) -> RegistryItem:
        normalized_names = tuple(self._normalize_name(name) for name in names)
        if not normalized_names:
            raise ValueError(f"{self._item_label} registry names must not be empty.")
        for normalized_name in normalized_names:
            self._items[normalized_name] = item
        return item

    def resolve(self, name: str) -> RegistryItem | None:
        return self._items.get(self._normalize_name(name))

    def list_unique(
        self,
        *,
        names: Iterable[str] | None = None,
        sort_key: Callable[[RegistryItem], str],
    ) -> tuple[RegistryItem, ...]:
        if names is None:
            items = tuple(self._items.values())
        else:
            items = tuple(
                item
                for name in names
                if (item := self.resolve(name)) is not None
            )
        unique_items = {id(item): item for item in items}
        return tuple(sorted(unique_items.values(), key=sort_key))

    def _normalize_name(self, name: str) -> str:
        normalized_name = name.strip().lower()
        if not normalized_name:
            raise ValueError(f"{self._item_label} registry name must not be empty.")
        return normalized_name
