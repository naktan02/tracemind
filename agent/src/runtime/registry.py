"""Small runtime registry primitive for agent capability adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from agent.src.runtime.registry_imports import (
    import_runtime_module_for_name,
    import_runtime_package_modules,
)
from shared.src.contracts.registry_catalog_metadata import (
    RegistryCatalogEntry,
    dedupe_registry_catalog_entries,
)

FactoryT = TypeVar("FactoryT")


@dataclass(slots=True)
class RuntimeRegistry(Generic[FactoryT]):
    """Package-local runtime adapter registry primitive."""

    package_name: str
    item_kind: str
    _registry: dict[str, tuple[FactoryT, RegistryCatalogEntry]] = field(
        default_factory=dict
    )

    def register(
        self,
        *item_names: str,
        catalog_entry: RegistryCatalogEntry,
        factory: FactoryT | None = None,
    ) -> Callable[[FactoryT], FactoryT] | FactoryT:
        """Factory 옆 decorator로 runtime adapter metadata를 등록한다."""

        def _decorator(factory: FactoryT) -> FactoryT:
            registered_item = (factory, catalog_entry)
            for item_name in item_names:
                self._registry[_normalize_registry_name(item_name)] = registered_item
            return factory

        if factory is not None:
            return _decorator(factory)
        return _decorator

    def get(self, item_name: str) -> tuple[FactoryT, RegistryCatalogEntry]:
        """이름에 맞는 factory와 catalog entry를 반환한다."""

        normalized_name = _normalize_registry_name(item_name)
        import_runtime_module_for_name(
            package_name=self.package_name,
            registered_name=normalized_name,
        )
        registered_item = self._registry.get(normalized_name)
        if registered_item is not None:
            return registered_item
        import_runtime_package_modules(package_name=self.package_name)
        registered_item = self._registry.get(normalized_name)
        if registered_item is not None:
            return registered_item
        raise ValueError(f"Unsupported {self.item_kind}: {item_name}.")

    def list_names(self) -> tuple[str, ...]:
        """등록된 item 이름을 정렬해서 반환한다."""

        import_runtime_package_modules(package_name=self.package_name)
        return tuple(sorted(self._registry))

    def list_catalog_entries(self) -> tuple[RegistryCatalogEntry, ...]:
        """등록된 catalog entry를 canonical item 기준으로 반환한다."""

        import_runtime_package_modules(package_name=self.package_name)
        return dedupe_registry_catalog_entries(
            catalog_entry for _factory, catalog_entry in self._registry.values()
        )


def _normalize_registry_name(item_name: str) -> str:
    return item_name.strip().lower()
