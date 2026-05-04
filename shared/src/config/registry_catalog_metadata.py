"""Registry-adjacent catalog metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

RegistryCatalogMetadataScalar = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class RegistryCatalogEntry:
    """Registry item을 developer catalog surface로 노출할 때 쓰는 메타데이터."""

    item_name: str
    display_name: str
    implementation_module: str
    core_method_name: str | None = None
    family_name: str | None = None
    supported_adapter_kinds: tuple[str, ...] = ()
    accepted_payload_formats: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, RegistryCatalogMetadataScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "supported_adapter_kinds",
            tuple(self.supported_adapter_kinds),
        )
        object.__setattr__(
            self,
            "accepted_payload_formats",
            tuple(self.accepted_payload_formats),
        )
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def dedupe_registry_catalog_entries(
    entries: Iterable[RegistryCatalogEntry],
) -> tuple[RegistryCatalogEntry, ...]:
    """alias가 섞인 registry entry들을 canonical item_name 기준으로 dedupe한다."""

    unique_entries: dict[str, RegistryCatalogEntry] = {}
    for entry in entries:
        unique_entries.setdefault(entry.item_name, entry)
    return tuple(
        sorted(
            unique_entries.values(),
            key=lambda entry: (
                entry.family_name or "",
                entry.item_name,
            ),
        )
    )
