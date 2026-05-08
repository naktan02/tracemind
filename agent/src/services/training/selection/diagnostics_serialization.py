"""Selection diagnostics dataclass 직렬화 helper."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum


def dataclass_to_diagnostics_mapping(
    instance: object,
    *,
    field_aliases: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """diagnostics dataclass를 JSON-friendly mapping으로 변환한다."""

    if not _is_dataclass_instance(instance):
        raise TypeError(
            "Diagnostics mapping serialization requires a dataclass instance."
        )

    aliases = {} if field_aliases is None else dict(field_aliases)
    return {
        aliases.get(field.name, field.name): diagnostics_value_to_mapping(
            getattr(instance, field.name)
        )
        for field in fields(instance)
    }


def diagnostics_value_to_mapping(value: object) -> object:
    """diagnostics payload 값의 canonical JSON shape를 만든다."""

    if hasattr(value, "to_mapping"):
        mapping = value.to_mapping()
        if not isinstance(mapping, Mapping):
            raise TypeError("Diagnostics to_mapping() must return a mapping.")
        return dict(mapping)
    if _is_dataclass_instance(value):
        return dataclass_to_diagnostics_mapping(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): diagnostics_value_to_mapping(mapped_value)
            for key, mapped_value in sorted(
                value.items(),
                key=lambda item: str(item[0]),
            )
        }
    if isinstance(value, tuple | list):
        return [diagnostics_value_to_mapping(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(
        f"Unsupported diagnostics serialization value: {type(value).__name__}."
    )


def _is_dataclass_instance(value: object) -> bool:
    return is_dataclass(value) and not isinstance(value, type)
