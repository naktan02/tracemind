"""Method config mapping parsing helpers."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from types import MappingProxyType

ConfigScalar = str | int | float | bool


def freeze_mapping(
    source: Mapping[str, ConfigScalar],
) -> Mapping[str, ConfigScalar]:
    """불변 view로 노출할 config mapping copy를 만든다."""

    return MappingProxyType(dict(source))


def validate_allowed_keys(
    source: Mapping[str, object] | None,
    *,
    allowed_keys: Collection[str],
    config_name: str,
) -> None:
    """알 수 없는 config key를 bootstrap 단계에서 거부한다."""

    if source is None:
        return
    unexpected_keys = sorted(key for key in source if key not in allowed_keys)
    if unexpected_keys:
        raise ValueError(f"Unsupported {config_name} key(s): {unexpected_keys}.")


def normalize_non_empty_str(value: str, *, field_name: str) -> str:
    """공백 문자열을 거부하고 strip한 값을 반환한다."""

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    return normalized


def set_normalized_str(
    instance: object,
    field_name: str,
    value: str,
    *,
    allow_empty: bool = False,
    field_prefix: str | None = None,
) -> None:
    """frozen dataclass string field를 strip한 값으로 정규화한다."""

    field_path = _field_path(field_name=field_name, field_prefix=field_prefix)
    normalized = value.strip()
    if not normalized and not allow_empty:
        raise ValueError(f"{field_path} must not be empty.")
    object.__setattr__(instance, field_name, normalized)


def read_str(
    source: Mapping[str, object],
    key: str,
    default: str | None = None,
    *,
    allow_empty: bool = False,
    field_prefix: str | None = None,
) -> str:
    """mapping에서 string config 값을 읽고 strip한다."""

    value = source.get(key, default)
    field_path = _field_path(field_name=key, field_prefix=field_prefix)
    if value is None:
        raise ValueError(f"{field_path} is required.")
    normalized = str(value).strip()
    if not normalized and not allow_empty:
        raise ValueError(f"{field_path} must not be empty.")
    return normalized


def read_positive_int(
    source: Mapping[str, object],
    key: str,
    default: int | None = None,
    *,
    field_prefix: str | None = None,
) -> int:
    """mapping에서 positive int config 값을 읽는다."""

    value = source.get(key, default)
    field_path = _field_path(field_name=key, field_prefix=field_prefix)
    if value is None:
        raise ValueError(f"{field_path} is required.")
    if isinstance(value, bool):
        raise ValueError(f"{field_path} must not be bool.")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_path} must be positive.")
    return parsed


def read_optional_positive_int(
    source: Mapping[str, object],
    key: str,
    *,
    field_prefix: str | None = None,
) -> int | None:
    """mapping에서 optional positive int config 값을 읽는다."""

    value = source.get(key)
    if value is None:
        return None
    field_path = _field_path(field_name=key, field_prefix=field_prefix)
    if isinstance(value, bool):
        raise ValueError(f"{field_path} must be int.")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_path} must be positive.")
    return parsed


def read_float(
    source: Mapping[str, object] | None,
    key: str,
    default: float | None = None,
    *,
    field_prefix: str | None = None,
) -> float:
    """mapping에서 float config 값을 읽는다."""

    field_path = _field_path(field_name=key, field_prefix=field_prefix)
    value = default if source is None else source.get(key, default)
    if value is None:
        raise ValueError(f"{field_path} must be a number.")
    if isinstance(value, bool):
        if field_prefix is None:
            raise ValueError(f"{field_path} must not be bool.")
        raise ValueError(f"{field_path} must be a number.")
    return float(value)


def read_unit_interval_float(
    source: Mapping[str, object],
    key: str,
    default: float,
) -> float:
    """mapping에서 0..1 범위의 float config 값을 읽는다."""

    parsed = read_float(source, key, default)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{key} must be between 0 and 1.")
    return parsed


def read_bool(
    source: Mapping[str, object],
    key: str,
    default: bool,
) -> bool:
    """mapping에서 bool config 값을 읽는다."""

    value = source.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"{key} must be bool.")


def read_str_tuple(
    source: Mapping[str, object],
    key: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    """mapping에서 comma-separated string 또는 string sequence를 tuple로 읽는다."""

    value = source.get(key)
    if value is None:
        return default
    if isinstance(value, str):
        return _split_str_tuple(value)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(
            item for raw_item in value for item in _split_str_tuple(str(raw_item))
        )
    return _split_str_tuple(str(value))


def _split_str_tuple(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _field_path(*, field_name: str, field_prefix: str | None) -> str:
    if field_prefix is None:
        return field_name
    return f"{field_prefix}.{field_name}"
