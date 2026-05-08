"""FL SSL experiment profile typed metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FederatedSslExperimentProfile:
    """Hydra experiment_profile compose preset metadata."""

    name: str
    method_name: str
    local_update_profile_name: str
    round_runtime_profile_name: str
    adapter_family_name: str
    aggregation_backend_name: str
    description: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "name",
            "method_name",
            "local_update_profile_name",
            "round_runtime_profile_name",
            "adapter_family_name",
            "aggregation_backend_name",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_non_empty(
                    getattr(self, field_name),
                    field_name=f"experiment_profile.{field_name}",
                ),
            )
        if self.description is not None:
            object.__setattr__(
                self,
                "description",
                _normalize_non_empty(
                    self.description,
                    field_name="experiment_profile.description",
                ),
            )

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, object],
    ) -> "FederatedSslExperimentProfile":
        """Hydra fl_profile mapping을 compose preset metadata로 해석한다."""

        unknown_keys = sorted(set(source) - _EXPERIMENT_PROFILE_KEYS)
        if unknown_keys:
            raise ValueError(f"Unsupported experiment_profile key(s): {unknown_keys}.")
        return cls(
            name=_str_value(source, "name"),
            method_name=_str_value(source, "method_name"),
            local_update_profile_name=_str_value(
                source,
                "local_update_profile_name",
            ),
            round_runtime_profile_name=_str_value(
                source,
                "round_runtime_profile_name",
            ),
            adapter_family_name=_str_value(source, "adapter_family_name"),
            aggregation_backend_name=_str_value(source, "aggregation_backend_name"),
            description=(
                None
                if source.get("description") is None
                else str(source["description"])
            ),
        )


def _normalize_non_empty(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    return normalized


def _str_value(source: Mapping[str, object], key: str) -> str:
    value = source.get(key)
    if value is None:
        raise ValueError(f"experiment_profile.{key} is required.")
    return str(value)


_EXPERIMENT_PROFILE_KEYS = frozenset(
    {
        "name",
        "method_name",
        "local_update_profile_name",
        "round_runtime_profile_name",
        "adapter_family_name",
        "aggregation_backend_name",
        "description",
    }
)
