"""FL SSL experiment profile typed metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from methods.common.config_reading import (
    normalize_non_empty_str,
    read_str,
    validate_allowed_keys,
)


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
                normalize_non_empty_str(
                    getattr(self, field_name),
                    field_name=f"experiment_profile.{field_name}",
                ),
            )
        if self.description is not None:
            object.__setattr__(
                self,
                "description",
                normalize_non_empty_str(
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

        validate_allowed_keys(
            source,
            allowed_keys=_EXPERIMENT_PROFILE_KEYS,
            config_name="experiment_profile",
        )
        return cls(
            name=read_str(
                source,
                "name",
                field_prefix="experiment_profile",
            ),
            method_name=read_str(
                source,
                "method_name",
                field_prefix="experiment_profile",
            ),
            local_update_profile_name=read_str(
                source,
                "local_update_profile_name",
                field_prefix="experiment_profile",
            ),
            round_runtime_profile_name=read_str(
                source,
                "round_runtime_profile_name",
                field_prefix="experiment_profile",
            ),
            adapter_family_name=read_str(
                source,
                "adapter_family_name",
                field_prefix="experiment_profile",
            ),
            aggregation_backend_name=read_str(
                source,
                "aggregation_backend_name",
                field_prefix="experiment_profile",
            ),
            description=(
                None
                if source.get("description") is None
                else str(source["description"])
            ),
        )


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
