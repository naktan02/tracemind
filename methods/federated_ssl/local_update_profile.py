"""FL SSL local update profile typed module."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from methods.common.config_reading import (
    read_optional_positive_int,
    read_str,
    set_normalized_str,
    validate_allowed_keys,
)
from shared.src.contracts.training_contracts import (
    TrainingConfigScalar,
    TrainingObjectiveConfig,
)


@dataclass(frozen=True, slots=True)
class LocalUpdateProfile:
    """agent local update 조합을 하나의 typed profile로 해석한다."""

    algorithm_profile_name: str
    training_scope: str
    training_backend_name: str
    example_generation_backend_name: str
    validation_scorer_backend_name: str
    validation_score_policy_name: str | None
    validation_score_top_k: int | None
    privacy_guard_name: str

    def __post_init__(self) -> None:
        for field_name in (
            "algorithm_profile_name",
            "training_scope",
            "training_backend_name",
            "example_generation_backend_name",
            "validation_scorer_backend_name",
            "privacy_guard_name",
        ):
            set_normalized_str(
                self,
                field_name,
                getattr(self, field_name),
                field_prefix="local_update_profile",
            )
        if self.validation_score_policy_name is not None:
            object.__setattr__(
                self,
                "validation_score_policy_name",
                _normalize_optional_str(
                    self.validation_score_policy_name,
                    field_name="validation_score_policy_name",
                ),
            )
        if self.validation_score_top_k is not None and self.validation_score_top_k <= 0:
            raise ValueError(
                "local_update_profile.validation_score_top_k must be positive."
            )

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, object],
    ) -> "LocalUpdateProfile":
        """Hydra local_update_profile mapping을 typed profile로 해석한다."""

        validate_allowed_keys(
            source,
            allowed_keys=_LOCAL_UPDATE_PROFILE_KEYS,
            config_name="local_update_profile",
        )
        return cls(
            algorithm_profile_name=read_str(
                source,
                "algorithm_profile_name",
                field_prefix="local_update_profile",
            ),
            training_scope=read_str(
                source,
                "training_scope",
                field_prefix="local_update_profile",
            ),
            training_backend_name=read_str(
                source,
                "training_backend_name",
                field_prefix="local_update_profile",
            ),
            example_generation_backend_name=read_str(
                source,
                "example_generation_backend_name",
                field_prefix="local_update_profile",
            ),
            validation_scorer_backend_name=read_str(
                source,
                "validation_scorer_backend_name",
                field_prefix="local_update_profile",
            ),
            validation_score_policy_name=_read_optional_str(
                source,
                "validation_score_policy_name",
                field_prefix="local_update_profile",
            ),
            validation_score_top_k=read_optional_positive_int(
                source,
                "validation_score_top_k",
                field_prefix="local_update_profile",
            ),
            privacy_guard_name=read_str(
                source,
                "privacy_guard_name",
                field_prefix="local_update_profile",
            ),
        )

    def to_training_objective_mapping(self) -> dict[str, TrainingConfigScalar]:
        """profile이 소유하는 TrainingObjectiveConfig field를 만든다."""

        result: dict[str, TrainingConfigScalar] = {
            "algorithm_profile_name": self.algorithm_profile_name,
            "training_backend_name": self.training_backend_name,
            "example_generation_backend_name": self.example_generation_backend_name,
            "privacy_guard_name": self.privacy_guard_name,
        }
        return result


def require_training_objective_matches_local_update_profile(
    *,
    objective_config: TrainingObjectiveConfig,
    local_update_profile: LocalUpdateProfile,
) -> None:
    """training objective가 local update profile과 drift되지 않았는지 검증한다."""

    expected = TrainingObjectiveConfig.from_mapping(
        local_update_profile.to_training_objective_mapping()
    ).to_mapping()
    actual = objective_config.to_mapping()
    mismatches = {
        key: (expected_value, actual.get(key))
        for key, expected_value in expected.items()
        if actual.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(
            f"training_task.objective must match local_update_profile: {mismatches}."
        )


_LOCAL_UPDATE_PROFILE_KEYS = frozenset(
    {
        "algorithm_profile_name",
        "training_scope",
        "training_backend_name",
        "example_generation_backend_name",
        "validation_scorer_backend_name",
        "validation_score_policy_name",
        "validation_score_top_k",
        "privacy_guard_name",
    }
)


def _read_optional_str(
    source: Mapping[str, object],
    key: str,
    *,
    field_prefix: str,
) -> str | None:
    value = source.get(key)
    if value is None:
        return None
    return _normalize_optional_str(value, field_name=f"{field_prefix}.{key}")


def _normalize_optional_str(value: object, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty when provided.")
    return normalized
