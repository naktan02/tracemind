"""FL SSL local update profile typed module."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from methods.common.config_reading import (
    read_float,
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
    confidence_threshold: float
    margin_threshold: float
    example_generation_backend_name: str
    evidence_backend_name: str
    scorer_backend_name: str
    score_policy_name: str
    score_top_k: int | None
    validation_scorer_backend_name: str
    validation_score_policy_name: str | None
    validation_score_top_k: int | None
    pseudo_label_algorithm_name: str
    acceptance_policy_name: str
    privacy_guard_name: str
    evidence_backend_temperature: float

    def __post_init__(self) -> None:
        for field_name in (
            "algorithm_profile_name",
            "training_scope",
            "training_backend_name",
            "example_generation_backend_name",
            "evidence_backend_name",
            "scorer_backend_name",
            "score_policy_name",
            "validation_scorer_backend_name",
            "pseudo_label_algorithm_name",
            "acceptance_policy_name",
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
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("local_update_profile.confidence_threshold invalid.")
        if self.margin_threshold < 0.0:
            raise ValueError("local_update_profile.margin_threshold invalid.")
        if self.score_top_k is not None and self.score_top_k <= 0:
            raise ValueError("local_update_profile.score_top_k must be positive.")
        if self.validation_score_top_k is not None and self.validation_score_top_k <= 0:
            raise ValueError(
                "local_update_profile.validation_score_top_k must be positive."
            )
        if self.evidence_backend_temperature <= 0.0:
            raise ValueError(
                "local_update_profile.evidence_backend_temperature must be positive."
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
            confidence_threshold=read_float(
                source,
                "confidence_threshold",
                field_prefix="local_update_profile",
            ),
            margin_threshold=read_float(
                source,
                "margin_threshold",
                field_prefix="local_update_profile",
            ),
            example_generation_backend_name=read_str(
                source,
                "example_generation_backend_name",
                field_prefix="local_update_profile",
            ),
            evidence_backend_name=read_str(
                source,
                "evidence_backend_name",
                field_prefix="local_update_profile",
            ),
            scorer_backend_name=read_str(
                source,
                "scorer_backend_name",
                field_prefix="local_update_profile",
            ),
            score_policy_name=read_str(
                source,
                "score_policy_name",
                field_prefix="local_update_profile",
            ),
            score_top_k=read_optional_positive_int(
                source,
                "score_top_k",
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
            pseudo_label_algorithm_name=read_str(
                source,
                "pseudo_label_algorithm_name",
                field_prefix="local_update_profile",
            ),
            acceptance_policy_name=read_str(
                source,
                "acceptance_policy_name",
                field_prefix="local_update_profile",
            ),
            privacy_guard_name=read_str(
                source,
                "privacy_guard_name",
                field_prefix="local_update_profile",
            ),
            evidence_backend_temperature=read_float(
                source,
                "evidence_backend_temperature",
                field_prefix="local_update_profile",
            ),
        )

    def to_training_objective_mapping(self) -> dict[str, TrainingConfigScalar]:
        """profile이 소유하는 TrainingObjectiveConfig field를 만든다."""

        result: dict[str, TrainingConfigScalar] = {
            "algorithm_profile_name": self.algorithm_profile_name,
            "training_backend_name": self.training_backend_name,
            "confidence_threshold": self.confidence_threshold,
            "margin_threshold": self.margin_threshold,
            "example_generation_backend_name": self.example_generation_backend_name,
            "evidence_backend_name": self.evidence_backend_name,
            "scorer_backend_name": self.scorer_backend_name,
            "score_policy_name": self.score_policy_name,
            "pseudo_label_algorithm_name": self.pseudo_label_algorithm_name,
            "acceptance_policy_name": self.acceptance_policy_name,
            "privacy_guard_name": self.privacy_guard_name,
            "evidence_backend.temperature": self.evidence_backend_temperature,
        }
        if self.score_top_k is not None:
            result["score_top_k"] = self.score_top_k
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
        "confidence_threshold",
        "margin_threshold",
        "example_generation_backend_name",
        "evidence_backend_name",
        "scorer_backend_name",
        "score_policy_name",
        "score_top_k",
        "validation_scorer_backend_name",
        "validation_score_policy_name",
        "validation_score_top_k",
        "pseudo_label_algorithm_name",
        "acceptance_policy_name",
        "privacy_guard_name",
        "evidence_backend_temperature",
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
