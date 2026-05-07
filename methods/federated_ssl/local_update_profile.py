"""FL SSL local update profile typed module."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

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
            "pseudo_label_algorithm_name",
            "acceptance_policy_name",
            "privacy_guard_name",
        ):
            _set_normalized_str(self, field_name, getattr(self, field_name))
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("local_update_profile.confidence_threshold invalid.")
        if self.margin_threshold < 0.0:
            raise ValueError("local_update_profile.margin_threshold invalid.")
        if self.score_top_k is not None and self.score_top_k <= 0:
            raise ValueError("local_update_profile.score_top_k must be positive.")
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

        unknown_keys = sorted(set(source) - _LOCAL_UPDATE_PROFILE_KEYS)
        if unknown_keys:
            raise ValueError(
                "Unsupported local_update_profile key(s): " f"{unknown_keys}."
            )
        return cls(
            algorithm_profile_name=_str_value(source, "algorithm_profile_name"),
            training_scope=_str_value(source, "training_scope"),
            training_backend_name=_str_value(source, "training_backend_name"),
            confidence_threshold=_float_value(source, "confidence_threshold"),
            margin_threshold=_float_value(source, "margin_threshold"),
            example_generation_backend_name=_str_value(
                source,
                "example_generation_backend_name",
            ),
            evidence_backend_name=_str_value(source, "evidence_backend_name"),
            scorer_backend_name=_str_value(source, "scorer_backend_name"),
            score_policy_name=_str_value(source, "score_policy_name"),
            score_top_k=_optional_positive_int_value(source, "score_top_k"),
            pseudo_label_algorithm_name=_str_value(
                source,
                "pseudo_label_algorithm_name",
            ),
            acceptance_policy_name=_str_value(source, "acceptance_policy_name"),
            privacy_guard_name=_str_value(source, "privacy_guard_name"),
            evidence_backend_temperature=_float_value(
                source,
                "evidence_backend_temperature",
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
            "training_task.objective must match local_update_profile: "
            f"{mismatches}."
        )


def _set_normalized_str(
    instance: LocalUpdateProfile,
    field_name: str,
    value: str,
) -> None:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"local_update_profile.{field_name} must not be empty.")
    object.__setattr__(instance, field_name, normalized)


def _str_value(source: Mapping[str, object], key: str) -> str:
    value = source.get(key)
    if value is None:
        raise ValueError(f"local_update_profile.{key} is required.")
    return str(value)


def _float_value(source: Mapping[str, object], key: str) -> float:
    value = source.get(key)
    if value is None or isinstance(value, bool):
        raise ValueError(f"local_update_profile.{key} must be a number.")
    return float(value)


def _optional_positive_int_value(
    source: Mapping[str, object],
    key: str,
) -> int | None:
    value = source.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"local_update_profile.{key} must be int.")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"local_update_profile.{key} must be positive.")
    return parsed


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
        "pseudo_label_algorithm_name",
        "acceptance_policy_name",
        "privacy_guard_name",
        "evidence_backend_temperature",
    }
)
