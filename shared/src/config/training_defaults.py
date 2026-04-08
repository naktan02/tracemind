"""공용 local training 기본 profile과 builder.

계약 파일은 필드 의미와 validation을 소유하고,
이 모듈은 운영/실험에서 재사용하는 canonical default profile을 소유한다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from shared.src.contracts.training_contracts import (
    SecureAggregationConfig,
    TrainingConfigScalar,
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
)


@dataclass(frozen=True, slots=True)
class TrainingDefaultsProfile:
    """운영/실험이 공유하는 local training 기본 profile 묶음."""

    profile_name: str
    objective_mapping: Mapping[str, TrainingConfigScalar]
    selection_mapping: Mapping[str, TrainingConfigScalar]
    secure_aggregation_mapping: Mapping[str, TrainingConfigScalar]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "objective_mapping",
            MappingProxyType(dict(self.objective_mapping)),
        )
        object.__setattr__(
            self,
            "selection_mapping",
            MappingProxyType(dict(self.selection_mapping)),
        )
        object.__setattr__(
            self,
            "secure_aggregation_mapping",
            MappingProxyType(dict(self.secure_aggregation_mapping)),
        )

    @property
    def acceptance_policy_name(self) -> str:
        return self._require_str(
            self.objective_mapping,
            "acceptance_policy_name",
        )

    @property
    def scorer_backend_name(self) -> str:
        return self._require_str(
            self.objective_mapping,
            "scorer_backend_name",
        )

    @property
    def score_policy_name(self) -> str:
        return self._require_str(
            self.objective_mapping,
            "score_policy_name",
        )

    @property
    def confidence_threshold(self) -> float:
        return self._require_float(
            self.objective_mapping,
            "confidence_threshold",
        )

    @property
    def margin_threshold(self) -> float:
        return self._require_float(
            self.objective_mapping,
            "margin_threshold",
        )

    @property
    def max_examples(self) -> int | None:
        value = self.selection_mapping.get("max_examples")
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("Default max_examples must not be bool.")
        return int(value)

    def build_objective_config(
        self,
        *,
        overrides: Mapping[str, TrainingConfigScalar] | None = None,
    ) -> TrainingObjectiveConfig:
        source = dict(self.objective_mapping)
        if overrides is not None:
            source.update(dict(overrides))
        return TrainingObjectiveConfig.from_mapping(source)

    def build_selection_policy(
        self,
        *,
        overrides: Mapping[str, TrainingConfigScalar] | None = None,
    ) -> TrainingSelectionPolicy:
        source = dict(self.selection_mapping)
        if overrides is not None:
            source.update(dict(overrides))
        return TrainingSelectionPolicy.from_mapping(source)

    def build_secure_aggregation_config(
        self,
        *,
        overrides: Mapping[str, TrainingConfigScalar] | None = None,
    ) -> SecureAggregationConfig:
        source = dict(self.secure_aggregation_mapping)
        if overrides is not None:
            source.update(dict(overrides))
        return SecureAggregationConfig.from_mapping(source)

    @staticmethod
    def _require_str(
        source: Mapping[str, TrainingConfigScalar],
        key: str,
    ) -> str:
        value = source.get(key)
        if value is None:
            raise ValueError(f"Default training profile is missing key: {key}")
        return str(value)

    @staticmethod
    def _require_float(
        source: Mapping[str, TrainingConfigScalar],
        key: str,
    ) -> float:
        value = source.get(key)
        if value is None:
            raise ValueError(f"Default training profile is missing key: {key}")
        if isinstance(value, bool):
            raise ValueError(f"Default training profile key must not be bool: {key}")
        return float(value)


PSEUDO_LABEL_SELF_TRAINING_V1_PROFILE = TrainingDefaultsProfile(
    profile_name="pseudo_label_self_training.v1",
    objective_mapping={
        "training_backend_name": "diagonal_scale_heuristic",
        "confidence_threshold": 0.6,
        "margin_threshold": 0.02,
        "example_generation_backend_name": "prototype_rescore",
        "scorer_backend_name": "prototype_similarity",
        "score_policy_name": "max_cosine",
        "acceptance_policy_name": "top1_margin_threshold",
        "privacy_guard_name": "diagonal_scale_clip_only",
    },
    selection_mapping={"max_examples": 128},
    secure_aggregation_mapping={"required": False},
)

DEFAULT_TRAINING_PROFILE = PSEUDO_LABEL_SELF_TRAINING_V1_PROFILE


def build_default_training_objective_config(
    *,
    overrides: Mapping[str, TrainingConfigScalar] | None = None,
) -> TrainingObjectiveConfig:
    """공용 기본 local training objective config를 조립한다."""

    return DEFAULT_TRAINING_PROFILE.build_objective_config(overrides=overrides)


def build_default_training_selection_policy(
    *,
    overrides: Mapping[str, TrainingConfigScalar] | None = None,
) -> TrainingSelectionPolicy:
    """공용 기본 local training selection policy를 조립한다."""

    return DEFAULT_TRAINING_PROFILE.build_selection_policy(overrides=overrides)


def build_default_secure_aggregation_config(
    *,
    overrides: Mapping[str, TrainingConfigScalar] | None = None,
) -> SecureAggregationConfig:
    """공용 기본 secure aggregation config를 조립한다."""

    return DEFAULT_TRAINING_PROFILE.build_secure_aggregation_config(
        overrides=overrides
    )
