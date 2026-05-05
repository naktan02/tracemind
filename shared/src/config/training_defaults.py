"""공용 local training 기본 profile과 builder.

계약 파일은 필드 의미와 validation을 소유하고,
이 모듈은 운영/실험에서 재사용하는 canonical default profile을 소유한다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from shared.src.config.training_default_values import (
    DEFAULT_TRAINING_OBJECTIVE_MAPPING,
    DEFAULT_TRAINING_SECURE_AGGREGATION_MAPPING,
    DEFAULT_TRAINING_SELECTION_MAPPING,
    DEFAULT_TRAINING_TASK_RUNTIME_DEFAULTS,
    PSEUDO_LABEL_SELF_TRAINING_V1_PROFILE_NAME,
    TrainingTaskRuntimeDefaults,
)
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
    task_runtime_defaults: TrainingTaskRuntimeDefaults

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
    def pseudo_label_algorithm_name(self) -> str:
        return self._require_str(
            self.objective_mapping,
            "pseudo_label_algorithm_name",
        )

    @property
    def training_backend_name(self) -> str:
        return self._require_str(
            self.objective_mapping,
            "training_backend_name",
        )

    @property
    def algorithm_profile_name(self) -> str:
        return self._require_str(
            self.objective_mapping,
            "algorithm_profile_name",
        )

    @property
    def example_generation_backend_name(self) -> str:
        return self._require_str(
            self.objective_mapping,
            "example_generation_backend_name",
        )

    @property
    def evidence_backend_name(self) -> str:
        return self._require_str(
            self.objective_mapping,
            "evidence_backend_name",
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
    def privacy_guard_name(self) -> str:
        return self._require_str(
            self.objective_mapping,
            "privacy_guard_name",
        )

    @property
    def max_examples(self) -> int | None:
        value = self.selection_mapping.get("max_examples")
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("Default max_examples must not be bool.")
        return int(value)

    @property
    def local_epochs(self) -> int:
        return self.task_runtime_defaults.local_epochs

    @property
    def batch_size(self) -> int:
        return self.task_runtime_defaults.batch_size

    @property
    def learning_rate(self) -> float:
        return self.task_runtime_defaults.learning_rate

    @property
    def max_steps(self) -> int:
        return self.task_runtime_defaults.max_steps

    @property
    def min_required_examples(self) -> int | None:
        return self.task_runtime_defaults.min_required_examples

    @property
    def gradient_clip_norm(self) -> float | None:
        return self.task_runtime_defaults.gradient_clip_norm

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
    profile_name=PSEUDO_LABEL_SELF_TRAINING_V1_PROFILE_NAME,
    objective_mapping=DEFAULT_TRAINING_OBJECTIVE_MAPPING,
    selection_mapping=DEFAULT_TRAINING_SELECTION_MAPPING,
    secure_aggregation_mapping=DEFAULT_TRAINING_SECURE_AGGREGATION_MAPPING,
    task_runtime_defaults=DEFAULT_TRAINING_TASK_RUNTIME_DEFAULTS,
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

    return DEFAULT_TRAINING_PROFILE.build_secure_aggregation_config(overrides=overrides)
