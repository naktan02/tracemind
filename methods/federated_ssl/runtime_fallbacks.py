"""FL SSL runtime fallback profile.

이 모듈은 API/runtime 요청이 명시 training 값을 주지 않았을 때만 쓰는
compatibility fallback을 소유한다. 실험 실행값의 source of truth는 Hydra
`conf/strategy_axes/fl/*`에 남긴다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from methods.adaptation.diagonal_scale.config import (
    DEFAULT_DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_CONFIG,
)
from shared.src.contracts.training_contracts import (
    SecureAggregationConfig,
    TrainingConfigScalar,
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
)


def _freeze_mapping(
    source: Mapping[str, TrainingConfigScalar],
) -> Mapping[str, TrainingConfigScalar]:
    return MappingProxyType(dict(source))


@dataclass(frozen=True, slots=True)
class RuntimeTrainingTaskDefaults:
    """round open/task 생성에 쓰는 top-level runtime fallback 값."""

    local_epochs: int
    batch_size: int
    learning_rate: float
    max_steps: int
    min_required_examples: int | None = None
    gradient_clip_norm: float | None = None


@dataclass(frozen=True, slots=True)
class RuntimeFallbackTrainingProfile:
    """명시 config가 없는 legacy/runtime 요청을 위한 fallback profile."""

    profile_name: str
    objective_mapping: Mapping[str, TrainingConfigScalar]
    selection_mapping: Mapping[str, TrainingConfigScalar]
    secure_aggregation_mapping: Mapping[str, TrainingConfigScalar]
    task_runtime_defaults: RuntimeTrainingTaskDefaults

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
            raise ValueError("Runtime fallback max_examples must not be bool.")
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
            raise ValueError(f"Runtime fallback training profile is missing: {key}")
        return str(value)

    @staticmethod
    def _require_float(
        source: Mapping[str, TrainingConfigScalar],
        key: str,
    ) -> float:
        value = source.get(key)
        if value is None:
            raise ValueError(f"Runtime fallback training profile is missing: {key}")
        if isinstance(value, bool):
            raise ValueError(f"Runtime fallback key must not be bool: {key}")
        return float(value)


PSEUDO_LABEL_SELF_TRAINING_V1_RUNTIME_FALLBACK_NAME = "pseudo_label_self_training.v1"

RUNTIME_FALLBACK_TRAINING_OBJECTIVE_MAPPING = _freeze_mapping(
    {
        "algorithm_profile_name": "prototype_pseudo_label_v1",
        "training_backend_name": "diagonal_scale_heuristic",
        "confidence_threshold": 0.6,
        "margin_threshold": 0.02,
        "example_generation_backend_name": "prototype_rescore",
        "evidence_backend_name": "prototype_similarity_evidence",
        "scorer_backend_name": "prototype_similarity",
        "score_policy_name": "max_cosine",
        "acceptance_policy_name": "top1_margin_threshold",
        "pseudo_label_algorithm_name": "top1_margin_threshold",
        "privacy_guard_name": "diagonal_scale_clip_only",
        **dict(
            DEFAULT_DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_CONFIG.to_scoped_mapping()
        ),
    }
)

RUNTIME_FALLBACK_TRAINING_SELECTION_MAPPING = _freeze_mapping({"max_examples": 128})

RUNTIME_FALLBACK_SECURE_AGGREGATION_MAPPING = _freeze_mapping({"required": False})

RUNTIME_FALLBACK_TRAINING_TASK_DEFAULTS = RuntimeTrainingTaskDefaults(
    local_epochs=1,
    batch_size=16,
    learning_rate=1e-4,
    max_steps=50,
    min_required_examples=None,
    gradient_clip_norm=None,
)

PSEUDO_LABEL_SELF_TRAINING_V1_RUNTIME_FALLBACK = RuntimeFallbackTrainingProfile(
    profile_name=PSEUDO_LABEL_SELF_TRAINING_V1_RUNTIME_FALLBACK_NAME,
    objective_mapping=RUNTIME_FALLBACK_TRAINING_OBJECTIVE_MAPPING,
    selection_mapping=RUNTIME_FALLBACK_TRAINING_SELECTION_MAPPING,
    secure_aggregation_mapping=RUNTIME_FALLBACK_SECURE_AGGREGATION_MAPPING,
    task_runtime_defaults=RUNTIME_FALLBACK_TRAINING_TASK_DEFAULTS,
)

RUNTIME_FALLBACK_TRAINING_PROFILE = PSEUDO_LABEL_SELF_TRAINING_V1_RUNTIME_FALLBACK


def build_runtime_fallback_training_objective_config(
    *,
    overrides: Mapping[str, TrainingConfigScalar] | None = None,
) -> TrainingObjectiveConfig:
    """명시 objective가 없는 runtime 요청용 fallback config를 조립한다."""

    return RUNTIME_FALLBACK_TRAINING_PROFILE.build_objective_config(overrides=overrides)


def build_runtime_fallback_training_selection_policy(
    *,
    overrides: Mapping[str, TrainingConfigScalar] | None = None,
) -> TrainingSelectionPolicy:
    """명시 selection policy가 없는 runtime 요청용 fallback을 조립한다."""

    return RUNTIME_FALLBACK_TRAINING_PROFILE.build_selection_policy(overrides=overrides)


def build_runtime_fallback_secure_aggregation_config(
    *,
    overrides: Mapping[str, TrainingConfigScalar] | None = None,
) -> SecureAggregationConfig:
    """명시 secure aggregation config가 없는 runtime 요청용 fallback을 조립한다."""

    return RUNTIME_FALLBACK_TRAINING_PROFILE.build_secure_aggregation_config(
        overrides=overrides
    )
