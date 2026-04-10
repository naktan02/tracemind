"""공용 local training raw default 값 묶음.

contract/payload builder에 의존하지 않는 canonical 기본값만 둔다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from shared.src.config.diagonal_scale_defaults import (
    DEFAULT_DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_CONFIG,
)

TrainingDefaultScalar = str | int | float | bool


def _freeze_mapping(
    source: Mapping[str, TrainingDefaultScalar],
) -> Mapping[str, TrainingDefaultScalar]:
    return MappingProxyType(dict(source))


@dataclass(frozen=True, slots=True)
class TrainingTaskRuntimeDefaults:
    """round open/task 생성에 쓰는 top-level runtime 기본값."""

    local_epochs: int
    batch_size: int
    learning_rate: float
    max_steps: int
    min_required_examples: int | None = None
    gradient_clip_norm: float | None = None


PSEUDO_LABEL_SELF_TRAINING_V1_PROFILE_NAME = "pseudo_label_self_training.v1"

PSEUDO_LABEL_SELF_TRAINING_V1_OBJECTIVE_MAPPING = _freeze_mapping(
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
        "privacy_guard_name": "diagonal_scale_clip_only",
        **dict(
            DEFAULT_DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_CONFIG.to_scoped_mapping()
        ),
    }
)

PSEUDO_LABEL_SELF_TRAINING_V1_SELECTION_MAPPING = _freeze_mapping(
    {"max_examples": 128}
)

PSEUDO_LABEL_SELF_TRAINING_V1_SECURE_AGGREGATION_MAPPING = _freeze_mapping(
    {"required": False}
)

PSEUDO_LABEL_SELF_TRAINING_V1_TASK_RUNTIME_DEFAULTS = TrainingTaskRuntimeDefaults(
    local_epochs=1,
    batch_size=16,
    learning_rate=1e-4,
    max_steps=50,
    min_required_examples=None,
    gradient_clip_norm=None,
)

DEFAULT_TRAINING_OBJECTIVE_MAPPING = PSEUDO_LABEL_SELF_TRAINING_V1_OBJECTIVE_MAPPING
DEFAULT_TRAINING_SELECTION_MAPPING = PSEUDO_LABEL_SELF_TRAINING_V1_SELECTION_MAPPING
DEFAULT_TRAINING_SECURE_AGGREGATION_MAPPING = (
    PSEUDO_LABEL_SELF_TRAINING_V1_SECURE_AGGREGATION_MAPPING
)
DEFAULT_TRAINING_TASK_RUNTIME_DEFAULTS = (
    PSEUDO_LABEL_SELF_TRAINING_V1_TASK_RUNTIME_DEFAULTS
)


__all__ = [
    "DEFAULT_TRAINING_OBJECTIVE_MAPPING",
    "DEFAULT_TRAINING_SECURE_AGGREGATION_MAPPING",
    "DEFAULT_TRAINING_SELECTION_MAPPING",
    "DEFAULT_TRAINING_TASK_RUNTIME_DEFAULTS",
    "PSEUDO_LABEL_SELF_TRAINING_V1_OBJECTIVE_MAPPING",
    "PSEUDO_LABEL_SELF_TRAINING_V1_PROFILE_NAME",
    "PSEUDO_LABEL_SELF_TRAINING_V1_SECURE_AGGREGATION_MAPPING",
    "PSEUDO_LABEL_SELF_TRAINING_V1_SELECTION_MAPPING",
    "PSEUDO_LABEL_SELF_TRAINING_V1_TASK_RUNTIME_DEFAULTS",
    "TrainingDefaultScalar",
    "TrainingTaskRuntimeDefaults",
]
