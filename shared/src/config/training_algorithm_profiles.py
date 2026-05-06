"""논문/알고리즘 단위 objective profile registry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from shared.src.config.training_default_values import DEFAULT_TRAINING_OBJECTIVE_MAPPING

TrainingAlgorithmScalar = str | int | float | bool

PROTOTYPE_PSEUDO_LABEL_V1_PROFILE_NAME = "prototype_pseudo_label_v1"
PROTOTYPE_TOP1_CONFIDENCE_V1_PROFILE_NAME = "prototype_top1_confidence_v1"
LORA_PSEUDO_LABEL_V1_PROFILE_NAME = "lora_pseudo_label_v1"


def _freeze_mapping(
    source: Mapping[str, TrainingAlgorithmScalar],
) -> Mapping[str, TrainingAlgorithmScalar]:
    return MappingProxyType(dict(source))


@dataclass(frozen=True, slots=True)
class TrainingAlgorithmProfile:
    """실험에서 선택 가능한 objective 조합 preset."""

    profile_name: str
    objective_mapping: Mapping[str, TrainingAlgorithmScalar]
    description: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "objective_mapping",
            MappingProxyType(dict(self.objective_mapping)),
        )


PROTOTYPE_PSEUDO_LABEL_V1_OBJECTIVE_MAPPING = _freeze_mapping(
    DEFAULT_TRAINING_OBJECTIVE_MAPPING
)

PROTOTYPE_TOP1_CONFIDENCE_V1_OBJECTIVE_MAPPING = _freeze_mapping(
    {
        **dict(DEFAULT_TRAINING_OBJECTIVE_MAPPING),
        "algorithm_profile_name": PROTOTYPE_TOP1_CONFIDENCE_V1_PROFILE_NAME,
        "margin_threshold": 0.0,
        "pseudo_label_algorithm_name": "top1_confidence_only",
        "acceptance_policy_name": "top1_confidence_only",
    }
)

LORA_PSEUDO_LABEL_V1_OBJECTIVE_MAPPING = _freeze_mapping(
    {
        "algorithm_profile_name": LORA_PSEUDO_LABEL_V1_PROFILE_NAME,
        "training_backend_name": "lora_classifier_trainer",
        "confidence_threshold": 0.6,
        "margin_threshold": 0.02,
        "example_generation_backend_name": "prototype_rescore",
        "evidence_backend_name": "prototype_similarity_evidence",
        "scorer_backend_name": "prototype_similarity",
        "score_policy_name": "max_cosine",
        "pseudo_label_algorithm_name": "top1_margin_threshold",
        "acceptance_policy_name": "top1_margin_threshold",
        "privacy_guard_name": "noop",
    }
)

PROTOTYPE_PSEUDO_LABEL_V1_PROFILE = TrainingAlgorithmProfile(
    profile_name=PROTOTYPE_PSEUDO_LABEL_V1_PROFILE_NAME,
    objective_mapping=PROTOTYPE_PSEUDO_LABEL_V1_OBJECTIVE_MAPPING,
    description=(
        "현재 production baseline에 해당하는 prototype similarity + "
        "top1 margin threshold pseudo-label self-training 조합."
    ),
)

PROTOTYPE_TOP1_CONFIDENCE_V1_PROFILE = TrainingAlgorithmProfile(
    profile_name=PROTOTYPE_TOP1_CONFIDENCE_V1_PROFILE_NAME,
    objective_mapping=PROTOTYPE_TOP1_CONFIDENCE_V1_OBJECTIVE_MAPPING,
    description=(
        "Prototype similarity evidence는 유지하고 acceptance만 "
        "top1 confidence gating으로 바꾼 비교용 조합."
    ),
)

LORA_PSEUDO_LABEL_V1_PROFILE = TrainingAlgorithmProfile(
    profile_name=LORA_PSEUDO_LABEL_V1_PROFILE_NAME,
    objective_mapping=LORA_PSEUDO_LABEL_V1_OBJECTIVE_MAPPING,
    description=(
        "LoRA-classifier trainer와 기존 prototype pseudo-label selection을 "
        "조합하는 FL simulation profile."
    ),
)

_TRAINING_ALGORITHM_PROFILE_REGISTRY: dict[str, TrainingAlgorithmProfile] = {
    PROTOTYPE_PSEUDO_LABEL_V1_PROFILE_NAME: PROTOTYPE_PSEUDO_LABEL_V1_PROFILE,
    PROTOTYPE_TOP1_CONFIDENCE_V1_PROFILE_NAME: (PROTOTYPE_TOP1_CONFIDENCE_V1_PROFILE),
    LORA_PSEUDO_LABEL_V1_PROFILE_NAME: LORA_PSEUDO_LABEL_V1_PROFILE,
}


def get_training_algorithm_profile(
    profile_name: str,
) -> TrainingAlgorithmProfile:
    """이름으로 objective profile을 조회한다."""

    normalized_name = profile_name.strip().lower()
    profile = _TRAINING_ALGORITHM_PROFILE_REGISTRY.get(normalized_name)
    if profile is not None:
        return profile
    raise ValueError(f"Unsupported training algorithm profile: {profile_name}.")


def expand_training_objective_mapping(
    source: Mapping[str, TrainingAlgorithmScalar] | None,
) -> dict[str, TrainingAlgorithmScalar]:
    """algorithm_profile_name이 있으면 canonical objective mapping으로 펼친다."""

    if source is None:
        return {}
    raw = dict(source)
    profile_name_value = raw.get("algorithm_profile_name")
    if profile_name_value is None:
        return raw
    profile = get_training_algorithm_profile(str(profile_name_value))
    merged = dict(profile.objective_mapping)
    merged.update(raw)
    return merged
