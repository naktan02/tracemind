"""Classifier-head family 기본값과 config helper."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

ClassifierHeadConfigScalar = str | int | float | bool

TRAINING_BACKEND_EXTRA_SCOPE = "training_backend"
CLASSIFIER_HEAD_FIXMATCH_TRAINING_BACKEND_EXTRA_KEYS = (
    "fixmatch.consistency_loss_weight",
    "fixmatch.step_scale_multiplier",
    "fixmatch.bias_learning_rate_multiplier",
)


@dataclass(frozen=True, slots=True)
class ClassifierHeadFixMatchTrainingBackendConfig:
    """Classifier-head FixMatch consistency backend 기본 설정."""

    consistency_loss_weight: float = 1.0
    step_scale_multiplier: float = 1.0
    bias_learning_rate_multiplier: float = 1.0

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, ClassifierHeadConfigScalar] | None,
    ) -> "ClassifierHeadFixMatchTrainingBackendConfig":
        if source is None:
            return cls()

        allowed_keys = {
            "consistency_loss_weight",
            "step_scale_multiplier",
            "bias_learning_rate_multiplier",
        }
        unsupported = sorted(set(source) - allowed_keys)
        if unsupported:
            raise ValueError(
                "Unsupported classifier-head FixMatch backend config key(s): "
                f"{unsupported}"
            )

        return cls(
            consistency_loss_weight=float(
                source.get("consistency_loss_weight", 1.0)
            ),
            step_scale_multiplier=float(source.get("step_scale_multiplier", 1.0)),
            bias_learning_rate_multiplier=float(
                source.get("bias_learning_rate_multiplier", 1.0)
            ),
        )

    def to_scoped_mapping(self) -> dict[str, float]:
        return {
            "training_backend.consistency_loss_weight": (
                self.consistency_loss_weight
            ),
            "training_backend.step_scale_multiplier": self.step_scale_multiplier,
            "training_backend.bias_learning_rate_multiplier": (
                self.bias_learning_rate_multiplier
            ),
        }


DEFAULT_CLASSIFIER_HEAD_FIXMATCH_TRAINING_BACKEND_CONFIG = (
    ClassifierHeadFixMatchTrainingBackendConfig()
)


__all__ = [
    "CLASSIFIER_HEAD_FIXMATCH_TRAINING_BACKEND_EXTRA_KEYS",
    "ClassifierHeadConfigScalar",
    "ClassifierHeadFixMatchTrainingBackendConfig",
    "DEFAULT_CLASSIFIER_HEAD_FIXMATCH_TRAINING_BACKEND_CONFIG",
    "TRAINING_BACKEND_EXTRA_SCOPE",
]
