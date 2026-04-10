"""Classifier-head FixMatch-style consistency training backend."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from shared.src.config.adapter_family_metadata import CLASSIFIER_HEAD_FAMILY_METADATA
from shared.src.config.classifier_head_defaults import (
    CLASSIFIER_HEAD_FIXMATCH_TRAINING_BACKEND_EXTRA_KEYS,
    DEFAULT_CLASSIFIER_HEAD_FIXMATCH_TRAINING_BACKEND_CONFIG,
    TRAINING_BACKEND_EXTRA_SCOPE,
    ClassifierHeadFixMatchTrainingBackendConfig,
)
from shared.src.contracts.adapter_contracts import (
    ClassifierHeadDelta,
    SharedAdapterUpdatePayload,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import (
    ClientMetricKeys,
    TrainingObjectiveConfig,
    TrainingTask,
)
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)

from .base import AcceptedTrainingExample


def build_classifier_head_fixmatch_training_backend_config(
    objective_config: TrainingObjectiveConfig | None,
) -> ClassifierHeadFixMatchTrainingBackendConfig:
    """objective config에서 classifier-head FixMatch backend 설정을 읽는다."""

    extras = (
        {}
        if objective_config is None
        else objective_config.get_component_extras(
            TRAINING_BACKEND_EXTRA_SCOPE,
            legacy_keys=CLASSIFIER_HEAD_FIXMATCH_TRAINING_BACKEND_EXTRA_KEYS,
        )
    )
    allowed_keys = {
        "consistency_loss_weight",
        "step_scale_multiplier",
        "bias_learning_rate_multiplier",
    }
    return ClassifierHeadFixMatchTrainingBackendConfig.from_mapping(
        {
            key: value
            for key, value in extras.items()
            if key in allowed_keys
        }
    )


@dataclass(slots=True)
class ClassifierHeadFixMatchConsistencyTrainingBackend:
    """weak/strong pair의 strong logits로 classifier-head delta를 만든다."""

    backend_name: str = "classifier_head_fixmatch_consistency"
    payload_format: str = CLASSIFIER_HEAD_FAMILY_METADATA.canonical_update_payload_format
    adapter_kind: str = CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind
    config: ClassifierHeadFixMatchTrainingBackendConfig = (
        DEFAULT_CLASSIFIER_HEAD_FIXMATCH_TRAINING_BACKEND_CONFIG
    )

    @classmethod
    def from_objective_config(
        cls,
        objective_config: TrainingObjectiveConfig | None,
    ) -> "ClassifierHeadFixMatchConsistencyTrainingBackend":
        return cls(
            config=build_classifier_head_fixmatch_training_backend_config(
                objective_config
            )
        )

    def build_update(
        self,
        *,
        training_task: TrainingTask,
        model_manifest: ModelManifest,
        accepted_examples: tuple[AcceptedTrainingExample, ...],
        created_at: datetime,
    ) -> ClassifierHeadDelta:
        if not accepted_examples:
            raise ValueError("accepted_examples must not be empty.")

        logits_labels = tuple(
            sorted(accepted_examples[0].update_scored_event.category_scores)
        )
        if not logits_labels:
            raise ValueError("Classifier-head FixMatch backend requires logits.")

        embedding_dim = len(accepted_examples[0].update_embedding)
        label_weight_deltas = {
            label: [0.0] * embedding_dim for label in logits_labels
        }
        label_bias_deltas = {label: 0.0 for label in logits_labels}
        label_counts: dict[str, int] = defaultdict(int)
        total_weight = 0.0
        total_confidence = 0.0
        total_margin = 0.0

        effective_scale = (
            training_task.learning_rate
            * max(training_task.local_epochs, 1)
            * max(training_task.max_steps, 1)
            * self.config.step_scale_multiplier
            * self.config.consistency_loss_weight
        )

        per_example_data: list[tuple[AcceptedTrainingExample, dict[str, float], float]] = []
        for example in accepted_examples:
            logits = example.update_scored_event.category_scores
            if tuple(sorted(logits)) != logits_labels:
                raise ValueError("All logits must share the same labels.")
            if len(example.update_embedding) != embedding_dim:
                raise ValueError(
                    "All accepted embeddings must share the same dimension."
                )
            if example.candidate is None:
                raise ValueError(
                    "Accepted example must carry a pseudo-label candidate."
                )
            if example.candidate.label not in logits:
                raise ValueError(
                    "Pseudo-label must exist in classifier logits labels."
                )
            sample_weight = max(example.candidate.sample_weight, 1e-6)
            total_weight += sample_weight
            total_confidence += example.candidate.confidence
            total_margin += example.candidate.margin
            label_counts[example.candidate.label] += 1
            per_example_data.append(
                (
                    example,
                    _softmax_distribution(logits),
                    sample_weight,
                )
            )

        for example, probabilities, sample_weight in per_example_data:
            assert example.candidate is not None
            scaled_weight = effective_scale * (sample_weight / total_weight)
            target_label = example.candidate.label
            for label in logits_labels:
                residual = (
                    1.0 if label == target_label else 0.0
                ) - probabilities[label]
                for index, value in enumerate(example.update_embedding):
                    label_weight_deltas[label][index] += (
                        scaled_weight * residual * float(value)
                    )
                label_bias_deltas[label] += (
                    scaled_weight
                    * residual
                    * self.config.bias_learning_rate_multiplier
                )

        return ClassifierHeadDelta(
            schema_version="classifier_head_delta.v1",
            model_id=model_manifest.model_id,
            base_model_revision=model_manifest.model_revision,
            training_scope=training_task.training_scope,
            label_weight_deltas=label_weight_deltas,
            label_bias_deltas=label_bias_deltas,
            example_count=len(accepted_examples),
            mean_confidence=total_confidence / len(accepted_examples),
            mean_margin=total_margin / len(accepted_examples),
            label_counts=dict(sorted(label_counts.items())),
            created_at=created_at,
            adapter_kind=self.adapter_kind,
        )

    def to_payload(self, update: SharedAdapterUpdate) -> SharedAdapterUpdatePayload:
        if not isinstance(update, ClassifierHeadDelta):
            raise TypeError(
                "ClassifierHeadFixMatchConsistencyTrainingBackend expects "
                f"ClassifierHeadDelta, got {type(update)!r}."
            )
        return update

    def build_client_metrics(self, update: SharedAdapterUpdate) -> dict[str, float]:
        if not isinstance(update, ClassifierHeadDelta):
            raise TypeError(
                "ClassifierHeadFixMatchConsistencyTrainingBackend expects "
                f"ClassifierHeadDelta, got {type(update)!r}."
            )
        return {
            ClientMetricKeys.MEAN_CONFIDENCE: update.mean_confidence,
            ClientMetricKeys.MEAN_MARGIN: update.mean_margin or 0.0,
            ClientMetricKeys.DELTA_L2_NORM: update.l2_norm(),
        }

    def matches_objective_config(
        self,
        objective_config: TrainingObjectiveConfig | None,
    ) -> bool:
        return self.config == build_classifier_head_fixmatch_training_backend_config(
            objective_config
        )


def _softmax_distribution(logits: dict[str, float]) -> dict[str, float]:
    max_logit = max(float(value) for value in logits.values())
    exp_values = {
        label: math.exp(float(value) - max_logit)
        for label, value in logits.items()
    }
    denominator = sum(exp_values.values())
    return {
        label: value / denominator for label, value in sorted(exp_values.items())
    }


__all__ = [
    "build_classifier_head_fixmatch_training_backend_config",
    "ClassifierHeadFixMatchConsistencyTrainingBackend",
]
