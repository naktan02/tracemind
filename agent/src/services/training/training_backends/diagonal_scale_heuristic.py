"""Diagonal-scale heuristic training backend."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from shared.src.config.adapter_family_metadata import DIAGONAL_SCALE_FAMILY_METADATA
from shared.src.config.diagonal_scale_defaults import (
    DEFAULT_DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_CONFIG,
    DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_EXTRA_KEYS,
    TRAINING_BACKEND_EXTRA_SCOPE,
    DiagonalScaleHeuristicTrainingBackendConfig,
)
from shared.src.contracts.adapter_contracts import (
    VECTOR_ADAPTER_DELTA_V1,
    SharedAdapterUpdatePayload,
    VectorAdapterDelta,
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


def build_diagonal_scale_heuristic_training_backend_config(
    objective_config: TrainingObjectiveConfig | None,
) -> DiagonalScaleHeuristicTrainingBackendConfig:
    """objective config에서 diagonal-scale heuristic backend 설정을 읽는다."""

    extras = (
        {}
        if objective_config is None
        else objective_config.get_component_extras(
            TRAINING_BACKEND_EXTRA_SCOPE,
            legacy_keys=DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_EXTRA_KEYS,
        )
    )
    return DiagonalScaleHeuristicTrainingBackendConfig.from_mapping(extras)


@dataclass(slots=True)
class DiagonalScaleHeuristicTrainingBackend:
    """결정적 통계 기반으로 diagonal scale adapter update를 만든다."""

    backend_name: str = "diagonal_scale_heuristic"
    payload_format: str = DIAGONAL_SCALE_FAMILY_METADATA.canonical_update_payload_format
    adapter_kind: str = DIAGONAL_SCALE_FAMILY_METADATA.adapter_kind
    config: DiagonalScaleHeuristicTrainingBackendConfig = (
        DEFAULT_DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_CONFIG
    )

    @classmethod
    def from_objective_config(
        cls,
        objective_config: TrainingObjectiveConfig | None,
    ) -> "DiagonalScaleHeuristicTrainingBackend":
        return cls(
            config=build_diagonal_scale_heuristic_training_backend_config(
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
    ) -> VectorAdapterDelta:
        if not accepted_examples:
            raise ValueError("accepted_examples must not be empty.")

        embedding_dim = len(accepted_examples[0].update_embedding)
        weighted_sums = [0.0] * embedding_dim
        label_counts: dict[str, int] = defaultdict(int)
        total_weight = 0.0
        total_confidence = 0.0
        total_margin = 0.0

        for example in accepted_examples:
            if len(example.update_embedding) != embedding_dim:
                raise ValueError(
                    "All accepted embeddings must share the same dimension."
                )

            if example.candidate is None:
                raise ValueError(
                    "Accepted example must carry a pseudo-label candidate."
                )

            weight = max(example.candidate.sample_weight, 1e-6)
            total_weight += weight
            total_confidence += example.candidate.confidence
            total_margin += example.candidate.margin
            label_counts[example.candidate.label] += 1
            for index, value in enumerate(example.update_embedding):
                weighted_sums[index] += weight * float(value)

        effective_scale = max(
            self.config.minimum_effective_scale,
            training_task.learning_rate
            * max(training_task.local_epochs, 1)
            * max(training_task.max_steps, 1)
            * self.config.delta_scale_multiplier,
        )
        dimension_deltas = [
            max(
                -self.config.max_abs_delta,
                min(
                    self.config.max_abs_delta,
                    (value / total_weight) * effective_scale,
                ),
            )
            for value in weighted_sums
        ]

        return VectorAdapterDelta(
            schema_version=VECTOR_ADAPTER_DELTA_V1,
            model_id=model_manifest.model_id,
            base_model_revision=model_manifest.model_revision,
            training_scope=training_task.training_scope,
            dimension_deltas=dimension_deltas,
            example_count=len(accepted_examples),
            mean_confidence=total_confidence / len(accepted_examples),
            mean_margin=total_margin / len(accepted_examples),
            label_counts=dict(sorted(label_counts.items())),
            created_at=created_at,
            adapter_kind=self.adapter_kind,
        )

    def to_payload(self, update: SharedAdapterUpdate) -> SharedAdapterUpdatePayload:
        if not isinstance(update, VectorAdapterDelta):
            raise TypeError(
                "DiagonalScaleHeuristicTrainingBackend expects VectorAdapterDelta "
                f"for payload conversion, got {type(update)!r}."
            )
        return update

    def build_client_metrics(self, update: SharedAdapterUpdate) -> dict[str, float]:
        if not isinstance(update, VectorAdapterDelta):
            raise TypeError(
                "DiagonalScaleHeuristicTrainingBackend expects VectorAdapterDelta "
                f"for metric extraction, got {type(update)!r}."
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
        return self.config == build_diagonal_scale_heuristic_training_backend_config(
            objective_config
        )


SyntheticVectorAdapterTrainingBackend = DiagonalScaleHeuristicTrainingBackend


__all__ = [
    "build_diagonal_scale_heuristic_training_backend_config",
    "DiagonalScaleHeuristicTrainingBackend",
    "SyntheticVectorAdapterTrainingBackend",
]
