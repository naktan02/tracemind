"""로컬 학습 update backend 구현과 resolver."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from shared.src.contracts.adapter_contracts import (
    SharedAdapterUpdatePayload,
    VectorAdapterDeltaPayload,
)
from shared.src.domain.entities.artifacts.model_manifest import ModelManifest
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)
from shared.src.domain.entities.training.training_task import TrainingTask
from shared.src.domain.entities.training.vector_adapter_delta import (
    VectorAdapterDelta,
)


class SharedAdapterTrainingBackend(Protocol):
    """채택된 로컬 예시를 shared adapter update로 바꾸는 backend 인터페이스."""

    backend_name: str
    payload_format: str
    adapter_kind: str

    def build_update(
        self,
        *,
        training_task: TrainingTask,
        model_manifest: ModelManifest,
        accepted_examples: tuple["EmbeddedTrainingExample", ...],
        created_at: datetime,
    ) -> SharedAdapterUpdate:
        """accepted local examples를 기반으로 shared adapter update를 계산한다."""

    def to_payload(self, update: SharedAdapterUpdate) -> SharedAdapterUpdatePayload:
        """저장 가능한 payload 형식으로 변환한다."""


TrainingBackend = SharedAdapterTrainingBackend


@dataclass(slots=True)
class DiagonalScaleHeuristicTrainingBackend:
    """결정적 통계 기반으로 diagonal scale adapter update를 만든다."""

    backend_name: str = "diagonal_scale_heuristic"
    payload_format: str = "diagonal_scale_update"
    adapter_kind: str = "diagonal_scale"
    delta_scale_multiplier: float = 10.0
    max_abs_delta: float = 0.05
    minimum_effective_scale: float = 1e-4

    def build_update(
        self,
        *,
        training_task: TrainingTask,
        model_manifest: ModelManifest,
        accepted_examples: tuple["EmbeddedTrainingExample", ...],
        created_at: datetime,
    ) -> VectorAdapterDelta:
        if not accepted_examples:
            raise ValueError("accepted_examples must not be empty.")

        embedding_dim = len(accepted_examples[0].embedding)
        weighted_sums = [0.0] * embedding_dim
        label_counts: dict[str, int] = defaultdict(int)
        total_weight = 0.0
        total_confidence = 0.0
        total_margin = 0.0

        for example in accepted_examples:
            if len(example.embedding) != embedding_dim:
                raise ValueError(
                    "All accepted embeddings must share the same dimension."
                )

            if example.candidate is None:
                raise ValueError("Accepted example must carry a pseudo-label candidate.")

            weight = max(example.candidate.confidence, 1e-6)
            total_weight += weight
            total_confidence += example.candidate.confidence
            total_margin += example.candidate.margin
            label_counts[example.candidate.label] += 1
            for index, value in enumerate(example.embedding):
                weighted_sums[index] += weight * float(value)

        effective_scale = max(
            self.minimum_effective_scale,
            training_task.learning_rate
            * max(training_task.local_epochs, 1)
            * max(training_task.max_steps, 1)
            * self.delta_scale_multiplier,
        )
        dimension_deltas = [
            max(
                -self.max_abs_delta,
                min(
                    self.max_abs_delta,
                    (value / total_weight) * effective_scale,
                ),
            )
            for value in weighted_sums
        ]

        return VectorAdapterDelta(
            schema_version="vector_adapter_delta.v1",
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
        return VectorAdapterDeltaPayload(
            schema_version=update.schema_version,
            adapter_kind=update.adapter_kind,
            model_id=update.model_id,
            base_model_revision=update.base_model_revision,
            training_scope=update.training_scope,
            dimension_deltas=update.dimension_deltas,
            example_count=update.example_count,
            mean_confidence=update.mean_confidence,
            created_at=update.created_at,
            mean_margin=update.mean_margin,
            label_counts=update.label_counts,
        )


SyntheticVectorAdapterTrainingBackend = DiagonalScaleHeuristicTrainingBackend


def build_shared_adapter_training_backend(
    backend_name: str,
) -> SharedAdapterTrainingBackend:
    """backend 이름으로 로컬 학습 backend를 생성한다."""

    normalized_name = backend_name.strip().lower()
    if normalized_name in {
        "diagonal_scale_heuristic",
        "synthetic_vector_adapter",
    }:
        return DiagonalScaleHeuristicTrainingBackend(backend_name=normalized_name)
    raise ValueError(f"Unsupported local training backend: {backend_name}.")
