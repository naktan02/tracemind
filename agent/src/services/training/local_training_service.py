"""로컬 pseudo-label 기반 update 생성 서비스."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Protocol
from uuid import uuid4

from agent.src.infrastructure.repositories.training_artifact_repository import (
    TrainingArtifactRepository,
)
from agent.src.services.training.privacy_guard_service import (
    DiagonalScaleClipOnlyPrivacyGuard,
    SharedAdapterPrivacyGuard,
)
from agent.src.services.training.pseudo_label_service import (
    PseudoLabelSelectionResult,
    PseudoLabelSelectionService,
)
from shared.src.contracts.adapter_contracts import (
    SharedAdapterUpdatePayload,
    VectorAdapterDeltaPayload,
)
from shared.src.domain.entities.artifacts.model_manifest import ModelManifest
from shared.src.domain.entities.inference.events import ScoredEvent
from shared.src.domain.entities.training.pseudo_label_candidate import (
    PseudoLabelCandidate,
)
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)
from shared.src.domain.entities.training.training_task import TrainingTask
from shared.src.domain.entities.training.training_update import TrainingUpdateEnvelope
from shared.src.domain.entities.training.vector_adapter_delta import (
    VectorAdapterDelta,
)


class SharedAdapterTrainingBackend(Protocol):
    """채택된 로컬 예시를 shared adapter update로 바꾸는 backend 인터페이스."""

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
class EmbeddedTrainingExample:
    """학습 후보가 된 로컬 scored event와 임베딩."""

    scored_event: ScoredEvent
    embedding: list[float]
    base_embedding: list[float] | None = None
    candidate: PseudoLabelCandidate | None = None


@dataclass(slots=True)
class DiagonalScaleHeuristicTrainingBackend:
    """결정적 통계 기반으로 diagonal scale adapter update를 만든다."""

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
        accepted_examples: tuple[EmbeddedTrainingExample, ...],
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


@dataclass(slots=True)
class LocalTrainingResult:
    """로컬 selection과 update 생성 결과."""

    selection_result: PseudoLabelSelectionResult
    update_envelope: TrainingUpdateEnvelope | None = None
    update_payload: SharedAdapterUpdate | None = None


@dataclass(slots=True)
class LocalTrainingRequest:
    """로컬 학습 실행 입력 묶음."""

    training_examples: (
        tuple[EmbeddedTrainingExample, ...] | list[EmbeddedTrainingExample]
    )
    training_task: TrainingTask
    model_manifest: ModelManifest
    created_at: datetime | None = None


@dataclass(slots=True)
class LocalTrainingService:
    """pseudo-label 선별과 update payload 생성을 조합한다."""

    selector: PseudoLabelSelectionService = field(
        default_factory=PseudoLabelSelectionService
    )
    repository: TrainingArtifactRepository = field(
        default_factory=TrainingArtifactRepository
    )
    backend: SharedAdapterTrainingBackend = field(
        default_factory=DiagonalScaleHeuristicTrainingBackend
    )
    privacy_guard: SharedAdapterPrivacyGuard = field(
        default_factory=DiagonalScaleClipOnlyPrivacyGuard
    )

    def run(self, request: LocalTrainingRequest) -> LocalTrainingResult:
        if (
            request.training_task.model_revision
            != request.model_manifest.model_revision
        ):
            raise ValueError("TrainingTask model_revision must match ModelManifest.")

        effective_created_at = request.created_at or datetime.now(timezone.utc)
        scored_events = [example.scored_event for example in request.training_examples]
        selection_result = self.selector.select(
            scored_events=scored_events,
            training_task=request.training_task,
        )
        accepted_by_event = {
            candidate.source_event_ref: candidate
            for candidate in selection_result.accepted_candidates
        }
        accepted_examples = tuple(
            replace(example, candidate=accepted_by_event[example.scored_event.query_id])
            for example in request.training_examples
            if example.scored_event.query_id in accepted_by_event
        )

        minimum_examples = request.training_task.min_required_examples or 1
        if len(accepted_examples) < minimum_examples:
            return LocalTrainingResult(selection_result=selection_result)

        update_payload = self.backend.build_update(
            training_task=request.training_task,
            model_manifest=request.model_manifest,
            accepted_examples=accepted_examples,
            created_at=effective_created_at,
        )
        protected_update = self.privacy_guard.protect(
            update=update_payload,
            training_task=request.training_task,
        )

        update_id = f"update_{request.training_task.round_id}_{uuid4().hex[:12]}"
        payload_path = self.repository.save_shared_adapter_update(
            update_id,
            self.backend.to_payload(protected_update.update),
        )
        update_envelope = TrainingUpdateEnvelope(
            schema_version="training_update_envelope.v1",
            update_id=update_id,
            round_id=request.training_task.round_id,
            task_id=request.training_task.task_id,
            model_id=request.model_manifest.model_id,
            base_model_revision=request.model_manifest.model_revision,
            training_scope=request.training_task.training_scope,
            payload_ref=str(payload_path),
            payload_format=self.backend.payload_format,
            example_count=len(accepted_examples),
            client_metrics={
                "accepted_ratio": selection_result.accepted_ratio,
                "mean_confidence": protected_update.update.mean_confidence,
                "mean_margin": protected_update.update.mean_margin or 0.0,
                "delta_l2_norm": protected_update.update.l2_norm(),
                "selected_examples": float(len(accepted_examples)),
            },
            created_at=effective_created_at,
            clipped=protected_update.clipped,
            dp_applied=protected_update.dp_applied,
        )
        return LocalTrainingResult(
            selection_result=selection_result,
            update_envelope=update_envelope,
            update_payload=protected_update.update,
        )

    def run_task(
        self,
        *,
        training_examples: tuple[EmbeddedTrainingExample, ...]
        | list[EmbeddedTrainingExample],
        training_task: TrainingTask,
        model_manifest: ModelManifest,
        created_at: datetime | None = None,
    ) -> LocalTrainingResult:
        return self.run(
            LocalTrainingRequest(
                training_examples=training_examples,
                training_task=training_task,
                model_manifest=model_manifest,
                created_at=created_at,
            )
        )
