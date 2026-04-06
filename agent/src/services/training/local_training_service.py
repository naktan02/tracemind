"""로컬 pseudo-label 기반 update 생성 서비스."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from uuid import uuid4

from agent.src.infrastructure.repositories.training_artifact_repository import (
    TrainingArtifactRepository,
)
from agent.src.services.training.privacy_guard_service import (
    DiagonalScaleClipOnlyPrivacyGuard,
    SharedAdapterPrivacyGuard,
    build_shared_adapter_privacy_guard,
)
from agent.src.services.training.pseudo_label_service import (
    PseudoLabelSelectionResult,
    PseudoLabelSelectionService,
)
from agent.src.services.training.training_backends import (
    DiagonalScaleHeuristicTrainingBackend,
    SharedAdapterTrainingBackend,
    build_shared_adapter_training_backend,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import (
    ClientMetricKeys,
    TrainingTask,
    TrainingUpdateEnvelope,
)
from shared.src.domain.entities.inference.events import ScoredEvent
from shared.src.domain.entities.training.pseudo_label_candidate import (
    PseudoLabelCandidate,
)
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)
from shared.src.domain.services.clock import Clock, SystemUtcClock


@dataclass(slots=True)
class EmbeddedTrainingExample:
    """학습 후보가 된 로컬 scored event와 임베딩."""

    scored_event: ScoredEvent
    embedding: list[float]
    base_embedding: list[float] | None = None
    candidate: PseudoLabelCandidate | None = None


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
    agent_id: str | None = None  # pseudonymous UUID, None이면 익명


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
    clock: Clock = field(default_factory=SystemUtcClock)

    def run(self, request: LocalTrainingRequest) -> LocalTrainingResult:
        if (
            request.training_task.model_revision
            != request.model_manifest.model_revision
        ):
            raise ValueError("TrainingTask model_revision must match ModelManifest.")

        effective_created_at = request.created_at or self.clock.now()
        backend = self._resolve_backend(training_task=request.training_task)
        privacy_guard = self._resolve_privacy_guard(training_task=request.training_task)
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

        update_payload = backend.build_update(
            training_task=request.training_task,
            model_manifest=request.model_manifest,
            accepted_examples=accepted_examples,
            created_at=effective_created_at,
        )
        protected_update = privacy_guard.protect(
            update=update_payload,
            training_task=request.training_task,
        )

        update_id = f"update_{request.training_task.round_id}_{uuid4().hex[:12]}"
        payload_path = self.repository.save_shared_adapter_update(
            update_id,
            backend.to_payload(protected_update.update),
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
            payload_format=backend.payload_format,
            example_count=len(accepted_examples),
            client_metrics={
                ClientMetricKeys.ACCEPTED_RATIO: selection_result.accepted_ratio,
                ClientMetricKeys.MEAN_CONFIDENCE: (
                    protected_update.update.mean_confidence
                ),
                ClientMetricKeys.MEAN_MARGIN: (
                    protected_update.update.mean_margin or 0.0
                ),
                ClientMetricKeys.DELTA_L2_NORM: protected_update.update.l2_norm(),
                ClientMetricKeys.SELECTED_EXAMPLES: float(len(accepted_examples)),
            },
            created_at=effective_created_at,
            clipped=protected_update.clipped,
            dp_applied=protected_update.dp_applied,
            agent_id=request.agent_id,
        )
        return LocalTrainingResult(
            selection_result=selection_result,
            update_envelope=update_envelope,
            update_payload=protected_update.update,
        )

    def _resolve_backend(
        self,
        *,
        training_task: TrainingTask,
    ) -> SharedAdapterTrainingBackend:
        backend_name = training_task.objective_config.loss
        if backend_name == self.backend.backend_name:
            return self.backend
        return build_shared_adapter_training_backend(backend_name)

    def _resolve_privacy_guard(
        self,
        *,
        training_task: TrainingTask,
    ) -> SharedAdapterPrivacyGuard:
        guard_name = training_task.objective_config.privacy_guard_name
        if guard_name is None or guard_name == self.privacy_guard.guard_name:
            return self.privacy_guard
        return build_shared_adapter_privacy_guard(guard_name)

    def run_task(
        self,
        *,
        training_examples: tuple[EmbeddedTrainingExample, ...]
        | list[EmbeddedTrainingExample],
        training_task: TrainingTask,
        model_manifest: ModelManifest,
        created_at: datetime | None = None,
        agent_id: str | None = None,
    ) -> LocalTrainingResult:
        return self.run(
            LocalTrainingRequest(
                training_examples=training_examples,
                training_task=training_task,
                model_manifest=model_manifest,
                created_at=created_at,
                agent_id=agent_id,
            )
        )
