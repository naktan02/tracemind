"""로컬 pseudo-label 기반 update 생성 서비스."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from uuid import uuid4

from agent.src.infrastructure.repositories.training_artifact_repository import (
    TrainingArtifactRepository,
)
from agent.src.services.training.backends.training.base import (
    SharedAdapterTrainingBackend,
)
from agent.src.services.training.backends.training.diagonal_scale_heuristic import (
    DiagonalScaleHeuristicTrainingBackend,
)
from agent.src.services.training.backends.training.registry import (
    build_shared_adapter_training_backend,
)
from agent.src.services.training.examples.models import (
    EmbeddedTrainingExample,
)
from agent.src.services.training.execution.privacy_guard_service import (
    NoOpSharedAdapterPrivacyGuard,
    SharedAdapterPrivacyGuard,
    build_shared_adapter_privacy_guard,
)
from agent.src.services.training.execution.runtime_compatibility import (
    validate_local_training_runtime,
)
from agent.src.services.training.selection.pseudo_label_service import (
    PseudoLabelSelectionResult,
    PseudoLabelSelectionService,
)
from shared.src.contracts.adapter_contracts import SharedAdapterUpdatePayload
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import (
    ClientMetricKeys,
    TrainingTask,
    TrainingUpdateEnvelope,
)
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)
from shared.src.domain.services.clock import Clock, SystemUtcClock
from shared.src.services.secure_update_codec import (
    NoOpSecureUpdateCodec,
    SecureUpdateCodec,
)


@dataclass(slots=True)
class LocalTrainingResult:
    """로컬 selection과 update 생성 결과."""

    selection_result: PseudoLabelSelectionResult
    update_envelope: TrainingUpdateEnvelope | None = None
    update_payload: SharedAdapterUpdatePayload | None = None


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
        default_factory=NoOpSharedAdapterPrivacyGuard
    )
    secure_update_codec: SecureUpdateCodec = field(
        default_factory=NoOpSecureUpdateCodec
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
        acceptance_policy_name = (
            request.training_task.objective_config.acceptance_policy_name
        )
        default_acceptance_policy = (
            self.selector.default_policy
            if acceptance_policy_name is None
            or acceptance_policy_name == self.selector.default_policy.policy_name
            else None
        )
        validate_local_training_runtime(
            request.training_task,
            default_acceptance_policy_name=(
                self.selector.default_acceptance_policy_name
            ),
            default_privacy_guard_name=self.privacy_guard.guard_name,
            training_backend=backend,
            acceptance_policy=default_acceptance_policy,
            privacy_guard=privacy_guard,
        )
        scored_events = [
            example.evidence_scored_event for example in request.training_examples
        ]
        selection_result = self.selector.select(
            scored_events=scored_events,
            training_task=request.training_task,
        )
        accepted_by_event = {
            candidate.source_event_ref: candidate
            for candidate in selection_result.accepted_candidates
        }
        accepted_examples = tuple(
            replace(example, candidate=accepted_by_event[example.selection_key])
            for example in request.training_examples
            if example.selection_key in accepted_by_event
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
        submission_payload = backend.to_payload(protected_update.update)

        update_id = f"update_{request.training_task.round_id}_{uuid4().hex[:12]}"
        self.repository.save_shared_adapter_update(
            update_id,
            submission_payload,
        )
        update_envelope = TrainingUpdateEnvelope(
            schema_version="training_update_envelope.v1",
            update_id=update_id,
            round_id=request.training_task.round_id,
            task_id=request.training_task.task_id,
            model_id=request.model_manifest.model_id,
            base_model_revision=request.model_manifest.model_revision,
            training_scope=request.training_task.training_scope,
            payload_ref=f"client-submission::{update_id}",
            payload_format=backend.payload_format,
            example_count=len(accepted_examples),
            client_metrics=self._build_client_metrics(
                backend=backend,
                update=protected_update.update,
                selection_result=selection_result,
                accepted_example_count=len(accepted_examples),
            ),
            created_at=effective_created_at,
            clipped=protected_update.clipped,
            dp_applied=protected_update.dp_applied,
            agent_id=request.agent_id,
        )
        update_envelope = self.secure_update_codec.encode_for_submission(
            envelope=update_envelope,
            training_task=request.training_task,
        )
        return LocalTrainingResult(
            selection_result=selection_result,
            update_envelope=update_envelope,
            update_payload=submission_payload,
        )

    def _resolve_backend(
        self,
        *,
        training_task: TrainingTask,
    ) -> SharedAdapterTrainingBackend:
        backend_name = training_task.objective_config.training_backend_name
        if backend_name == self.backend.backend_name and (
            self.backend.matches_objective_config(training_task.objective_config)
        ):
            return self.backend
        return build_shared_adapter_training_backend(
            backend_name,
            objective_config=training_task.objective_config,
        )

    def _resolve_privacy_guard(
        self,
        *,
        training_task: TrainingTask,
    ) -> SharedAdapterPrivacyGuard:
        guard_name = training_task.objective_config.privacy_guard_name
        if guard_name is None or guard_name == self.privacy_guard.guard_name:
            return self.privacy_guard
        return build_shared_adapter_privacy_guard(guard_name)

    @staticmethod
    def _build_client_metrics(
        *,
        backend: SharedAdapterTrainingBackend,
        update: SharedAdapterUpdate,
        selection_result: PseudoLabelSelectionResult,
        accepted_example_count: int,
    ) -> dict[str, float]:
        client_metrics = {
            ClientMetricKeys.ACCEPTED_RATIO: selection_result.accepted_ratio,
            ClientMetricKeys.SELECTED_EXAMPLES: float(accepted_example_count),
        }
        client_metrics.update(backend.build_client_metrics(update))
        return client_metrics

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
