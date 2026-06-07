"""로컬 pseudo-label 기반 update 생성 서비스."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime

from agent.src.infrastructure.repositories.training_artifact_repository import (
    TrainingArtifactRepository,
)
from agent.src.services.training.examples.models import (
    EmbeddedTrainingExample,
)
from agent.src.services.training.execution.local_update_executor import (
    LocalUpdateExecutionRequest,
    LocalUpdateExecutor,
)
from agent.src.services.training.execution.runtime_compatibility import (
    validate_local_training_runtime,
)
from agent.src.services.training.selection.pseudo_label_service import (
    PseudoLabelSelectionResult,
    PseudoLabelSelectionService,
)
from methods.adaptation.local_update_backend import (
    SharedAdapterTrainingBackend,
)
from methods.adaptation.privacy_guards.base import (
    SharedAdapterPrivacyGuard,
)
from methods.adaptation.privacy_guards.noop import (
    NoOpSharedAdapterPrivacyGuard,
)
from shared.src.contracts.adapter_contract_families.base import (
    SharedAdapterUpdatePayload,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import (
    TrainingTask,
    TrainingUpdateEnvelope,
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
    backend: SharedAdapterTrainingBackend | None = None
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
        analysis_events = [
            example.evidence_analysis_event for example in request.training_examples
        ]
        selection_result = self.selector.select(
            analysis_events=analysis_events,
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

        update_result = self._build_update_executor().execute(
            LocalUpdateExecutionRequest(
                training_task=request.training_task,
                model_manifest=request.model_manifest,
                accepted_examples=accepted_examples,
                selection_result=selection_result,
                created_at=effective_created_at,
                agent_id=request.agent_id,
            ),
            backend=backend,
            privacy_guard=privacy_guard,
        )
        return LocalTrainingResult(
            selection_result=selection_result,
            update_envelope=update_result.update_envelope,
            update_payload=update_result.update_payload,
        )

    def _resolve_backend(
        self,
        *,
        training_task: TrainingTask,
    ) -> SharedAdapterTrainingBackend:
        return self._build_update_executor().resolve_backend(
            training_task=training_task
        )

    def _resolve_privacy_guard(
        self,
        *,
        training_task: TrainingTask,
    ) -> SharedAdapterPrivacyGuard:
        return self._build_update_executor().resolve_privacy_guard(
            training_task=training_task
        )

    def _build_update_executor(self) -> LocalUpdateExecutor:
        return LocalUpdateExecutor(
            repository=self.repository,
            backend=self.backend,
            privacy_guard=self.privacy_guard,
            secure_update_codec=self.secure_update_codec,
        )

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
