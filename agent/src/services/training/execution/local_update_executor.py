"""Local update execution port for accepted training examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from agent.src.infrastructure.repositories.training_artifact_repository import (
    TrainingArtifactRepository,
)
from agent.src.services.training.execution.privacy_guards.base import (
    SharedAdapterPrivacyGuard,
)
from agent.src.services.training.execution.privacy_guards.noop import (
    NoOpSharedAdapterPrivacyGuard,
)
from agent.src.services.training.execution.privacy_guards.registry import (
    build_shared_adapter_privacy_guard,
)
from agent.src.services.training.selection.pseudo_label_service import (
    PseudoLabelSelectionResult,
)
from methods.adaptation.local_update_backend import (
    AcceptedTrainingExample,
    SharedAdapterTrainingBackend,
)
from methods.adaptation.local_update_registry import (
    build_shared_adapter_training_backend,
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
from shared.src.services.secure_update_codec import (
    NoOpSecureUpdateCodec,
    SecureUpdateCodec,
)


@dataclass(frozen=True, slots=True)
class LocalUpdateExecutionRequest:
    """Accepted examples를 update submission으로 바꾸기 위한 입력."""

    training_task: TrainingTask
    model_manifest: ModelManifest
    accepted_examples: tuple[AcceptedTrainingExample, ...]
    selection_result: PseudoLabelSelectionResult
    created_at: datetime
    agent_id: str | None = None


@dataclass(frozen=True, slots=True)
class LocalUpdateExecutionResult:
    """Local update 실행 결과."""

    update_envelope: TrainingUpdateEnvelope
    update_payload: SharedAdapterUpdatePayload


@dataclass(slots=True)
class LocalUpdateExecutor:
    """선택된 method core를 agent-local submission으로 연결한다."""

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

    def resolve_backend(
        self,
        *,
        training_task: TrainingTask,
    ) -> SharedAdapterTrainingBackend:
        """TrainingTask가 선택한 local update backend를 resolve한다."""

        backend_name = training_task.objective_config.training_backend_name
        if self.backend is not None and backend_name == self.backend.backend_name:
            if self.backend.matches_objective_config(training_task.objective_config):
                return self.backend
        return build_shared_adapter_training_backend(
            backend_name,
            objective_config=training_task.objective_config,
        )

    def resolve_privacy_guard(
        self,
        *,
        training_task: TrainingTask,
    ) -> SharedAdapterPrivacyGuard:
        """TrainingTask가 선택한 privacy guard를 resolve한다."""

        guard_name = training_task.objective_config.privacy_guard_name
        if guard_name is None or guard_name == self.privacy_guard.guard_name:
            return self.privacy_guard
        return build_shared_adapter_privacy_guard(guard_name)

    def execute(
        self,
        request: LocalUpdateExecutionRequest,
        *,
        backend: SharedAdapterTrainingBackend | None = None,
        privacy_guard: SharedAdapterPrivacyGuard | None = None,
    ) -> LocalUpdateExecutionResult:
        """Accepted examples를 protected payload와 update envelope로 변환한다."""

        resolved_backend = backend or self.resolve_backend(
            training_task=request.training_task
        )
        resolved_privacy_guard = privacy_guard or self.resolve_privacy_guard(
            training_task=request.training_task
        )
        update = resolved_backend.build_update(
            training_task=request.training_task,
            model_manifest=request.model_manifest,
            accepted_examples=request.accepted_examples,
            created_at=request.created_at,
        )
        protected_update = resolved_privacy_guard.protect(
            update=update,
            training_task=request.training_task,
        )
        submission_payload = resolved_backend.to_payload(protected_update.update)

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
            payload_format=resolved_backend.payload_format,
            example_count=len(request.accepted_examples),
            client_metrics=self._build_client_metrics(
                backend=resolved_backend,
                update=protected_update.update,
                selection_result=request.selection_result,
                accepted_example_count=len(request.accepted_examples),
            ),
            created_at=request.created_at,
            clipped=protected_update.clipped,
            dp_applied=protected_update.dp_applied,
            agent_id=request.agent_id,
        )
        encoded_envelope = self.secure_update_codec.encode_for_submission(
            envelope=update_envelope,
            training_task=request.training_task,
        )
        return LocalUpdateExecutionResult(
            update_envelope=encoded_envelope,
            update_payload=submission_payload,
        )

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
