"""Local update execution port for accepted training examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from agent.src.infrastructure.repositories.training_artifact_repository import (
    TrainingArtifactRepository,
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
from methods.adaptation.privacy_guards.base import (
    SharedAdapterPrivacyGuard,
)
from methods.adaptation.privacy_guards.noop import (
    NoOpSharedAdapterPrivacyGuard,
)
from methods.adaptation.privacy_guards.registry import (
    build_shared_adapter_privacy_guard,
)
from shared.src.contracts.adapter_contract_families.base import (
    SharedAdapterUpdatePayload,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import (
    TrainingTask,
    TrainingUpdateEnvelope,
)
from shared.src.services.secure_update_codec import (
    NoOpSecureUpdateCodec,
    SecureUpdateCodec,
)

_PRIVATE_UPDATE_METADATA_FIELDS = ("mean_confidence", "mean_margin")


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
        server_visible_payload = _server_visible_update_payload(submission_payload)

        update_id = f"update_{request.training_task.round_id}_{uuid4().hex[:12]}"
        self.repository.save_shared_adapter_update(
            update_id,
            server_visible_payload,
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
            example_count=server_visible_payload.example_count,
            client_metrics={},
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
            update_payload=server_visible_payload,
        )


def _server_visible_update_payload(
    payload: SharedAdapterUpdatePayload,
) -> SharedAdapterUpdatePayload:
    updates: dict[str, object] = {
        "example_count": _server_visible_example_count(payload.example_count)
    }
    for field_name in _PRIVATE_UPDATE_METADATA_FIELDS:
        if hasattr(payload, field_name):
            updates[field_name] = None
    return payload.model_copy(update=updates)


def _server_visible_example_count(raw_count: int) -> int:
    return 0 if raw_count <= 0 else 1
